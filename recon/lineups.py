"""On-floor state machine and minutes reconciliation from the event log.

This is the key *minutes gate* of the reconstruction pipeline. Working from each
period's starting five (``recon_period_starters``), it walks that period's actions
in chronological order, maintaining the exact 5-man on-floor set for each team,
and applying every substitution as a swap (remove outgoing, add incoming). Each
maximal interval during which both fives stay constant is a *stint*; summing a
player's stint durations gives their reconstructed on-floor seconds, which must
reconcile to ``stg_box_player.minutes_seconds``.

Design notes
------------
* **Order, not clock, is authoritative.** Several events can share a clock value
  (a made basket and the substitution that follows it can both read "PT6M00.00S").
  The PBP's own ``order`` (action sequence number) is the chronological key; the
  clock is used only to measure elapsed time via ``recon.clock``.
* **Substitution waves collapse.** When multiple swaps happen at the same
  game-time, the intermediate lineups have zero duration and produce no stint —
  only distinct-time boundaries open a new stint.
* **The 5-man invariant is enforced.** A substitution whose outgoing player is
  not currently on the floor raises ``OnFloorError``; any swap that would leave a
  team with other than five players raises ``LineupSizeError``. These are the
  data-integrity asserts that make a clean reconciliation meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from recon.clock import (
    cumulative_elapsed,
    period_length_seconds,
    period_start_cumulative,
)

SUBSTITUTION = "Substitution"
LINEUP_SIZE = 5


class OnFloorError(ValueError):
    """A substitution removed a player who was not on the floor."""


class LineupSizeError(ValueError):
    """A team's on-floor set was not exactly five players."""


@dataclass(frozen=True)
class Action:
    """One normalized play-by-play event.

    ``order`` is the PBP's action sequence number — the authoritative
    chronological key, since several events can share a clock value. For a
    substitution, ``sub_out``/``sub_in`` carry the outgoing/incoming person_id and
    ``team_id`` says whose lineup changes; for every other action they are None.
    ``person_id`` is the acting player on non-substitution events (used only to
    back-infer period starters).
    """

    order: int
    period: int
    clock: str
    team_id: int | None = None
    action_type: str = ""
    person_id: int | None = None
    sub_out: int | None = None
    sub_in: int | None = None

    @property
    def is_substitution(self) -> bool:
        return self.action_type == SUBSTITUTION


@dataclass(frozen=True)
class Stint:
    """One team's on-floor interval during which its five stayed constant.

    Two rows are emitted per interval (home and away), each carrying both fives
    for matchup context but keyed on its own ``team_id`` / ``lineup_id``.
    """

    game_id: str
    period: int
    team_id: int
    lineup_id: str
    home_five: tuple[int, ...]
    away_five: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class TaggedAction:
    """An action annotated with the lineup in effect at the moment it occurred."""

    action: Action
    home_five: tuple[int, ...]
    away_five: tuple[int, ...]
    lineup_id: str  # combined "home|away" identifier


@dataclass(frozen=True)
class ReconDiff:
    """A per-player mismatch between reconstructed and box-score seconds."""

    person_id: int
    reconstructed_seconds: float
    box_seconds: float
    delta_seconds: float


@dataclass(frozen=True)
class LineupReconstruction:
    game_id: str
    stints: tuple[Stint, ...]
    tagged_actions: tuple[TaggedAction, ...]
    player_seconds: Mapping[int, float]


def lineup_id(five: Iterable[int]) -> str:
    """Stable, content-based id for a five: sorted person_ids joined by '-'."""
    return "-".join(str(pid) for pid in sorted(five))


def _combined_lineup_id(home: Iterable[int], away: Iterable[int]) -> str:
    return f"{lineup_id(home)}|{lineup_id(away)}"


def _require_five(five: Iterable[int], period: int, team_id: int, when: str) -> None:
    members = set(five)
    if len(members) != LINEUP_SIZE:
        raise LineupSizeError(
            f"team {team_id} has {len(members)} players on the floor "
            f"({sorted(members)}) {when} in period {period}; expected {LINEUP_SIZE}"
        )


def reconstruct_lineups(
    game_id: str,
    home_team_id: int,
    away_team_id: int,
    period_starters: Mapping[tuple[int, int], frozenset[int]],
    actions: Sequence[Action],
) -> LineupReconstruction:
    """Walk the event log and reconstruct stints and per-player on-floor seconds.

    ``period_starters`` maps ``(period, team_id)`` to the frozenset of five
    players on the floor at that period's tip-off. ``actions`` is the full event
    log; it is grouped by period and ordered by ``Action.order`` internally.
    """
    periods = sorted({a.period for a in actions})
    stints: list[Stint] = []
    tagged: list[TaggedAction] = []
    player_seconds: dict[int, float] = defaultdict(float)

    for period in periods:
        home_on = set(_starters(period_starters, period, home_team_id))
        away_on = set(_starters(period_starters, period, away_team_id))
        _require_five(home_on, period, home_team_id, "at tip-off")
        _require_five(away_on, period, away_team_id, "at tip-off")

        period_start = period_start_cumulative(period)
        period_end = period_start + period_length_seconds(period)
        stint_start = float(period_start)

        period_actions = sorted(
            (a for a in actions if a.period == period), key=lambda a: a.order
        )
        for action in period_actions:
            tagged.append(
                TaggedAction(
                    action=action,
                    home_five=tuple(sorted(home_on)),
                    away_five=tuple(sorted(away_on)),
                    lineup_id=_combined_lineup_id(home_on, away_on),
                )
            )
            if not action.is_substitution:
                continue

            change_time = cumulative_elapsed(action.period, action.clock)
            # A real interval elapsed under the current five: close its stint
            # before the swap. Same-time waves (change_time == stint_start) skip
            # this and just apply, collapsing zero-duration intermediate lineups.
            if change_time > stint_start:
                _emit_interval(
                    stints,
                    player_seconds,
                    game_id,
                    period,
                    home_team_id,
                    away_team_id,
                    home_on,
                    away_on,
                    stint_start,
                    change_time,
                )
                stint_start = change_time

            _apply_substitution(action, home_team_id, away_team_id, home_on, away_on)

        if period_end > stint_start:
            _emit_interval(
                stints,
                player_seconds,
                game_id,
                period,
                home_team_id,
                away_team_id,
                home_on,
                away_on,
                stint_start,
                float(period_end),
            )

    return LineupReconstruction(
        game_id=game_id,
        stints=tuple(stints),
        tagged_actions=tuple(tagged),
        player_seconds=dict(player_seconds),
    )


def _starters(
    period_starters: Mapping[tuple[int, int], frozenset[int]],
    period: int,
    team_id: int,
) -> frozenset[int]:
    try:
        return period_starters[(period, team_id)]
    except KeyError as exc:
        raise LineupSizeError(
            f"no starting five provided for team {team_id} in period {period}"
        ) from exc


def _apply_substitution(
    action: Action,
    home_team_id: int,
    away_team_id: int,
    home_on: set[int],
    away_on: set[int],
) -> None:
    if action.team_id == home_team_id:
        team_set = home_on
    elif action.team_id == away_team_id:
        team_set = away_on
    else:
        raise ValueError(
            f"substitution (order {action.order}) has team_id {action.team_id}, "
            f"neither home ({home_team_id}) nor away ({away_team_id})"
        )

    if action.sub_out not in team_set:
        raise OnFloorError(
            f"substitution (order {action.order}, period {action.period}) removes "
            f"player {action.sub_out} who is not on the floor for team "
            f"{action.team_id}; on floor: {sorted(team_set)}"
        )
    team_set.discard(action.sub_out)
    if action.sub_in is not None:
        team_set.add(action.sub_in)
    _require_five(
        team_set,
        action.period,
        action.team_id,
        f"after substitution at order {action.order}",
    )


def _emit_interval(
    stints: list[Stint],
    player_seconds: dict[int, float],
    game_id: str,
    period: int,
    home_team_id: int,
    away_team_id: int,
    home_on: set[int],
    away_on: set[int],
    start: float,
    end: float,
) -> None:
    home_five = tuple(sorted(home_on))
    away_five = tuple(sorted(away_on))
    duration = end - start
    stints.append(
        Stint(
            game_id=game_id,
            period=period,
            team_id=home_team_id,
            lineup_id=lineup_id(home_five),
            home_five=home_five,
            away_five=away_five,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration,
        )
    )
    stints.append(
        Stint(
            game_id=game_id,
            period=period,
            team_id=away_team_id,
            lineup_id=lineup_id(away_five),
            home_five=home_five,
            away_five=away_five,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration,
        )
    )
    for pid in home_five:
        player_seconds[pid] += duration
    for pid in away_five:
        player_seconds[pid] += duration


def recon_period_starters(
    home_team_id: int,
    away_team_id: int,
    actions: Sequence[Action],
) -> dict[tuple[int, int], frozenset[int]]:
    """Infer each team's five on the floor at every period's tip-off.

    Standard back-inference from the event log: within a period a player was a
    starter if they either (a) took part in a non-substitution action before any
    of their own substitutions, or (b) were substituted OUT without first being
    substituted IN. A player whose first appearance in the period is a sub-IN came
    off the bench and did not start.

    The result is intended as input to :func:`reconstruct_lineups`, which
    validates that each returned set is exactly five.
    """
    periods = sorted({a.period for a in actions})
    result: dict[tuple[int, int], frozenset[int]] = {}
    for period in periods:
        period_actions = sorted(
            (a for a in actions if a.period == period), key=lambda a: a.order
        )
        for team_id in (home_team_id, away_team_id):
            result[(period, team_id)] = _infer_team_starters(period_actions, team_id)
    return result


def _infer_team_starters(
    period_actions: Sequence[Action], team_id: int
) -> frozenset[int]:
    starters: set[int] = set()
    entered: set[int] = set()  # subbed in this period (came off the bench)
    for action in period_actions:
        if action.team_id != team_id:
            continue
        if action.is_substitution:
            if action.sub_out is not None and action.sub_out not in entered:
                starters.add(action.sub_out)
            if action.sub_in is not None:
                entered.add(action.sub_in)
        elif action.person_id is not None and action.person_id not in entered:
            starters.add(action.person_id)
    return frozenset(starters)


def reconcile_minutes(
    player_seconds: Mapping[int, float],
    box_seconds: Mapping[int, float],
    tolerance_seconds: float = 0.0,
) -> list[ReconDiff]:
    """Return per-player mismatches beyond ``tolerance_seconds``.

    Compares the union of both key sets, so a player present in only one source
    surfaces as a full mismatch. An empty list means the reconstructed minutes
    reconcile to the box score within tolerance — the minutes gate passes.
    """
    diffs: list[ReconDiff] = []
    for pid in sorted(set(player_seconds) | set(box_seconds)):
        recon = player_seconds.get(pid, 0.0)
        box = box_seconds.get(pid, 0.0)
        delta = recon - box
        if abs(delta) > tolerance_seconds:
            diffs.append(
                ReconDiff(
                    person_id=pid,
                    reconstructed_seconds=recon,
                    box_seconds=box,
                    delta_seconds=delta,
                )
            )
    return diffs
