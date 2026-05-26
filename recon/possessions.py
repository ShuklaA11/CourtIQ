"""Possession segmentation from a lineup-tagged action log.

This is the reconstruction stage that turns the ordered event log (already
annotated with the five players on the floor for each team by the
`onfloor-minutes-recon` stage) into *possessions*: contiguous stretches where a
single team has the ball. Everything downstream — pace, offensive/defensive
rating, RAPM stints — is defined per possession, so the boundaries have to be
right.

Why not just trust the box score's possession count? `box_advanced.possessions`
is a *formula estimate*: FGA - OREB + TOV + 0.44*FTA and its cousins. That is a
fine sanity check (a loose gate — we should land within a few percent), but it
is not ground truth. The ground truth we hold ourselves to is stronger and
exact: **the points we attribute across all possessions must sum, per team, to
the official final box score.** Every made basket and free throw has to land in
exactly one possession owned by the team that scored it. That is the hard gate
(`points_by_team` / `assert_reconciles`).

Where a possession ends
-----------------------
A possession flips to the other team on:
  * a made field goal that is NOT an and-1 continuation,
  * a defensive rebound,
  * a turnover (steals, bad passes, offensive fouls, shot-clock violations),
  * the made FINAL free throw of a trip,
  * the end of a period.

A possession is *extended* (does not flip) by:
  * a missed shot awaiting a rebound,
  * an offensive rebound (same team keeps the ball),
  * an and-1: a made FG plus a shooting foul — the ensuing free throw belongs to
    the same possession, so the made FG does not close it; the made final FT does,
  * a non-final free throw of a trip,
  * a missed final free throw (the rebound then decides),
  * a flagrant foul's free throws (the fouled team shoots AND retains the ball),
  * a technical free throw (a dead-ball freebie; play resumes with whoever had
    the ball — it must not spawn a "phantom" possession).

The two foul-shot exceptions above are the subtle ones. And-1 and three-shot
fouls are handled structurally (see `_is_and1` and the free-throw branch), not by
trusting an upstream flag, so the logic is self-contained and testable.

Input contract (the lineup-tagged action log)
---------------------------------------------
`recon_possessions` consumes an *ordered* list of action dicts. Recognised keys:

  period          int    -- 1..4 (5+ for OT)
  seconds_elapsed float  -- seconds since the start of THIS period (monotone up)
  event           str    -- one of the normalised events below
  team_id         int    -- team credited with the action (the shooter/rebounder)
  points          int    -- points this action scored (0 unless a make)
  lineups         dict   -- {team_id: (p1..p5)} for BOTH teams at this moment
  ft_index        int    -- free throws only: 1-based index within the trip
  ft_total        int    -- free throws only: attempts in the trip
  ft_kind         str    -- free throws only: "regular" | "technical" | "flagrant"
  made            bool   -- free throws only: whether it went in (falls back to points>0)

Recognised `event` values:
  "made_shot", "missed_shot", "rebound", "turnover", "free_throw",
  "period_end", and any number of boundary-neutral events ("foul",
  "substitution", "timeout", "jump_ball", ...) that are ignored for segmentation.

Rebounds are classified structurally (rebounder's team vs. the current offense),
which means team rebounds — a `team_id` with no player — fall out correctly with
no special case.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Possession:
    """One possession — the `recon_possessions` row schema.

    `points` is what the *offense* scored during the possession (field goals and
    free throws, including and-1 and flagrant free throws). `start_seconds` and
    `end_seconds` are seconds elapsed within the period.
    """

    game_id: str
    period: int
    possession_number: int
    offense_team_id: int | None
    defense_team_id: int | None
    points: int
    start_seconds: float
    end_seconds: float
    offense_five: tuple[int, ...]
    defense_five: tuple[int, ...]
    home_score_before: int | None = None
    away_score_before: int | None = None


# Events that may appear between a made basket and its and-1 free throw, or that
# otherwise never move a possession boundary. Skipped when we look ahead to
# decide whether a make was an and-1.
_BOUNDARY_NEUTRAL = frozenset(
    {"foul", "substitution", "timeout", "jump_ball", "ejection", "violation", "delay"}
)


def recon_possessions(game_id: str, actions: list[dict]) -> list[Possession]:
    """Segment one game's lineup-tagged action log into possessions.

    Walks the log once, maintaining the in-progress possession as a mutable dict
    and emitting a frozen `Possession` each time one closes. Possessions are
    numbered from 1 within each period.
    """
    raw_rows: list[dict] = []
    poss: dict | None = None
    poss_num = 0
    cur_period: int | None = None
    last_seconds = 0.0
    last_lineups: dict | None = None
    # Technical free throws can be shot by a team that is on defense in the
    # surrounding possession (or at a dead ball with no possession open). Their
    # points still count on the official score, so we park them here and fold
    # them into that team's next possession — no phantom possession row.
    pending_tech: dict[int, int] = defaultdict(int)

    def close(seconds: float) -> None:
        nonlocal poss
        if poss is not None:
            poss["end"] = seconds
            raw_rows.append(poss)
            poss = None

    def open_(offense: int | None, defense: int | None, seconds: float,
              lineups: dict | None, action: dict | None = None) -> None:
        nonlocal poss, poss_num
        poss_num += 1
        off5 = tuple(lineups.get(offense, ())) if lineups else ()
        def5 = tuple(lineups.get(defense, ())) if lineups else ()
        poss = {
            "period": cur_period,
            "num": poss_num,
            "offense": offense,
            "defense": defense,
            "points": pending_tech.pop(offense, 0),  # fold any parked technicals
            "start": seconds,
            "end": seconds,
            "off5": off5,
            "def5": def5,
            "home_score_before": (
                action.get("score_home_before") if action is not None else None
            ),
            "away_score_before": (
                action.get("score_away_before") if action is not None else None
            ),
        }

    def ensure_offense(
        team: int | None, seconds: float, lineups: dict | None, action: dict
    ) -> None:
        """Guarantee an open possession owned by `team`.

        Opening the first possession, and defensively re-syncing if the log hands
        us an offensive action by a team that isn't the current offense (a scored
        steal with no explicit turnover row, or a gap in the log).
        """
        nonlocal poss
        if poss is None or poss["offense"] != team:
            if poss is not None:
                close(seconds)
            open_(team, _other_team(lineups, team), seconds, lineups, action)

    for i, action in enumerate(actions):
        period = action["period"]
        seconds = float(action.get("seconds_elapsed", 0.0))
        event = action.get("event")
        team = action.get("team_id")
        lineups = action.get("lineups")

        if period != cur_period:
            close(last_seconds)  # buzzer: close whatever the old period left open
            cur_period = period
            poss_num = 0

        if event == "made_shot":
            ensure_offense(team, seconds, lineups, action)
            poss["points"] += action.get("points", 0)
            if not _is_and1(actions, i):
                close(seconds)

        elif event == "missed_shot":
            ensure_offense(team, seconds, lineups, action)  # stays open, awaits rebound

        elif event == "turnover":
            ensure_offense(team, seconds, lineups, action)
            close(seconds)

        elif event == "free_throw":
            kind = action.get("ft_kind", "regular")
            points = action.get("points", 0)
            made = action.get("made", points > 0)
            if kind == "technical":
                # Dead-ball freebie: never opens or closes a possession. Credit
                # the points to the shooting team without moving the boundary; if
                # they aren't the current offense, park the points for their next
                # possession (see pending_tech).
                if made:
                    if poss is not None and poss["offense"] == team:
                        poss["points"] += points
                    else:
                        pending_tech[team] += points
            elif kind == "flagrant":
                # Fouled team shoots AND keeps the ball: score it, never close.
                ensure_offense(team, seconds, lineups, action)
                if made:
                    poss["points"] += points
            else:
                # Regular trip: the made FINAL attempt flips the possession; a
                # missed final FT does not — the ensuing rebound decides.
                ensure_offense(team, seconds, lineups, action)
                if made:
                    poss["points"] += points
                if made and action.get("ft_index") == action.get("ft_total"):
                    close(seconds)

        elif event == "rebound":
            if poss is None:
                open_(team, _other_team(lineups, team), seconds, lineups, action)
            elif team == poss["offense"]:
                poss["end"] = seconds  # offensive rebound: same team keeps the ball
            else:
                close(seconds)  # defensive rebound ends it; the grab starts the flip
                open_(team, _other_team(lineups, team), seconds, lineups, action)

        elif event == "period_end":
            close(seconds)

        # else: boundary-neutral event (foul, substitution, timeout, ...) — ignore.

        last_seconds = seconds
        if lineups:
            last_lineups = lineups

    close(last_seconds)  # log ended mid-possession (final buzzer)

    # Any technicals never claimed by a later possession (e.g. shot at the final
    # buzzer) still have to reconcile — the hard gate is exact per-team points.
    for team, pts in pending_tech.items():
        if pts:
            _flush_pending(raw_rows, team, pts, last_seconds, last_lineups)

    return [
        Possession(
            game_id=game_id,
            period=r["period"],
            possession_number=r["num"],
            offense_team_id=r["offense"],
            defense_team_id=r["defense"],
            points=r["points"],
            start_seconds=r["start"],
            end_seconds=r["end"],
            offense_five=r["off5"],
            defense_five=r["def5"],
            home_score_before=r["home_score_before"],
            away_score_before=r["away_score_before"],
        )
        for r in raw_rows
    ]


def _is_and1(actions: list[dict], i: int) -> bool:
    """True if the made shot at index `i` is an and-1 (a shooting foul follows).

    Structural, not flag-based: scan forward past boundary-neutral events (the
    foul itself, substitutions); if the first real action is a REGULAR free throw
    by the same team, the make and that free throw share a possession.
    """
    team = actions[i].get("team_id")
    for action in actions[i + 1:]:
        event = action.get("event")
        if event in _BOUNDARY_NEUTRAL:
            continue
        return (
            event == "free_throw"
            and action.get("team_id") == team
            and action.get("ft_kind", "regular") == "regular"
        )
    return False


def _other_team(lineups: dict | None, team: int | None) -> int | None:
    """The opponent's team id, read off the two-team lineup map."""
    if not lineups:
        return None
    for key in lineups:
        if key != team:
            return key
    return None


def _flush_pending(raw_rows: list[dict], team: int, pts: int,
                   last_seconds: float, last_lineups: dict | None) -> None:
    """Land leftover technical points on a possession OWNED by `team`.

    Normal case: the team has a real possession — add the points there. Genuinely
    orphaned case (the team scored only via a technical and never possessed, which
    does not happen in a real game but must still reconcile): mint one minimal
    possession for that team so per-team points stay exact. Crediting the opponent
    instead would break the hard gate.
    """
    for row in reversed(raw_rows):
        if row["offense"] == team:
            row["points"] += pts
            return
    defense = _other_team(last_lineups, team)
    period = raw_rows[-1]["period"] if raw_rows else 1
    num = max((r["num"] for r in raw_rows if r["period"] == period), default=0) + 1
    raw_rows.append({
        "period": period,
        "num": num,
        "offense": team,
        "defense": defense,
        "points": pts,
        "start": last_seconds,
        "end": last_seconds,
        "off5": tuple(last_lineups.get(team, ())) if last_lineups else (),
        "def5": tuple(last_lineups.get(defense, ())) if last_lineups else (),
        "home_score_before": None,
        "away_score_before": None,
    })


# --------------------------------------------------------------------------- #
# Reconciliation — the hard gate.
# --------------------------------------------------------------------------- #

def points_by_team(possessions: list[Possession]) -> dict[int, int]:
    """Total offensive points per team across all possessions."""
    totals: dict[int, int] = defaultdict(int)
    for p in possessions:
        totals[p.offense_team_id] += p.points
    return dict(totals)


def assert_reconciles(possessions: list[Possession],
                      final_score: dict[int, int]) -> None:
    """Raise unless per-team possession points equal the official final score.

    `final_score` maps team_id -> official points. This is the exact gate the
    reconstruction must pass for every game; the possession *count* is only
    sanity-checked against the box-score formula estimate elsewhere.
    """
    got = points_by_team(possessions)
    # Compare on the union of teams so a team that scored zero in one source but
    # not the other is still caught.
    for team in set(got) | set(final_score):
        expected = final_score.get(team, 0)
        actual = got.get(team, 0)
        if actual != expected:
            raise AssertionError(
                f"possession points do not reconcile for team {team}: "
                f"summed {actual} != final box {expected}"
            )
