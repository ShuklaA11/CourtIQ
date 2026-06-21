"""Availability-adjusted pre-game team strength from PRIOR-season RAPM.

The pre-tip market prices *who is actually on the floor*: a star ruled out an
hour before tip drops a team's expected strength more than any season-pooled
rating. This module turns the leakage-safe availability layer
(``winprob.availability``) into that signal, entirely from quantities known
before tip-off:

* **Ratings are prior-season.** A game in season ``S`` reads each player's
  ``net_rating`` from season ``S - 1``; a player with no prior rating contributes
  at the replacement prior (``net_rating`` 0), mirroring the mart's existing
  replacement convention (``ablation.add_team_strength_columns``).
* **Role weights are prior-season possessions.** A player's weight is their
  season ``S - 1`` ``n_possessions`` — fully leakage-safe, and a player with no
  prior rating gets weight 0, so they never move a team's number.
* **Rosters are season-level.** A ``(team_id, season)`` roster is every player who
  appears for that team that season in the availability frame — the FULL roster,
  regardless of who is out on any given night.

From those three pieces the module computes, per team per game:

* ``expected_team_strength`` — the prior-possession-weighted mean ``net_rating``
  over the team's FULL season roster (everyone).
* ``available_team_strength`` — the same weighted mean with that game's
  UNAVAILABLE players removed and the weights renormalized over the remaining
  available rated players.
* ``injury_hit = available - expected`` — the headline feature. It is ``<= 0``
  when good players sit (removing an above-average, heavily-weighted star pulls
  the mean down) and ``~0`` when only fringe players (little weight) are out.

``add_availability_features`` attaches the two available strengths, the two
injury hits, and ``injury_hit_diff`` (home minus away) to a pre-game frame.

Pure numpy/pandas; every function returns a new object and never mutates its
input. The scalar strength functions are the single definition of the math; the
frame builder calls them so the vectorized path can never drift from the tested
scalar path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

# A player with no prior-season rating contributes at the replacement prior: a
# neutral ``net_rating`` and zero weight, so they never influence a team's mean.
REPLACEMENT_NET_RATING: float = 0.0
REPLACEMENT_WEIGHT: float = 0.0

# Columns read off the ``ratings`` frame (prior-season per-player RAPM).
_RATING_COLUMNS: tuple[str, ...] = ("player_id", "season", "net_rating", "n_possessions")

# Columns read off the ``availability`` frame (see ``winprob.availability``).
_AVAILABILITY_COLUMNS: tuple[str, ...] = ("game_id", "team_id", "player_id", "available")

# Identity columns read off the pre-game ``games`` frame.
_GAME_COLUMNS: tuple[str, ...] = ("game_id", "season", "home_team_id", "away_team_id")

# The NEW additive columns ``add_availability_features`` appends, in fixed order.
AVAILABILITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_available_strength",
    "away_available_strength",
    "home_injury_hit",
    "away_injury_hit",
    "injury_hit_diff",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], ctx: str) -> None:
    """Fail fast if any required column is absent, naming the offenders."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{ctx}: missing columns {missing}")


# --------------------------------------------------------------------------
# The immutable team roster and the scalar strength math (single source of truth).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamRoster:
    """A ``(team_id, season)`` roster with each player's prior-season role.

    ``player_ids``, ``weights`` (prior-season ``n_possessions``), and
    ``net_ratings`` (prior-season ``net_rating``) are positionally aligned tuples
    — one entry per player who appeared for the team that season. Unrated players
    carry the replacement prior (weight 0, ``net_rating`` 0). Frozen and built
    from tuples, so the value object is fully immutable.
    """

    team_id: int
    season: int
    player_ids: tuple[int, ...]
    weights: tuple[float, ...]
    net_ratings: tuple[float, ...]


def _weighted_strength(
    net_ratings: tuple[float, ...], weights: tuple[float, ...]
) -> float:
    """Possession-weighted mean ``net_rating``; replacement when no weight remains.

    ``sum(w * net) / sum(w)`` over the given players. When the total weight is
    non-positive — every player unrated (weight 0) — the mean is undefined and
    the team falls back to the replacement prior (``REPLACEMENT_NET_RATING``).
    """
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return REPLACEMENT_NET_RATING
    weighted_sum = float(sum(w * n for w, n in zip(weights, net_ratings)))
    return weighted_sum / total_weight


def expected_team_strength(roster: TeamRoster) -> float:
    """Prior-possession-weighted mean ``net_rating`` over the team's FULL roster.

    Everyone on the season roster is counted, regardless of availability — this is
    the strength the team would field with its whole roster healthy.
    """
    return _weighted_strength(roster.net_ratings, roster.weights)


def available_team_strength(
    roster: TeamRoster, unavailable_player_ids: frozenset[int]
) -> float:
    """Weighted mean ``net_rating`` with this game's UNAVAILABLE players removed.

    The ``unavailable_player_ids`` are the game's out list for this team; they are
    dropped and the weights renormalize over the remaining available rated players
    (``_weighted_strength`` divides by the surviving weight). With nobody out this
    equals ``expected_team_strength`` exactly.
    """
    kept_nets: list[float] = []
    kept_weights: list[float] = []
    for player_id, weight, net in zip(
        roster.player_ids, roster.weights, roster.net_ratings
    ):
        if player_id in unavailable_player_ids:
            continue
        kept_nets.append(net)
        kept_weights.append(weight)
    return _weighted_strength(tuple(kept_nets), tuple(kept_weights))


def injury_hit(roster: TeamRoster, unavailable_player_ids: frozenset[int]) -> float:
    """``available_team_strength - expected_team_strength`` for one team-game.

    ``<= 0`` when above-average, heavily-weighted players sit (their removal pulls
    the mean down); ``~0`` when only fringe players (little prior weight) are out.
    """
    return available_team_strength(roster, unavailable_player_ids) - (
        expected_team_strength(roster)
    )


# --------------------------------------------------------------------------
# Building the prior-rating lookup and the season rosters.
# --------------------------------------------------------------------------

def build_prior_rating_lookup(
    ratings: pd.DataFrame,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Index the ratings frame as ``(player_id, season) -> (net_rating, weight)``.

    The weight is the season's ``n_possessions``. Callers look up a game's season
    ``S`` player at key ``(player_id, S - 1)`` to get the prior-season role; a miss
    is the replacement prior. Returns a plain dict; the input is never mutated.
    """
    _require_columns(ratings, _RATING_COLUMNS, "build_prior_rating_lookup")
    lookup: dict[tuple[int, int], tuple[float, float]] = {}
    for row in ratings.itertuples(index=False):
        key = (int(row.player_id), int(row.season))
        lookup[key] = (float(row.net_rating), float(row.n_possessions))
    return lookup


def build_team_rosters(
    availability: pd.DataFrame,
    game_seasons: Mapping[str, int],
    rating_lookup: Mapping[tuple[int, int], tuple[float, float]],
) -> dict[tuple[int, int], TeamRoster]:
    """Derive every ``(team_id, season)`` roster from the availability frame.

    Each availability row's season comes from ``game_seasons`` (the game_id ->
    season map from the pre-game frame); rows whose game is not in that map are
    skipped. For a team-season, the roster is the DISTINCT players who appeared for
    that team that season. Each player's weight and ``net_rating`` are read from the
    PRIOR season (``season - 1``) via ``rating_lookup``, defaulting to the
    replacement prior when absent. Returns a ``(team_id, season) -> TeamRoster``
    dict; inputs are never mutated.
    """
    _require_columns(availability, ("game_id", "team_id", "player_id"), "build_team_rosters")

    # Collect distinct player_ids per (team_id, season) deterministically.
    members: dict[tuple[int, int], list[int]] = {}
    seen: set[tuple[int, int, int]] = set()
    for row in availability.itertuples(index=False):
        season = game_seasons.get(str(row.game_id))
        if season is None:
            continue
        team_id = int(row.team_id)
        player_id = int(row.player_id)
        member_key = (team_id, season, player_id)
        if member_key in seen:
            continue
        seen.add(member_key)
        members.setdefault((team_id, season), []).append(player_id)

    rosters: dict[tuple[int, int], TeamRoster] = {}
    for (team_id, season), player_ids in members.items():
        ordered = tuple(sorted(player_ids))
        weights: list[float] = []
        net_ratings: list[float] = []
        for player_id in ordered:
            net, weight = rating_lookup.get(
                (player_id, season - 1),
                (REPLACEMENT_NET_RATING, REPLACEMENT_WEIGHT),
            )
            net_ratings.append(net)
            weights.append(weight)
        rosters[(team_id, season)] = TeamRoster(
            team_id=team_id,
            season=season,
            player_ids=ordered,
            weights=tuple(weights),
            net_ratings=tuple(net_ratings),
        )
    return rosters


def build_unavailable_sets(
    availability: pd.DataFrame,
) -> dict[tuple[str, int], frozenset[int]]:
    """Index the out list per game-team as ``(game_id, team_id) -> frozenset``.

    Only rows flagged ``available == False`` contribute. A game-team with nobody
    out simply has no entry (the empty out list). Returns a plain dict; the input
    is never mutated.
    """
    _require_columns(availability, _AVAILABILITY_COLUMNS, "build_unavailable_sets")
    out_rows = availability.loc[~availability["available"].astype(bool)]
    result: dict[tuple[str, int], set[int]] = {}
    for row in out_rows.itertuples(index=False):
        key = (str(row.game_id), int(row.team_id))
        result.setdefault(key, set()).add(int(row.player_id))
    return {key: frozenset(ids) for key, ids in result.items()}


# --------------------------------------------------------------------------
# Public feature builder.
# --------------------------------------------------------------------------

def _empty_roster(team_id: int, season: int) -> TeamRoster:
    """A roster with no players — strength falls back to the replacement prior."""
    return TeamRoster(team_id=team_id, season=season, player_ids=(), weights=(), net_ratings=())


def _side_strengths(
    games: pd.DataFrame,
    team_col: str,
    rosters: Mapping[tuple[int, int], TeamRoster],
    unavailable: Mapping[tuple[str, int], frozenset[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Available strength and injury hit for one side, aligned to ``games`` rows."""
    available = np.empty(len(games), dtype=np.float64)
    hit = np.empty(len(games), dtype=np.float64)
    no_out: frozenset[int] = frozenset()
    for position, row in enumerate(games.itertuples(index=False)):
        team_id = int(getattr(row, team_col))
        season = int(row.season)
        roster = rosters.get((team_id, season)) or _empty_roster(team_id, season)
        out_ids = unavailable.get((str(row.game_id), team_id), no_out)
        available[position] = available_team_strength(roster, out_ids)
        hit[position] = available[position] - expected_team_strength(roster)
    return available, hit


def add_availability_features(
    games: pd.DataFrame, availability: pd.DataFrame, ratings: pd.DataFrame
) -> pd.DataFrame:
    """Attach availability-adjusted pre-game strength columns to a pre-game frame.

    ``games`` is one pre-tip row per game carrying identity (``game_id``,
    ``season``, ``home_team_id``, ``away_team_id``); ``availability`` is the
    leakage-safe pre-tip availability frame (``winprob.availability``); ``ratings``
    is the per-player prior-season RAPM (``bayes_ratings``). Rosters and weights are
    built from the PRIOR season (``season - 1``), so the whole computation is known
    before tip-off. Emits the ``AVAILABILITY_FEATURE_COLUMNS`` additively:
    ``home_available_strength``, ``away_available_strength``, ``home_injury_hit``,
    ``away_injury_hit``, and ``injury_hit_diff`` (home minus away). Returns a NEW
    frame and never mutates its inputs.
    """
    _require_columns(games, _GAME_COLUMNS, "add_availability_features: games")
    _require_columns(
        availability, _AVAILABILITY_COLUMNS, "add_availability_features: availability"
    )
    _require_columns(ratings, _RATING_COLUMNS, "add_availability_features: ratings")

    game_seasons = {
        str(game_id): int(season)
        for game_id, season in zip(games["game_id"], games["season"])
    }
    rating_lookup = build_prior_rating_lookup(ratings)
    rosters = build_team_rosters(availability, game_seasons, rating_lookup)
    unavailable = build_unavailable_sets(availability)

    home_available, home_hit = _side_strengths(
        games, "home_team_id", rosters, unavailable
    )
    away_available, away_hit = _side_strengths(
        games, "away_team_id", rosters, unavailable
    )

    out = games.copy()
    out["home_available_strength"] = home_available
    out["away_available_strength"] = away_available
    out["home_injury_hit"] = home_hit
    out["away_injury_hit"] = away_hit
    out["injury_hit_diff"] = home_hit - away_hit
    return out
