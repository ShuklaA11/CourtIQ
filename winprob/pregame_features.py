"""Sprint-4 leakage-safe CURRENT-SEASON FORM features.

The market prices things a season-pooled RAPM cannot — and the single biggest
closeable gap is *this* season's form: a team can be a very different team by
January than its prior-season player ratings imply. This module turns the
possession mart into per-game current-season form signals WITHOUT leaking the
game's own outcome, or any future game, into its features.

Two functions, both pure (a new frame out, the input never mutated):

* ``season_game_results`` collapses the possession mart to one row per game — the
  final score (``max`` of the running ``home_score`` / ``away_score``) and a
  per-game possession count proxy (``max`` of ``possession_number``). It reuses
  the mart's own scores; it never re-reconstructs anything.
* ``add_current_season_form`` attaches, for each game G, form computed from ONLY
  the games that are strictly earlier in the SAME season. For each side it emits
  games played to date, win pct to date, a net rating to date, and an
  Empirical-Bayes ``form_strength`` that shrinks that net rating toward the
  team's prior-season strength: ``w*current_net + (1-w)*prior`` with
  ``w = n/(n+SHRINK_K)``. With no games yet (``n == 0``) the weight is exactly 0,
  so ``form_strength`` equals the prior-season strength EXACTLY.

*Net rating definition.* When the game-results table carries a usable possession
count for every game (all present and strictly positive), the net rating to date
is the points-per-100-possession margin:
``(points_for - points_against) / possessions * 100``, summed over the prior
games. Otherwise it falls back to the plain per-game point differential
``(points_for - points_against) / games_played``. The choice is made once per call
and applied to every row, so a single frame is always scored one consistent way.

*Leakage safety.* Only rows with ``game_date`` STRICTLY LESS THAN G's date and the
SAME ``season`` feed G's features. Strict inequality drops G's own row (and any
same-day game, whose result is not safely known pre-tip), and the same-season
filter prevents last season's results from bleeding into an early-season number.
The prior it shrinks toward is a prior-season quantity, so the blend is leakage-
safe end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Empirical-Bayes shrinkage strength: the number of games at which the current
# season and the prior receive equal weight (w = n/(n+SHRINK_K) = 0.5 at n=K).
SHRINK_K: float = 10.0
POINTS_PER_100: float = 100.0

# Rest / schedule knobs. Calendar rest days are capped at REST_CAP so a long layoff
# collapses to one "well rested" value; <= BACK_TO_BACK_REST_DAYS days off is a b2b.
REST_CAP: int = 7
BACK_TO_BACK_REST_DAYS: int = 1

# Possession-mart columns ``season_game_results`` reads (never recomputes).
_RESULT_SOURCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "possession_number",
)

# The game-results table's exact columns, in fixed order.
RESULT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_final_points",
    "away_final_points",
    "possessions",
)

# Prior-season strength columns expected on the pre-game ``games`` frame; these
# are what ``winprob.pregame.build_pregame_table`` already carries.
HOME_PRIOR_STRENGTH = "home_team_strength"
AWAY_PRIOR_STRENGTH = "away_team_strength"

# Identity columns ``add_current_season_form`` reads off the ``games`` frame.
_GAME_KEY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
)

# The NEW additive columns ``add_current_season_form`` appends.
FORM_COLUMNS: tuple[str, ...] = (
    "home_games_played",
    "away_games_played",
    "home_win_pct",
    "away_win_pct",
    "home_form_net",
    "away_form_net",
    "form_net_diff",
    "home_form_strength",
    "away_form_strength",
    "form_strength_diff",
)

# The NEW additive columns ``add_rest_features`` appends.
REST_COLUMNS: tuple[str, ...] = (
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    "rest_diff",
    "home_is_season_opener",
    "away_is_season_opener",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], ctx: str) -> None:
    """Fail fast if any required column is absent, naming the offenders."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{ctx}: missing columns {missing}")


# --------------------------------------------------------------------------
# Game-grain results from the possession mart.
# --------------------------------------------------------------------------

def season_game_results(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: identity, FINAL score, and a possession count proxy.

    The running ``home_score`` / ``away_score`` peak at the final score, so the
    per-game ``max`` is the final points; ``max`` of ``possession_number`` is the
    game's possession count proxy. Identity columns are constant within a game and
    taken from its first row. Returns a NEW frame with exactly ``RESULT_COLUMNS``;
    the input is never mutated.
    """
    _require_columns(df, _RESULT_SOURCE_COLUMNS, "season_game_results")
    results = (
        df.groupby("game_id", sort=False)
        .agg(
            season=("season", "first"),
            game_date=("game_date", "first"),
            home_team_id=("home_team_id", "first"),
            away_team_id=("away_team_id", "first"),
            home_final_points=("home_score", "max"),
            away_final_points=("away_score", "max"),
            possessions=("possession_number", "max"),
        )
        .reset_index()
    )
    return results.loc[:, list(RESULT_COLUMNS)]


# --------------------------------------------------------------------------
# Long per-team view + net-rating mode.
# --------------------------------------------------------------------------

def _team_game_long(results: pd.DataFrame) -> pd.DataFrame:
    """Explode game results to two team-oriented rows per game.

    Each game contributes one row for the home team and one for the away team,
    with that team's ``points_for`` / ``points_against`` / ``possessions`` and a
    ``won`` flag. This is the per-(team, game) ledger the as-of form sums over.
    """
    game_date = pd.to_datetime(results["game_date"])
    home = pd.DataFrame({
        "season": results["season"].to_numpy(),
        "game_date": game_date.to_numpy(),
        "team_id": results["home_team_id"].to_numpy(),
        "points_for": results["home_final_points"].to_numpy(dtype=np.float64),
        "points_against": results["away_final_points"].to_numpy(dtype=np.float64),
        "possessions": results["possessions"].to_numpy(dtype=np.float64),
    })
    away = pd.DataFrame({
        "season": results["season"].to_numpy(),
        "game_date": game_date.to_numpy(),
        "team_id": results["away_team_id"].to_numpy(),
        "points_for": results["away_final_points"].to_numpy(dtype=np.float64),
        "points_against": results["home_final_points"].to_numpy(dtype=np.float64),
        "possessions": results["possessions"].to_numpy(dtype=np.float64),
    })
    long = pd.concat([home, away], ignore_index=True)
    long["won"] = (long["points_for"] > long["points_against"]).astype(np.float64)
    return long


def _has_possession_data(results: pd.DataFrame) -> bool:
    """True iff every game carries a present, strictly-positive possession count.

    Only then is the per-100 net rating well defined for every row; otherwise the
    caller falls back to the plain per-game point differential.
    """
    if "possessions" not in results.columns:
        return False
    poss = results["possessions"]
    return bool(poss.notna().all() and (poss > 0).all())


# --------------------------------------------------------------------------
# As-of form accumulation.
# --------------------------------------------------------------------------

def _form_to_date(
    long: pd.DataFrame, queries: pd.DataFrame, use_possessions: bool
) -> pd.DataFrame:
    """Accumulate each query's strictly-earlier same-season games.

    ``queries`` has one row per (game, side) with a unique ``_key`` and the
    ``season`` / ``team_id`` / ``cutoff_date`` to look back from. Rows of ``long``
    with a matching team and season and ``game_date`` STRICTLY BEFORE the cutoff
    are summed. Teams with no prior games get ``games_played == 0``, ``win_pct``
    and ``form_net`` both ``0.0`` (a neutral placeholder that makes the downstream
    shrinkage fall back exactly to the prior). Returned frame is aligned to
    ``queries.index``.
    """
    merged = queries.merge(long, on=["season", "team_id"], how="left")
    prior = merged.loc[merged["game_date"] < merged["cutoff_date"]]
    agg = (
        prior.groupby("_key")
        .agg(
            n=("won", "size"),
            wins=("won", "sum"),
            pf=("points_for", "sum"),
            pa=("points_against", "sum"),
            poss=("possessions", "sum"),
        )
        .reindex(queries["_key"].to_numpy())
    )

    n = np.nan_to_num(agg["n"].to_numpy(dtype=np.float64))
    wins = np.nan_to_num(agg["wins"].to_numpy(dtype=np.float64))
    pf = np.nan_to_num(agg["pf"].to_numpy(dtype=np.float64))
    pa = np.nan_to_num(agg["pa"].to_numpy(dtype=np.float64))
    poss = np.nan_to_num(agg["poss"].to_numpy(dtype=np.float64))

    played = n > 0
    safe_n = np.where(played, n, 1.0)
    win_pct = np.where(played, wins / safe_n, 0.0)
    margin = pf - pa
    if use_possessions:
        safe_poss = np.where(poss > 0, poss, 1.0)
        net = np.where(played, margin / safe_poss * POINTS_PER_100, 0.0)
    else:
        net = np.where(played, margin / safe_n, 0.0)

    return pd.DataFrame(
        {
            "games_played": n.astype(np.int64),
            "win_pct": win_pct,
            "form_net": net,
        },
        index=queries.index,
    )


def _side_queries(games: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Per-game lookback keys for one side (home or away)."""
    return pd.DataFrame(
        {
            "_key": np.arange(len(games)),
            "season": games["season"].to_numpy(),
            "team_id": games[team_col].to_numpy(),
            "cutoff_date": pd.to_datetime(games["game_date"]).to_numpy(),
        },
        index=games.index,
    )


# --------------------------------------------------------------------------
# Public feature builder.
# --------------------------------------------------------------------------

def add_current_season_form(
    games: pd.DataFrame, all_games: pd.DataFrame, shrink_k: float = SHRINK_K
) -> pd.DataFrame:
    """Attach leakage-safe current-season form columns to a pre-game frame.

    ``games`` is one pre-tip row per game carrying game identity and the two
    prior-season strengths (``home_team_strength`` / ``away_team_strength``);
    ``all_games`` is a ``season_game_results`` table used as the read-only history.
    For each game and each side, form is computed from ``all_games`` rows that are
    strictly earlier and in the same season (see ``_form_to_date``), then blended
    with the prior-season strength by Empirical-Bayes shrinkage
    ``w*current_net + (1-w)*prior`` with ``w = n/(n+shrink_k)``. A team's first
    game of a season has ``n == 0`` and therefore ``form_strength == prior``
    exactly. Emits the ``FORM_COLUMNS`` additively; returns a NEW frame and never
    mutates its inputs.
    """
    _require_columns(games, _GAME_KEY_COLUMNS, "add_current_season_form: games")
    _require_columns(
        games,
        (HOME_PRIOR_STRENGTH, AWAY_PRIOR_STRENGTH),
        "add_current_season_form: games",
    )
    _require_columns(all_games, RESULT_COLUMNS, "add_current_season_form: all_games")
    if shrink_k <= 0:
        raise ValueError(f"shrink_k must be positive, got {shrink_k}")

    long = _team_game_long(all_games)
    use_possessions = _has_possession_data(all_games)

    home = _form_to_date(long, _side_queries(games, "home_team_id"), use_possessions)
    away = _form_to_date(long, _side_queries(games, "away_team_id"), use_possessions)

    home_prior = games[HOME_PRIOR_STRENGTH].to_numpy(dtype=np.float64)
    away_prior = games[AWAY_PRIOR_STRENGTH].to_numpy(dtype=np.float64)
    home_n = home["games_played"].to_numpy(dtype=np.float64)
    away_n = away["games_played"].to_numpy(dtype=np.float64)
    w_home = home_n / (home_n + shrink_k)
    w_away = away_n / (away_n + shrink_k)

    home_net = home["form_net"].to_numpy()
    away_net = away["form_net"].to_numpy()
    home_strength = w_home * home_net + (1.0 - w_home) * home_prior
    away_strength = w_away * away_net + (1.0 - w_away) * away_prior

    out = games.copy()
    out["home_games_played"] = home["games_played"].to_numpy()
    out["away_games_played"] = away["games_played"].to_numpy()
    out["home_win_pct"] = home["win_pct"].to_numpy()
    out["away_win_pct"] = away["win_pct"].to_numpy()
    out["home_form_net"] = home_net
    out["away_form_net"] = away_net
    out["form_net_diff"] = home_net - away_net
    out["home_form_strength"] = home_strength
    out["away_form_strength"] = away_strength
    out["form_strength_diff"] = home_strength - away_strength
    return out


# --------------------------------------------------------------------------
# Rest / schedule features (strictly-prior).
# --------------------------------------------------------------------------

def _rest_to_date(long: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Days of rest before each query's game, from strictly-earlier same-season play.

    ``queries`` has one row per (game, side) with a unique ``_key``, the
    ``season`` / ``team_id`` to match, and the ``cutoff_date`` to look back from.
    A team's most recent ``long`` game with a matching team and season and
    ``game_date`` STRICTLY BEFORE the cutoff sets the rest: ``(cutoff - prev)``
    in whole calendar days, capped at ``REST_CAP``. A team with no prior
    same-season game is a season opener — rest is set to ``REST_CAP`` (fully
    rested) and ``is_opener`` is True. Returned frame is aligned to
    ``queries.index``.
    """
    merged = queries.merge(long, on=["season", "team_id"], how="left")
    prior = merged.loc[merged["game_date"] < merged["cutoff_date"]]
    last_played = (
        prior.groupby("_key")["game_date"]
        .max()
        .reindex(queries["_key"].to_numpy())
        .to_numpy()
    )

    cutoff = queries["cutoff_date"].to_numpy()
    is_opener = np.isnat(last_played)
    raw_days = (cutoff - last_played) / np.timedelta64(1, "D")
    rest = np.where(is_opener, float(REST_CAP), raw_days)
    rest = np.minimum(rest, REST_CAP).astype(np.int64)

    return pd.DataFrame(
        {"rest_days": rest, "is_opener": is_opener},
        index=queries.index,
    )


def add_rest_features(games: pd.DataFrame, all_games: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly-prior rest / schedule columns to a pre-game frame.

    ``games`` is one pre-tip row per game carrying game identity; ``all_games`` is
    a ``season_game_results`` table used as the read-only schedule history. For each
    game and each side, rest is the calendar days since that team's previous
    SAME-SEASON game strictly earlier than tip-off, capped at ``REST_CAP`` (see
    ``_rest_to_date``). A team's first game of a season has no such prior game, so
    its rest is ``REST_CAP`` and its opener flag is True; a game with
    ``rest_days <= BACK_TO_BACK_REST_DAYS`` is a back-to-back. Emits the
    ``REST_COLUMNS`` additively; returns a NEW frame and never mutates its inputs.
    """
    _require_columns(games, _GAME_KEY_COLUMNS, "add_rest_features: games")
    _require_columns(all_games, RESULT_COLUMNS, "add_rest_features: all_games")

    long = _team_game_long(all_games)
    home = _rest_to_date(long, _side_queries(games, "home_team_id"))
    away = _rest_to_date(long, _side_queries(games, "away_team_id"))

    home_rest = home["rest_days"].to_numpy()
    away_rest = away["rest_days"].to_numpy()

    out = games.copy()
    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    out["home_back_to_back"] = home_rest <= BACK_TO_BACK_REST_DAYS
    out["away_back_to_back"] = away_rest <= BACK_TO_BACK_REST_DAYS
    out["rest_diff"] = home_rest - away_rest
    out["home_is_season_opener"] = home["is_opener"].to_numpy()
    out["away_is_season_opener"] = away["is_opener"].to_numpy()
    return out
