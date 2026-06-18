"""Tests for leakage-safe current-season form features (`winprob.pregame_features`).

These pin the two contracts the rest of Sprint 4's pre-game features stack on:

1. `season_game_results` collapses the possession mart to one row per game with
   the FINAL score (max of the running scores) and a possession count proxy,
   without mutating its input.
2. `add_current_season_form` computes, for each game, form from ONLY strictly
   earlier same-season games — games played, win pct, a per-100 net rating — and
   an Empirical-Bayes `form_strength` that equals the prior EXACTLY on a team's
   first game of the season and shrinks toward the current net as games accrue.

Synthetic, deterministic, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winprob import pregame_features as pf

A, B, C = 101, 102, 103  # team ids


def _possession_game(gid, season, date, home, away, home_pts, away_pts, n=50):
    """One synthetic game as a possession-grain frame with a rising score."""
    home_score = np.round(np.linspace(0, home_pts, n)).astype(int)
    away_score = np.round(np.linspace(0, away_pts, n)).astype(int)
    return pd.DataFrame({
        "game_id": gid,
        "season": season,
        "game_date": pd.Timestamp(date),
        "period": np.clip((np.arange(n) // (n // 4)) + 1, 1, 4),
        "possession_number": np.arange(n),
        "home_team_id": home,
        "away_team_id": away,
        "home_score": home_score,
        "away_score": away_score,
    })


def _result(gid, season, date, home, away, home_pts, away_pts, poss=100):
    """One row of a `season_game_results`-shaped table."""
    return {
        "game_id": gid, "season": season, "game_date": date,
        "home_team_id": home, "away_team_id": away,
        "home_final_points": home_pts, "away_final_points": away_pts,
        "possessions": poss,
    }


def _results_frame(rows):
    return pd.DataFrame(rows, columns=list(pf.RESULT_COLUMNS))


# --------------------------------------------------------------------------
# 1. season_game_results.
# --------------------------------------------------------------------------

def test_season_game_results_one_row_per_game_with_final_scores():
    # Arrange
    df = pd.concat([
        _possession_game("g1", 2025, "2025-11-01", A, B, 110, 100),
        _possession_game("g2", 2025, "2025-11-02", B, A, 95, 120),
    ], ignore_index=True)

    # Act
    results = pf.season_game_results(df)

    # Assert
    assert list(results.columns) == list(pf.RESULT_COLUMNS)
    assert len(results) == 2
    g1 = results[results["game_id"] == "g1"].iloc[0]
    assert g1["home_final_points"] == 110  # max of the rising home score
    assert g1["away_final_points"] == 100
    assert g1["possessions"] == 49  # max possession_number for n=50


def test_season_game_results_does_not_mutate_input():
    # Arrange
    df = _possession_game("g1", 2025, "2025-11-01", A, B, 110, 100)
    before_cols, before_rows = list(df.columns), len(df)

    # Act
    pf.season_game_results(df)

    # Assert
    assert list(df.columns) == before_cols
    assert len(df) == before_rows


def test_season_game_results_rejects_missing_columns():
    # Arrange — a frame missing home_score/away_score.
    df = pd.DataFrame({"game_id": ["g1"], "season": [2025]})

    # Act / Assert
    with pytest.raises(ValueError):
        pf.season_game_results(df)


# --------------------------------------------------------------------------
# 2. add_current_season_form — column contract + immutability.
# --------------------------------------------------------------------------

def _games_frame(rows):
    cols = list(pf._GAME_KEY_COLUMNS) + [pf.HOME_PRIOR_STRENGTH, pf.AWAY_PRIOR_STRENGTH]
    return pd.DataFrame(rows, columns=cols)


def test_add_current_season_form_emits_additive_columns():
    # Arrange
    all_games = _results_frame([_result("g1", 2025, "2025-11-10", A, B, 110, 100)])
    games = _games_frame([["g1", 2025, "2025-11-10", A, B, 5.0, -5.0]])

    # Act
    out = pf.add_current_season_form(games, all_games)

    # Assert — originals kept, every form column added.
    for col in pf._GAME_KEY_COLUMNS:
        assert col in out.columns
    for col in pf.FORM_COLUMNS:
        assert col in out.columns


def test_add_current_season_form_does_not_mutate_inputs():
    # Arrange
    all_games = _results_frame([_result("g1", 2025, "2025-11-10", A, B, 110, 100)])
    games = _games_frame([["g1", 2025, "2025-11-10", A, B, 5.0, -5.0]])
    games_cols, ag_cols = list(games.columns), list(all_games.columns)

    # Act
    pf.add_current_season_form(games, all_games)

    # Assert
    assert list(games.columns) == games_cols
    assert list(all_games.columns) == ag_cols


# --------------------------------------------------------------------------
# 3. First-game invariant: n == 0 => form_strength == prior EXACTLY.
# --------------------------------------------------------------------------

def test_first_game_of_season_falls_back_exactly_to_prior():
    # Arrange — team C has no prior 2025 games at all.
    all_games = _results_frame([_result("gq", 2025, "2025-11-10", C, A, 90, 100)])
    games = _games_frame([["gq", 2025, "2025-11-10", C, A, 7.5, -3.25]])

    # Act
    out = pf.add_current_season_form(games, all_games)
    row = out.iloc[0]

    # Assert — no games, weight 0, strength is the untouched prior.
    assert row["home_games_played"] == 0
    assert row["away_games_played"] == 0
    assert row["home_form_net"] == 0.0
    assert row["home_form_strength"] == 7.5   # exactly the home prior
    assert row["away_form_strength"] == -3.25  # exactly the away prior


# --------------------------------------------------------------------------
# 4. Net rating, win pct, and the shrinkage blend on known numbers.
# --------------------------------------------------------------------------

def test_net_rating_and_shrinkage_blend_are_exact():
    # Arrange — team A wins two prior same-season games, +30 total margin over
    # 200 possessions => net = 30/200*100 = 15; win_pct = 1.0; n = 2.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-01", A, B, 110, 100, poss=100),
        _result("g2", 2025, "2025-11-05", A, B, 120, 100, poss=100),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100, poss=100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 1.0]])

    # Act
    out = pf.add_current_season_form(games, all_games, shrink_k=10.0)
    row = out.iloc[0]

    # Assert
    assert row["home_games_played"] == 2
    assert row["home_win_pct"] == pytest.approx(1.0)
    assert row["home_form_net"] == pytest.approx(15.0)
    w = 2.0 / (2.0 + 10.0)  # = 1/6
    expected = w * 15.0 + (1.0 - w) * 5.0
    assert row["home_form_strength"] == pytest.approx(expected)
    # away team C is cold-started here => strength is its exact prior.
    assert row["away_form_strength"] == pytest.approx(1.0)
    assert row["form_strength_diff"] == pytest.approx(expected - 1.0)


def test_shrink_k_controls_weight_on_current_form():
    # Arrange — same history, larger K trusts the prior more.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-01", A, B, 120, 100, poss=100),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100, poss=100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 0.0]])

    # Act
    small_k = pf.add_current_season_form(games, all_games, shrink_k=1.0).iloc[0]
    large_k = pf.add_current_season_form(games, all_games, shrink_k=50.0).iloc[0]

    # Assert — net (+20) is above the prior (5), so more weight => higher strength.
    assert small_k["home_form_strength"] > large_k["home_form_strength"]
    assert large_k["home_form_strength"] > 5.0  # still pulled above the prior


# --------------------------------------------------------------------------
# 5. Leakage safety: strict-less-than date and same-season only.
# --------------------------------------------------------------------------

def test_same_day_and_own_game_are_excluded():
    # Arrange — a game on the SAME date as the query must not count, and the
    # query's own row must not count itself.
    all_games = _results_frame([
        _result("g_same", 2025, "2025-11-10", A, B, 130, 90, poss=100),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100, poss=100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 0.0]])

    # Act
    row = pf.add_current_season_form(games, all_games).iloc[0]

    # Assert — nothing strictly earlier => cold start, prior exactly.
    assert row["home_games_played"] == 0
    assert row["home_form_strength"] == 5.0


def test_prior_season_games_are_excluded():
    # Arrange — team A has a big 2024 game that must NOT feed a 2025 query.
    all_games = _results_frame([
        _result("g_old", 2024, "2024-11-01", A, B, 130, 90, poss=100),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100, poss=100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 0.0]])

    # Act
    row = pf.add_current_season_form(games, all_games).iloc[0]

    # Assert — same-season filter drops 2024, so A is cold-started in 2025.
    assert row["home_games_played"] == 0
    assert row["home_form_strength"] == 5.0


# --------------------------------------------------------------------------
# 6. Possession fallback to per-game point differential.
# --------------------------------------------------------------------------

def test_falls_back_to_per_game_differential_without_possessions():
    # Arrange — possessions unusable (all zero) => net = margin / games.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-01", A, B, 120, 100, poss=0),
        _result("g2", 2025, "2025-11-05", A, B, 110, 100, poss=0),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100, poss=0),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 0.0]])

    # Act
    row = pf.add_current_season_form(games, all_games).iloc[0]

    # Assert — (+20 and +10) over 2 games => mean margin 15.
    assert row["home_form_net"] == pytest.approx(15.0)


def test_add_current_season_form_rejects_missing_prior_strength():
    # Arrange — games frame without the prior-strength columns.
    all_games = _results_frame([_result("g1", 2025, "2025-11-10", A, B, 110, 100)])
    games = pd.DataFrame([{
        "game_id": "g1", "season": 2025, "game_date": "2025-11-10",
        "home_team_id": A, "away_team_id": B,
    }])

    # Act / Assert
    with pytest.raises(ValueError):
        pf.add_current_season_form(games, all_games)


# --------------------------------------------------------------------------
# 7. Rest / schedule features (strictly-prior).
# --------------------------------------------------------------------------

def test_add_rest_features_emits_additive_columns_without_mutation():
    # Arrange
    all_games = _results_frame([_result("g1", 2025, "2025-11-10", A, B, 110, 100)])
    games = _games_frame([["g1", 2025, "2025-11-10", A, B, 5.0, -5.0]])
    games_cols, ag_cols = list(games.columns), list(all_games.columns)

    # Act
    out = pf.add_rest_features(games, all_games)

    # Assert — originals kept, every rest column added, inputs untouched.
    for col in pf._GAME_KEY_COLUMNS:
        assert col in out.columns
    for col in pf.REST_COLUMNS:
        assert col in out.columns
    assert list(games.columns) == games_cols
    assert list(all_games.columns) == ag_cols


def test_rest_days_count_calendar_days_since_prior_same_season_game():
    # Arrange — A last played 2025-11-05, query is 2025-11-10 => 5 days rest.
    # C last played 2025-11-08 => 2 days rest.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-05", A, B, 110, 100),
        _result("g2", 2025, "2025-11-08", C, B, 100, 90),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 1.0]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert
    assert row["home_rest_days"] == 5
    assert row["away_rest_days"] == 2
    assert row["rest_diff"] == 3
    assert not row["home_back_to_back"]
    assert not row["away_back_to_back"]
    assert not row["home_is_season_opener"]
    assert not row["away_is_season_opener"]


def test_prior_game_found_when_team_played_as_away():
    # Arrange — A's only prior game is as the AWAY team; must still count.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-06", B, A, 100, 111),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 1.0]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert — 2025-11-06 -> 2025-11-10 is 4 days.
    assert row["home_rest_days"] == 4
    assert not row["home_is_season_opener"]


def test_back_to_back_flag_at_one_day_rest():
    # Arrange — A played yesterday (1 day rest => back-to-back).
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-09", A, B, 110, 100),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 1.0]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert
    assert row["home_rest_days"] == 1
    assert row["home_back_to_back"]


def test_rest_days_capped_at_rest_cap():
    # Arrange — a 19-day layoff must clamp to REST_CAP.
    all_games = _results_frame([
        _result("g1", 2025, "2025-11-01", A, B, 110, 100),
        _result("gq", 2025, "2025-11-20", A, C, 100, 100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-20", A, C, 5.0, 1.0]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert
    assert row["home_rest_days"] == pf.REST_CAP
    assert not row["home_back_to_back"]


def test_season_opener_sets_rest_cap_and_opener_flag():
    # Arrange — team C has no prior 2025 game at all.
    all_games = _results_frame([_result("gq", 2025, "2025-11-10", C, A, 90, 100)])
    games = _games_frame([["gq", 2025, "2025-11-10", C, A, 7.5, -3.25]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert — both sides are cold-started: rested to REST_CAP, opener True.
    assert row["home_rest_days"] == pf.REST_CAP
    assert row["away_rest_days"] == pf.REST_CAP
    assert row["home_is_season_opener"]
    assert row["away_is_season_opener"]
    assert not row["home_back_to_back"]


def test_rest_excludes_same_day_own_game_and_prior_seasons():
    # Arrange — a same-day game, the query's own row, and a big 2024 game must all
    # be ignored; only strictly-earlier SAME-season play sets rest.
    all_games = _results_frame([
        _result("g_old", 2024, "2024-11-09", A, B, 130, 90),
        _result("g_same", 2025, "2025-11-10", A, B, 130, 90),
        _result("gq", 2025, "2025-11-10", A, C, 100, 100),
    ])
    games = _games_frame([["gq", 2025, "2025-11-10", A, C, 5.0, 1.0]])

    # Act
    row = pf.add_rest_features(games, all_games).iloc[0]

    # Assert — nothing strictly earlier in 2025 => season opener.
    assert row["home_is_season_opener"]
    assert row["home_rest_days"] == pf.REST_CAP


def test_add_rest_features_rejects_missing_columns():
    # Arrange — games frame missing the identity columns.
    all_games = _results_frame([_result("g1", 2025, "2025-11-10", A, B, 110, 100)])
    games = pd.DataFrame({"game_id": ["g1"]})

    # Act / Assert
    with pytest.raises(ValueError):
        pf.add_rest_features(games, all_games)
