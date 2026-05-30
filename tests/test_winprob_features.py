"""Unit tests for the pure win-probability feature transform (winprob.features).

Warehouse-free: every test builds a tiny hand-constructed DataFrame with the
columns of `data/winprob/fct_game_states.parquet` and checks the transform's
output directly. No DuckDB, no parquet reads, no model fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winprob.features import (
    TIME_KNOTS,
    build_design,
    build_design_matrix,
    margin_over_sqrt_time,
    playoff_indicator,
    season_dummies,
    standardize_columns,
    time_knot_basis,
)

# Columns from the source mart that MUST NEVER appear in `feature_names`: the
# `home_win` target (the label), the player-rating columns from the ratings
# pipeline (which would leak the outcome / couple this baseline to that fit), and
# the raw pre-reconciliation `feed_*` scoreboard fields. The purity guarantee is
# a property of the transform's *output*, so the blocklist lives here in the test
# rather than in the pure module — keeping the module source provably free of any
# rating-column references.
FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "home_win",
    "home_lineup_off_rapm",
    "home_lineup_def_rapm",
    "home_lineup_net_rapm",
    "away_lineup_off_rapm",
    "away_lineup_def_rapm",
    "away_lineup_net_rapm",
    "lineup_net_rapm_differential",
    "feed_home_score_before",
    "feed_away_score_before",
)


# --------------------------------------------------------------------------
# Helpers: a minimal game-state frame with all mart columns present (including
# the forbidden ones, to prove they are dropped rather than merely absent).
# --------------------------------------------------------------------------

def _frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "game_id": "0022100001",
        "season": 2021,
        "home_score_differential": 0,
        "regulation_seconds_remaining": 2880.0,
        "home_has_possession": True,
        # Forbidden columns present in the source mart:
        "home_win": True,
        "home_lineup_off_rapm": 1.0,
        "home_lineup_def_rapm": 1.0,
        "home_lineup_net_rapm": 1.0,
        "away_lineup_off_rapm": 1.0,
        "away_lineup_def_rapm": 1.0,
        "away_lineup_net_rapm": 1.0,
        "lineup_net_rapm_differential": 1.0,
        "feed_home_score_before": 0,
        "feed_away_score_before": 0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# --------------------------------------------------------------------------
# margin / sqrt(time) basis — the near-sufficient statistic.
# --------------------------------------------------------------------------

def test_margin_over_sqrt_time_matches_hand_computation():
    # margin +6 with 99 seconds left => 6 / sqrt(99 + 1) = 6 / 10 = 0.6.
    margin = np.array([6.0])
    reg_sec = np.array([99.0])
    got = margin_over_sqrt_time(margin, reg_sec)
    assert got.tolist() == pytest.approx([0.6])


def test_margin_over_sqrt_time_defined_at_buzzer():
    # reg_seconds_remaining == 0 => denominator sqrt(0 + 1) == 1, no divide-by-zero.
    got = margin_over_sqrt_time(np.array([5.0]), np.array([0.0]))
    assert got.tolist() == pytest.approx([5.0])
    assert np.isfinite(got).all()


def test_margin_over_sqrt_time_grows_as_clock_winds_down():
    # For a fixed +4 lead, the basis magnitude rises as time falls.
    reg = np.array([2000.0, 200.0, 20.0])
    got = margin_over_sqrt_time(np.array([4.0, 4.0, 4.0]), reg)
    assert got[0] < got[1] < got[2]


# --------------------------------------------------------------------------
# time knot hinge basis.
# --------------------------------------------------------------------------

def test_time_knot_basis_is_relu_of_knot_minus_time():
    # knots (720, 180, 30); at t=200 seconds remaining:
    #   relu(720-200)=520, relu(180-200)=0, relu(30-200)=0.
    basis = time_knot_basis(np.array([200.0]))
    assert basis.shape == (1, 3)
    assert basis[0].tolist() == pytest.approx([520.0, 0.0, 0.0])


def test_time_knot_basis_zero_when_ample_time_remains():
    basis = time_knot_basis(np.array([2880.0]))
    assert basis[0].tolist() == pytest.approx([0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# playoff indicator from the game-id prefix.
# --------------------------------------------------------------------------

def test_playoff_indicator_from_game_id_prefix():
    ids = np.array(["0042100301", "0022100001", "0042400111"])
    got = playoff_indicator(ids)
    assert got.tolist() == [1.0, 0.0, 1.0]


# --------------------------------------------------------------------------
# season dummies — reference level dropped.
# --------------------------------------------------------------------------

def test_season_dummies_drop_first_level_as_reference():
    season = np.array([2021, 2022, 2023, 2021])
    dummies, names = season_dummies(season, [2021, 2022, 2023])
    assert names == ["season_2022", "season_2023"]
    # 2021 is the reference -> all-zero dummy row; others one-hot.
    assert dummies.tolist() == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_season_dummies_single_level_has_no_columns():
    dummies, names = season_dummies(np.array([2024, 2024]), [2024])
    assert names == []
    assert dummies.shape == (2, 0)


# --------------------------------------------------------------------------
# standardize_columns — continuous centered/scaled, others pass through.
# --------------------------------------------------------------------------

def test_standardize_only_touches_continuous_columns():
    m = np.array([[1.0, 10.0], [3.0, 10.0]])
    mask = np.array([True, False])
    z, means, stds = standardize_columns(m, mask)
    # Column 0 continuous: mean 2, population std 1 -> [-1, +1].
    assert z[:, 0].tolist() == pytest.approx([-1.0, 1.0])
    # Column 1 not continuous: unchanged, identity stats.
    assert z[:, 1].tolist() == [10.0, 10.0]
    assert means.tolist() == pytest.approx([2.0, 0.0])
    assert stds.tolist() == pytest.approx([1.0, 1.0])


def test_standardize_floors_zero_variance_std():
    m = np.array([[5.0], [5.0]])
    z, means, stds = standardize_columns(m, np.array([True]))
    # Constant continuous column -> centered to zeros, std floored to 1 (no NaN).
    assert z[:, 0].tolist() == [0.0, 0.0]
    assert stds.tolist() == [1.0]
    assert np.isfinite(z).all()


# --------------------------------------------------------------------------
# build_design_matrix — hand-built rows -> expected feature values.
# --------------------------------------------------------------------------

def test_hand_built_rows_map_to_expected_values():
    df = _frame(
        [
            {"game_id": "0022100001", "season": 2021,
             "home_score_differential": 6, "regulation_seconds_remaining": 99.0,
             "home_has_possession": True},
            {"game_id": "0042100301", "season": 2022,
             "home_score_differential": -6, "regulation_seconds_remaining": 99.0,
             "home_has_possession": False},
        ]
    )
    design = build_design(df)
    names = design.feature_names
    col = {name: design.X[:, i] for i, name in enumerate(names)}

    # Intercept passes through as ones (not standardized).
    assert col["intercept"].tolist() == [1.0, 1.0]

    # margin_over_sqrt_time raw values are +0.6 and -0.6 (6/sqrt(100)); after
    # standardization (mean 0, symmetric) they become +1 / -1.
    assert col["margin_over_sqrt_time"].tolist() == pytest.approx([1.0, -1.0])

    # home_has_possession is a passthrough 0/1 indicator.
    assert col["home_has_possession"].tolist() == [1.0, 0.0]

    # is_playoff: row 0 is regular ('002'), row 1 is playoff ('004').
    assert col["is_playoff"].tolist() == [0.0, 1.0]

    # season dummy: 2021 is reference, so only season_2022 exists; row1 is 2022.
    assert "season_2022" in names
    assert "season_2021" not in names
    assert col["season_2022"].tolist() == [0.0, 1.0]


def test_margin_sqrt_time_column_matches_hand_computation():
    # Two rows chosen so the raw basis is exactly +2 and -2 => standardized +1/-1.
    df = _frame(
        [
            {"home_score_differential": 20, "regulation_seconds_remaining": 99.0},
            {"home_score_differential": -20, "regulation_seconds_remaining": 99.0},
        ]
    )
    design = build_design(df)
    idx = design.feature_names.index("margin_over_sqrt_time")
    # raw = +-20 / sqrt(100) = +-2.0; population mean 0, std 2 -> +1 / -1.
    assert design.X[:, idx].tolist() == pytest.approx([1.0, -1.0])


# --------------------------------------------------------------------------
# No NaN / inf anywhere in the output.
# --------------------------------------------------------------------------

def test_output_has_no_nan_or_inf():
    df = _frame(
        [
            {"home_score_differential": 3, "regulation_seconds_remaining": 0.0},
            {"home_score_differential": -8, "regulation_seconds_remaining": 720.0},
            {"game_id": "0042100301", "season": 2023,
             "home_score_differential": 15, "regulation_seconds_remaining": 2880.0},
        ]
    )
    X, _ = build_design_matrix(df)
    assert np.isfinite(X).all()


# --------------------------------------------------------------------------
# feature_names excludes every forbidden RAPM/lineup/target/feed column.
# --------------------------------------------------------------------------

def test_feature_names_exclude_forbidden_columns():
    df = _frame(
        [
            {"season": 2021, "home_score_differential": 1},
            {"season": 2022, "home_score_differential": -1},
        ]
    )
    _, names = build_design_matrix(df)
    for forbidden in FORBIDDEN_COLUMNS:
        assert forbidden not in names
    # Spot-check the specific families the spec calls out.
    assert not any("rapm" in n for n in names)
    assert not any("lineup" in n for n in names)
    assert not any(n.startswith("feed_") for n in names)
    assert "home_win" not in names


# --------------------------------------------------------------------------
# Column order is identical across repeated calls (determinism).
# --------------------------------------------------------------------------

def test_column_order_stable_across_calls():
    df = _frame(
        [
            {"game_id": "0022100001", "season": 2021, "home_score_differential": 4},
            {"game_id": "0042100301", "season": 2023, "home_score_differential": -4},
            {"game_id": "0022400001", "season": 2022, "home_score_differential": 9},
        ]
    )
    _, names_a = build_design_matrix(df)
    _, names_b = build_design_matrix(df)
    assert names_a == names_b

    # Structural prefix is the fixed, documented order; dummies follow in
    # ascending season order (2021 reference dropped -> 2022, 2023).
    expected_prefix = [
        "intercept",
        "home_score_differential",
        "margin_over_sqrt_time",
        "regulation_seconds_remaining",
        *[f"time_knot_{int(k)}" for k in TIME_KNOTS],
        "home_has_possession",
        "is_playoff",
    ]
    assert names_a[: len(expected_prefix)] == expected_prefix
    assert names_a[len(expected_prefix):] == ["season_2022", "season_2023"]


def test_matrix_shape_matches_names_and_rows():
    df = _frame(
        [
            {"season": 2021, "home_score_differential": 2},
            {"season": 2022, "home_score_differential": -2},
        ]
    )
    X, names = build_design_matrix(df)
    assert X.shape == (2, len(names))
