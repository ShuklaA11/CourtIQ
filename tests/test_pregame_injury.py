"""Tests for the P4 availability ladder tier (`winprob.pregame_injury`).

These pin the Sprint-5 contracts:

1. P4 strictly nests P3 — its feature list is P3's followed by exactly the five
   availability columns, so the paired P4-vs-P3 comparison is a clean ablation.
2. `_gap_close` re-measures the market gap against the frozen tier-E baseline: its
   `baseline_gap` is `brier(p_model) - brier(market)` and `fraction_of_gap_closed`
   is the share of THAT gap the tier closes — the identical stick Sprint 4 recorded.
3. The gates encode the Sprint-4 philosophy: `gate_availability_beats_form` is the
   P4-vs-P3 held-out Brier CI upper bound below zero; calibration and the structural
   integrity checks are separate; beating the market is never a gate.
4. The verdict states the null plainly when P4 ~= P3.

Parts 1-4 are synthetic and deterministic. A final real-data integration test fits
the true P4 on the mart, asserts the structural gates hold, every prediction is in
(0, 1), and the baseline gap reproduces Sprint 4's recorded value on the same join.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from winprob import evaluate, model, pregame, pregame_availability, pregame_ladder
from winprob import pregame_injury as pi

DATA_DIR = Path("data/winprob")
TARGET = model.TARGET_COLUMN


# --------------------------------------------------------------------------
# 1. P4 strictly nests P3.
# --------------------------------------------------------------------------

def test_p4_strictly_nests_p3_then_adds_availability():
    p3 = pregame_ladder.TIER_P3_FEATURES
    # Act / Assert: P4 begins with the entire P3 contract, in order...
    assert pi.TIER_P4_FEATURES[: len(p3)] == p3
    # ...and its tail is exactly the five availability columns.
    assert pi.TIER_P4_FEATURES[len(p3):] == pregame_availability.AVAILABILITY_FEATURE_COLUMNS
    assert pi.AVAILABILITY_FEATURES == pregame_availability.AVAILABILITY_FEATURE_COLUMNS


def test_full_p0_through_p4_ladder_is_emitted():
    # The artifact reports the whole forward-chaining ladder, not just P3/P4: P0..P3
    # are reused verbatim from Sprint 4's ladder, with P4 appended on top.
    assert pi.INJURY_TIERS == pregame_ladder.LADDER_TIERS + ("P4",)
    assert set(pi.TIER_FEATURES) == {"P0", "P1", "P2", "P3", "P4"}
    for tier in pregame_ladder.LADDER_TIERS:
        assert pi.TIER_FEATURES[tier] == pregame_ladder.TIER_FEATURES[tier]
        assert pi.TIER_ADDS[tier] == pregame_ladder.TIER_ADDS[tier]


def test_availability_columns_are_registered_continuous_in_the_fitter():
    # The minimal extension to pregame_ladder must mark every availability column
    # continuous, or the shared fitter would pass them through unstandardized.
    for column in pi.AVAILABILITY_FEATURES:
        assert column in pregame_ladder.LADDER_CONTINUOUS_FEATURES


# --------------------------------------------------------------------------
# 2. _gap_close re-measures the market gap against the frozen tier-E baseline.
# --------------------------------------------------------------------------

def _covered_frame() -> pd.DataFrame:
    # Four covered games: the frozen tier-E model (`p_model`) and the market.
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            TARGET: [1.0, 0.0, 1.0, 0.0],
            "p_model": [0.60, 0.45, 0.55, 0.40],
            "market_home_prob": [0.70, 0.35, 0.65, 0.30],
        }
    )


def test_gap_close_reports_the_frozen_tier_e_baseline_gap():
    # Arrange
    covered = _covered_frame()
    preds = pd.Series([0.62, 0.44, 0.58, 0.36], index=pd.Index(covered["game_id"], name="game_id"))
    y = covered[TARGET].to_numpy()

    # Act
    out = pi._gap_close(covered, preds, "p4")

    # Assert: baseline_gap is tier-E Brier minus market Brier, exactly.
    expected_baseline = evaluate.brier_score(
        y, covered["p_model"].to_numpy()
    ) - evaluate.brier_score(y, covered["market_home_prob"].to_numpy())
    assert out["baseline_gap"] == pytest.approx(expected_baseline)
    assert out["new_gap"] == pytest.approx(
        evaluate.brier_score(y, preds.to_numpy())
        - evaluate.brier_score(y, covered["market_home_prob"].to_numpy())
    )
    assert out["n_games"] == 4
    assert "p4_brier" in out and "correlation_p4_market" in out


def test_gap_close_fraction_matches_closed_form():
    # Arrange: give the tier the market's own probabilities, so it closes the gap
    # entirely and the fraction is exactly 1.0.
    covered = _covered_frame()
    market = pd.Series(
        covered["market_home_prob"].to_numpy(),
        index=pd.Index(covered["game_id"], name="game_id"),
    )

    # Act
    out = pi._gap_close(covered, market, "p4")

    # Assert
    assert out["fraction_of_gap_closed"] == pytest.approx(1.0)
    assert out["new_gap"] == pytest.approx(0.0)


def test_gap_close_raises_on_missing_prediction():
    covered = _covered_frame()
    incomplete = pd.Series([0.6], index=pd.Index(["g1"], name="game_id"))
    with pytest.raises(ValueError, match="missing a p4 prediction"):
        pi._gap_close(covered, incomplete, "p4")


# --------------------------------------------------------------------------
# 3. Gates encode the Sprint-4 philosophy.
# --------------------------------------------------------------------------

def _base_metrics(
    *,
    diff_hi: float,
    intercept: float = 0.0,
    slope: float = 1.0,
    pmin: float = 0.05,
    pmax: float = 0.95,
    splits=("train", "validation"),
) -> dict:
    tier = {
        "calibration": {"intercept": intercept, "slope": slope},
        "predictions_min": pmin,
        "predictions_max": pmax,
        "splits_used": list(splits),
    }
    return {
        "tiers": {"P3": dict(tier), "P4": dict(tier)},
        "paired_diff": {"P4_minus_P3": {"brier": {"point": -0.001, "lo": -0.004, "hi": diff_hi}}},
    }


def test_gate_availability_beats_form_requires_ci_upper_bound_below_zero():
    # Upper bound below zero: availability clearly beats form.
    beats = pi.compute_injury_gates(_base_metrics(diff_hi=-0.0005))
    assert beats["gate_availability_beats_form"] is True
    # Upper bound above zero (the real-data null): not a win.
    ties = pi.compute_injury_gates(_base_metrics(diff_hi=0.0006))
    assert ties["gate_availability_beats_form"] is False


def test_gate_pregame_calibrated_uses_shared_tolerances():
    ok = pi.compute_injury_gates(_base_metrics(diff_hi=0.0, intercept=0.0, slope=1.0))
    assert ok["gate_pregame_calibrated"] is True
    bad_slope = pi.compute_injury_gates(
        _base_metrics(diff_hi=0.0, slope=evaluate.CALIB_SLOPE_HI + 0.1)
    )
    assert bad_slope["gate_pregame_calibrated"] is False


def test_structural_gates_flag_predictions_and_touched_test_season():
    leaked = pi.compute_injury_gates(
        _base_metrics(diff_hi=0.0, splits=("train", "validation", "test"))
    )
    assert leaked["gate_test_season_untouched"] is False
    degenerate = pi.compute_injury_gates(_base_metrics(diff_hi=0.0, pmax=1.0))
    assert degenerate["gate_predictions_in_open_interval"] is False


# --------------------------------------------------------------------------
# 4. Verdict states the null plainly when P4 ~= P3.
# --------------------------------------------------------------------------

def _verdict_metrics(*, beats: bool) -> dict:
    diff_hi = -0.0005 if beats else 0.0006
    ci = {"lo": 0.02, "hi": 0.88, "point": 0.5}
    ortho = {"model_coefficient": 0.0, "model_coefficient_ci": {"lo": -0.4, "hi": 0.35, "point": 0.0}}
    return {
        "gates": {
            "gate_availability_beats_form": beats,
            "gate_pregame_calibrated": True,
        },
        "gap_close": {
            "fraction_of_gap_closed": 0.507,
            "fraction_of_gap_closed_ci": ci,
            "orthogonality": ortho,
        },
        "form_gap_close": {"fraction_of_gap_closed": 0.496},
        "extra_gap_closed": 0.011,
        "paired_diff": {"P4_minus_P3": {"brier": {"point": -0.0005, "lo": -0.0017, "hi": diff_hi}}},
    }


def test_verdict_states_the_null_when_availability_ties_form():
    verdict = pi.injury_verdict(_verdict_metrics(beats=False))
    assert verdict["availability_beats_form"] is False
    assert "does NOT beat" in verdict["summary"]
    assert "null" in verdict["summary"]


def test_verdict_reports_a_win_without_the_null_clause():
    verdict = pi.injury_verdict(_verdict_metrics(beats=True))
    assert verdict["availability_beats_form"] is True
    assert "beats current-season form" in verdict["summary"]
    assert "null" not in verdict["summary"]


# --------------------------------------------------------------------------
# 5. Real-data integration: the true P4 fit.
# --------------------------------------------------------------------------

def _read_or_skip(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"missing {path}; run the winprob pipeline to materialize it")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def real_metrics() -> dict:
    mart = DATA_DIR / "fct_game_states.parquet"
    odds = DATA_DIR / pregame.ODDS_CSV_NAME
    if not (mart.exists() and odds.exists()):
        pytest.skip("missing mart or odds; run the winprob pipeline to materialize them")
    df = _read_or_skip(mart)
    availability = _read_or_skip(DATA_DIR / pi.AVAILABILITY_PARQUET_NAME)
    ratings = _read_or_skip(Path("data/rapm/bayes_ratings.parquet"))
    covered = pregame.covered_games_frame(df, odds)
    return pi.evaluate_injury_ladder(df, availability, ratings, covered)


def test_real_structural_gates_pass(real_metrics):
    assert real_metrics["structural_gates_pass"] is True
    assert set(real_metrics["tiers"]) == {"P0", "P1", "P2", "P3", "P4"}
    for tier in real_metrics["tiers"].values():
        assert 0.0 < tier["predictions_min"] and tier["predictions_max"] < 1.0
        assert set(tier["splits_used"]) == {"train", "validation"}


def test_real_p4_nests_p3_and_scores_finite(real_metrics):
    n_p3 = len(pregame_ladder.TIER_P3_FEATURES)
    assert real_metrics["tiers"]["P4"]["features"][:n_p3] == list(pregame_ladder.TIER_P3_FEATURES)
    assert np.isfinite(real_metrics["tiers"]["P4"]["brier"])
    assert real_metrics["tiers"]["P4"]["n_features"] == len(pi.TIER_P4_FEATURES)


def test_real_baseline_gap_reproduces_sprint4_on_the_same_join(real_metrics):
    sprint4 = DATA_DIR / "pregame_metrics.json"
    if not sprint4.exists():
        pytest.skip("missing Sprint 4 pregame_metrics.json to compare against")
    import json

    recorded = json.loads(sprint4.read_text())["gap_close"]
    # Same covered join => the frozen tier-E baseline gap is identical, and P4's
    # fraction is measured against exactly that stick.
    assert real_metrics["gap_close"]["n_games"] == recorded["n_games"]
    assert real_metrics["gap_close"]["baseline_gap"] == pytest.approx(
        recorded["baseline_gap"], abs=1e-12
    )
    assert real_metrics["form_gap_close"]["fraction_of_gap_closed"] == pytest.approx(
        recorded["fraction_of_gap_closed"], abs=1e-9
    )
