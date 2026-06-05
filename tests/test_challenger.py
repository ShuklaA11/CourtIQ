"""Tests for the Sprint-3 Phase-4 gradient-boosted challenger.

The heavy real-data run lives in `python -m winprob.challenger` (and
`./challenger.sh`); these tests pin the contract on small synthetic frames so
they stay fast and deterministic. They lock what Phase 4 rests on:

1. The GBM races the tier-E logistic on the IDENTICAL feature information — the
   apples-to-apples nonlinearity test — using the tier-E columns minus the
   constant intercept (which the GBM's base score subsumes).
2. Fitting is leakage-safe: selection reads train, monitors validation, refits on
   train+validation, and NEVER touches the 2025 test / 2021 audit rows.
3. Both models' predictions stay strictly inside (0, 1); the paired
   game-clustered bootstrap plumbing yields a gbm-minus-logistic difference CI.
4. Gates separate STRUCTURAL integrity (predictions valid, holdout untouched,
   ratings strictly prior) — which drives the exit code — from the SCIENTIFIC
   adoption verdict (GBM beats the logistic AND stays calibrated). A null is a
   valid result: when the GBM does not clearly beat the logistic, the verdict is
   RETAIN the logistic, and the run still exits 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob import challenger


# --------------------------------------------------------------------------
# Synthetic frame following the pinned split definition, with enough test games
# for a non-degenerate game-clustered bootstrap.
# --------------------------------------------------------------------------

def _game(gid, season, split, n, edge, home_id, away_id, home_net, away_net, rng):
    reg_sec = np.linspace(2880.0, 0.0, n)
    margin = edge * np.linspace(0.0, 12.0, n) + rng.normal(0.0, 1.0, n)
    home_win = bool(margin[-1] + edge > 0.0)
    period = np.clip((4 - np.floor(reg_sec / 720.0)).astype(int), 1, 4)
    return pd.DataFrame(
        {
            "game_id": gid,
            "season": season,
            "period": period,
            "home_score_differential": margin,
            "regulation_seconds_remaining": reg_sec,
            "home_has_possession": rng.integers(0, 2, n).astype(bool),
            "home_win": home_win,
            "split": split,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_lineup_net_rapm": float(home_net),
            "away_lineup_net_rapm": float(away_net),
            "lineup_net_rapm_differential": float(home_net - away_net),
            "home_rated_players": 5,
            "away_rated_players": 5,
            "rapm_source_season": float(season - 1),
        }
    )


def _frame(seed: int = 0, games_per_slot: int = 6) -> pd.DataFrame:
    """Multi-season frame; team 1 strong (+4), team 2 weak (-4), both venues."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    counter = [0]

    def add(season, split, home_id, away_id):
        home_net = 4.0 if home_id == 1 else -4.0
        away_net = 4.0 if away_id == 1 else -4.0
        edge = 1.0 if home_net > away_net else -1.0
        gid = f"002-{split[:2]}-{counter[0]}"
        counter[0] += 1
        parts.append(_game(gid, season, split, 40, edge, home_id, away_id,
                           home_net, away_net, rng))

    for season in (2022, 2023):
        for _ in range(games_per_slot):
            add(season, "train", 1, 2)
            add(season, "train", 2, 1)
    for _ in range(games_per_slot):
        add(2024, "validation", 1, 2)
        add(2024, "validation", 2, 1)
    for _ in range(games_per_slot):
        add(2025, "test", 1, 2)
        add(2025, "test", 2, 1)
    add(2021, "audit_only", 1, 2)
    return pd.concat(parts, ignore_index=True)


# A small, fast selection grid for the end-to-end tests.
_TINY_GRID = challenger.SelectionGrid(
    learning_rates=(0.1,),
    max_depths=(2,),
    n_trees_cap=8,
    n_bins=8,
    leaf_l2=1.0,
    min_samples_leaf=20,
)


# --------------------------------------------------------------------------
# 1. Feature contract: apples-to-apples with the tier-E logistic.
# --------------------------------------------------------------------------

def test_gbm_features_are_tier_e_minus_intercept():
    from winprob import ablation

    assert "intercept" not in challenger.GBM_FEATURES
    # Everything else in tier E is present, in order.
    expected = tuple(f for f in ablation.TIER_E_FEATURES if f != "intercept")
    assert challenger.GBM_FEATURES == expected


def test_gbm_features_carry_the_rating_columns():
    for name in ("home_team_strength", "away_team_strength",
                 "lineup_net_rapm_differential", "home_rated_players"):
        assert name in challenger.GBM_FEATURES


# --------------------------------------------------------------------------
# 2. Leakage safety.
# --------------------------------------------------------------------------

def test_selection_never_touches_holdout():
    df = _frame(seed=1)
    prepared = challenger.add_team_strength_columns(df)
    sel = challenger.select_gbm_config(prepared, grid=_TINY_GRID)
    assert set(sel.splits_used).issubset({"train", "validation"})
    assert sel.holdout_rows_excluded > 0  # 2025 + 2021 rows were excluded.


def test_metrics_record_holdout_excluded_and_splits_used():
    df = _frame(seed=2)
    metrics = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    assert metrics["holdout_rows_excluded"] > 0
    assert set(metrics["splits_used_for_fit"]).issubset({"train", "validation"})


# --------------------------------------------------------------------------
# 3. Prediction guarantees + paired bootstrap plumbing.
# --------------------------------------------------------------------------

def test_both_models_predict_inside_unit_interval():
    df = _frame(seed=3)
    metrics = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    for key in ("gbm", "logistic"):
        assert metrics[key]["predictions_min"] > 0.0
        assert metrics[key]["predictions_max"] < 1.0


def test_paired_diff_has_gbm_minus_logistic_interval():
    df = _frame(seed=4)
    metrics = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    diff = metrics["paired_diff"]["gbm_minus_logistic"]
    for metric in ("brier", "log_loss"):
        assert set(diff[metric]) == {"lo", "hi", "point"}
        assert diff[metric]["lo"] <= diff[metric]["point"] <= diff[metric]["hi"]


def test_evaluation_is_deterministic():
    df = _frame(seed=5)
    m1 = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    m2 = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    assert m1["gbm"]["brier"] == m2["gbm"]["brier"]
    assert m1["logistic"]["brier"] == m2["logistic"]["brier"]


# --------------------------------------------------------------------------
# 4. Gates + verdict logic (on hand-built metrics dicts, deterministic).
# --------------------------------------------------------------------------

def _metrics_stub(
    beats: bool, calibrated: bool, miscal: list, provenance: bool = True,
    splits=("train", "validation"), gbm_pmin=0.01, gbm_pmax=0.99,
) -> dict:
    """A minimal metrics dict exercising just the gate/verdict branches."""
    hi = -0.001 if beats else 0.001
    intercept = 0.0 if calibrated else 1.0
    slope = 1.0 if calibrated else 0.2
    return {
        "gbm": {
            "predictions_min": gbm_pmin,
            "predictions_max": gbm_pmax,
            "calibration": {"intercept": intercept, "slope": slope},
        },
        "logistic": {"predictions_min": 0.01, "predictions_max": 0.99},
        "paired_diff": {
            "gbm_minus_logistic": {
                "brier": {"lo": hi - 0.001, "hi": hi, "point": hi - 0.0005},
                "log_loss": {"lo": 0.001, "hi": 0.002, "point": 0.0015},
            }
        },
        "material_phase_miscalibration": miscal,
        "rating_provenance_ok": provenance,
        "splits_used_for_fit": list(splits),
    }


def test_gates_pass_when_gbm_beats_and_calibrated():
    gates = challenger.compute_challenger_gates(
        _metrics_stub(beats=True, calibrated=True, miscal=[])
    )
    assert gates["gate_gbm_beats_logistic"] is True
    assert gates["gate_gbm_calibrated_global"] is True
    assert gates["gate_gbm_no_material_phase_miscalibration"] is True
    assert gates["gate_predictions_in_open_interval"] is True
    assert gates["gate_holdout_untouched"] is True
    assert gates["gate_every_rating_strictly_prior"] is True


def test_beats_gate_reads_upper_bound_below_zero():
    beat = challenger.compute_challenger_gates(_metrics_stub(True, True, []))
    null = challenger.compute_challenger_gates(_metrics_stub(False, True, []))
    assert beat["gate_gbm_beats_logistic"] is True
    assert null["gate_gbm_beats_logistic"] is False


def test_verdict_retains_logistic_on_null():
    metrics = _metrics_stub(beats=False, calibrated=True, miscal=[])
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    verdict = challenger.challenger_verdict(metrics)
    assert verdict["adopt_gbm"] is False
    assert verdict["retain_logistic"] is True


def test_verdict_adopts_gbm_only_when_beats_and_calibrated():
    metrics = _metrics_stub(beats=True, calibrated=True, miscal=[])
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    assert challenger.challenger_verdict(metrics)["adopt_gbm"] is True


def test_miscalibrated_gbm_is_not_adopted_even_if_it_beats():
    metrics = _metrics_stub(beats=True, calibrated=True, miscal=[{"group": "x"}])
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    assert challenger.challenger_verdict(metrics)["adopt_gbm"] is False


# --------------------------------------------------------------------------
# 5. Structural gates drive exit; a null is not a structural failure.
# --------------------------------------------------------------------------

def test_structural_gates_independent_of_scientific_finding():
    # Null finding, but integrity intact -> structural gates still pass (exit 0).
    metrics = _metrics_stub(beats=False, calibrated=True, miscal=[])
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    assert challenger.structural_gates_pass(metrics) is True


def test_structural_failure_on_holdout_leak():
    metrics = _metrics_stub(beats=True, calibrated=True, miscal=[],
                            splits=("train", "validation", "test"))
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    assert metrics["gates"]["gate_holdout_untouched"] is False
    assert challenger.structural_gates_pass(metrics) is False


def test_structural_failure_on_rating_leak():
    metrics = _metrics_stub(beats=True, calibrated=True, miscal=[], provenance=False)
    metrics["gates"] = challenger.compute_challenger_gates(metrics)
    assert challenger.structural_gates_pass(metrics) is False


# --------------------------------------------------------------------------
# 6. End-to-end assembly on synthetic data runs and stays structurally sound.
# --------------------------------------------------------------------------

def test_end_to_end_structural_gates_pass_on_synthetic():
    df = _frame(seed=7)
    metrics = challenger.evaluate_challenger(df, grid=_TINY_GRID)
    assert challenger.structural_gates_pass(metrics)
    # The verdict key is always present and internally consistent.
    assert metrics["verdict"]["retain_logistic"] == (not metrics["verdict"]["adopt_gbm"])
