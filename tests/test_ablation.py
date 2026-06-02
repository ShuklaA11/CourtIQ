"""Tests for the Sprint-3 Phase-3 leakage-safe RAPM lineup ablation.

The heavy real-data end-to-end run lives in `python -m winprob.ablation` (and
`./ablation.sh`); these tests pin the contract on small synthetic frames so they
run fast and deterministically. They lock the four things Phase 3 rests on:

1. The A..E tiers are strictly nested and A/B are byte-identical to the Phase-2
   score+time / score+time+possession baselines — so the ladder genuinely builds
   on Phase 2 rather than re-deriving a parallel baseline.
2. `home/away_team_strength` (tier C) is a leakage-safe per-(team, season) pooled
   mean: because `season` is part of the key and splits partition on season, a
   test-season team's strength is a function of test-season rows only — never of
   the fitted train/validation rows.
3. `fit_tier` inherits the Phase-2 leakage guard (never reads test/audit rows)
   and lambda selection (grid argmin of validation game-clustered log-loss), and
   its predictions stay strictly inside (0, 1).
4. The gates separate STRUCTURAL integrity (predictions in (0,1), late-game
   calibration not degraded C->D, every rating strictly prior) from the
   SCIENTIFIC finding (D beats C beyond bootstrap noise, reproduced across rolling
   folds). Only structural failure is an error; a null RAPM finding is a valid
   result, so `structural_gates_pass` — not the scientific verdict — drives exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winprob import ablation, evaluate


# --------------------------------------------------------------------------
# Synthetic frame builders.
# --------------------------------------------------------------------------

def _game_rows(
    game_id: str,
    season: int,
    split: str,
    n: int,
    home_edge: float,
    home_team_id: int,
    away_team_id: int,
    home_net: float,
    away_net: float,
    rng: np.random.Generator,
    home_rated: int = 5,
    away_rated: int = 5,
) -> pd.DataFrame:
    """One synthetic game with every column the ablation reads.

    `home_edge` biases the margin so the home team tends to win when positive.
    `home_net`/`away_net` are the (constant-within-game) on-court lineup net RAPM
    for the two teams, which makes each team-season pooled mean trivial to assert.
    """
    reg_sec = np.linspace(2880.0, 0.0, n)
    margin = home_edge * np.linspace(0.0, 12.0, n) + rng.normal(0.0, 1.0, n)
    home_win = bool(margin[-1] + home_edge > 0.0)
    period = np.clip((4 - np.floor(reg_sec / 720.0)).astype(int), 1, 4)
    return pd.DataFrame(
        {
            "game_id": game_id,
            "season": season,
            "period": period,
            "home_score_differential": margin,
            "regulation_seconds_remaining": reg_sec,
            "home_has_possession": rng.integers(0, 2, n).astype(bool),
            "home_win": home_win,
            "split": split,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_lineup_net_rapm": float(home_net),
            "away_lineup_net_rapm": float(away_net),
            "lineup_net_rapm_differential": float(home_net - away_net),
            "home_rated_players": int(home_rated),
            "away_rated_players": int(away_rated),
            "rapm_source_season": float(season - 1),
        }
    )


def _synthetic_frame(seed: int = 0, include_holdout: bool = True) -> pd.DataFrame:
    """Multi-season train/validation frame following the pinned split definition.

    Team 1 is the strong side (net +4 wherever it plays), team 2 the weak side
    (net -4). Each season pairs a game with team 1 home against one with team 1
    away, so every team has both home and away appearances in every season — which
    is what exercises the pooled team-season mean.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    counter = [0]

    def add(season, split, home_id, away_id):
        home_net = 4.0 if home_id == 1 else -4.0
        away_net = 4.0 if away_id == 1 else -4.0
        edge = 1.0 if home_net > away_net else -1.0
        gid = f"002-{split[:2]}-{counter[0]}"
        counter[0] += 1
        parts.append(
            _game_rows(gid, season, split, 40, edge, home_id, away_id,
                       home_net, away_net, rng)
        )

    for season in (2022, 2023):
        for _ in range(2):
            add(season, "train", 1, 2)
            add(season, "train", 2, 1)
    for _ in range(2):
        add(2024, "validation", 1, 2)
        add(2024, "validation", 2, 1)
    if include_holdout:
        add(2025, "test", 1, 2)
        add(2025, "test", 2, 1)
        add(2021, "audit_only", 1, 2)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# 1. Tier nesting + Phase-2 identity.
# --------------------------------------------------------------------------

def test_tiers_are_strictly_nested():
    order = ["A", "B", "C", "D", "E"]
    for lo, hi in zip(order, order[1:]):
        cols_lo = ablation.TIER_FEATURES[lo]
        cols_hi = ablation.TIER_FEATURES[hi]
        # Each higher tier is the lower tier's columns as a prefix, plus more.
        assert cols_hi[: len(cols_lo)] == cols_lo
        assert len(cols_hi) > len(cols_lo)


def test_tier_a_and_b_match_phase2_baselines():
    # The ladder must nest ON Phase 2, not fork a parallel baseline.
    assert ablation.TIER_FEATURES["A"] == evaluate.SCORE_TIME_FEATURES
    assert ablation.TIER_FEATURES["B"] == evaluate.SCORE_TIME_POSSESSION_FEATURES


def test_baseline_tiers_carry_no_rating_columns():
    rating_markers = ("rapm", "team_strength", "rated_players", "lineup")
    for tier in ("A", "B"):
        for name in ablation.TIER_FEATURES[tier]:
            assert not any(mark in name for mark in rating_markers)


def test_rapm_columns_enter_at_the_expected_tiers():
    assert "home_team_strength" in ablation.TIER_FEATURES["C"]
    assert "away_team_strength" in ablation.TIER_FEATURES["C"]
    assert "home_team_strength" not in ablation.TIER_FEATURES["B"]
    assert "lineup_net_rapm_differential" in ablation.TIER_FEATURES["D"]
    assert "lineup_net_rapm_differential" not in ablation.TIER_FEATURES["C"]
    assert "home_rated_players" in ablation.TIER_FEATURES["E"]
    assert "home_rated_players" not in ablation.TIER_FEATURES["D"]


# --------------------------------------------------------------------------
# 2. Team-strength construction + leakage safety.
# --------------------------------------------------------------------------

def test_team_strength_is_pooled_mean_over_home_and_away_appearances():
    df = _synthetic_frame(seed=1)
    out = ablation.add_team_strength_columns(df)

    # Team 1 in 2022: it is the strong side (+4 net) whether home or away, so its
    # pooled team-season mean is exactly 4.0; team 2 is -4.0.
    m22 = out[out["season"] == 2022]
    t1_home = m22[m22["home_team_id"] == 1]["home_team_strength"]
    t1_away = m22[m22["away_team_id"] == 1]["away_team_strength"]
    assert t1_home.to_numpy() == pytest.approx(4.0)
    assert t1_away.to_numpy() == pytest.approx(4.0)
    t2_home = m22[m22["home_team_id"] == 2]["home_team_strength"]
    assert t2_home.to_numpy() == pytest.approx(-4.0)


def test_add_team_strength_columns_is_immutable():
    df = _synthetic_frame(seed=2)
    before = df.copy(deep=True)
    _ = ablation.add_team_strength_columns(df)
    pd.testing.assert_frame_equal(df, before)  # input untouched


def test_team_strength_has_no_cross_split_leakage():
    # A test-season team's strength must be a function of test-season rows only.
    # Since `season` is part of the key, computing on the full frame must equal
    # computing on the test slice alone for every test row.
    df = _synthetic_frame(seed=3)
    full = ablation.add_team_strength_columns(df)
    test_only = ablation.add_team_strength_columns(
        df[df["split"] == "test"].reset_index(drop=True)
    )
    full_test = full[full["split"] == "test"].reset_index(drop=True)
    for col in ("home_team_strength", "away_team_strength"):
        assert full_test[col].to_numpy() == pytest.approx(
            test_only[col].to_numpy()
        )


def test_new_continuous_features_are_standardized():
    # Team strength, the lineup differential, and coverage counts are continuous
    # and must be marked for standardization; the intercept and possession must not.
    for name in ("home_team_strength", "lineup_net_rapm_differential",
                 "home_rated_players"):
        assert name in ablation.ABLATION_CONTINUOUS_FEATURES
    assert "intercept" not in ablation.ABLATION_CONTINUOUS_FEATURES
    assert "home_has_possession" not in ablation.ABLATION_CONTINUOUS_FEATURES


# --------------------------------------------------------------------------
# 3. Per-tier fit: leakage guard, lambda selection, valid predictions.
# --------------------------------------------------------------------------

def test_fit_tier_never_reads_test_or_audit_rows():
    df = _synthetic_frame(seed=4, include_holdout=True)
    prepared = ablation.add_team_strength_columns(df)
    fit = ablation.fit_tier(prepared, "D", ablation.TIER_FEATURES["D"],
                            grid=(0.01, 1.0))
    assert fit.splits_used == frozenset({"train", "validation"})
    assert fit.holdout_rows_excluded > 0


def test_fit_tier_predictions_strictly_in_open_interval():
    df = ablation.add_team_strength_columns(_synthetic_frame(seed=5))
    test = df[df["split"] == "test"].reset_index(drop=True)
    for tier in ("A", "C", "D", "E"):
        fit = ablation.fit_tier(df, tier, ablation.TIER_FEATURES[tier],
                                grid=(0.01, 1.0))
        p = ablation.predict_tier(fit, test)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)


def test_fit_tier_selects_lambda_as_grid_argmin_of_validation_log_loss():
    df = ablation.add_team_strength_columns(_synthetic_frame(seed=6))
    grid = (1e-4, 1e-2, 1e0, 1e2)
    fit = ablation.fit_tier(df, "D", ablation.TIER_FEATURES["D"], grid=grid)
    argmin = int(np.argmin(fit.validation_log_loss))
    assert fit.chosen_index == argmin
    assert fit.chosen_lambda == grid[argmin]


def test_fit_tier_coefficients_are_finite_and_aligned():
    df = ablation.add_team_strength_columns(_synthetic_frame(seed=7))
    fit = ablation.fit_tier(df, "E", ablation.TIER_FEATURES["E"], grid=(0.1, 1.0))
    assert np.all(np.isfinite(fit.beta))
    assert fit.beta.shape[0] == len(fit.feature_names)


# --------------------------------------------------------------------------
# 4. Analysis outputs: Brier decomposition + where RAPM helped.
# --------------------------------------------------------------------------

def test_brier_decomposition_reconstructs_the_brier_score():
    # Murphy decomposition: Brier = Reliability - Resolution + Uncertainty, with
    # Uncertainty a property of the outcomes alone. The binned reconstruction must
    # recover the raw Brier to within binning error.
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(np.float64)
    d = ablation.brier_decomposition(y, p)

    base = float(y.mean())
    assert d["uncertainty"] == pytest.approx(base * (1.0 - base))
    assert d["brier_reconstructed"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"]
    )
    assert d["brier_reconstructed"] == pytest.approx(d["brier"], abs=0.01)


def test_subgroup_improvement_flags_where_rapm_helped():
    # The "and where" deliverable: per-subgroup C-brier minus D-brier, so a
    # positive improvement means the lineup tier D scored better in that bucket.
    bd_c = {"by_time_remaining": [
        {"bucket": "final_2min", "n": 300, "brier": 0.20, "log_loss": 0.60},
        {"bucket": "24to48min", "n": 900, "brier": 0.12, "log_loss": 0.40},
    ]}
    bd_d = {"by_time_remaining": [
        {"bucket": "final_2min", "n": 300, "brier": 0.18, "log_loss": 0.55},
        {"bucket": "24to48min", "n": 900, "brier": 0.121, "log_loss": 0.401},
    ]}
    imp = ablation.subgroup_improvement(bd_c, bd_d)
    rows = {r["bucket"]: r for r in imp["by_time_remaining"]}
    assert rows["final_2min"]["brier_improvement"] == pytest.approx(0.02)
    assert rows["final_2min"]["log_loss_improvement"] == pytest.approx(0.05)
    # A bucket where D is essentially unchanged is not counted as a help.
    assert rows["24to48min"]["brier_improvement"] == pytest.approx(-0.001)


def test_subgroup_improvement_skips_empty_buckets():
    bd_c = {"by_period": [{"bucket": "OT", "n": 0}]}
    bd_d = {"by_period": [{"bucket": "OT", "n": 0}]}
    imp = ablation.subgroup_improvement(bd_c, bd_d)
    assert imp["by_period"] == []


# --------------------------------------------------------------------------
# 5. Gates: structural integrity vs the scientific finding.
# --------------------------------------------------------------------------

def _clean_metrics(d_beats_c: bool, reproduces: bool) -> dict:
    """Hand-built metrics dict exercising the gate logic without a full run.

    A negative `*_diff` point/hi means D scored lower (better) than C.
    """
    diff_hi = -0.001 if d_beats_c else 0.002
    fold_point = -0.001 if reproduces else 0.001
    return {
        "tiers": {
            t: {"predictions_min": 0.02, "predictions_max": 0.98}
            for t in ablation.TIER_NAMES
        },
        "paired_diff": {
            "D_minus_C": {
                "brier": {"lo": -0.003, "hi": diff_hi, "point": -0.002},
                "log_loss": {"lo": -0.01, "hi": 0.005, "point": -0.003},
            }
        },
        "rolling": [
            {"fold": "A", "d_minus_c_brier_point": -0.001},
            {"fold": "B", "d_minus_c_brier_point": fold_point},
        ],
        "late_game_calibration": {"C": 0.03, "D": 0.03},
        "rating_provenance_ok": True,
    }


def test_structural_gates_pass_on_clean_metrics():
    metrics = _clean_metrics(d_beats_c=True, reproduces=True)
    gates = ablation.compute_ablation_gates(metrics)
    for name in ablation.STRUCTURAL_GATE_NAMES:
        assert gates[name] is True
    assert ablation.structural_gates_pass(metrics) is True


def test_scientific_gate_flags_rapm_improvement_when_ci_excludes_zero():
    beats = ablation.compute_ablation_gates(_clean_metrics(True, True))
    null = ablation.compute_ablation_gates(_clean_metrics(False, True))
    assert beats["gate_rapm_beats_team_strength"] is True
    assert null["gate_rapm_beats_team_strength"] is False


def test_scientific_gate_requires_reproduction_across_folds():
    reproduced = ablation.compute_ablation_gates(_clean_metrics(True, True))
    flipped = ablation.compute_ablation_gates(_clean_metrics(True, False))
    assert reproduced["gate_reproduces_rolling"] is True
    assert flipped["gate_reproduces_rolling"] is False


def test_a_null_rapm_finding_is_not_a_structural_failure():
    # RAPM adding nothing (scientific gate False) must still leave structural
    # integrity intact — the run reports the null, it does not error out.
    metrics = _clean_metrics(d_beats_c=False, reproduces=False)
    assert ablation.structural_gates_pass(metrics) is True
    gates = ablation.compute_ablation_gates(metrics)
    assert gates["gate_rapm_beats_team_strength"] is False


def test_late_game_calibration_degradation_fails_structural_gate():
    metrics = _clean_metrics(True, True)
    metrics["late_game_calibration"] = {"C": 0.03, "D": 0.09}  # D much worse late
    gates = ablation.compute_ablation_gates(metrics)
    assert gates["gate_late_game_calibration_not_degraded"] is False
    assert ablation.structural_gates_pass(metrics) is False


def test_rating_provenance_violation_fails_structural_gate():
    metrics = _clean_metrics(True, True)
    metrics["rating_provenance_ok"] = False
    gates = ablation.compute_ablation_gates(metrics)
    assert gates["gate_every_rating_strictly_prior"] is False
    assert ablation.structural_gates_pass(metrics) is False


def test_predictions_out_of_interval_fails_structural_gate():
    metrics = _clean_metrics(True, True)
    metrics["tiers"]["D"]["predictions_max"] = 1.0  # saturated
    gates = ablation.compute_ablation_gates(metrics)
    assert gates["gate_predictions_in_open_interval"] is False
    assert ablation.structural_gates_pass(metrics) is False
