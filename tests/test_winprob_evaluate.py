"""Tests for the out-of-sample win-probability evaluation + Phase-2 gates.

The heavy real-data run lives in `python -m winprob.evaluate`; these tests pin
the properties the spec requires on hand examples and small synthetic frames:
Brier and log loss match a by-hand computation, the calibration recalibration
recovers intercept 0 / slope 1 on perfectly-calibrated predictions, the
game-clustered bootstrap resampler resamples whole games (never rows), and the
emitted `gates` dict is present with strictly boolean values.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from winprob import evaluate, model


# --------------------------------------------------------------------------
# Brier and log loss on a hand example.
# --------------------------------------------------------------------------

def test_brier_score_matches_hand_computation():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.8, 0.3, 0.6])
    # (0.8-1)^2 + (0.3-0)^2 + (0.6-1)^2 = 0.04 + 0.09 + 0.16 = 0.29, /3.
    expected = (0.04 + 0.09 + 0.16) / 3.0
    assert evaluate.brier_score(y, p) == pytest.approx(expected)


def test_log_loss_matches_hand_computation():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.8, 0.3, 0.6])
    expected = -(np.log(0.8) + np.log(1.0 - 0.3) + np.log(0.6)) / 3.0
    assert evaluate.mean_log_loss(y, p) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Calibration: perfectly-calibrated predictions -> intercept 0, slope 1.
# --------------------------------------------------------------------------

def test_calibration_recovers_intercept_zero_slope_one_when_perfect():
    # Exact construction: for each p_k = k/100 (k = 1..99), emit 100 rows with
    # exactly k ones. The empirical rate at every predicted probability equals
    # the prediction, so the logistic-recalibration score equations are solved
    # exactly at (intercept, slope) = (0, 1) — the unique MLE.
    p_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for k in range(1, 100):
        p_k = k / 100.0
        p_parts.append(np.full(100, p_k))
        y_parts.append(np.concatenate([np.ones(k), np.zeros(100 - k)]))
    p = np.concatenate(p_parts)
    y = np.concatenate(y_parts)

    intercept, slope = evaluate.fit_calibration(y, p)
    assert intercept == pytest.approx(0.0, abs=1e-3)
    assert slope == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# Game-clustered bootstrap resampler resamples by game, not by row.
# --------------------------------------------------------------------------

def test_resampler_resamples_whole_games_not_rows():
    game_ids = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"])
    game_rows = {"A": [0, 1, 2], "B": [3, 4], "C": [5, 6, 7, 8, 9]}
    sizes = {g: len(r) for g, r in game_rows.items()}

    rng = np.random.default_rng(0)
    idx = evaluate.resample_game_indices(game_ids, rng)
    row_counts = Counter(idx.tolist())

    # Every row of a game appears the SAME number of times -> whole-game blocks,
    # never a partial (row-level) resample.
    appearances: dict[str, int] = {}
    for g, rows in game_rows.items():
        counts = {row_counts.get(r, 0) for r in rows}
        assert len(counts) == 1, f"game {g} was split across rows: {counts}"
        appearances[g] = counts.pop()

    # k games drawn with replacement out of k unique games.
    assert sum(appearances.values()) == len(game_rows)
    # Total rows == sum of the drawn games' sizes (rows come in game-sized blocks).
    assert len(idx) == sum(appearances[g] * sizes[g] for g in sizes)


def test_resampler_can_draw_a_game_more_than_once():
    # With replacement, some game must repeat across many seeds (proving draws
    # are with replacement over games).
    game_ids = np.array(["A", "A", "B", "B", "C", "C"])
    saw_repeat = False
    for seed in range(20):
        rng = np.random.default_rng(seed)
        idx = evaluate.resample_game_indices(game_ids, rng)
        drawn = Counter(game_ids[idx].tolist())
        # counts are in multiples of the game size (2 here); >2 means repeated.
        if any(c > 2 for c in drawn.values()):
            saw_repeat = True
            break
    assert saw_repeat


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    n_games, per_game = 30, 10
    game_ids = np.repeat([f"g{i}" for i in range(n_games)], per_game)
    y = rng.integers(0, 2, n_games * per_game).astype(float)
    p = np.full_like(y, 0.5)
    boot = evaluate.game_clustered_bootstrap(
        y, {"model": p}, game_ids, n_boot=200, seed=7
    )
    ci = boot["brier"]["model"]
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert "n_boot" in boot and boot["n_boot"] == 200


# --------------------------------------------------------------------------
# End-to-end: gates dict present, every value a bool.
# --------------------------------------------------------------------------

def _game_rows(game_id: str, season: int, split: str, n: int, home_edge: float,
               rng: np.random.Generator) -> pd.DataFrame:
    reg_sec = np.linspace(2880.0, 0.0, n)
    margin = home_edge * np.linspace(0.0, 12.0, n) + rng.normal(0.0, 1.0, n)
    home_win = bool(margin[-1] + home_edge > 0.0)
    return pd.DataFrame({
        "game_id": game_id,
        "season": season,
        "period": np.clip((4 * np.arange(n) / n).astype(int) + 1, 1, 4),
        "home_score_differential": margin,
        "regulation_seconds_remaining": reg_sec,
        "home_has_possession": rng.integers(0, 2, n).astype(bool),
        "home_win": home_win,
        "split": split,
    })


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for g in range(6):
        parts.append(_game_rows(f"002-tr-{g}", 2022 + g % 2, "train", 40,
                                1.0 if g % 2 == 0 else -1.0, rng))
    for g in range(4):
        parts.append(_game_rows(f"002-va-{g}", 2024, "validation", 40,
                                1.0 if g % 2 == 0 else -1.0, rng))
    for g in range(4):
        parts.append(_game_rows(f"002-te-{g}", 2025, "test", 40,
                                1.0 if g % 2 == 0 else -1.0, rng))
    parts.append(_game_rows("002-au-0", 2021, "audit_only", 40, -1.0, rng))
    return pd.concat(parts, ignore_index=True)


def _fit_model_doc(df: pd.DataFrame) -> dict:
    result = model.select_and_fit(df, grid=(0.01, 1.0))
    return model.model_payload(result, dataset_sha256="deadbeef")


def test_gates_dict_present_and_all_boolean():
    df = _synthetic_frame(seed=2)
    model_doc = _fit_model_doc(df)
    metrics = evaluate.evaluate(df, model_doc)

    assert "gates" in metrics
    expected_gates = {
        "gate_brier_beats_score_time",
        "gate_logloss_beats_score_time",
        "gate_calibration_intercept_near_zero",
        "gate_calibration_slope_near_one",
        "gate_predictions_in_open_interval",
        "gate_no_material_phase_miscalibration",
    }
    assert set(metrics["gates"]) == expected_gates
    for name, value in metrics["gates"].items():
        assert isinstance(value, bool), f"gate {name} is not a bool: {type(value)}"


def test_predictions_gate_reflects_open_interval():
    df = _synthetic_frame(seed=3)
    model_doc = _fit_model_doc(df)
    metrics = evaluate.evaluate(df, model_doc)
    # The guarded sigmoid guarantees strict (0, 1); the gate must observe it.
    assert 0.0 < metrics["predictions_min"]
    assert metrics["predictions_max"] < 1.0
    assert metrics["gates"]["gate_predictions_in_open_interval"] is True


def test_evaluate_scores_all_three_baselines_leakage_safe():
    df = _synthetic_frame(seed=4)
    model_doc = _fit_model_doc(df)
    metrics = evaluate.evaluate(df, model_doc)
    assert set(metrics["baselines"]) == {
        "base_rate", "score_time", "score_time_possession"
    }
    # The base-rate baseline is a single constant probability on every test row.
    test = df.loc[df["split"] == "test"].reset_index(drop=True)
    base = evaluate.fit_baseline(
        model.working_frame(df).frame, "base_rate", evaluate.BASE_RATE_FEATURES
    )
    p_base = evaluate.predict_with_baseline(base, test)
    assert np.allclose(p_base, p_base[0])
