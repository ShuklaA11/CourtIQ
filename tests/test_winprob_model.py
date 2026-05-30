"""Tests for the win-probability logistic fit + lambda selection.

The heavy real-data end-to-end run lives in `python -m winprob.model`; these
tests pin the properties the spec requires on small synthetic frames so they run
fast and deterministically: predictions strictly in (0, 1), finite coefficients,
the selected lambda equals the grid argmin of validation game-clustered log-loss,
the fit never reads test/audit_only rows, and a separable dataset is learned with
low training loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winprob import model


# --------------------------------------------------------------------------
# Synthetic frame builders.
# --------------------------------------------------------------------------

def _game_rows(game_id: str, season: int, split: str, n: int, home_edge: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """A single synthetic game's worth of game-state rows.

    `home_edge` biases the margin so the home team tends to win when positive.
    Only the columns `winprob.features.REQUIRED_COLUMNS` needs plus the target
    and `split` are produced.
    """
    reg_sec = np.linspace(2880.0, 0.0, n)
    margin = home_edge * np.linspace(0.0, 12.0, n) + rng.normal(0.0, 1.0, n)
    home_win = bool(margin[-1] + home_edge > 0.0)
    return pd.DataFrame(
        {
            "game_id": game_id,
            "season": season,
            "home_score_differential": margin,
            "regulation_seconds_remaining": reg_sec,
            "home_has_possession": rng.integers(0, 2, n).astype(bool),
            "home_win": home_win,
            "split": split,
        }
    )


def _synthetic_frame(seed: int = 0, include_holdout: bool = True) -> pd.DataFrame:
    """Multi-season train/validation frame, optionally with test/audit rows.

    Seasons follow the pinned split definition (2022-2023 train, 2024 validation,
    2025 test, 2021 audit) so season dummies and split filtering behave as in
    production.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for g in range(6):
        parts.append(_game_rows(f"002-tr-{g}", 2022 + g % 2, "train", 40,
                                 home_edge=1.0 if g % 2 == 0 else -1.0, rng=rng))
    for g in range(4):
        parts.append(_game_rows(f"002-va-{g}", 2024, "validation", 40,
                                 home_edge=1.0 if g % 2 == 0 else -1.0, rng=rng))
    if include_holdout:
        parts.append(_game_rows("002-te-0", 2025, "test", 40, home_edge=1.0, rng=rng))
        parts.append(_game_rows("002-au-0", 2021, "audit_only", 40, home_edge=-1.0, rng=rng))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Property tests.
# --------------------------------------------------------------------------

def test_predictions_strictly_in_open_unit_interval():
    df = _synthetic_frame(seed=1)
    result = model.select_and_fit(df, grid=(0.01, 1.0))
    # Rebuild the working design and predict on every fit row.
    work = model.working_frame(df).frame
    from winprob import features

    X, _ = features.build_design_matrix(work)
    p = model.predict_proba(X, result.beta)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)


def test_guarded_sigmoid_never_saturates_to_zero_or_one():
    z = np.array([-1e6, -50.0, 0.0, 50.0, 1e6])
    p = model.guarded_sigmoid(z)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)
    assert np.all(np.isfinite(p))


def test_coefficients_are_finite():
    df = _synthetic_frame(seed=2)
    result = model.select_and_fit(df, grid=(0.001, 0.1, 10.0))
    assert np.all(np.isfinite(result.beta))
    assert result.beta.shape[0] == len(result.feature_names)


def test_selected_lambda_is_grid_argmin_of_validation_log_loss():
    df = _synthetic_frame(seed=3)
    grid = (1e-4, 1e-2, 1e0, 1e2)
    result = model.select_and_fit(df, grid=grid)
    argmin = int(np.argmin(result.validation_log_loss))
    assert result.chosen_index == argmin
    assert result.chosen_lambda == grid[argmin]
    assert len(result.validation_log_loss) == len(grid)


def test_model_payload_exposes_lambda_and_lambda_grid_contract_keys():
    # The serialized document must carry top-level `lambda` (the chosen scalar)
    # and `lambda_grid` (per-lambda validation log-loss), with the chosen lambda
    # being the argmin of the log-loss recorded in `lambda_grid`.
    df = _synthetic_frame(seed=6)
    grid = (1e-4, 1e-2, 1e0, 1e2)
    result = model.select_and_fit(df, grid=grid)
    doc = model.model_payload(result, dataset_sha256="deadbeef")

    assert "lambda" in doc
    assert "lambda_grid" in doc
    assert doc["lambda"] == result.chosen_lambda

    losses = [entry["validation_log_loss"] for entry in doc["lambda_grid"]]
    argmin = int(np.argmin(losses))
    assert doc["lambda_grid"][argmin]["lambda"] == doc["lambda"]
    assert [entry["lambda"] for entry in doc["lambda_grid"]] == list(grid)


def test_fit_never_reads_test_or_audit_rows():
    df = _synthetic_frame(seed=4, include_holdout=True)
    n_holdout = int(df["split"].isin(model.HOLDOUT_SPLITS).sum())
    assert n_holdout > 0  # the frame really does contain holdout rows

    result = model.select_and_fit(df, grid=(0.1, 1.0))
    # By construction the working frame carries only train/validation...
    assert result.splits_used == frozenset({"train", "validation"})
    # ...and every holdout row was excluded and counted.
    assert result.holdout_rows_excluded == n_holdout
    assert result.n_fit == len(df) - n_holdout


def test_working_frame_rejects_leaked_holdout_split():
    # A frame whose filtering was bypassed must never silently pass through.
    df = _synthetic_frame(seed=5)
    wf = model.working_frame(df)
    assert not (wf.splits_used & model.HOLDOUT_SPLITS)


def test_separable_dataset_learned_with_low_training_loss():
    # A cleanly separable design (intercept + one strong feature) should reach
    # near-zero training log-loss at light regularization.
    rng = np.random.default_rng(7)
    n = 400
    signal = rng.normal(0.0, 1.0, n)
    X = np.column_stack([np.ones(n), signal])  # intercept + feature
    y = (signal > 0.0).astype(np.float64)
    pen = model.penalty_mask(["intercept", "signal"])

    beta = model.fit_l2_logistic(X, y, lam=1e-6, pen_mask=pen)
    p = model.predict_proba(X, beta)
    train_loss = float(np.mean(model.row_log_loss(y, p)))
    assert np.all(np.isfinite(beta))
    assert train_loss < 0.05


def test_intercept_is_not_penalized():
    # With only an intercept column and a base rate p != 0.5, an intercept-free
    # penalty must still recover the log-odds even at large lambda.
    n = 1000
    y = np.zeros(n)
    y[: int(0.8 * n)] = 1.0  # base rate 0.8
    X = np.ones((n, 1))
    pen = model.penalty_mask(["intercept"])
    beta = model.fit_l2_logistic(X, y, lam=100.0, pen_mask=pen)
    # log-odds of 0.8 ~= 1.386; an unpenalized intercept recovers it.
    assert beta[0] == pytest.approx(np.log(0.8 / 0.2), abs=0.05)


def test_game_clustered_log_loss_weights_games_equally():
    # Two games of very different length must contribute equally, not per-row.
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # last row (game B) is wrong
    games = np.array(["A", "A", "A", "A", "A", "B"])
    loss = model.game_clustered_log_loss(y, p, games)
    loss_a = -np.log(0.9)
    loss_b = -np.log(0.1)
    assert loss == pytest.approx((loss_a + loss_b) / 2.0)
