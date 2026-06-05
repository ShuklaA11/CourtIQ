"""Tests for the hand-rolled histogram Newton gradient-boosting classifier.

The engine in `winprob.gbm` is the Phase-4 nonlinear challenger's core: a pure
numpy, second-order (Newton) gradient-boosted tree ensemble on the log-loss, with
histogram-binned split search so it stays tractable on ~800k rows without any
third-party ML dependency. These tests pin the contract on small synthetic data:

1. Binning is deterministic and monotone, and `apply_bins` reproduces the
   training bins so predict-time mapping matches fit-time exactly.
2. Predictions are guarded strictly inside (0, 1).
3. The Newton leaf value is the closed-form second-order step -G / (H + lambda).
4. Boosting actually learns: final training log-loss beats the constant base-rate
   model on a learnable target.
5. THE CRUX — a pure-interaction (XOR) target that any additive model scores at
   chance is learned by a depth->=2 GBM. This is the whole reason the challenger
   exists: to capture interactions the additive logistic structurally cannot.
6. The fit is deterministic (no randomness), monitor scores track per round, and
   `truncate` yields the prefix ensemble exactly.
7. Config is validated at the boundary (fail fast on bad hyperparameters).
"""

from __future__ import annotations

import numpy as np
import pytest

from winprob import gbm


# --------------------------------------------------------------------------
# Synthetic data builders.
# --------------------------------------------------------------------------

def _separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A linearly-learnable target: P(y=1) rises smoothly with a single feature."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, 2))
    logit = 1.5 * x[:, 0] - 0.5 * x[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return x, y


def _xor(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pure interaction: y = 1 iff sign(x1) != sign(x2). No additive signal.

    Each feature is marginally independent of y (both classes equally likely at
    any single-feature value), so an additive model can only score chance; the
    signal lives entirely in the x1*x2 interaction a depth->=2 tree can split on.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.float64)
    return x, y


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# --------------------------------------------------------------------------
# Binning.
# --------------------------------------------------------------------------

def test_bins_are_in_range_and_reproducible():
    x, _ = _separable(500, seed=1)
    edges = gbm.bin_edges(x, n_bins=16)
    binned = gbm.apply_bins(x, edges)
    assert binned.dtype.kind == "i"
    assert binned.min() >= 0 and binned.max() < 16
    # Re-mapping the same rows through the stored edges is byte-identical.
    again = gbm.apply_bins(x, edges)
    assert np.array_equal(binned, again)


def test_binning_is_monotone_per_feature():
    x = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    edges = gbm.bin_edges(x, n_bins=10)
    binned = gbm.apply_bins(x, edges)
    # A larger raw value never lands in a lower bin.
    assert np.all(np.diff(binned[:, 0]) >= 0)


def test_constant_feature_bins_without_error():
    x = np.column_stack([np.ones(100), np.linspace(0, 1, 100)])
    edges = gbm.bin_edges(x, n_bins=8)
    binned = gbm.apply_bins(x, edges)
    # Degenerate column collapses to a single bin rather than crashing.
    assert len(np.unique(binned[:, 0])) == 1


# --------------------------------------------------------------------------
# Newton leaf value.
# --------------------------------------------------------------------------

def test_leaf_value_is_second_order_step():
    g = np.array([0.2, -0.4, 0.1])
    h = np.array([0.25, 0.24, 0.20])
    lam = 1.0
    expected = -g.sum() / (h.sum() + lam)
    assert gbm.leaf_value(float(g.sum()), float(h.sum()), lam) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Prediction guarantees.
# --------------------------------------------------------------------------

def test_predictions_strictly_inside_unit_interval():
    x, y = _separable(400, seed=2)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=30, max_depth=3, n_bins=16))
    p = gbm.predict_proba(out.model, x)
    assert p.min() > 0.0 and p.max() < 1.0


def test_base_score_matches_base_rate_logodds():
    x, y = _separable(400, seed=3)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=1, max_depth=2, n_bins=8))
    ybar = float(y.mean())
    assert out.model.base_score == pytest.approx(np.log(ybar / (1.0 - ybar)))


# --------------------------------------------------------------------------
# Learning.
# --------------------------------------------------------------------------

def test_boosting_beats_constant_base_rate():
    x, y = _separable(1500, seed=4)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=80, max_depth=3, learning_rate=0.1))
    p = gbm.predict_proba(out.model, x)
    ybar = float(y.mean())
    base_ll = _log_loss(y, np.full_like(y, ybar))
    assert _log_loss(y, p) < base_ll


def test_learns_pure_interaction_xor():
    """The crux: depth-2 GBM cracks XOR that any additive model scores at chance."""
    x, y = _xor(2000, seed=5)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=60, max_depth=2, learning_rate=0.2))
    p = gbm.predict_proba(out.model, x)
    # Random/additive would sit at ~0.693; the interaction is fully learnable.
    assert _log_loss(y, p) < 0.35
    acc = float(np.mean((p > 0.5) == (y > 0.5)))
    assert acc > 0.9


def test_depth_one_cannot_crack_xor():
    """Sanity on the crux: a depth-1 stump ensemble (additive) stays near chance."""
    x, y = _xor(2000, seed=6)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=60, max_depth=1, learning_rate=0.2))
    p = gbm.predict_proba(out.model, x)
    assert _log_loss(y, p) > 0.6  # depth-1 is additive -> no interaction captured


# --------------------------------------------------------------------------
# Determinism, monitoring, truncation.
# --------------------------------------------------------------------------

def test_fit_is_deterministic():
    x, y = _separable(600, seed=7)
    cfg = gbm.GBMConfig(n_trees=25, max_depth=3)
    p1 = gbm.predict_proba(gbm.fit_gbm(x, y, cfg).model, x)
    p2 = gbm.predict_proba(gbm.fit_gbm(x, y, cfg).model, x)
    assert np.array_equal(p1, p2)


def test_monitor_scores_recorded_per_round():
    x, y = _separable(800, seed=8)
    xv, yv = _separable(400, seed=9)
    monitor = gbm.MonitorSet(xv, yv, _log_loss)
    out = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=20, max_depth=2), monitor=monitor)
    assert len(out.monitor_scores) == 20
    # Validation log-loss should improve from the first tree to the last here.
    assert out.monitor_scores[-1] < out.monitor_scores[0]


def test_truncate_is_prefix_ensemble():
    x, y = _separable(600, seed=10)
    full = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=30, max_depth=2)).model
    cut = gbm.truncate(full, 12)
    assert len(cut.trees) == 12
    # A model fit for exactly 12 trees equals the full model truncated to 12.
    short = gbm.fit_gbm(x, y, gbm.GBMConfig(n_trees=12, max_depth=2)).model
    assert np.array_equal(
        gbm.predict_proba(cut, x), gbm.predict_proba(short, x)
    )


# --------------------------------------------------------------------------
# Config validation at the boundary.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_trees": 0},
        {"max_depth": 0},
        {"n_bins": 1},
        {"learning_rate": 0.0},
        {"min_samples_leaf": 0},
    ],
)
def test_config_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        gbm.GBMConfig(**kwargs)
