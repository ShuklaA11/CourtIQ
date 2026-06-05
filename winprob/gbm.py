"""Histogram-based Newton gradient-boosting classifier — hand-rolled on numpy.

The Phase-4 nonlinear challenger to the additive win-probability logistic. The
logistic is additive in its features; it hand-builds a couple of nonlinear terms
(``margin/sqrt(time)``, time knots) but structurally cannot represent an
interaction like ``margin x team_strength`` — the exact shape comeback dynamics
take (a *strong* team down six behaves unlike a *weak* one in the same state).
A shallow boosted tree ensemble can learn those interaction surfaces. This module
is that engine, with no third-party ML dependency, mirroring the project's
scipy/numpy hand-rolled ethos (``rapm.ridge``, ``winprob.model``).

Design, and why each piece.

*Second-order (Newton) boosting on the log-loss.* The model is an additive score
in log-odds space, ``F(x) = F0 + lr * sum_m tree_m(x)`` with ``p = sigmoid(F)``.
``F0`` is the base-rate log-odds (the calibration anchor). Each round fits a
regression tree to the current residuals using the log-loss gradient
``g = p - y`` and hessian ``h = p (1 - p)``; a leaf's optimal value under the
second-order Taylor expansion is the closed form ``-sum(g) / (sum(h) + lambda)``
(XGBoost-style). Newton steps directly minimize log-loss per leaf and tend to be
well-calibrated, unlike plain gradient boosting that fits MSE to residuals.

*Histogram binning — the tractability move.* On ~800k rows a pure-numpy
exact-greedy split search over every threshold every round is far too slow. Each
continuous feature is binned ONCE into ``n_bins`` quantile bins; split search then
accumulates per-bin gradient/hessian sums with ``np.bincount`` (vectorized) and
scans cumulative sums for the best threshold. Per-node cost drops to
``O(n_node * features)`` histogram builds plus ``O(bins * features)`` gain scans.

*Deterministic.* No row/feature subsampling and no randomness anywhere, so a fit
is a pure function of its inputs and its predictions hash-reproduce like the rest
of the pipeline. Split ties break toward the first feature and lowest threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Strict open-interval clamp so guarded probabilities never hit exactly 0 or 1
# (keeps downstream log-loss finite), matching ``winprob.model.PROB_EPS``.
_PROB_EPS = 1e-12


# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GBMConfig:
    """Hyperparameters for one boosted fit; validated at construction."""

    learning_rate: float = 0.1
    max_depth: int = 3
    n_trees: int = 300
    n_bins: int = 64
    leaf_l2: float = 1.0
    min_samples_leaf: int = 200
    min_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {self.n_trees}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.leaf_l2 < 0.0:
            raise ValueError(f"leaf_l2 must be >= 0, got {self.leaf_l2}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}")
        if self.min_gain < 0.0:
            raise ValueError(f"min_gain must be >= 0, got {self.min_gain}")


# --------------------------------------------------------------------------
# Guarded sigmoid.
# --------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid clipped into the open interval (0, 1)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.clip(out, _PROB_EPS, 1.0 - _PROB_EPS)


# --------------------------------------------------------------------------
# Histogram binning: fit quantile edges once, map raw -> bin consistently.
# --------------------------------------------------------------------------

def bin_edges(X: np.ndarray, n_bins: int) -> tuple[np.ndarray, ...]:
    """Per-feature interior quantile edges defining ``n_bins`` histogram bins.

    Each feature gets up to ``n_bins - 1`` interior split points at evenly-spaced
    quantiles of the observed values; duplicate quantile values (few distinct
    levels, e.g. a binary or constant column) collapse so a degenerate column
    yields a single bin rather than crashing. Edges are what both the training
    split search and prediction-time mapping share, keeping them consistent.
    """
    X = np.asarray(X, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges: list[np.ndarray] = []
    for j in range(X.shape[1]):
        col_edges = np.unique(np.quantile(X[:, j], quantiles))
        edges.append(np.asarray(col_edges, dtype=np.float64))
    return tuple(edges)


def apply_bins(X: np.ndarray, edges: tuple[np.ndarray, ...]) -> np.ndarray:
    """Map raw features to integer bin indices via stored edges (searchsorted).

    ``bin = searchsorted(edges, x, side='right')`` places ``x`` into ``[0, n_bins)``
    monotonically, and out-of-range values clamp into the end bins — so a test row
    beyond the training range is scored in the nearest trained bin, never NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    binned = np.empty(X.shape, dtype=np.int32)
    for j, col_edges in enumerate(edges):
        binned[:, j] = np.searchsorted(col_edges, X[:, j], side="right")
    return binned


# --------------------------------------------------------------------------
# Newton leaf value.
# --------------------------------------------------------------------------

def leaf_value(g_sum: float, h_sum: float, leaf_l2: float) -> float:
    """Closed-form second-order-optimal leaf step ``-G / (H + lambda)``."""
    return -g_sum / (h_sum + leaf_l2)


def _split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    g_total: float,
    h_total: float,
    leaf_l2: float,
) -> float:
    """Loss reduction from a split, ``0.5 * (GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l))``."""
    def term(g: float, h: float) -> float:
        return g * g / (h + leaf_l2)

    return 0.5 * (term(g_left, h_left) + term(g_right, h_right) - term(g_total, h_total))


# --------------------------------------------------------------------------
# Tree: flat-array representation with a vectorized descent.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tree:
    """One regression tree as parallel node arrays (leaves have ``feature == -1``).

    ``threshold`` is a bin index: a row goes LEFT when ``binned[:, feature] <=
    threshold``. Leaf rows carry the Newton step in ``value``; internal rows carry
    the child indices in ``left``/``right``. Flat arrays make prediction a
    vectorized descent rather than per-row Python recursion — essential at 800k
    rows.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    max_depth: int

    def predict(self, binned: np.ndarray) -> np.ndarray:
        """Leaf value for each row via a vectorized top-down descent."""
        n = binned.shape[0]
        node = np.zeros(n, dtype=np.int64)
        rows = np.arange(n)
        # A tree of depth d has at most d internal levels; bound the descent.
        for _ in range(self.max_depth + 1):
            feat = self.feature[node]
            active = feat >= 0
            if not active.any():
                break
            go_left = binned[rows, feat] <= self.threshold[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(active, nxt, node)
        return self.value[node]


# Mutable scratch node used only while growing a tree, before freezing to arrays.
@dataclass
class _GrowNode:
    rows: np.ndarray
    depth: int
    g_sum: float
    h_sum: float


def _best_split(
    binned: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    rows: np.ndarray,
    g_sum: float,
    h_sum: float,
    cfg: GBMConfig,
) -> tuple[int, int, float] | None:
    """Best (feature, bin-threshold, gain) over all features, or None if no gain.

    For each feature, ``np.bincount`` accumulates the per-bin gradient, hessian,
    and count over ``rows`` in one vectorized pass; cumulative sums then give the
    left/right statistics at every candidate threshold at once. The best gain is
    taken subject to ``min_samples_leaf`` on both sides and ``min_gain``.
    """
    n_features = binned.shape[1]
    best_feat = -1
    best_thr = -1
    best_gain = cfg.min_gain
    for j in range(n_features):
        col = binned[rows, j]
        length = int(col.max()) + 1 if col.size else 0
        if length <= 1:
            continue  # single occupied bin -> no threshold to split on.
        gj = np.bincount(col, weights=g[rows], minlength=length)
        hj = np.bincount(col, weights=h[rows], minlength=length)
        cj = np.bincount(col, minlength=length)
        g_left = np.cumsum(gj)[:-1]
        h_left = np.cumsum(hj)[:-1]
        c_left = np.cumsum(cj)[:-1]
        g_right = g_sum - g_left
        h_right = h_sum - h_left
        c_right = int(rows.size) - c_left
        valid = (c_left >= cfg.min_samples_leaf) & (c_right >= cfg.min_samples_leaf)
        if not valid.any():
            continue
        gains = np.where(
            valid,
            0.5
            * (
                g_left * g_left / (h_left + cfg.leaf_l2)
                + g_right * g_right / (h_right + cfg.leaf_l2)
                - g_sum * g_sum / (h_sum + cfg.leaf_l2)
            ),
            -np.inf,
        )
        thr = int(np.argmax(gains))
        if gains[thr] > best_gain:
            best_gain = float(gains[thr])
            best_feat = j
            best_thr = thr
    if best_feat < 0:
        return None
    return best_feat, best_thr, best_gain


def _fit_tree(binned: np.ndarray, g: np.ndarray, h: np.ndarray, cfg: GBMConfig) -> Tree:
    """Grow one depth-limited Newton regression tree over pre-binned features."""
    features: list[int] = []
    thresholds: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    values: list[float] = []

    def new_slot() -> int:
        """Append a fresh (as-yet-unclassified) node and return its array index."""
        features.append(-1)
        thresholds.append(-1)
        lefts.append(-1)
        rights.append(-1)
        values.append(0.0)
        return len(features) - 1

    all_rows = np.arange(binned.shape[0])
    root = _GrowNode(all_rows, 0, float(g.sum()), float(h.sum()))
    # Explicit stack holds (grow node, index of its slot in the flat arrays).
    stack: list[tuple[_GrowNode, int]] = [(root, new_slot())]

    while stack:
        node, slot = stack.pop()
        make_leaf = (
            node.depth >= cfg.max_depth
            or node.rows.size < 2 * cfg.min_samples_leaf
        )
        split = (
            None
            if make_leaf
            else _best_split(binned, g, h, node.rows, node.g_sum, node.h_sum, cfg)
        )
        if split is None:
            values[slot] = leaf_value(node.g_sum, node.h_sum, cfg.leaf_l2)
            continue

        feat, thr, _ = split
        col = binned[node.rows, feat]
        left_mask = col <= thr
        left_rows = node.rows[left_mask]
        right_rows = node.rows[~left_mask]

        left_slot = new_slot()
        right_slot = new_slot()
        features[slot] = feat
        thresholds[slot] = thr
        lefts[slot] = left_slot
        rights[slot] = right_slot

        gl = float(g[left_rows].sum())
        hl = float(h[left_rows].sum())
        stack.append((_GrowNode(left_rows, node.depth + 1, gl, hl), left_slot))
        stack.append(
            (_GrowNode(right_rows, node.depth + 1, node.g_sum - gl, node.h_sum - hl),
             right_slot)
        )

    return Tree(
        feature=np.asarray(features, dtype=np.int64),
        threshold=np.asarray(thresholds, dtype=np.int64),
        left=np.asarray(lefts, dtype=np.int64),
        right=np.asarray(rights, dtype=np.int64),
        value=np.asarray(values, dtype=np.float64),
        max_depth=cfg.max_depth,
    )


# --------------------------------------------------------------------------
# Boosted model + prediction.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoostedModel:
    """A fitted ensemble: base log-odds, trees, shared bin edges, learning rate."""

    base_score: float
    trees: tuple[Tree, ...]
    edges: tuple[np.ndarray, ...]
    learning_rate: float
    n_features: int


def predict_margin(model: BoostedModel, X: np.ndarray) -> np.ndarray:
    """Log-odds score ``F0 + lr * sum_m tree_m(x)`` for raw feature rows ``X``."""
    binned = apply_bins(X, model.edges)
    margin = np.full(X.shape[0], model.base_score, dtype=np.float64)
    for tree in model.trees:
        margin += model.learning_rate * tree.predict(binned)
    return margin


def predict_proba(model: BoostedModel, X: np.ndarray) -> np.ndarray:
    """Guarded win probabilities ``sigmoid(margin)`` for raw feature rows ``X``."""
    return _sigmoid(predict_margin(model, X))


def truncate(model: BoostedModel, n_trees: int) -> BoostedModel:
    """A new model using only the first ``n_trees`` trees (the prefix ensemble)."""
    if n_trees < 0 or n_trees > len(model.trees):
        raise ValueError(
            f"n_trees to keep must be in [0, {len(model.trees)}], got {n_trees}"
        )
    return BoostedModel(
        base_score=model.base_score,
        trees=tuple(model.trees[:n_trees]),
        edges=model.edges,
        learning_rate=model.learning_rate,
        n_features=model.n_features,
    )


# --------------------------------------------------------------------------
# Fit with optional per-round validation monitoring.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitorSet:
    """A held-out set scored after each boosting round for n-tree selection."""

    X: np.ndarray
    y: np.ndarray
    score_fn: Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class FitOutput:
    """A fitted model plus, if monitored, its per-round validation scores."""

    model: BoostedModel
    monitor_scores: tuple[float, ...]


def _base_log_odds(y: np.ndarray) -> float:
    """Base-rate log-odds ``log(ybar / (1 - ybar))``, guarded away from 0/1."""
    ybar = float(np.clip(np.mean(y), _PROB_EPS, 1.0 - _PROB_EPS))
    return float(np.log(ybar / (1.0 - ybar)))


def fit_gbm(
    X: np.ndarray,
    y: np.ndarray,
    config: GBMConfig,
    monitor: MonitorSet | None = None,
) -> FitOutput:
    """Fit a Newton gradient-boosted classifier; optionally monitor a held-out set.

    Bins the features once, seeds the score at the base-rate log-odds, and adds
    ``config.n_trees`` shrunk Newton trees, each fit to the current gradient and
    hessian. When ``monitor`` is given, the held-out margin is updated
    incrementally and scored after every round (``O(n_val)`` per round, no full
    re-prediction), so the caller can pick the validation-optimal tree count.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = bin_edges(X, config.n_bins)
    binned = apply_bins(X, edges)

    base = _base_log_odds(y)
    margin = np.full(X.shape[0], base, dtype=np.float64)

    monitor_binned = None
    monitor_margin = None
    monitor_scores: list[float] = []
    if monitor is not None:
        monitor_binned = apply_bins(np.asarray(monitor.X, dtype=np.float64), edges)
        monitor_margin = np.full(monitor.X.shape[0], base, dtype=np.float64)

    trees: list[Tree] = []
    for _ in range(config.n_trees):
        p = _sigmoid(margin)
        g = p - y
        h = p * (1.0 - p)
        tree = _fit_tree(binned, g, h, config)
        trees.append(tree)
        margin += config.learning_rate * tree.predict(binned)
        if monitor is not None:
            monitor_margin += config.learning_rate * tree.predict(monitor_binned)
            p_val = _sigmoid(monitor_margin)
            monitor_scores.append(float(monitor.score_fn(monitor.y, p_val)))

    model = BoostedModel(
        base_score=base,
        trees=tuple(trees),
        edges=edges,
        learning_rate=config.learning_rate,
        n_features=X.shape[1],
    )
    return FitOutput(model=model, monitor_scores=tuple(monitor_scores))
