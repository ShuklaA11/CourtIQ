"""Pure feature transform: game-state rows -> logistic design matrix.

Warehouse-free building blocks for the Sprint-3 win-probability fit. Everything
here is arrays/DataFrame in, arrays out — no DuckDB, no `data/winprob/` reads, no
model fitting — so each helper is independently importable and unit-testable on a
hand-built DataFrame. It mirrors the arrays-in/arrays-out discipline of
`rapm.ridge`, including the mean/std standardization the ridge fit relies on to
stay well-conditioned.

What the design encodes, and why these features. Win probability at a mid-game
state is driven almost entirely by *how far ahead* the home team is relative to
*how much time is left to give the lead back*. So the workhorse feature is the
nonlinear basis `margin / sqrt(regulation_seconds_remaining + 1)`: a lead of +6
with two minutes left is far safer than +6 with a full half to play, and the
`1/sqrt(time)` shape is the near-sufficient statistic for that intuition (a
random-walk score difference has standard deviation proportional to `sqrt(time)`,
so the margin measured in standard-deviations is `margin / sqrt(time)`). Around
that we add the raw margin, a compact piecewise-linear time basis (the level plus
three hinge knots so the model can bend the time response near crunch time),
whether the home team currently has the ball, a playoff indicator, and season
dummies to absorb league-wide scoring-environment drift.

What is deliberately excluded. The player-rating columns, the raw `feed_*`
scoreboard columns, and the `home_win` target never enter the design —
`home_win` is the label, and letting any of the others in would either leak the
answer or couple this baseline to the ratings pipeline. The transform reads only
`REQUIRED_COLUMNS` and builds every output column from them, so those columns are
structurally absent from `feature_names`; the test suite pins that guarantee.

Determinism. Column order is fixed and identical across calls on the same input:
the structural columns come first in a hard-coded order, then season dummies in
ascending season order. That stability is what lets a fitted `beta` be applied to
a freshly built matrix without re-aligning columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Source columns the transform reads. Validated up front so a stale mart fails
# fast with a clear message instead of a downstream KeyError.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "home_score_differential",
    "regulation_seconds_remaining",
    "home_has_possession",
)

# Knots (in regulation-seconds-remaining) for the piecewise-linear time basis.
# Each contributes a hinge relu(knot - t) that is zero while plenty of time
# remains and ramps up as the clock winds toward that mark: 12:00, 3:00, 0:30
# left. Three knots is "compact" per the spec while still letting the time
# response bend around the moments that matter most for win probability.
TIME_KNOTS: tuple[float, ...] = (720.0, 180.0, 30.0)

# Canonical column names for the time-knot hinges, derived once from TIME_KNOTS.
# `build_design` below names its knot columns from this tuple, and downstream
# consumers (e.g. `winprob.evaluate`) import it rather than re-spelling the
# `time_knot_<k>` strings, so the knot schema has a single source of truth.
TIME_KNOT_NAMES: tuple[str, ...] = tuple(f"time_knot_{int(k)}" for k in TIME_KNOTS)

# Canonical set of continuous (standardized) feature names, in build order.
# These are the only columns `build_design` marks continuous; every other column
# (intercept, binary indicators, season dummies) passes standardization through
# as an identity. Consumers that need the standardization mask or the score+time
# baseline contract import this instead of maintaining a parallel literal list.
CONTINUOUS_FEATURE_NAMES: tuple[str, ...] = (
    "home_score_differential",
    "margin_over_sqrt_time",
    "regulation_seconds_remaining",
) + TIME_KNOT_NAMES

# Game-id prefixes: NBA stats ids encode the season type in the first three
# digits — '004' is playoffs, '002' is regular season.
PLAYOFF_PREFIX = "004"
REGULAR_PREFIX = "002"

# Ridge-mirroring standardization guard: a zero-variance continuous column (e.g.
# a single-row frame) would divide by zero, so its std is treated as 1.0, leaving
# the centered column at all-zeros rather than NaN/inf.
_STD_FLOOR = 1.0


# --------------------------------------------------------------------------
# Pure per-feature builders. Each returns a fresh float64 array of length n and
# never mutates its inputs.
# --------------------------------------------------------------------------

def constant_column(n: int) -> np.ndarray:
    """The explicit intercept: an all-ones column of length `n`."""
    return np.ones(n, dtype=np.float64)


def margin_column(home_score_differential: np.ndarray) -> np.ndarray:
    """The home margin (home minus away score) as float64."""
    return np.asarray(home_score_differential, dtype=np.float64)


def margin_over_sqrt_time(
    margin: np.ndarray, regulation_seconds_remaining: np.ndarray
) -> np.ndarray:
    """The near-sufficient statistic `margin / sqrt(reg_seconds_remaining + 1)`.

    The `+1` keeps the denominator finite and well-behaved at the buzzer
    (`reg_seconds_remaining == 0`), so the basis is defined for every row and
    never divides by zero. Grows in magnitude as time runs out for a fixed lead,
    which is exactly how a lead's safety scales.
    """
    margin = np.asarray(margin, dtype=np.float64)
    t = np.asarray(regulation_seconds_remaining, dtype=np.float64)
    return margin / np.sqrt(t + 1.0)


def time_knot_basis(
    regulation_seconds_remaining: np.ndarray, knots: tuple[float, ...] = TIME_KNOTS
) -> np.ndarray:
    """Piecewise-linear hinge basis on time: columns `relu(knot - t)` per knot.

    Returns an (n, len(knots)) array whose column k is `max(0, knot_k - t)`. Each
    hinge is zero while more than `knot_k` seconds remain and rises linearly as
    the clock passes that mark, letting a linear model bend its time response at
    the knots without a full spline. Column order follows `knots`.
    """
    t = np.asarray(regulation_seconds_remaining, dtype=np.float64)
    cols = [np.maximum(0.0, float(k) - t) for k in knots]
    return np.column_stack(cols) if cols else np.empty((len(t), 0), dtype=np.float64)


def home_possession_column(home_has_possession: np.ndarray) -> np.ndarray:
    """Home-has-the-ball indicator as 0.0/1.0."""
    return np.asarray(home_has_possession, dtype=bool).astype(np.float64)


def playoff_indicator(game_id: np.ndarray) -> np.ndarray:
    """Playoff flag derived from the game-id prefix: '004' -> 1.0, else 0.0.

    Regular-season ids start '002'; only the playoff prefix flips the indicator,
    so any non-playoff season type reads as 0.0.
    """
    ids = pd.Series(game_id).astype(str)
    return ids.str.startswith(PLAYOFF_PREFIX).to_numpy(dtype=np.float64)


def season_levels(season: np.ndarray) -> list[int]:
    """Ascending unique seasons present — the dummy levels, in stable order."""
    return sorted({int(s) for s in np.asarray(season)})


def season_dummies(
    season: np.ndarray, levels: list[int]
) -> tuple[np.ndarray, list[str]]:
    """One-hot season dummies with the first level dropped as the reference.

    Dropping the reference level keeps the dummies linearly independent of the
    explicit intercept (no dummy-variable trap), which is what keeps the normal
    equations well-conditioned. Returns an (n, max(0, len(levels)-1)) array and
    the matching `season_<year>` names. With zero or one level present there are
    no dummy columns.
    """
    s = np.asarray(season).astype(int)
    dummy_levels = levels[1:]  # drop the smallest season as the reference
    names = [f"season_{lvl}" for lvl in dummy_levels]
    if not dummy_levels:
        return np.empty((len(s), 0), dtype=np.float64), names
    cols = [(s == lvl).astype(np.float64) for lvl in dummy_levels]
    return np.column_stack(cols), names


# --------------------------------------------------------------------------
# Standardization — mirrors the ridge scaling so the fit is well-conditioned.
# --------------------------------------------------------------------------

def standardize_columns(
    matrix: np.ndarray, continuous_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center-and-scale the continuous columns; pass the rest through unchanged.

    For every column marked continuous, subtract its mean and divide by its std
    (population std; a zero std is floored to 1.0 so a constant column becomes
    all-zeros rather than NaN/inf). Binary and intercept columns keep an identity
    transform (mean 0, std 1) so the returned `means`/`stds` are full-length and
    a fitted model can reapply `(raw - means) / stds` uniformly to any of them.

    Pure: returns a fresh matrix and fresh stat vectors; inputs are not mutated.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    mask = np.asarray(continuous_mask, dtype=bool)
    n, p = matrix.shape
    if mask.shape != (p,):
        raise ValueError(f"continuous_mask must have shape ({p},), got {mask.shape}")

    means = np.zeros(p, dtype=np.float64)
    stds = np.ones(p, dtype=np.float64)
    means[mask] = matrix[:, mask].mean(axis=0)
    raw_std = matrix[:, mask].std(axis=0)  # population std (ddof=0)
    stds[mask] = np.where(raw_std > 0.0, raw_std, _STD_FLOOR)

    standardized = (matrix - means) / stds
    return standardized, means, stds


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignMatrix:
    """A built design plus the scaling used, so a fit can be reapplied exactly.

    `X` is (n, p) standardized; `feature_names` labels its columns in the same
    fixed order across calls. `means`/`stds` are length-p (identity on the
    non-standardized columns) and `continuous_mask` marks which columns were
    standardized. `season_levels` records the dummy levels seen in the input.
    """

    X: np.ndarray
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    continuous_mask: np.ndarray
    season_levels: list[int]


def _validate_columns(df: pd.DataFrame) -> None:
    """Fail fast if a required source column is missing (stale/partial mart)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input DataFrame missing required columns: {missing}")


def build_design(df: pd.DataFrame) -> DesignMatrix:
    """Assemble the full standardized design from game-state rows.

    Builds the columns in a fixed order — intercept, margin,
    margin/sqrt(time), the time level and its three hinge knots, home
    possession, playoff flag, then season dummies — standardizes the continuous
    ones, and returns everything (matrix, names, scaling) as a `DesignMatrix`.
    Pure: reads only from `df`, writes nothing, mutates nothing.
    """
    _validate_columns(df)
    n = len(df)

    margin = margin_column(df["home_score_differential"].to_numpy())
    reg_sec = np.asarray(df["regulation_seconds_remaining"].to_numpy(), dtype=np.float64)

    # Structural columns and whether each is a continuous (standardized) feature.
    columns: list[np.ndarray] = []
    names: list[str] = []
    continuous: list[bool] = []

    def add(name: str, values: np.ndarray, is_continuous: bool) -> None:
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)
        continuous.append(is_continuous)

    add("intercept", constant_column(n), False)
    add("home_score_differential", margin, True)
    add("margin_over_sqrt_time", margin_over_sqrt_time(margin, reg_sec), True)
    add("regulation_seconds_remaining", reg_sec, True)

    knots = time_knot_basis(reg_sec)
    for name, knot_col in zip(TIME_KNOT_NAMES, knots.T):
        add(name, knot_col, True)

    add("home_has_possession", home_possession_column(df["home_has_possession"].to_numpy()), False)
    add("is_playoff", playoff_indicator(df["game_id"].to_numpy()), False)

    levels = season_levels(df["season"].to_numpy())
    dummies, dummy_names = season_dummies(df["season"].to_numpy(), levels)
    for i, dummy_name in enumerate(dummy_names):
        add(dummy_name, dummies[:, i], False)

    matrix = np.column_stack(columns) if columns else np.empty((n, 0), dtype=np.float64)
    continuous_mask = np.array(continuous, dtype=bool)
    X, means, stds = standardize_columns(matrix, continuous_mask)

    return DesignMatrix(
        X=X,
        feature_names=list(names),
        means=means,
        stds=stds,
        continuous_mask=continuous_mask,
        season_levels=levels,
    )


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Public transform: game-state rows -> (standardized X, feature_names).

    Thin wrapper over `build_design` for callers that only need the matrix and
    its column labels; the scaling stats live on the `DesignMatrix` from
    `build_design` when a fit needs to reapply them.
    """
    design = build_design(df)
    return design.X, design.feature_names
