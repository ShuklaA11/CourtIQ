"""Fit + lambda selection for the Sprint-3 win-probability logistic model.

This is the training entrypoint that stands on the two pure building blocks from
earlier in the sprint: `winprob.features.build_design` turns game-state rows into
a standardized logistic design matrix, and `winprob.design` supplies the
deterministic hashing helpers (`canonical_hash`, `file_hash`) and the pinned
`SPLIT_DEFINITION`. Nothing here reaches back into DuckDB or any third-party ML
framework — the fit is scipy-only L2-regularized logistic regression on the
exported Parquet mart.

Modeling choices, and why.

*One design matrix over train+validation, split by row mask.* The season dummies
in `winprob.features` drop the smallest season as the reference level, so the
column set a build produces depends on which seasons are present. Building `train`
(2022-2023) and `validation` (2024) separately would yield DIFFERENT columns
(validation alone has a single season and therefore no dummy), and a coefficient
vector fit on one could not be applied to the other. Building once on the
train+validation working frame fixes a single column order and a single
standardization, then the fit indexes train-only or train+validation rows out of
that shared matrix. The standardization means/stds use only feature columns — the
`home_win` label never enters them — so sharing them across the selection split is
conditioning, not target leakage, and the saved stats are exactly the ones the
final (train+validation) model was fit under.

*Selective L2 penalty, intercept free.* The objective is the mean negative
log-likelihood plus `0.5 * lambda * ||beta_penalized||^2`, mirroring the
selective-penalty discipline of `rapm.ridge`: every coefficient is shrunk toward
zero EXCEPT the intercept, which carries a real global offset (the base home-win
rate) and must stay unpenalized. L-BFGS-B minimizes it with an analytic gradient.

*Guarded sigmoid.* Predicted probabilities are clipped into a strict open
interval so `log(p)` and `log(1-p)` are always finite and every stored prediction
is strictly in (0, 1), as the tests pin.

*Game-clustered selection.* Possession rows within a game are massively
correlated (the same score margin persists across dozens of consecutive states),
so scoring lambda by row-wise log-loss would count a blowout game hundreds of
times and a nail-biter just as often — rows are not iid. Selection instead
averages log-loss WITHIN each game first, then averages those per-game means, so
every game contributes equally regardless of length.

*Holdout is never touched.* `test` (2025) and `audit_only` (2021) rows are
dropped before the design matrix is ever built; the working frame is asserted to
contain only train/validation, and the count of excluded holdout rows is recorded
so the guarantee is observable, not just asserted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from winprob import features
from winprob.design import (
    FEATURE_SCHEMA_VERSION,
    SPLIT_DEFINITION,
    canonical_hash,
    file_hash,
)

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
MODEL_JSON_NAME = "logistic_model.json"
MANIFEST_JSON_NAME = "winprob_model_manifest.json"

TARGET_COLUMN = "home_win"
INTERCEPT_NAME = "intercept"

# Splits the fit is permitted to read. `test`/`audit_only` are structurally
# excluded before any matrix is built so nothing leaks in during selection.
ALLOWED_FIT_SPLITS: frozenset[str] = frozenset({"train", "validation"})
HOLDOUT_SPLITS: frozenset[str] = frozenset({"test", "audit_only"})

# Fixed regularization grid searched for lambda. Geometric from near-unpenalized
# to heavy shrinkage; the mean-NLL objective makes lambda scale-free in n.
LAMBDA_GRID: tuple[float, ...] = tuple(np.logspace(-4.0, 2.0, 13))

# Strict open-interval clamp so guarded probabilities never hit exactly 0 or 1.
PROB_EPS = 1e-12
# L-BFGS-B iteration cap; the objective is smooth and low-dimensional so this is
# ample headroom, not a tight budget.
MAX_ITER = 500
# Finite box on each coefficient. Legitimate standardized coefficients are O(1);
# this bound is astronomically loose and never binds on real (non-separable)
# data, but it stops a perfectly separable input from driving the MLE toward
# +/-infinity and overflowing the linear predictor.
COEF_BOUND = 1e6


# --------------------------------------------------------------------------
# Numerics: guarded sigmoid, penalized objective, and the fit.
# --------------------------------------------------------------------------

# Apple's Accelerate BLAS leaks FPU status flags out of its vectorized
# matrix-vector `matmul` kernel, so NumPy raises spurious "divide by zero" /
# "overflow" / "invalid value encountered in matmul" RuntimeWarnings even on a
# trivially-finite product. The linear predictor here is always finite (X is
# finite and coefficients are bounded), predictions are clipped, and the fit
# raises on any non-finite result — so silencing those float flags around the
# linear predictor hides only the known-spurious warning, not a real defect.
_MATMUL_ERRSTATE = dict(divide="ignore", over="ignore", invalid="ignore")


def _linear_predictor(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """`X @ beta` with the spurious Accelerate matmul float flags silenced."""
    with np.errstate(**_MATMUL_ERRSTATE):
        return np.asarray(X, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)


def guarded_sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid clipped into the strict open interval (0, 1).

    `scipy.special.expit` is already overflow-safe; the clip guarantees the
    output never reaches exactly 0.0 or 1.0 so downstream `log(p)`/`log(1-p)`
    stay finite and every stored prediction is strictly inside (0, 1).
    """
    p = expit(np.asarray(z, dtype=np.float64))
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def penalty_mask(feature_names: list[str]) -> np.ndarray:
    """1.0 for every penalized coefficient, 0.0 for the free intercept.

    The intercept column carries the unshrunk base home-win rate and must not be
    penalized; every other coefficient is shrunk toward zero.
    """
    return np.array(
        [0.0 if name == INTERCEPT_NAME else 1.0 for name in feature_names],
        dtype=np.float64,
    )


def penalized_objective(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    pen_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean penalized negative log-likelihood and its gradient.

    Loss is `mean(-[y log p + (1-y) log(1-p)]) + 0.5 lambda ||pen_mask * beta||^2`
    with `p` the guarded sigmoid of `X beta`. Returns `(loss, grad)` for
    `minimize(..., jac=True)`; the intercept is excluded from the penalty via
    `pen_mask`. Averaging over rows keeps lambda scale-free in the row count.
    """
    n = X.shape[0]
    p = guarded_sigmoid(_linear_predictor(X, beta))
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    penalized_beta = pen_mask * beta
    with np.errstate(**_MATMUL_ERRSTATE):
        grad = X.T @ (p - y) / n + lam * penalized_beta
    loss = nll + 0.5 * lam * float(penalized_beta @ penalized_beta)
    return float(loss), grad


def fit_l2_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    pen_mask: np.ndarray,
    max_iter: int = MAX_ITER,
) -> np.ndarray:
    """Fit L2-penalized logistic regression via L-BFGS-B; return the coefficients.

    Minimizes `penalized_objective` from an all-zeros start with the analytic
    gradient. The intercept stays unpenalized (`pen_mask` is 0 there). Raises if
    the optimizer returns a non-finite solution so a broken fit can never be
    silently serialized.
    """
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = np.zeros(X.shape[1], dtype=np.float64)
    bounds = [(-COEF_BOUND, COEF_BOUND)] * X.shape[1]
    result = minimize(
        penalized_objective,
        x0,
        args=(X, y, float(lam), pen_mask),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )
    beta = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(beta)):
        raise RuntimeError(f"non-finite coefficients from fit at lambda={lam}")
    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Guarded win probabilities for design rows `X` under coefficients `beta`."""
    return guarded_sigmoid(_linear_predictor(X, beta))


# --------------------------------------------------------------------------
# Scoring: row log-loss and the game-clustered aggregate.
# --------------------------------------------------------------------------

def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row binary cross-entropy `-[y log p + (1-y) log(1-p)]` (guarded p)."""
    p = np.clip(np.asarray(p, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def game_clustered_log_loss(
    y: np.ndarray, p: np.ndarray, game_ids: np.ndarray
) -> float:
    """Log-loss averaged within each game, then averaged across games.

    Possession rows inside a game are heavily autocorrelated, so treating them
    as iid would weight long games more than short ones. Collapsing to a per-game
    mean first gives every game equal weight — the honest unit of an NBA
    prediction is the game, not the possession.
    """
    losses = row_log_loss(y, p)
    per_game = pd.Series(losses).groupby(np.asarray(game_ids)).mean()
    return float(per_game.mean())


# --------------------------------------------------------------------------
# Working-frame partition — the leakage guard.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkingFrame:
    """Train+validation rows only, plus the count of holdout rows excluded."""

    frame: pd.DataFrame
    holdout_rows_excluded: int
    splits_used: frozenset[str]


def working_frame(df: pd.DataFrame) -> WorkingFrame:
    """Drop `test`/`audit_only`; keep only train+validation and prove it.

    Filters to `ALLOWED_FIT_SPLITS` before any design matrix touches the data,
    records how many holdout rows were dropped, and asserts the surviving frame
    carries no holdout split. The returned `splits_used` is what tests inspect to
    confirm the fit never reads test/audit rows by construction.
    """
    if "split" not in df.columns:
        raise ValueError("input frame is missing the 'split' column")
    keep = df["split"].isin(ALLOWED_FIT_SPLITS)
    work = df.loc[keep].reset_index(drop=True)
    splits_used = frozenset(work["split"].unique().tolist())
    if splits_used & HOLDOUT_SPLITS:
        raise AssertionError(f"holdout split leaked into working frame: {splits_used}")
    return WorkingFrame(
        frame=work,
        holdout_rows_excluded=int((~keep).sum()),
        splits_used=splits_used,
    )


# --------------------------------------------------------------------------
# Selection + final refit.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    """Everything needed to serialize and audit the fitted model."""

    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    lambda_grid: list[float]
    validation_log_loss: list[float]
    chosen_index: int
    chosen_lambda: float
    beta: np.ndarray
    n_train: int
    n_validation: int
    n_fit: int
    train_games: int
    validation_games: int
    splits_used: frozenset[str]
    holdout_rows_excluded: int


def select_and_fit(
    df: pd.DataFrame, grid: tuple[float, ...] = LAMBDA_GRID
) -> FitResult:
    """Select lambda by game-clustered validation log-loss, then refit final.

    Pipeline: drop holdout rows, build one shared standardized design over
    train+validation, fit each grid lambda on TRAIN rows, score game-clustered
    log-loss on VALIDATION rows, pick the grid argmin, and refit at that lambda on
    train+validation combined. Never reads test/audit rows.
    """
    if len(grid) == 0:
        raise ValueError("lambda grid must be non-empty")

    work = working_frame(df)
    frame = work.frame
    design = features.build_design(frame)
    X = design.X
    feature_names = list(design.feature_names)
    pen_mask = penalty_mask(feature_names)

    y = frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = frame["game_id"].to_numpy()
    is_train = (frame["split"] == "train").to_numpy()
    is_val = (frame["split"] == "validation").to_numpy()

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val, games_val = X[is_val], y[is_val], game_ids[is_val]

    validation_log_loss: list[float] = []
    for lam in grid:
        beta = fit_l2_logistic(X_train, y_train, float(lam), pen_mask)
        p_val = predict_proba(X_val, beta)
        validation_log_loss.append(game_clustered_log_loss(y_val, p_val, games_val))

    chosen_index = int(np.argmin(validation_log_loss))
    chosen_lambda = float(grid[chosen_index])

    # Final model: refit at the chosen lambda on train+validation combined.
    beta_final = fit_l2_logistic(X, y, chosen_lambda, pen_mask)

    return FitResult(
        feature_names=feature_names,
        means=np.asarray(design.means, dtype=np.float64),
        stds=np.asarray(design.stds, dtype=np.float64),
        lambda_grid=[float(lam) for lam in grid],
        validation_log_loss=validation_log_loss,
        chosen_index=chosen_index,
        chosen_lambda=chosen_lambda,
        beta=beta_final,
        n_train=int(is_train.sum()),
        n_validation=int(is_val.sum()),
        n_fit=int(len(frame)),
        train_games=int(pd.Series(game_ids[is_train]).nunique()),
        validation_games=int(pd.Series(games_val).nunique()),
        splits_used=work.splits_used,
        holdout_rows_excluded=work.holdout_rows_excluded,
    )


# --------------------------------------------------------------------------
# Serialization.
# --------------------------------------------------------------------------

def feature_schema_hash(feature_names: list[str]) -> str:
    """Stable hash pinning the model to its feature contract.

    Combines the schema version, the source columns the transform reads, and the
    produced column order, so any change to the feature space changes the hash.
    Reuses `winprob.design.canonical_hash`.
    """
    return canonical_hash(
        {
            "version": FEATURE_SCHEMA_VERSION,
            "source_required_columns": list(features.REQUIRED_COLUMNS),
            "feature_names": list(feature_names),
        }
    )


def model_payload(result: FitResult, dataset_sha256: str) -> dict:
    """Assemble the `logistic_model.json` document from a fit result."""
    schema_hash = feature_schema_hash(result.feature_names)
    coefficients = [float(b) for b in result.beta]
    # `lambda_grid` is the canonical, self-describing selection record: one entry
    # per grid point pairing the regularization strength with the validation
    # game-clustered log-loss it earned, so `lambda` (the chosen scalar) is
    # verifiably the argmin of the per-lambda log-loss recorded here.
    lambda_grid = [
        {"lambda": float(lam), "validation_log_loss": float(loss)}
        for lam, loss in zip(result.lambda_grid, result.validation_log_loss)
    ]
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": schema_hash,
        "dataset_sha256": dataset_sha256,
        "feature_names": result.feature_names,
        "coefficients": coefficients,
        "coefficients_by_name": dict(zip(result.feature_names, coefficients)),
        "standardization": {
            "means": [float(m) for m in result.means],
            "stds": [float(s) for s in result.stds],
        },
        "lambda": result.chosen_lambda,
        "lambda_grid": lambda_grid,
        "chosen_lambda": result.chosen_lambda,
        "lambda_selection": {
            "grid": result.lambda_grid,
            "validation_game_clustered_log_loss": result.validation_log_loss,
            "chosen_index": result.chosen_index,
            "chosen_lambda": result.chosen_lambda,
            "chosen_validation_log_loss": result.validation_log_loss[result.chosen_index],
            "metric": "game_clustered_log_loss",
            "selection_split": "validation",
            "fit_split": "train",
        },
        "objective": "mean_penalized_negative_log_likelihood",
        "penalty": "l2_intercept_free",
    }


def manifest_payload(
    result: FitResult, model_doc: dict, dataset_sha256: str
) -> dict:
    """Assemble the `winprob_model_manifest.json` provenance document."""
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": model_doc["feature_schema_hash"],
        "dataset_sha256": dataset_sha256,
        "model_hash": canonical_hash(model_doc),
        "split_definition": SPLIT_DEFINITION,
        "split_hash": canonical_hash(SPLIT_DEFINITION),
        "splits_used_for_fit": sorted(result.splits_used),
        "holdout_splits_untouched": sorted(HOLDOUT_SPLITS),
        "holdout_rows_excluded": result.holdout_rows_excluded,
        "chosen_lambda": result.chosen_lambda,
        "lambda_grid": result.lambda_grid,
        "validation_game_clustered_log_loss": result.validation_log_loss,
        "chosen_validation_log_loss": result.validation_log_loss[result.chosen_index],
        "n_features": len(result.feature_names),
        "n_train_rows": result.n_train,
        "n_validation_rows": result.n_validation,
        "n_fit_rows": result.n_fit,
        "n_train_games": result.train_games,
        "n_validation_games": result.validation_games,
        "objective": "mean_penalized_negative_log_likelihood",
        "penalty": "l2_intercept_free",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(
    data_dir: Path = DEFAULT_DATA_DIR, grid: tuple[float, ...] = LAMBDA_GRID
) -> FitResult:
    """Load the mart, fit + select lambda, and write both JSON artifacts."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run `python -m winprob.design` first"
        )
    df = pd.read_parquet(parquet_path)
    dataset_sha256 = file_hash(parquet_path)

    result = select_and_fit(df, grid)
    model_doc = model_payload(result, dataset_sha256)
    manifest = manifest_payload(result, model_doc, dataset_sha256)

    _write_json(data_dir / MODEL_JSON_NAME, model_doc)
    _write_json(data_dir / MANIFEST_JSON_NAME, manifest)
    return result


def _print_summary(result: FitResult) -> None:
    print(
        f"winprob logistic: fit on {result.n_fit:,} train+validation rows "
        f"({len(result.feature_names)} features), "
        f"{result.holdout_rows_excluded:,} holdout rows excluded"
    )
    print(
        f"  train rows {result.n_train:,} ({result.train_games:,} games); "
        f"validation rows {result.n_validation:,} ({result.validation_games:,} games)"
    )
    print("  lambda grid (validation game-clustered log-loss):")
    for i, (lam, loss) in enumerate(
        zip(result.lambda_grid, result.validation_log_loss)
    ):
        marker = " <- chosen" if i == result.chosen_index else ""
        print(f"    lambda={lam:.6g}  log_loss={loss:.6f}{marker}")
    print(
        f"  chosen lambda={result.chosen_lambda:.6g} "
        f"(validation log-loss {result.validation_log_loss[result.chosen_index]:.6f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the win-probability logistic model with lambda selection"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    result = run(Path(args.data_dir))
    _print_summary(result)


if __name__ == "__main__":
    main()
