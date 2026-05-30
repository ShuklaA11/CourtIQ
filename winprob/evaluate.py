"""Out-of-sample evaluation + Phase-2 gates for the win-probability model.

This is the honest-broker step that stands on the two builders already in the
package: ``winprob.features`` supplies the pure per-feature column math and the
mean/std standardization, and ``winprob.model`` supplies the guarded sigmoid, the
intercept-free L2 logistic fit, the row/game-clustered log-loss, the leakage
guard ``working_frame``, and the ``canonical_hash``/``file_hash`` hashing pattern
(re-exported there from ``winprob.design``). Nothing here re-derives any of that.

What it does, and why each piece.

*Test split only, applied leakage-safe.* The fitted model was trained on
train+validation (2022-2024) and its holdout (2025) was never touched. Evaluation
reads ONLY ``split == 'test'``. The fitted coefficients live on standardized
columns, so a test row is scored by rebuilding its RAW feature columns in the
model's exact ``feature_names`` order, reapplying the model's SAVED means/stds
(never test-derived ones), and pushing the result through the guarded sigmoid.
Season dummies for 2025 (a season the model never saw) are all-zero, i.e. the
reference-season baseline — the only defensible extrapolation.

*Baselines derived the same leakage-safe way.* Three references are FIT on the
same train+validation working frame with the same intercept-free L2 logistic and
the same standardization discipline, then applied to test: (i) an intercept-only
model — the constant home base rate; (ii) score+time only; (iii)
score+time+possession. Fitting them fresh (rather than reading a stored number)
is what makes "the same way" literally true, and light regularization gives each
baseline its strongest fair form so a gate the model passes is earned, not rigged.

*Game-clustered bootstrap.* Possession rows within a game are massively
autocorrelated, so a row-wise bootstrap would badly understate uncertainty.
Confidence intervals resample GAME IDS with replacement and take every row of
each drawn game, so the resampling unit is the game. The same game draws are
reused across the model and every baseline (a paired bootstrap), which also
yields an interval on the model-minus-baseline difference the gates rest on.

*Gates.* The Phase-2 exit criteria are encoded as explicit booleans under the
``gates`` key of the metrics JSON, each the single source of truth for one
pass/fail line in the printed summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import features, model
from winprob.design import canonical_hash, file_hash

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
MODEL_JSON_NAME = "logistic_model.json"
METRICS_JSON_NAME = "winprob_metrics.json"
AUDIT_JSON_NAME = "winprob_audit.json"

TEST_SPLIT = "test"
TARGET_COLUMN = "home_win"

# Continuous (standardized) feature names — everything else (intercept, binary
# indicators, season dummies) passes through the standardization as an identity.
# Sourced from `winprob.features` (the module that owns the design schema) so a
# knot or feature-name change in one place cannot silently desync this module's
# standardization mask or baseline contract.
CONTINUOUS_FEATURES: frozenset[str] = frozenset(features.CONTINUOUS_FEATURE_NAMES)

# Feature subsets for the three leakage-safe baselines, in fixed column order.
# score+time is exactly the intercept plus every continuous feature, kept in the
# canonical build order so the baseline design matches the model's columns.
BASE_RATE_FEATURES: tuple[str, ...] = ("intercept",)
SCORE_TIME_FEATURES: tuple[str, ...] = ("intercept",) + features.CONTINUOUS_FEATURE_NAMES
SCORE_TIME_POSSESSION_FEATURES: tuple[str, ...] = SCORE_TIME_FEATURES + (
    "home_has_possession",
)

# Baselines are lightly regularized so each fits in its strongest fair form; a
# heavily-shrunk baseline would make the model's gate cheap to pass.
BASELINE_LAMBDA = 1e-4

# Bootstrap: fixed seed makes the emitted metrics (and therefore their hash)
# reproducible run to run.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20250728
BOOTSTRAP_ALPHA = 0.05

# Reliability diagram resolution.
RELIABILITY_BINS = 10

# Calibration gate tolerances: a perfectly calibrated model has intercept 0 and
# slope 1 when the true outcome is logistically regressed on the model's logit.
CALIB_INTERCEPT_TOL = 0.15
CALIB_SLOPE_LO = 0.85
CALIB_SLOPE_HI = 1.15

# Phase-miscalibration gate: a bucket with at least this many rows whose mean
# predicted probability differs from its empirical rate by more than this
# threshold is a gross, material calibration failure.
MISCAL_THRESHOLD = 0.10
MIN_BUCKET_N = 250

# Regulation-seconds-remaining buckets; the final two minutes are called out
# explicitly per the Phase-2 spec.
TIME_BUCKET_EDGES: tuple[float, ...] = (120.0, 360.0, 720.0, 1440.0)
TIME_BUCKET_LABELS: tuple[str, ...] = (
    "final_2min", "2to6min", "6to12min", "12to24min", "24to48min",
)
# Absolute-margin bands (inclusive upper edge; last band is open).
MARGIN_BAND_EDGES: tuple[float, ...] = (3.0, 6.0, 10.0, 15.0)
MARGIN_BAND_LABELS: tuple[str, ...] = ("0-3", "4-6", "7-10", "11-15", "16+")


# --------------------------------------------------------------------------
# Scalar metrics.
# --------------------------------------------------------------------------

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability forecast, `mean((p - y)^2)`."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def mean_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Row-wise mean binary cross-entropy (guarded), reusing `model.row_log_loss`."""
    return float(np.mean(model.row_log_loss(y, p)))


def fit_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of the outcome on the model's logit.

    Fits `logit(P(y=1)) = intercept + slope * logit(p)` by unpenalized MLE
    (reusing `model.fit_l2_logistic` at lambda 0). A perfectly calibrated model
    yields intercept 0 and slope 1; the coefficient bounds inside the fit keep a
    perfectly separating input from diverging. Returns `(intercept, slope)`.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), model.PROB_EPS, 1.0 - model.PROB_EPS)
    z = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones_like(z), z])
    y = np.asarray(y, dtype=np.float64)
    beta = model.fit_l2_logistic(X, y, lam=0.0, pen_mask=np.zeros(2))
    return float(beta[0]), float(beta[1])


# --------------------------------------------------------------------------
# Design assembly — apply a fitted model / fit a baseline, leakage-safe.
# --------------------------------------------------------------------------

def raw_feature_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw (pre-standardization) structural columns keyed by feature name.

    Reuses the pure per-feature builders in `winprob.features` so this module
    never forks the feature math. Season dummies are handled by the caller
    (parsed from `season_<year>` names) since their set depends on the frame.
    """
    n = len(df)
    margin = features.margin_column(df["home_score_differential"].to_numpy())
    reg_sec = np.asarray(
        df["regulation_seconds_remaining"].to_numpy(), dtype=np.float64
    )
    cols: dict[str, np.ndarray] = {
        "intercept": features.constant_column(n),
        "home_score_differential": margin,
        "margin_over_sqrt_time": features.margin_over_sqrt_time(margin, reg_sec),
        "regulation_seconds_remaining": reg_sec,
        "home_has_possession": features.home_possession_column(
            df["home_has_possession"].to_numpy()
        ),
        "is_playoff": features.playoff_indicator(df["game_id"].to_numpy()),
    }
    knots = features.time_knot_basis(reg_sec)
    for name, knot_col in zip(features.TIME_KNOT_NAMES, knots.T):
        cols[name] = knot_col
    return cols


def assemble_raw_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Raw design matrix with columns in `feature_names` order.

    Structural columns come from `raw_feature_columns`; any `season_<year>` name
    becomes a one-hot on that season (all-zero for a season absent from the
    frame, e.g. the 2025 test rows against a model trained through 2024).
    """
    raw = raw_feature_columns(df)
    season = np.asarray(df["season"].to_numpy()).astype(int)
    columns: list[np.ndarray] = []
    for name in feature_names:
        if name in raw:
            columns.append(raw[name])
        elif name.startswith("season_"):
            year = int(name.split("_", 1)[1])
            columns.append((season == year).astype(np.float64))
        else:
            raise ValueError(f"unknown feature name in model contract: {name!r}")
    return np.column_stack(columns)


def continuous_mask_for(feature_names: list[str]) -> np.ndarray:
    """Boolean mask marking which named columns are standardized continuous ones."""
    return np.array(
        [name in CONTINUOUS_FEATURES for name in feature_names], dtype=bool
    )


def predict_with_model(df: pd.DataFrame, model_doc: dict) -> np.ndarray:
    """Guarded win probabilities for `df` under a serialized model document.

    Rebuilds the raw columns in the model's `feature_names` order, reapplies the
    model's SAVED standardization (`(raw - means) / stds`), and pushes the
    standardized rows through the fitted coefficients. Using the saved stats — not
    stats recomputed on `df` — is what keeps test scoring leakage-safe.
    """
    feature_names = list(model_doc["feature_names"])
    beta = np.asarray(model_doc["coefficients"], dtype=np.float64)
    means = np.asarray(model_doc["standardization"]["means"], dtype=np.float64)
    stds = np.asarray(model_doc["standardization"]["stds"], dtype=np.float64)
    raw = assemble_raw_matrix(df, feature_names)
    X = (raw - means) / stds
    return model.predict_proba(X, beta)


@dataclass(frozen=True)
class Baseline:
    """A leakage-safe baseline: its feature contract, scaling, and coefficients."""

    name: str
    feature_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    beta: np.ndarray


def fit_baseline(
    work: pd.DataFrame, name: str, feature_names: tuple[str, ...]
) -> Baseline:
    """Fit one baseline on the train+validation working frame, leakage-safe.

    Standardizes the subset's continuous columns on `work`, fits the same
    intercept-free L2 logistic used for the full model at a light lambda, and
    returns the coefficients plus the scaling so it can be applied to test with
    the SAME transform.
    """
    raw = assemble_raw_matrix(work, list(feature_names))
    mask = continuous_mask_for(list(feature_names))
    X, means, stds = features.standardize_columns(raw, mask)
    y = work[TARGET_COLUMN].to_numpy().astype(np.float64)
    pen = model.penalty_mask(list(feature_names))
    beta = model.fit_l2_logistic(X, y, BASELINE_LAMBDA, pen)
    return Baseline(name, feature_names, means, stds, beta)


def predict_with_baseline(baseline: Baseline, df: pd.DataFrame) -> np.ndarray:
    """Apply a fitted baseline to `df` using its stored scaling and coefficients."""
    raw = assemble_raw_matrix(df, list(baseline.feature_names))
    X = (raw - baseline.means) / baseline.stds
    return model.predict_proba(X, baseline.beta)


# --------------------------------------------------------------------------
# Game-clustered bootstrap.
# --------------------------------------------------------------------------

def build_game_groups(game_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by game, as one block of positions per distinct game.

    This is the invariant structure a game-clustered bootstrap draws from: it
    depends only on `game_ids`, not on the random draw, so it is computed ONCE and
    reused across every bootstrap iteration rather than rebuilt inside the loop.
    Groups are returned in first-appearance order and their concatenation covers
    every row exactly once, so the mapping is a pure partition of the rows.
    """
    game_ids = np.asarray(game_ids)
    positions = pd.Series(np.arange(len(game_ids)))
    # sort=False keeps groups in first-appearance order; one O(n) pass total.
    grouped = positions.groupby(game_ids, sort=False)
    return [block.to_numpy() for _, block in grouped]


def draw_game_rows(
    groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """One game-clustered resample: draw `k` of the `k` game blocks, with replacement.

    Takes the precomputed `groups` from `build_game_groups` and returns the row
    indices of the drawn games as whole blocks (a game drawn twice contributes all
    its rows twice). Only the random draw and the concatenation run per call — the
    expensive per-game index map is built once by the caller.
    """
    k = len(groups)
    chosen = rng.integers(0, k, size=k)
    parts = [groups[i] for i in chosen]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def resample_game_indices(
    game_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample GAMES with replacement; return the row indices of the drawn games.

    The resampling unit is the game, not the row: `k` unique games are drawn with
    replacement (`k` = number of distinct games) and EVERY row of each drawn game
    is taken as a whole block. A game drawn twice contributes all of its rows
    twice. This is the correct cluster bootstrap for possession rows that are
    heavily autocorrelated within a game. Thin wrapper over `build_game_groups` +
    `draw_game_rows` for single-shot callers; the bootstrap loop builds the group
    map once and calls `draw_game_rows` directly.
    """
    return draw_game_rows(build_game_groups(game_ids), rng)


def _percentile_ci(values: list[float], alpha: float) -> dict[str, float]:
    """Two-sided percentile interval `[alpha/2, 1 - alpha/2]` of a bootstrap sample."""
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return {"lo": lo, "hi": hi, "point": float(np.mean(arr))}


def game_clustered_bootstrap(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    game_ids: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    """Paired game-clustered CIs for Brier and log loss of several forecasts.

    Draws one set of game clusters per iteration and scores EVERY forecast in
    `preds` on it, so the model and its baselines share the same resampled games.
    Returns per-forecast Brier/log-loss intervals plus, for each non-`model`
    forecast, an interval on the paired `model - baseline` difference (negative =
    the model is better on that draw).
    """
    y = np.asarray(y, dtype=np.float64)
    keys = list(preds)
    briers: dict[str, list[float]] = {k: [] for k in keys}
    loglosses: dict[str, list[float]] = {k: [] for k in keys}

    # The per-game row-index map is invariant across iterations, so build it once
    # here instead of rebuilding it inside every one of the `n_boot` draws.
    groups = build_game_groups(game_ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = draw_game_rows(groups, rng)
        yb = y[idx]
        for k in keys:
            pb = preds[k][idx]
            briers[k].append(brier_score(yb, pb))
            loglosses[k].append(mean_log_loss(yb, pb))

    out: dict = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "brier": {k: _percentile_ci(briers[k], alpha) for k in keys},
        "log_loss": {k: _percentile_ci(loglosses[k], alpha) for k in keys},
    }
    if "model" in preds:
        diffs: dict[str, dict] = {}
        bm = np.asarray(briers["model"])
        lm = np.asarray(loglosses["model"])
        for k in keys:
            if k == "model":
                continue
            diffs[k] = {
                "brier": _percentile_ci(
                    (bm - np.asarray(briers[k])).tolist(), alpha
                ),
                "log_loss": _percentile_ci(
                    (lm - np.asarray(loglosses[k])).tolist(), alpha
                ),
            }
        out["model_minus_baseline"] = diffs
    return out


# --------------------------------------------------------------------------
# Reliability table and phase breakdowns.
# --------------------------------------------------------------------------

def reliability_table(
    y: np.ndarray, p: np.ndarray, n_bins: int = RELIABILITY_BINS
) -> list[dict]:
    """Binned predicted-vs-empirical reliability rows over `n_bins` equal-width bins."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize on the interior edges so p in [edge_b, edge_{b+1}) lands in bin b.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    table: list[dict] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        table.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(p[mask].mean()) if n else None,
            "empirical": float(y[mask].mean()) if n else None,
        })
    return table


def _bucket_metrics(
    labels: np.ndarray, order: tuple[str, ...], y: np.ndarray, p: np.ndarray
) -> list[dict]:
    """Per-bucket metrics for a categorical labelling, in a fixed label order."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    rows: list[dict] = []
    for lab in order:
        mask = labels == lab
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": lab, "n": 0})
            continue
        mean_pred = float(p[mask].mean())
        empirical = float(y[mask].mean())
        rows.append({
            "bucket": lab,
            "n": n,
            "mean_predicted": mean_pred,
            "empirical": empirical,
            "calibration_error": mean_pred - empirical,
            "brier": brier_score(y[mask], p[mask]),
            "log_loss": mean_log_loss(y[mask], p[mask]),
        })
    return rows


def period_labels(period: np.ndarray) -> np.ndarray:
    """Map period numbers to `Q1..Q4` and a single `OT` bucket for period >= 5."""
    period = np.asarray(period).astype(int)
    return np.where(period <= 4, np.char.add("Q", period.astype(str)), "OT")


def time_bucket_labels(reg_sec: np.ndarray) -> np.ndarray:
    """Bucket regulation-seconds-remaining, isolating the final two minutes."""
    reg_sec = np.asarray(reg_sec, dtype=np.float64)
    idx = np.digitize(reg_sec, TIME_BUCKET_EDGES, right=True)
    return np.asarray(TIME_BUCKET_LABELS, dtype=object)[idx]


def margin_band_labels(margin: np.ndarray) -> np.ndarray:
    """Bucket absolute home margin into fixed bands."""
    absmargin = np.abs(np.asarray(margin, dtype=np.float64))
    idx = np.digitize(absmargin, MARGIN_BAND_EDGES, right=True)
    return np.asarray(MARGIN_BAND_LABELS, dtype=object)[idx]


def phase_breakdowns(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    """All three Phase-2 breakdowns: by period, by time bucket, by margin band."""
    periods = period_labels(df["period"].to_numpy())
    period_order = ("Q1", "Q2", "Q3", "Q4", "OT")
    times = time_bucket_labels(df["regulation_seconds_remaining"].to_numpy())
    margins = margin_band_labels(df["home_score_differential"].to_numpy())
    return {
        "by_period": _bucket_metrics(periods, period_order, y, p),
        "by_time_remaining": _bucket_metrics(times, TIME_BUCKET_LABELS, y, p),
        "by_absolute_margin": _bucket_metrics(margins, MARGIN_BAND_LABELS, y, p),
    }


def material_miscalibration(breakdowns: dict) -> list[dict]:
    """Buckets whose calibration failure is both large and well-sampled.

    A bucket counts as a gross failure when it holds at least `MIN_BUCKET_N` rows
    and its mean predicted probability differs from its empirical rate by more
    than `MISCAL_THRESHOLD`. Returns the offending buckets (empty = gate passes).
    """
    offenders: list[dict] = []
    for group, rows in breakdowns.items():
        for row in rows:
            if row.get("n", 0) < MIN_BUCKET_N:
                continue
            if abs(row["calibration_error"]) > MISCAL_THRESHOLD:
                offenders.append({"group": group, **row})
    return offenders


# --------------------------------------------------------------------------
# Assembly: metrics, gates.
# --------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, model_doc: dict) -> dict:
    """Score the model + three baselines out-of-sample on the test split.

    Returns the full metrics dict (headline scalars, calibration, reliability
    table, phase breakdowns, baseline comparison, bootstrap CIs, and the `gates`
    booleans). Pure with respect to `df`/`model_doc`: reads them, writes nothing.
    """
    test = df.loc[df["split"] == TEST_SPLIT].reset_index(drop=True)
    if len(test) == 0:
        raise ValueError("no rows with split == 'test' to evaluate on")
    work = model.working_frame(df).frame  # train+validation, leakage guard.

    y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = test["game_id"].to_numpy()

    p_model = predict_with_model(test, model_doc)
    baselines = {
        "base_rate": fit_baseline(work, "base_rate", BASE_RATE_FEATURES),
        "score_time": fit_baseline(work, "score_time", SCORE_TIME_FEATURES),
        "score_time_possession": fit_baseline(
            work, "score_time_possession", SCORE_TIME_POSSESSION_FEATURES
        ),
    }
    baseline_preds = {
        name: predict_with_baseline(bl, test) for name, bl in baselines.items()
    }
    preds = {"model": p_model, **baseline_preds}

    intercept, slope = fit_calibration(y, p_model)
    breakdowns = phase_breakdowns(test, y, p_model)
    bootstrap = game_clustered_bootstrap(y, preds, game_ids)

    metrics: dict = {
        "n_test_rows": int(len(test)),
        "n_test_games": int(pd.Series(game_ids).nunique()),
        "home_win_rate_test": float(y.mean()),
        "model": {
            "brier": brier_score(y, p_model),
            "log_loss": mean_log_loss(y, p_model),
            "game_clustered_log_loss": model.game_clustered_log_loss(
                y, p_model, game_ids
            ),
        },
        "baselines": {
            name: {
                "brier": brier_score(y, baseline_preds[name]),
                "log_loss": mean_log_loss(y, baseline_preds[name]),
                "features": list(baselines[name].feature_names),
            }
            for name in baselines
        },
        "calibration": {
            "intercept": intercept,
            "slope": slope,
            "reliability_table": reliability_table(y, p_model),
        },
        "phase_breakdowns": breakdowns,
        "bootstrap": bootstrap,
        "predictions_min": float(p_model.min()),
        "predictions_max": float(p_model.max()),
    }
    metrics["gates"] = compute_gates(metrics)
    return metrics


def compute_gates(metrics: dict) -> dict[str, bool]:
    """Encode the Phase-2 exit criteria as explicit booleans.

    Every value is a plain `bool`, so the metrics JSON is a machine-checkable
    contract and the printed summary reads straight off this dict.
    """
    model_brier = metrics["model"]["brier"]
    model_logloss = metrics["model"]["log_loss"]
    score_time = metrics["baselines"]["score_time"]
    intercept = metrics["calibration"]["intercept"]
    slope = metrics["calibration"]["slope"]
    offenders = material_miscalibration(metrics["phase_breakdowns"])
    pmin, pmax = metrics["predictions_min"], metrics["predictions_max"]
    return {
        "gate_brier_beats_score_time": bool(model_brier < score_time["brier"]),
        "gate_logloss_beats_score_time": bool(model_logloss < score_time["log_loss"]),
        "gate_calibration_intercept_near_zero": bool(
            abs(intercept) < CALIB_INTERCEPT_TOL
        ),
        "gate_calibration_slope_near_one": bool(
            CALIB_SLOPE_LO <= slope <= CALIB_SLOPE_HI
        ),
        "gate_predictions_in_open_interval": bool(pmin > 0.0 and pmax < 1.0),
        "gate_no_material_phase_miscalibration": bool(len(offenders) == 0),
    }


def all_gates_pass(metrics: dict) -> bool:
    """True iff every gate boolean holds — the single source of truth for exit code."""
    return all(bool(v) for v in metrics["gates"].values())


# --------------------------------------------------------------------------
# Serialization + provenance audit.
# --------------------------------------------------------------------------

def audit_payload(
    metrics: dict, parquet_path: Path, model_path: Path, model_doc: dict
) -> dict:
    """Provenance document reusing the canonical_hash/file_hash hashing pattern."""
    return {
        "metrics_hash": canonical_hash(metrics),
        "dataset_parquet_sha256": file_hash(parquet_path),
        "model_json_sha256": file_hash(model_path),
        "model_feature_schema_hash": model_doc.get("feature_schema_hash"),
        "model_dataset_sha256": model_doc.get("dataset_sha256"),
        "chosen_lambda": model_doc.get("lambda"),
        "n_test_rows": metrics["n_test_rows"],
        "n_test_games": metrics["n_test_games"],
        "bootstrap": {
            "n_boot": metrics["bootstrap"]["n_boot"],
            "seed": metrics["bootstrap"]["seed"],
            "alpha": metrics["bootstrap"]["alpha"],
        },
        "gates": metrics["gates"],
        "all_gates_pass": all_gates_pass(metrics),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart + fitted model, evaluate on test, write metrics + audit JSON."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    model_path = data_dir / MODEL_JSON_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run `python -m winprob.design` first"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing fitted model at {model_path}; run `python -m winprob.model` first"
        )
    df = pd.read_parquet(parquet_path)
    model_doc = json.loads(model_path.read_text())

    metrics = evaluate(df, model_doc)
    audit = audit_payload(metrics, parquet_path, model_path, model_doc)

    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    m = metrics["model"]
    boot = metrics["bootstrap"]
    print(
        f"winprob evaluate: {metrics['n_test_rows']:,} test rows / "
        f"{metrics['n_test_games']:,} games "
        f"(home win rate {metrics['home_win_rate_test']:.3f})"
    )
    print("\n  metric                            model     base_rate  score_time  score_time_poss")
    briers = {"model": m["brier"], **{k: v["brier"] for k, v in metrics["baselines"].items()}}
    lls = {"model": m["log_loss"], **{k: v["log_loss"] for k, v in metrics["baselines"].items()}}
    print(
        f"  Brier                          {briers['model']:9.5f} {briers['base_rate']:10.5f} "
        f"{briers['score_time']:11.5f} {briers['score_time_possession']:15.5f}"
    )
    print(
        f"  log loss                       {lls['model']:9.5f} {lls['base_rate']:10.5f} "
        f"{lls['score_time']:11.5f} {lls['score_time_possession']:15.5f}"
    )
    cb = boot["brier"]["model"]
    cl = boot["log_loss"]["model"]
    print(
        f"\n  model Brier 95% CI    [{cb['lo']:.5f}, {cb['hi']:.5f}]   "
        f"log loss 95% CI [{cl['lo']:.5f}, {cl['hi']:.5f}]  "
        f"(game-clustered, {boot['n_boot']} draws)"
    )
    cal = metrics["calibration"]
    print(
        f"  calibration intercept {cal['intercept']:+.4f}   slope {cal['slope']:.4f}"
    )

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    n_pass = sum(bool(v) for v in metrics["gates"].values())
    print(f"  {n_pass}/{len(metrics['gates'])} gates pass")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the win-probability model out-of-sample and run its gates"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not all_gates_pass(metrics):
        raise SystemExit("gate failure: one or more Phase-2 gates did not pass")


if __name__ == "__main__":
    main()
