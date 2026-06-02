"""Sprint-3 Phase-3 leakage-safe RAPM lineup ablation.

The project's headline question: once you know the score, the clock, and the
possession, does knowing *which players are on the floor* — via prior-season RAPM
— tell you anything more about who wins? This module answers it with a nested
A..E ladder scored out of sample on the untouched 2025 test season, reusing the
Phase-2 machinery wholesale rather than forking a parallel pipeline:

    ``winprob.features``  — the pure per-feature column math + mean/std scaling
    ``winprob.model``     — guarded sigmoid, intercept-free L2 fit, working-frame
                            leakage guard, game-clustered log-loss
    ``winprob.evaluate``  — Brier / log-loss, reliability table, phase breakdowns,
                            the game-clustered bootstrap primitives
    ``winprob.design``    — canonical/file hashing + the pinned split definition

The tiers (each strictly nests in the next):

    A  score + time                        (== Phase-2 score_time baseline)
    B  + possession                        (== Phase-2 score_time_possession)
    C  + pregame team strength             (team-season mean on-court net RAPM)
    D  + current-lineup net RAPM           (lineup_net_rapm_differential)
    E  + RAPM coverage (rated-player counts)

*Why the confound control (C before D).* Lineup RAPM is high when good teams
play, and good teams are usually already ahead — a signal ``margin`` already
sees. Tier C absorbs "this is a good team" at the team level, so tier D only
earns its keep if the *specific five on the floor right now* (starters vs bench,
foul trouble, a stagger) adds held-out accuracy *beyond* the team's baseline
strength. That is the real, non-obvious thing lineup-level RAPM could contribute
that a team rating cannot.

*Leakage safety.* Ratings in the mart are already prior-season (season S uses
S-1) by construction. Team strength is a per-(team, season) pooled mean, and
because ``season`` is part of the key and the splits partition on season, a
test-season team's strength is a function of test-season rows only — never of the
fitted train/validation rows. Standardization means/stds come from the
train+validation working frame only; 2025 never enters any fit or lambda
selection.

*Honest nulls.* The gates separate STRUCTURAL integrity (predictions in (0,1),
late-game calibration not degraded C->D, every rating strictly prior) from the
SCIENTIFIC finding (D beats C beyond game-clustered bootstrap noise, reproduced
across rolling folds). Only structural failure is an error — a null RAPM result
is a valid, publishable finding, so ``structural_gates_pass`` drives the exit
code, not the scientific verdict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import design, evaluate, features, model

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
METRICS_JSON_NAME = "ablation_metrics.json"
AUDIT_JSON_NAME = "ablation_audit.json"

TEST_SPLIT = "test"
TARGET_COLUMN = model.TARGET_COLUMN

# --------------------------------------------------------------------------
# Tier feature contracts. A/B are byte-identical to the Phase-2 baselines so the
# ladder nests ON Phase 2; C/D/E append the RAPM columns in fixed order.
# --------------------------------------------------------------------------

TIER_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E")

TEAM_STRENGTH_FEATURES: tuple[str, ...] = ("home_team_strength", "away_team_strength")
LINEUP_RAPM_FEATURES: tuple[str, ...] = ("lineup_net_rapm_differential",)
UNCERTAINTY_FEATURES: tuple[str, ...] = ("home_rated_players", "away_rated_players")

TIER_A_FEATURES: tuple[str, ...] = evaluate.SCORE_TIME_FEATURES
TIER_B_FEATURES: tuple[str, ...] = evaluate.SCORE_TIME_POSSESSION_FEATURES
TIER_C_FEATURES: tuple[str, ...] = TIER_B_FEATURES + TEAM_STRENGTH_FEATURES
TIER_D_FEATURES: tuple[str, ...] = TIER_C_FEATURES + LINEUP_RAPM_FEATURES
TIER_E_FEATURES: tuple[str, ...] = TIER_D_FEATURES + UNCERTAINTY_FEATURES

TIER_FEATURES: dict[str, tuple[str, ...]] = {
    "A": TIER_A_FEATURES,
    "B": TIER_B_FEATURES,
    "C": TIER_C_FEATURES,
    "D": TIER_D_FEATURES,
    "E": TIER_E_FEATURES,
}

# The RAPM columns read from the mart, all standardized as continuous features
# alongside the Phase-2 continuous set (margin, margin/sqrt(time), knots, ...).
RAPM_MART_COLUMNS: tuple[str, ...] = (
    LINEUP_RAPM_FEATURES + UNCERTAINTY_FEATURES
)
ABLATION_CONTINUOUS_FEATURES: frozenset[str] = (
    frozenset(features.CONTINUOUS_FEATURE_NAMES)
    | set(TEAM_STRENGTH_FEATURES)
    | set(LINEUP_RAPM_FEATURES)
    | set(UNCERTAINTY_FEATURES)
)

# Rolling forward-chaining folds for the reproduction gate. 2025 appears only as
# a TEST season (fold B), never in any fold's train/validation, so the untouched
# holdout is never fit on. 2021 is all cold-start and cannot be a RAPM season.
FOLDS: dict[str, dict[int, str]] = {
    "A": {2022: "train", 2023: "validation", 2024: "test"},
    "B": {2022: "train", 2023: "train", 2024: "validation", 2025: "test"},
}

# Allowed late-game (final 2 min) calibration degradation from C to D before the
# structural gate trips: D's |calibration error| may exceed C's by at most this.
LATE_GAME_CALIB_TOL = 0.02

STRUCTURAL_GATE_NAMES: tuple[str, ...] = (
    "gate_predictions_in_open_interval",
    "gate_late_game_calibration_not_degraded",
    "gate_every_rating_strictly_prior",
)
SCIENTIFIC_GATE_NAMES: tuple[str, ...] = (
    "gate_rapm_beats_team_strength",
    "gate_reproduces_rolling",
)


# --------------------------------------------------------------------------
# Team strength (tier C) — leakage-safe per-(team, season) pooled mean.
# --------------------------------------------------------------------------

def add_team_strength_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a NEW frame with `home_team_strength`/`away_team_strength`.

    Team strength is the possession-weighted mean on-court net RAPM the team
    fielded that season, pooled over its home AND away appearances. Because the
    aggregation key is (team_id, season) and splits partition on season, a
    test-season team's strength depends only on test-season rows — no train/val
    row can leak into it. Pure: the input frame is copied, never mutated.
    """
    home = df[["home_team_id", "season", "home_lineup_net_rapm"]].rename(
        columns={"home_team_id": "team_id", "home_lineup_net_rapm": "net"}
    )
    away = df[["away_team_id", "season", "away_lineup_net_rapm"]].rename(
        columns={"away_team_id": "team_id", "away_lineup_net_rapm": "net"}
    )
    pooled = pd.concat([home, away], ignore_index=True)
    strength = pooled.groupby(["team_id", "season"])["net"].mean()

    home_key = pd.MultiIndex.from_arrays([df["home_team_id"], df["season"]])
    away_key = pd.MultiIndex.from_arrays([df["away_team_id"], df["season"]])
    out = df.copy()
    out["home_team_strength"] = strength.reindex(home_key).to_numpy()
    out["away_team_strength"] = strength.reindex(away_key).to_numpy()
    return out


# --------------------------------------------------------------------------
# Design assembly for an arbitrary tier — reuse Phase-2 column math, add RAPM.
# --------------------------------------------------------------------------

def raw_ablation_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw (pre-standardization) columns keyed by name: Phase-2 set + RAPM.

    The structural columns come straight from `evaluate.raw_feature_columns`
    (intercept, margin, margin/sqrt(time), reg_sec, possession, playoff, knots);
    the team-strength and RAPM-mart columns are read from `df` when present.
    """
    cols: dict[str, np.ndarray] = dict(evaluate.raw_feature_columns(df))
    for name in TEAM_STRENGTH_FEATURES + RAPM_MART_COLUMNS:
        if name in df.columns:
            cols[name] = np.asarray(df[name].to_numpy(), dtype=np.float64)
    return cols


def continuous_mask_for(feature_names: tuple[str, ...]) -> np.ndarray:
    """Boolean mask marking which named columns are standardized continuous ones."""
    return np.array(
        [name in ABLATION_CONTINUOUS_FEATURES for name in feature_names], dtype=bool
    )


def assemble_tier_matrix(
    df: pd.DataFrame, feature_names: tuple[str, ...]
) -> np.ndarray:
    """Raw design matrix with columns in `feature_names` order for one tier."""
    raw = raw_ablation_columns(df)
    columns: list[np.ndarray] = []
    for name in feature_names:
        if name not in raw:
            raise ValueError(f"unknown feature name in tier contract: {name!r}")
        columns.append(raw[name])
    return np.column_stack(columns)


# --------------------------------------------------------------------------
# Per-tier fit: lambda selection on validation, refit on train+validation.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TierFit:
    """Everything needed to apply a fitted tier to a fresh (e.g. test) frame."""

    name: str
    feature_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    beta: np.ndarray
    lambda_grid: tuple[float, ...]
    validation_log_loss: list[float]
    chosen_index: int
    chosen_lambda: float
    splits_used: frozenset[str]
    holdout_rows_excluded: int


def fit_tier(
    df: pd.DataFrame,
    name: str,
    feature_names: tuple[str, ...],
    grid: tuple[float, ...] = model.LAMBDA_GRID,
) -> TierFit:
    """Fit one tier leakage-safe: select lambda on validation, refit on train+val.

    Mirrors `model.select_and_fit` on an arbitrary feature subset. Drops holdout
    rows via the shared `working_frame` guard, standardizes the tier's continuous
    columns on the train+validation frame, fits each grid lambda on TRAIN rows,
    scores game-clustered log-loss on VALIDATION rows, and refits at the grid
    argmin on train+validation combined. `df` must already carry the team-strength
    columns (`add_team_strength_columns`) for tiers C and up.
    """
    if len(grid) == 0:
        raise ValueError("lambda grid must be non-empty")

    work = model.working_frame(df)
    frame = work.frame
    raw = assemble_tier_matrix(frame, feature_names)
    mask = continuous_mask_for(feature_names)
    X, means, stds = features.standardize_columns(raw, mask)
    pen = model.penalty_mask(list(feature_names))

    y = frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = frame["game_id"].to_numpy()
    is_train = (frame["split"] == "train").to_numpy()
    is_val = (frame["split"] == "validation").to_numpy()

    validation_log_loss: list[float] = []
    for lam in grid:
        beta = model.fit_l2_logistic(X[is_train], y[is_train], float(lam), pen)
        p_val = model.predict_proba(X[is_val], beta)
        validation_log_loss.append(
            model.game_clustered_log_loss(y[is_val], p_val, game_ids[is_val])
        )

    chosen_index = int(np.argmin(validation_log_loss))
    chosen_lambda = float(grid[chosen_index])
    beta_final = model.fit_l2_logistic(X, y, chosen_lambda, pen)

    return TierFit(
        name=name,
        feature_names=tuple(feature_names),
        means=np.asarray(means, dtype=np.float64),
        stds=np.asarray(stds, dtype=np.float64),
        beta=beta_final,
        lambda_grid=tuple(float(lam) for lam in grid),
        validation_log_loss=validation_log_loss,
        chosen_index=chosen_index,
        chosen_lambda=chosen_lambda,
        splits_used=work.splits_used,
        holdout_rows_excluded=work.holdout_rows_excluded,
    )


def predict_tier(fit: TierFit, df: pd.DataFrame) -> np.ndarray:
    """Guarded win probabilities for `df` under a fitted tier's saved scaling."""
    raw = assemble_tier_matrix(df, fit.feature_names)
    X = (raw - fit.means) / fit.stds
    return model.predict_proba(X, fit.beta)


# --------------------------------------------------------------------------
# Analysis outputs: Brier decomposition + where RAPM helped.
# --------------------------------------------------------------------------

def brier_decomposition(
    y: np.ndarray, p: np.ndarray, n_bins: int = evaluate.RELIABILITY_BINS
) -> dict:
    """Murphy calibration-refinement decomposition of the Brier score.

    Brier = Reliability - Resolution + Uncertainty, where Uncertainty is a
    property of the OUTCOMES alone (identical across models), Resolution measures
    how far each bin's empirical rate sits from the base rate (higher = better
    sorting), and Reliability is the within-bin calibration error (lower =
    better). The binned reconstruction recovers the raw Brier to within binning
    error, which is reported so the approximation is observable.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    table = evaluate.reliability_table(y, p, n_bins)
    n_total = len(y)
    base_rate = float(y.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    reliability = 0.0
    resolution = 0.0
    for row in table:
        n = row["n"]
        if n == 0:
            continue
        weight = n / n_total
        reliability += weight * (row["mean_predicted"] - row["empirical"]) ** 2
        resolution += weight * (row["empirical"] - base_rate) ** 2

    reconstructed = reliability - resolution + uncertainty
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_reconstructed": reconstructed,
        "brier": evaluate.brier_score(y, p),
        "n_bins": int(n_bins),
    }


def subgroup_improvement(breakdowns_c: dict, breakdowns_d: dict) -> dict:
    """Per-subgroup C-minus-D Brier/log-loss: where did the lineup tier help?

    A positive improvement means tier D scored lower (better) than tier C in that
    bucket. Buckets empty in either tier are skipped. This is the "and where"
    half of the "did lineup strength help, and where" deliverable.
    """
    out: dict[str, list[dict]] = {}
    for group, rows_c in breakdowns_c.items():
        by_bucket_d = {r["bucket"]: r for r in breakdowns_d.get(group, [])}
        improvements: list[dict] = []
        for row_c in rows_c:
            bucket = row_c["bucket"]
            row_d = by_bucket_d.get(bucket)
            if row_c.get("n", 0) == 0 or row_d is None or row_d.get("n", 0) == 0:
                continue
            improvements.append(
                {
                    "bucket": bucket,
                    "n": row_c["n"],
                    "brier_improvement": row_c["brier"] - row_d["brier"],
                    "log_loss_improvement": row_c["log_loss"] - row_d["log_loss"],
                }
            )
        out[group] = improvements
    return out


# --------------------------------------------------------------------------
# Paired game-clustered bootstrap across tiers.
# --------------------------------------------------------------------------

def paired_diff_ci(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    game_ids: np.ndarray,
    pairs: list[tuple[str, str]],
    n_boot: int = evaluate.N_BOOTSTRAP,
    seed: int = evaluate.BOOTSTRAP_SEED,
    alpha: float = evaluate.BOOTSTRAP_ALPHA,
) -> dict:
    """Adjacent-tier difference CIs from one shared set of game-cluster draws.

    Every tier is scored on the SAME resampled games each iteration (a paired
    bootstrap), so the `hi - lo` difference interval is the honest test of whether
    the richer tier `hi` genuinely beats `lo` rather than two marginal intervals
    that overlap by construction. A negative interval means `hi` scored lower
    (better). Reuses the `evaluate` bootstrap primitives so the resampling unit is
    the game, never the row.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = evaluate.build_game_groups(game_ids)
    rng = np.random.default_rng(seed)
    briers: dict[str, list[float]] = {k: [] for k in preds}
    loglosses: dict[str, list[float]] = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = evaluate.draw_game_rows(groups, rng)
        yb = y[idx]
        for k in preds:
            pb = preds[k][idx]
            briers[k].append(evaluate.brier_score(yb, pb))
            loglosses[k].append(evaluate.mean_log_loss(yb, pb))

    out: dict[str, dict] = {}
    for hi, lo in pairs:
        b_diff = (np.asarray(briers[hi]) - np.asarray(briers[lo])).tolist()
        l_diff = (np.asarray(loglosses[hi]) - np.asarray(loglosses[lo])).tolist()
        out[f"{hi}_minus_{lo}"] = {
            "brier": evaluate._percentile_ci(b_diff, alpha),
            "log_loss": evaluate._percentile_ci(l_diff, alpha),
        }
    return out


# --------------------------------------------------------------------------
# Rolling folds + rating provenance.
# --------------------------------------------------------------------------

def fold_frame(prepared: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    """Reslice a prepared frame to one rolling fold via a season->split mapping.

    Team strength is per-(team, season) and so is unaffected by the split
    reassignment, which is why `prepared` (already carrying strength columns) can
    be resliced directly. Seasons absent from `mapping` are dropped.
    """
    keep = prepared["season"].isin(mapping)
    out = prepared.loc[keep].copy()
    out["split"] = out["season"].map(mapping)
    return out.reset_index(drop=True)


def rolling_fold_diffs(
    prepared: pd.DataFrame, grid: tuple[float, ...] = model.LAMBDA_GRID
) -> list[dict]:
    """Fit C and D on each rolling fold; report the D-minus-C test-season deltas."""
    results: list[dict] = []
    for fold_name, mapping in FOLDS.items():
        fold = fold_frame(prepared, mapping)
        fit_c = fit_tier(fold, "C", TIER_C_FEATURES, grid)
        fit_d = fit_tier(fold, "D", TIER_D_FEATURES, grid)
        test = fold.loc[fold["split"] == "test"].reset_index(drop=True)
        y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
        p_c = predict_tier(fit_c, test)
        p_d = predict_tier(fit_d, test)
        results.append(
            {
                "fold": fold_name,
                "test_seasons": sorted(int(s) for s in test["season"].unique()),
                "c_brier": evaluate.brier_score(y, p_c),
                "d_brier": evaluate.brier_score(y, p_d),
                "d_minus_c_brier_point": evaluate.brier_score(y, p_d)
                - evaluate.brier_score(y, p_c),
                "d_minus_c_logloss_point": evaluate.mean_log_loss(y, p_d)
                - evaluate.mean_log_loss(y, p_c),
            }
        )
    return results


def rating_provenance_ok(df: pd.DataFrame) -> bool:
    """True iff every rated row draws its RAPM strictly from the prior season."""
    source = df["rapm_source_season"]
    rated = source.notna()
    if not rated.any():
        return True
    return bool((df.loc[rated, "rapm_source_season"] == df.loc[rated, "season"] - 1).all())


# --------------------------------------------------------------------------
# Gates: structural integrity vs the scientific finding.
# --------------------------------------------------------------------------

def _final_2min_calibration_error(breakdowns: dict) -> float:
    """Absolute final-two-minute calibration error from a tier's breakdowns."""
    for row in breakdowns.get("by_time_remaining", []):
        if row["bucket"] == "final_2min" and row.get("n", 0) > 0:
            return abs(row["calibration_error"])
    return 0.0


def compute_ablation_gates(metrics: dict) -> dict[str, bool]:
    """Encode the Phase-3 exit criteria as explicit booleans.

    Structural gates guard integrity (a failure is a real defect); scientific
    gates report the finding (a null is a valid result, not an error). The
    D-minus-C difference is negative when D scored lower (better), so
    "beats" means the difference CI's upper bound sits strictly below zero.
    """
    tiers = metrics["tiers"]
    predictions_ok = all(
        t["predictions_min"] > 0.0 and t["predictions_max"] < 1.0
        for t in tiers.values()
    )

    late = metrics["late_game_calibration"]
    late_game_ok = abs(late["D"]) <= abs(late["C"]) + LATE_GAME_CALIB_TOL

    provenance_ok = bool(metrics["rating_provenance_ok"])

    diff = metrics["paired_diff"]["D_minus_C"]
    beats = (diff["brier"]["hi"] < 0.0) or (diff["log_loss"]["hi"] < 0.0)

    fold_points = [f["d_minus_c_brier_point"] for f in metrics["rolling"]]
    reproduces = len(fold_points) > 0 and all(pt < 0.0 for pt in fold_points)

    return {
        "gate_predictions_in_open_interval": bool(predictions_ok),
        "gate_late_game_calibration_not_degraded": bool(late_game_ok),
        "gate_every_rating_strictly_prior": bool(provenance_ok),
        "gate_rapm_beats_team_strength": bool(beats),
        "gate_reproduces_rolling": bool(reproduces),
    }


def structural_gates_pass(metrics: dict) -> bool:
    """True iff every structural gate holds — the single source of truth for exit."""
    gates = compute_ablation_gates(metrics)
    return all(bool(gates[name]) for name in STRUCTURAL_GATE_NAMES)


def ablation_verdict(metrics: dict) -> dict:
    """The explicit 'did lineup strength help, and where' statement."""
    gates = metrics["gates"]
    helped = (
        gates["gate_rapm_beats_team_strength"]
        and gates["gate_reproduces_rolling"]
    )
    # Buckets where D's POINT estimate nudged below C. These carry no
    # per-subgroup CI, so when the overall D-vs-C finding is null they are noise
    # wiggles, not established subgroup wins — the key name says so explicitly to
    # keep the "and where" statement honest rather than overclaiming.
    point_favored: list[str] = []
    for group, rows in metrics["where_rapm_helped"].items():
        for row in rows:
            if row["brier_improvement"] > 0.0 or row["log_loss_improvement"] > 0.0:
                point_favored.append(f"{group}:{row['bucket']}")
    return {
        "lineup_strength_helped": bool(helped),
        "beats_team_strength_ci_excludes_zero": bool(
            gates["gate_rapm_beats_team_strength"]
        ),
        "reproduced_across_folds": bool(gates["gate_reproduces_rolling"]),
        "point_favored_d_subgroups_not_significance_tested": point_favored,
    }


# --------------------------------------------------------------------------
# Assembly: score all tiers out of sample on the test split.
# --------------------------------------------------------------------------

def evaluate_ablation(
    df: pd.DataFrame, grid: tuple[float, ...] = model.LAMBDA_GRID
) -> dict:
    """Fit and score the full A..E ladder out of sample; assemble the metrics.

    Prepares team strength, fits every tier leakage-safe, scores each on the test
    split, runs the paired game-clustered bootstrap over adjacent tiers, and
    computes the Brier decompositions, the where-RAPM-helped table, the rolling
    folds, and the gates + verdict. Pure with respect to `df`.
    """
    prepared = add_team_strength_columns(df)
    test = prepared.loc[prepared["split"] == TEST_SPLIT].reset_index(drop=True)
    if len(test) == 0:
        raise ValueError("no rows with split == 'test' to evaluate on")

    y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = test["game_id"].to_numpy()

    fits = {t: fit_tier(prepared, t, TIER_FEATURES[t], grid) for t in TIER_NAMES}
    preds = {t: predict_tier(fits[t], test) for t in TIER_NAMES}

    tiers: dict[str, dict] = {}
    breakdowns: dict[str, dict] = {}
    for t in TIER_NAMES:
        p = preds[t]
        breakdowns[t] = evaluate.phase_breakdowns(test, y, p)
        intercept, slope = evaluate.fit_calibration(y, p)
        tiers[t] = {
            "features": list(TIER_FEATURES[t]),
            "n_features": len(TIER_FEATURES[t]),
            "chosen_lambda": fits[t].chosen_lambda,
            "brier": evaluate.brier_score(y, p),
            "log_loss": evaluate.mean_log_loss(y, p),
            "brier_decomposition": brier_decomposition(y, p),
            "calibration": {"intercept": intercept, "slope": slope},
            "reliability_table": evaluate.reliability_table(y, p),
            "phase_breakdowns": breakdowns[t],
            "predictions_min": float(p.min()),
            "predictions_max": float(p.max()),
        }

    adjacent_pairs = [("B", "A"), ("C", "B"), ("D", "C"), ("E", "D")]
    paired = paired_diff_ci(y, preds, game_ids, adjacent_pairs)

    metrics: dict = {
        "n_test_rows": int(len(test)),
        "n_test_games": int(pd.Series(game_ids).nunique()),
        "home_win_rate_test": float(y.mean()),
        "tiers": tiers,
        "paired_diff": paired,
        "where_rapm_helped": subgroup_improvement(breakdowns["C"], breakdowns["D"]),
        "late_game_calibration": {
            "C": _final_2min_calibration_error(breakdowns["C"]),
            "D": _final_2min_calibration_error(breakdowns["D"]),
        },
        "rolling": rolling_fold_diffs(prepared, grid),
        "rating_provenance_ok": rating_provenance_ok(prepared),
        "bootstrap": {
            "n_boot": int(evaluate.N_BOOTSTRAP),
            "seed": int(evaluate.BOOTSTRAP_SEED),
            "alpha": float(evaluate.BOOTSTRAP_ALPHA),
        },
    }
    metrics["gates"] = compute_ablation_gates(metrics)
    metrics["verdict"] = ablation_verdict(metrics)
    return metrics


# --------------------------------------------------------------------------
# Serialization + provenance audit.
# --------------------------------------------------------------------------

def audit_payload(metrics: dict, parquet_path: Path) -> dict:
    """Provenance document reusing the canonical/file hashing pattern."""
    return {
        "metrics_hash": design.canonical_hash(metrics),
        "dataset_parquet_sha256": design.file_hash(parquet_path),
        "split_definition": design.SPLIT_DEFINITION,
        "split_hash": design.canonical_hash(design.SPLIT_DEFINITION),
        "n_test_rows": metrics["n_test_rows"],
        "n_test_games": metrics["n_test_games"],
        "tier_brier": {t: metrics["tiers"][t]["brier"] for t in TIER_NAMES},
        "tier_log_loss": {t: metrics["tiers"][t]["log_loss"] for t in TIER_NAMES},
        "paired_D_minus_C": metrics["paired_diff"]["D_minus_C"],
        "rolling": [
            {
                "fold": f["fold"],
                "test_seasons": f["test_seasons"],
                "d_minus_c_brier_point": f["d_minus_c_brier_point"],
                "d_minus_c_logloss_point": f["d_minus_c_logloss_point"],
            }
            for f in metrics["rolling"]
        ],
        "gates": metrics["gates"],
        "structural_gates_pass": structural_gates_pass(metrics),
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart, run the ablation, and write metrics + audit JSON."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run `python -m winprob.design` first"
        )
    df = pd.read_parquet(parquet_path)
    metrics = evaluate_ablation(df)
    audit = audit_payload(metrics, parquet_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    print(
        f"winprob ablation: {metrics['n_test_rows']:,} test rows / "
        f"{metrics['n_test_games']:,} games "
        f"(home win rate {metrics['home_win_rate_test']:.3f})"
    )
    print("\n  tier  features  lambda      Brier     log_loss   resolution")
    for t in TIER_NAMES:
        tier = metrics["tiers"][t]
        dec = tier["brier_decomposition"]
        print(
            f"  {t}     {tier['n_features']:>3}      {tier['chosen_lambda']:9.4g} "
            f"{tier['brier']:9.5f} {tier['log_loss']:10.5f} {dec['resolution']:11.5f}"
        )

    dc = metrics["paired_diff"]["D_minus_C"]
    print(
        f"\n  D - C paired  Brier {dc['brier']['point']:+.5f} "
        f"[{dc['brier']['lo']:+.5f}, {dc['brier']['hi']:+.5f}]   "
        f"log_loss {dc['log_loss']['point']:+.5f} "
        f"[{dc['log_loss']['lo']:+.5f}, {dc['log_loss']['hi']:+.5f}]  "
        f"(negative = D better)"
    )
    print("  rolling folds (D - C Brier):")
    for f in metrics["rolling"]:
        print(
            f"    fold {f['fold']} (test {f['test_seasons']}): "
            f"{f['d_minus_c_brier_point']:+.5f}"
        )

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        kind = "structural" if name in STRUCTURAL_GATE_NAMES else "scientific"
        print(f"    [{'PASS' if passed else 'FAIL'}] ({kind}) {name}")

    verdict = metrics["verdict"]
    helped = "YES" if verdict["lineup_strength_helped"] else "NO"
    print(f"\n  lineup strength helped out of sample: {helped}")
    point_favored = verdict["point_favored_d_subgroups_not_significance_tested"]
    if point_favored:
        print(
            "    D point-favored (within noise, not significance-tested) in: "
            + ", ".join(point_favored)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe RAPM lineup ablation and its gates"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not structural_gates_pass(metrics):
        raise SystemExit("structural gate failure: Phase-3 integrity check did not pass")


if __name__ == "__main__":
    main()
