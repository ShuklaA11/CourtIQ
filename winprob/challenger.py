"""Sprint-3 Phase-4 gradient-boosted challenger to the win-probability logistic.

The question Phase 4 settles: given the IDENTICAL leakage-safe features, does a
nonlinear model with free interactions beat the additive logistic out of sample?
The logistic is additive in {margin, margin/sqrt(time), time knots, possession,
team strength, lineup RAPM, coverage counts}; it hand-builds a little nonlinearity
but structurally cannot represent an interaction like ``margin x team_strength``
(a strong team down six behaves unlike a weak one in the same state). A shallow
boosted tree can. This module races the two, leakage-safe, on the untouched 2025
season and reports an honest verdict.

The comparison is deliberately apples-to-apples. The challenger is a histogram
Newton GBM (``winprob.gbm``) on the tier-E feature set MINUS the constant
intercept (the GBM's base score is its intercept). Its opponent is the tier-E
LOGISTIC — the exact Phase-3 tier-E fit (``winprob.ablation.fit_tier``) — so the
*only* difference between them is nonlinearity/interactions, not the feature set.
Racing the GBM against the sparse Phase-2 score+time logistic would conflate "more
features" with "nonlinearity"; that is not the question.

Everything downstream reuses the sprint's machinery rather than forking it:
``winprob.ablation`` for the leakage-safe tier-E design, team strength, rating
provenance, and the paired game-clustered bootstrap; ``winprob.evaluate`` for
Brier / log-loss / calibration / phase breakdowns / material-miscalibration;
``winprob.model`` for the working-frame leakage guard and game-clustered log-loss;
``winprob.design`` for canonical/file hashing and the pinned split definition.

*Leakage safety.* GBM hyperparameters (learning rate, depth, tree count) are
selected on validation only: each config fits on TRAIN and is scored by
game-clustered log-loss on VALIDATION, and the chosen config is refit on
train+validation combined — mirroring the logistic's lambda discipline exactly.
2025 never enters fitting, selection, or calibration.

*Honest nulls.* Structural gates (predictions in (0,1), holdout untouched, every
rating strictly prior) guard integrity and drive the exit code. The adoption
verdict (the GBM beats the logistic beyond game-clustered bootstrap noise AND
stays calibrated globally and by game phase) is REPORTED, not gated on exit: when
the GBM does not clearly beat the calibrated logistic, the correct, publishable
result is "retain the logistic" — exit 0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import ablation, design, evaluate, gbm, model
from winprob.ablation import add_team_strength_columns  # re-exported for callers/tests

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
METRICS_JSON_NAME = "challenger_metrics.json"
AUDIT_JSON_NAME = "challenger_audit.json"

TEST_SPLIT = "test"
TARGET_COLUMN = model.TARGET_COLUMN

# The challenger's features: tier E without the constant intercept (subsumed by
# the GBM's base score). Same information as the tier-E logistic it races.
GBM_FEATURES: tuple[str, ...] = tuple(
    name for name in ablation.TIER_E_FEATURES if name != "intercept"
)

STRUCTURAL_GATE_NAMES: tuple[str, ...] = (
    "gate_predictions_in_open_interval",
    "gate_holdout_untouched",
    "gate_every_rating_strictly_prior",
)
SCIENTIFIC_GATE_NAMES: tuple[str, ...] = (
    "gate_gbm_beats_logistic",
    "gate_gbm_calibrated_global",
    "gate_gbm_no_material_phase_miscalibration",
)


# --------------------------------------------------------------------------
# Hyperparameter selection grid.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionGrid:
    """The small, disciplined GBM search; tests pass a tiny override."""

    learning_rates: tuple[float, ...] = (0.05, 0.1)
    max_depths: tuple[int, ...] = (2, 3)
    n_trees_cap: int = 300
    n_bins: int = 64
    leaf_l2: float = 1.0
    min_samples_leaf: int = 200


DEFAULT_GRID = SelectionGrid()


@dataclass(frozen=True)
class SelectionResult:
    """The validation-selected GBM configuration and its selection record."""

    learning_rate: float
    max_depth: int
    n_trees: int
    validation_log_loss: float
    records: tuple[dict, ...]
    splits_used: frozenset[str]
    holdout_rows_excluded: int


# --------------------------------------------------------------------------
# Feature assembly (raw — trees are invariant to monotone per-feature scaling).
# --------------------------------------------------------------------------

def gbm_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw (unstandardized) tier-E-minus-intercept design for the GBM."""
    return ablation.assemble_tier_matrix(df, GBM_FEATURES)


# --------------------------------------------------------------------------
# Leakage-safe hyperparameter selection: fit on train, score on validation.
# --------------------------------------------------------------------------

def select_gbm_config(
    prepared: pd.DataFrame, grid: SelectionGrid = DEFAULT_GRID
) -> SelectionResult:
    """Pick (learning_rate, max_depth, n_trees) by validation game-clustered log-loss.

    For each (learning_rate, max_depth) the GBM fits on TRAIN rows with the
    validation set monitored per round; the round count is taken at the validation
    argmin (early stopping), and the config with the lowest validation
    game-clustered log-loss wins. Reads only train+validation via the shared
    leakage guard — 2025/2021 are dropped before any matrix is built.
    """
    work = model.working_frame(prepared)
    frame = work.frame
    X = gbm_matrix(frame)
    y = frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = frame["game_id"].to_numpy()
    is_train = (frame["split"] == "train").to_numpy()
    is_val = (frame["split"] == "validation").to_numpy()

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val, games_val = X[is_val], y[is_val], game_ids[is_val]

    def val_game_log_loss(yv: np.ndarray, pv: np.ndarray) -> float:
        return model.game_clustered_log_loss(yv, pv, games_val)

    monitor = gbm.MonitorSet(X_val, y_val, val_game_log_loss)

    records: list[dict] = []
    best: tuple[float, float, int, float] | None = None
    for lr in grid.learning_rates:
        for depth in grid.max_depths:
            cfg = gbm.GBMConfig(
                learning_rate=lr,
                max_depth=depth,
                n_trees=grid.n_trees_cap,
                n_bins=grid.n_bins,
                leaf_l2=grid.leaf_l2,
                min_samples_leaf=grid.min_samples_leaf,
            )
            out = gbm.fit_gbm(X_train, y_train, cfg, monitor=monitor)
            best_round = int(np.argmin(out.monitor_scores))
            best_ll = float(out.monitor_scores[best_round])
            records.append(
                {
                    "learning_rate": lr,
                    "max_depth": depth,
                    "best_n_trees": best_round + 1,
                    "validation_log_loss": best_ll,
                }
            )
            if best is None or best_ll < best[3]:
                best = (lr, depth, best_round + 1, best_ll)

    assert best is not None  # grid is always non-empty (dataclass defaults)
    return SelectionResult(
        learning_rate=best[0],
        max_depth=best[1],
        n_trees=best[2],
        validation_log_loss=best[3],
        records=tuple(records),
        splits_used=work.splits_used,
        holdout_rows_excluded=work.holdout_rows_excluded,
    )


def fit_final_gbm(
    prepared: pd.DataFrame, selection: SelectionResult, grid: SelectionGrid
) -> gbm.BoostedModel:
    """Refit the selected GBM config on train+validation combined, leakage-safe."""
    work = model.working_frame(prepared)
    X = gbm_matrix(work.frame)
    y = work.frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    cfg = gbm.GBMConfig(
        learning_rate=selection.learning_rate,
        max_depth=selection.max_depth,
        n_trees=selection.n_trees,
        n_bins=grid.n_bins,
        leaf_l2=grid.leaf_l2,
        min_samples_leaf=grid.min_samples_leaf,
    )
    return gbm.fit_gbm(X, y, cfg).model


# --------------------------------------------------------------------------
# Gates + verdict.
# --------------------------------------------------------------------------

def compute_challenger_gates(metrics: dict) -> dict[str, bool]:
    """Encode the Phase-4 exit criteria as explicit booleans.

    Structural gates guard integrity (a failure is a real defect); scientific
    gates report the adoption finding (a null is a valid result). The
    gbm-minus-logistic difference is negative when the GBM scored lower (better),
    so "beats" means the difference CI's upper bound sits strictly below zero on
    Brier or log loss.
    """
    gbm_m = metrics["gbm"]
    log_m = metrics["logistic"]
    predictions_ok = (
        gbm_m["predictions_min"] > 0.0
        and gbm_m["predictions_max"] < 1.0
        and log_m["predictions_min"] > 0.0
        and log_m["predictions_max"] < 1.0
    )

    splits = set(metrics["splits_used_for_fit"])
    holdout_untouched = splits.issubset(model.ALLOWED_FIT_SPLITS)

    provenance_ok = bool(metrics["rating_provenance_ok"])

    diff = metrics["paired_diff"]["gbm_minus_logistic"]
    beats = (diff["brier"]["hi"] < 0.0) or (diff["log_loss"]["hi"] < 0.0)

    cal = gbm_m["calibration"]
    calibrated = (
        abs(cal["intercept"]) < evaluate.CALIB_INTERCEPT_TOL
        and evaluate.CALIB_SLOPE_LO <= cal["slope"] <= evaluate.CALIB_SLOPE_HI
    )

    no_miscal = len(metrics["material_phase_miscalibration"]) == 0

    return {
        "gate_predictions_in_open_interval": bool(predictions_ok),
        "gate_holdout_untouched": bool(holdout_untouched),
        "gate_every_rating_strictly_prior": bool(provenance_ok),
        "gate_gbm_beats_logistic": bool(beats),
        "gate_gbm_calibrated_global": bool(calibrated),
        "gate_gbm_no_material_phase_miscalibration": bool(no_miscal),
    }


def structural_gates_pass(metrics: dict) -> bool:
    """True iff every structural gate holds — the single source of truth for exit."""
    gates = metrics.get("gates") or compute_challenger_gates(metrics)
    return all(bool(gates[name]) for name in STRUCTURAL_GATE_NAMES)


def challenger_verdict(metrics: dict) -> dict:
    """The explicit adopt-GBM-or-retain-logistic decision.

    The GBM is adopted ONLY if it beats the logistic beyond game-clustered
    bootstrap noise AND remains calibrated globally and by game phase. Anything
    short of that — including a well-calibrated GBM whose edge is within noise —
    means retain the logistic, which is a valid Phase-4 result, not a failure.
    """
    gates = metrics["gates"]
    beats = gates["gate_gbm_beats_logistic"]
    calibrated = (
        gates["gate_gbm_calibrated_global"]
        and gates["gate_gbm_no_material_phase_miscalibration"]
    )
    adopt = bool(beats and calibrated)
    if adopt:
        reason = "GBM beats the logistic out of sample and stays calibrated"
    elif not beats:
        reason = "GBM does not clearly beat the logistic (difference CI includes 0)"
    else:
        reason = "GBM beats on point estimate but is not adequately calibrated"
    return {
        "adopt_gbm": adopt,
        "retain_logistic": not adopt,
        "beats_logistic_ci_excludes_zero": bool(beats),
        "gbm_calibrated": bool(calibrated),
        "reason": reason,
    }


# --------------------------------------------------------------------------
# Assembly: fit both models, score out of sample on the test split.
# --------------------------------------------------------------------------

def evaluate_challenger(
    df: pd.DataFrame, grid: SelectionGrid = DEFAULT_GRID
) -> dict:
    """Fit the tier-E logistic and the GBM leakage-safe, race them on 2025.

    Prepares team strength, selects + refits the GBM, refits the tier-E logistic
    (reusing the Phase-3 machinery), scores both on the untouched test split, runs
    the paired game-clustered bootstrap on the gbm-minus-logistic difference, and
    computes the calibration / phase breakdowns / gates / verdict. Pure w.r.t.
    ``df``.
    """
    prepared = add_team_strength_columns(df)
    test = prepared.loc[prepared["split"] == TEST_SPLIT].reset_index(drop=True)
    if len(test) == 0:
        raise ValueError("no rows with split == 'test' to evaluate on")

    y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = test["game_id"].to_numpy()

    # GBM: select on validation, refit on train+validation, score on test.
    selection = select_gbm_config(prepared, grid)
    gbm_model = fit_final_gbm(prepared, selection, grid)
    p_gbm = gbm.predict_proba(gbm_model, gbm_matrix(test))

    # Logistic opponent: the exact Phase-3 tier-E fit.
    fit_e = ablation.fit_tier(prepared, "E", ablation.TIER_E_FEATURES, model.LAMBDA_GRID)
    p_log = ablation.predict_tier(fit_e, test)

    preds = {"gbm": p_gbm, "logistic": p_log}
    paired = ablation.paired_diff_ci(y, preds, game_ids, [("gbm", "logistic")])

    gbm_intercept, gbm_slope = evaluate.fit_calibration(y, p_gbm)
    log_intercept, log_slope = evaluate.fit_calibration(y, p_log)
    gbm_breakdowns = evaluate.phase_breakdowns(test, y, p_gbm)

    metrics: dict = {
        "n_test_rows": int(len(test)),
        "n_test_games": int(pd.Series(game_ids).nunique()),
        "home_win_rate_test": float(y.mean()),
        "gbm": {
            "features": list(GBM_FEATURES),
            "n_features": len(GBM_FEATURES),
            "config": {
                "learning_rate": selection.learning_rate,
                "max_depth": selection.max_depth,
                "n_trees": selection.n_trees,
                "n_bins": grid.n_bins,
                "leaf_l2": grid.leaf_l2,
                "min_samples_leaf": grid.min_samples_leaf,
            },
            "selection": {
                "validation_log_loss": selection.validation_log_loss,
                "records": list(selection.records),
                "metric": "game_clustered_log_loss",
                "selection_split": "validation",
                "fit_split": "train",
                "refit_split": "train+validation",
            },
            "brier": evaluate.brier_score(y, p_gbm),
            "log_loss": evaluate.mean_log_loss(y, p_gbm),
            "brier_decomposition": ablation.brier_decomposition(y, p_gbm),
            "calibration": {"intercept": gbm_intercept, "slope": gbm_slope},
            "reliability_table": evaluate.reliability_table(y, p_gbm),
            "phase_breakdowns": gbm_breakdowns,
            "predictions_min": float(p_gbm.min()),
            "predictions_max": float(p_gbm.max()),
        },
        "logistic": {
            "features": list(ablation.TIER_E_FEATURES),
            "n_features": len(ablation.TIER_E_FEATURES),
            "chosen_lambda": fit_e.chosen_lambda,
            "brier": evaluate.brier_score(y, p_log),
            "log_loss": evaluate.mean_log_loss(y, p_log),
            "calibration": {"intercept": log_intercept, "slope": log_slope},
            "predictions_min": float(p_log.min()),
            "predictions_max": float(p_log.max()),
        },
        "paired_diff": paired,
        "material_phase_miscalibration": evaluate.material_miscalibration(gbm_breakdowns),
        "rating_provenance_ok": ablation.rating_provenance_ok(prepared),
        "holdout_rows_excluded": selection.holdout_rows_excluded,
        "splits_used_for_fit": sorted(selection.splits_used),
        "bootstrap": {
            "n_boot": int(evaluate.N_BOOTSTRAP),
            "seed": int(evaluate.BOOTSTRAP_SEED),
            "alpha": float(evaluate.BOOTSTRAP_ALPHA),
        },
    }
    metrics["gates"] = compute_challenger_gates(metrics)
    metrics["verdict"] = challenger_verdict(metrics)
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
        "gbm_config": metrics["gbm"]["config"],
        "gbm_brier": metrics["gbm"]["brier"],
        "gbm_log_loss": metrics["gbm"]["log_loss"],
        "logistic_brier": metrics["logistic"]["brier"],
        "logistic_log_loss": metrics["logistic"]["log_loss"],
        "paired_gbm_minus_logistic": metrics["paired_diff"]["gbm_minus_logistic"],
        "splits_used_for_fit": metrics["splits_used_for_fit"],
        "holdout_rows_excluded": metrics["holdout_rows_excluded"],
        "gates": metrics["gates"],
        "structural_gates_pass": structural_gates_pass(metrics),
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR, grid: SelectionGrid = DEFAULT_GRID) -> dict:
    """Load the mart, run the challenger, and write metrics + audit JSON."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run `./game_states.sh` first"
        )
    df = pd.read_parquet(parquet_path)
    metrics = evaluate_challenger(df, grid)
    audit = audit_payload(metrics, parquet_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    g, lg = metrics["gbm"], metrics["logistic"]
    print(
        f"winprob challenger: {metrics['n_test_rows']:,} test rows / "
        f"{metrics['n_test_games']:,} games "
        f"(home win rate {metrics['home_win_rate_test']:.3f})"
    )
    cfg = g["config"]
    print(
        f"\n  GBM config: lr={cfg['learning_rate']}, depth={cfg['max_depth']}, "
        f"n_trees={cfg['n_trees']} (validation log-loss "
        f"{g['selection']['validation_log_loss']:.6f})"
    )
    print("\n  model      Brier      log_loss   calib(intercept/slope)")
    print(
        f"  logistic  {lg['brier']:9.5f} {lg['log_loss']:10.5f}   "
        f"{lg['calibration']['intercept']:+.3f} / {lg['calibration']['slope']:.3f}"
    )
    print(
        f"  GBM       {g['brier']:9.5f} {g['log_loss']:10.5f}   "
        f"{g['calibration']['intercept']:+.3f} / {g['calibration']['slope']:.3f}"
    )

    diff = metrics["paired_diff"]["gbm_minus_logistic"]
    print(
        f"\n  GBM - logistic paired  Brier {diff['brier']['point']:+.5f} "
        f"[{diff['brier']['lo']:+.5f}, {diff['brier']['hi']:+.5f}]   "
        f"log_loss {diff['log_loss']['point']:+.5f} "
        f"[{diff['log_loss']['lo']:+.5f}, {diff['log_loss']['hi']:+.5f}]  "
        f"(negative = GBM better)"
    )

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        kind = "structural" if name in STRUCTURAL_GATE_NAMES else "scientific"
        print(f"    [{'PASS' if passed else 'FAIL'}] ({kind}) {name}")

    verdict = metrics["verdict"]
    decision = "ADOPT GBM" if verdict["adopt_gbm"] else "RETAIN LOGISTIC"
    print(f"\n  verdict: {decision} — {verdict['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase-4 gradient-boosted challenger and its gates"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not structural_gates_pass(metrics):
        raise SystemExit("structural gate failure: Phase-4 integrity check did not pass")


if __name__ == "__main__":
    main()
