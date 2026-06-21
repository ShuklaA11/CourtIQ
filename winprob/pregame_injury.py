"""Pre-game ladder tier P4: availability-adjusted strength + re-measured market gap.

Sprint 4's ladder (``winprob.pregame_ladder``) closed ~half the tier-E-to-market
Brier gap with current-season FORM. Tier P4 asks the next forward-chaining
question on the same stick: does telling the model *who is on the floor* — the
leakage-safe availability signal (``winprob.pregame_availability``) — earn its keep
OVER form out of sample, and how much MORE of the market gap does it close?

P4 strictly nests P3: it is P3 (home + prior strength + current-season form + rest)
plus the five availability columns (home/away available strength, home/away injury
hit, injury_hit_diff). It does NOT fork the fitter — it builds the P0..P3 ladder
frame with ``pregame_ladder.build_ladder_frame`` at the SAME validation-selected
``shrink_k`` Sprint 4 used, attaches availability with
``pregame_availability.add_availability_features``, and fits P3 and P4 through the
identical leakage-safe ``pregame_ladder.fit_ladder_tier`` (lambda on validation,
refit on train+validation, the 2025 test season scored once). The gap is
re-measured on the SAME covered join Sprint 4 pinned
(``pregame.covered_games_frame``), so P4's ``fraction_of_gap_closed`` rides the
identical frozen tier-E ``baseline_gap``.

Gates follow Sprint-4 philosophy: ``gate_availability_beats_form`` is the P4-vs-P3
held-out Brier paired-diff CI upper bound below zero; ``gate_pregame_calibrated``
checks P4 is itself calibrated; structural gates guard integrity (predictions in
(0, 1); test season untouched). Beating the market is NOT a gate — the fraction is
REPORTED with a game-clustered CI, and the verdict states the null plainly when
P4 ~= P3. Pure numpy/pandas; every function returns a new object.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import (
    ablation,
    availability,
    design,
    evaluate,
    model,
    pregame,
    pregame_availability,
    pregame_ladder,
)

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = pregame.PARQUET_NAME
ODDS_CSV_NAME = pregame.ODDS_CSV_NAME
AVAILABILITY_PARQUET_NAME = availability.AVAILABILITY_PARQUET_NAME
RATINGS_SUBPATH = Path("rapm") / "bayes_ratings.parquet"
METRICS_JSON_NAME = "injury_metrics.json"
AUDIT_JSON_NAME = "injury_audit.json"

TARGET_COLUMN = model.TARGET_COLUMN
TEST_SPLIT = pregame_ladder.TEST_SPLIT

# The availability columns P4 layers on top of P3, and the nested P4 contract.
AVAILABILITY_FEATURES: tuple[str, ...] = pregame_availability.AVAILABILITY_FEATURE_COLUMNS
TIER_P4_FEATURES: tuple[str, ...] = pregame_ladder.TIER_P3_FEATURES + AVAILABILITY_FEATURES

# The full P0..P4 ladder is emitted: P0..P3 are Sprint 4's forward-chaining tiers
# (reused wholesale from ``pregame_ladder`` so the two artifacts share one definition),
# and P4 layers availability on top of P3. The headline paired comparison stays P4
# vs P3 (does availability beat form?), but every rung is fit, scored, and reported.
INJURY_TIERS: tuple[str, ...] = pregame_ladder.LADDER_TIERS + ("P4",)
TIER_FEATURES: dict[str, tuple[str, ...]] = {
    **pregame_ladder.TIER_FEATURES,
    "P4": TIER_P4_FEATURES,
}
TIER_ADDS: dict[str, str] = {
    **pregame_ladder.TIER_ADDS,
    "P4": "prior-season availability (injury) adjustment",
}


# --- Game-grain P4 frame: the P0..P3 ladder frame plus availability columns.

def build_injury_ladder_frame(
    df: pd.DataFrame,
    availability: pd.DataFrame,
    ratings: pd.DataFrame,
    shrink_k: float,
) -> pd.DataFrame:
    """The P0..P3 ladder frame with the five availability columns attached.

    Delegates to ``pregame_ladder.build_ladder_frame`` for the identity, split,
    outcome, prior-strength, form, and rest columns at ``shrink_k``, then attaches
    the leakage-safe availability strengths and injury hits with
    ``pregame_availability.add_availability_features`` (rosters and weights read from
    the PRIOR season, so the whole frame is known before tip-off). Returns a NEW
    frame; the inputs are never mutated.
    """
    base = pregame_ladder.build_ladder_frame(df, shrink_k)
    return pregame_availability.add_availability_features(base, availability, ratings)


# --- Gap-close vs market on the covered games (reuses the ladder's primitives).

def _gap_close(covered: pd.DataFrame, preds_by_game: pd.Series, label: str) -> dict:
    """A tier's Brier vs market on the covered games: the gap, the gap closed, more.

    ``covered`` is ``pregame.covered_games_frame`` output (one row per covered test
    game with the outcome, the frozen tier-E probability ``p_model``, and the
    vig-free ``market_home_prob``); ``preds_by_game`` maps ``game_id`` to the tier's
    pre-game probability. ``baseline_gap`` is the tier-E-to-market Brier gap — the
    SAME frozen stick Sprint 4 recorded, since the covered join is identical — and
    ``fraction_of_gap_closed`` is how much of it this tier closes. Reuses the
    ladder's game-clustered ``_fraction_of_gap_closed``, ``_bootstrap_game_ci``, and
    ``orthogonality`` primitives, plus ``pregame.score_forecast``. Pure.
    """
    joined = covered.assign(_pred=covered["game_id"].map(preds_by_game))
    if joined["_pred"].isna().any():
        raise ValueError(f"covered game missing a {label} prediction (game_id mismatch)")

    y = joined[TARGET_COLUMN].to_numpy().astype(np.float64)
    p_baseline = joined["p_model"].to_numpy(dtype=np.float64)
    p_market = joined["market_home_prob"].to_numpy(dtype=np.float64)
    p = joined["_pred"].to_numpy(dtype=np.float64)
    game_ids = joined["game_id"].to_numpy()

    market_brier = evaluate.brier_score(y, p_market)
    baseline_brier = evaluate.brier_score(y, p_baseline)
    pred_brier = evaluate.brier_score(y, p)

    def _fraction(idx: np.ndarray) -> float:
        return pregame_ladder._fraction_of_gap_closed(
            y[idx], p_baseline[idx], p[idx], p_market[idx]
        )

    return {
        "n_games": len(joined),
        "label": label,
        "score": pregame.score_forecast(y, p, game_ids),
        "market_brier": market_brier,
        "baseline_model_brier": baseline_brier,
        f"{label}_brier": pred_brier,
        "baseline_gap": float(baseline_brier - market_brier),
        "new_gap": float(pred_brier - market_brier),
        "fraction_of_gap_closed": pregame_ladder._fraction_of_gap_closed(
            y, p_baseline, p, p_market
        ),
        "fraction_of_gap_closed_ci": pregame_ladder._bootstrap_game_ci(game_ids, _fraction),
        f"correlation_{label}_market": float(np.corrcoef(p, p_market)[0, 1]),
        "orthogonality": pregame_ladder.orthogonality(y, p, p_market, game_ids),
    }


# --- Gates + verdict.

def compute_injury_gates(metrics: dict) -> dict[str, bool]:
    """Encode the P4 exit criteria as explicit booleans (Sprint-4 philosophy).

    Structural gates guard integrity: every P3/P4 prediction strictly inside (0, 1),
    and the test season entered no fit or selection. ``gate_availability_beats_form``
    is the P4-vs-P3 held-out Brier paired-difference CI upper bound below zero
    (negative means P4 better). ``gate_pregame_calibrated`` checks the top tier P4 is
    itself calibrated. Beating the market is deliberately NOT a gate.
    """
    tiers = metrics["tiers"]
    predictions_ok = all(
        t["predictions_min"] > 0.0 and t["predictions_max"] < 1.0
        for t in tiers.values()
    )
    untouched_ok = all(
        set(t["splits_used"]) == {"train", "validation"} for t in tiers.values()
    )

    avail_diff = metrics["paired_diff"]["P4_minus_P3"]
    availability_beats = avail_diff["brier"]["hi"] < 0.0

    cal = tiers["P4"]["calibration"]
    calibrated = (
        abs(cal["intercept"]) < evaluate.CALIB_INTERCEPT_TOL
        and evaluate.CALIB_SLOPE_LO <= cal["slope"] <= evaluate.CALIB_SLOPE_HI
    )

    return {
        "gate_predictions_in_open_interval": bool(predictions_ok),
        "gate_test_season_untouched": bool(untouched_ok),
        "gate_availability_beats_form": bool(availability_beats),
        "gate_pregame_calibrated": bool(calibrated),
    }


def injury_verdict(metrics: dict) -> dict:
    """The honest 'did availability beat form, and how much extra gap closed' string."""
    gates = metrics["gates"]
    p4 = metrics["gap_close"]
    p3 = metrics["form_gap_close"]
    frac4 = p4["fraction_of_gap_closed"]
    frac3 = p3["fraction_of_gap_closed"]
    extra = metrics["extra_gap_closed"]
    ci = p4["fraction_of_gap_closed_ci"]
    diff = metrics["paired_diff"]["P4_minus_P3"]["brier"]
    beats = gates["gate_availability_beats_form"]
    ortho = p4["orthogonality"]["model_coefficient_ci"]
    adds_signal = ortho["lo"] > 0.0

    null_clause = (
        "" if beats
        else " — availability is statistically indistinguishable from form here, "
        "so the null (P4 ~= P3) stands"
    )
    summary = (
        f"availability {'beats' if beats else 'does NOT beat'} current-season form out "
        f"of sample (P4 - P3 held-out Brier {diff['point']:+.5f} "
        f"[{diff['lo']:+.5f}, {diff['hi']:+.5f}]{null_clause}); P4 closes {frac4:+.1%} of "
        f"the tier-E-to-market Brier gap versus form's {frac3:+.1%} ({extra:+.1%} extra, "
        f"CI [{ci['lo']:+.1%}, {ci['hi']:+.1%}]); P4 "
        f"{'adds' if adds_signal else 'does not add'} signal orthogonal to the market; "
        f"the model {'is' if gates['gate_pregame_calibrated'] else 'is NOT'} calibrated pre-game"
    )
    return {
        "availability_beats_form": bool(beats),
        "pregame_calibrated": bool(gates["gate_pregame_calibrated"]),
        "fraction_of_gap_closed": float(frac4),
        "form_fraction_of_gap_closed": float(frac3),
        "extra_gap_closed": float(extra),
        "p4_adds_orthogonal_signal": bool(adds_signal),
        "summary": summary,
    }


# --- Assembly: fit P3 + P4, score on test, re-measure the market gap.

def _score_tier(fit: pregame_ladder.LadderFit, test: pd.DataFrame, y: np.ndarray) -> dict:
    """One tier's out-of-sample summary in the ladder's fixed metric shape."""
    p = pregame_ladder.predict_ladder(fit, test)
    intercept, slope = evaluate.fit_calibration(y, p)
    return {
        "adds": TIER_ADDS[fit.name],
        "features": list(TIER_FEATURES[fit.name]),
        "n_features": len(TIER_FEATURES[fit.name]),
        "chosen_lambda": fit.chosen_lambda,
        "brier": evaluate.brier_score(y, p),
        "log_loss": evaluate.mean_log_loss(y, p),
        "calibration": {"intercept": intercept, "slope": slope},
        "predictions_min": float(p.min()),
        "predictions_max": float(p.max()),
        "splits_used": sorted(fit.splits_used),
        "holdout_rows_excluded": fit.holdout_rows_excluded,
    }


def evaluate_injury_ladder(
    df: pd.DataFrame,
    availability: pd.DataFrame,
    ratings: pd.DataFrame,
    covered: pd.DataFrame,
) -> dict:
    """Fit P3 + P4, score them on 2025, and re-measure P4's gap-close vs the market.

    Selects ``shrink_k`` on validation via ``pregame_ladder.select_shrink_k`` (the
    SAME value Sprint 4's P3 used, so the two tiers are built on an identical frame),
    fits P3 and P4 leakage-safe through the shared ``fit_ladder_tier``, scores each on
    the untouched test split, runs the paired game-clustered bootstrap over P4 vs P3,
    and measures both tiers' gap-close on the ``covered`` games. Pure with respect to
    every input.
    """
    selection = pregame_ladder.select_shrink_k(df)
    chosen_k = selection["chosen_k"]
    frame = build_injury_ladder_frame(df, availability, ratings, chosen_k)

    test = frame.loc[frame["split"] == TEST_SPLIT].reset_index(drop=True)
    if len(test) == 0:
        raise ValueError("no rows with split == 'test' to evaluate on")
    y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = test["game_id"].to_numpy()

    fits = {
        t: pregame_ladder.fit_ladder_tier(frame, t, TIER_FEATURES[t]) for t in INJURY_TIERS
    }
    preds = {t: pregame_ladder.predict_ladder(fits[t], test) for t in INJURY_TIERS}
    tiers = {t: _score_tier(fits[t], test, y) for t in INJURY_TIERS}

    paired = ablation.paired_diff_ci(y, preds, game_ids, [("P4", "P3")])

    index = pd.Index(game_ids, name="game_id")
    p4_gap = _gap_close(covered, pd.Series(preds["P4"], index=index), "p4")
    p3_gap = _gap_close(covered, pd.Series(preds["P3"], index=index), "p3")
    extra = p4_gap["fraction_of_gap_closed"] - p3_gap["fraction_of_gap_closed"]

    metrics: dict = {
        "n_test_games": int(pd.Series(game_ids).nunique()),
        "home_win_rate_test": float(y.mean()),
        "shrink_k_selection": selection,
        "chosen_k": chosen_k,
        "tier_p4_features": list(TIER_P4_FEATURES),
        "availability_features": list(AVAILABILITY_FEATURES),
        "tiers": tiers,
        "paired_diff": paired,
        "gap_close": p4_gap,
        "form_gap_close": p3_gap,
        "extra_gap_closed": float(extra),
        "bootstrap": {
            "n_boot": int(evaluate.N_BOOTSTRAP),
            "seed": int(evaluate.BOOTSTRAP_SEED),
            "alpha": float(evaluate.BOOTSTRAP_ALPHA),
        },
    }
    metrics["gates"] = compute_injury_gates(metrics)
    metrics["structural_gates_pass"] = pregame_ladder.structural_gates_pass(metrics)
    metrics["verdict"] = injury_verdict(metrics)
    return metrics


# --- Entry point + provenance audit.

def audit_payload(
    metrics: dict,
    parquet_path: Path,
    ratings_path: Path,
    availability_path: Path,
) -> dict:
    """Provenance document pinning the mart, ratings, and availability by sha256."""
    return {
        "metrics_hash": design.canonical_hash(metrics),
        "dataset_parquet_sha256": design.file_hash(parquet_path),
        "bayes_ratings_sha256": design.file_hash(ratings_path),
        "game_availability_sha256": design.file_hash(availability_path),
        "split_definition": design.SPLIT_DEFINITION,
        "split_hash": design.canonical_hash(design.SPLIT_DEFINITION),
        "chosen_k": metrics["chosen_k"],
        "n_test_games": metrics["n_test_games"],
        "tier_brier": {t: metrics["tiers"][t]["brier"] for t in metrics["tiers"]},
        "paired_P4_minus_P3": metrics["paired_diff"]["P4_minus_P3"],
        "fraction_of_gap_closed": metrics["gap_close"]["fraction_of_gap_closed"],
        "fraction_of_gap_closed_ci": metrics["gap_close"]["fraction_of_gap_closed_ci"],
        "extra_gap_closed": metrics["extra_gap_closed"],
        "gates": metrics["gates"],
        "structural_gates_pass": metrics["structural_gates_pass"],
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart, odds, availability, and ratings; run P4; write metrics + audit."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    odds_path = data_dir / ODDS_CSV_NAME
    availability_path = data_dir / AVAILABILITY_PARQUET_NAME
    ratings_path = data_dir.parent / RATINGS_SUBPATH

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run ./game_states.sh first"
        )
    if not odds_path.exists():
        raise FileNotFoundError(
            f"missing odds file at {odds_path}; see market.sh for how to fetch it"
        )
    if not availability_path.exists():
        raise FileNotFoundError(
            f"missing availability artifact at {availability_path}; run "
            "python -m winprob.availability first"
        )
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"missing prior-season ratings at {ratings_path}; run the RAPM pipeline first"
        )

    df = pd.read_parquet(parquet_path)
    availability = pd.read_parquet(availability_path)
    ratings = pd.read_parquet(ratings_path)
    covered = pregame.covered_games_frame(df, odds_path)

    metrics = evaluate_injury_ladder(df, availability, ratings, covered)
    audit = audit_payload(metrics, parquet_path, ratings_path, availability_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics
