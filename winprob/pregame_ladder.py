"""Sprint-4 pre-game ABLATION LADDER + gap-close-vs-market measurement.

The Sprint-4 harness (``winprob.pregame``) fixes the measuring stick — the tier-E
opening-state logistic versus the vig-free market line on the covered games. This
module asks the forward-chaining question that stick was built for: as you feed a
GAME-GRAIN pre-game model more information, how much of the model-to-market gap
closes, and does each new signal earn its keep out of sample on the untouched 2025
test season?

The ladder is a nested L2-logistic at the GAME grain (one pre-tip row per game),
each tier strictly nesting in the next:

    P0  intercept only                    (the base home-win rate / home court)
    P1  + prior-season team strength      (approximates today's model pre-game)
    P2  + current-season form             (``add_current_season_form``, EB-shrunk)
    P3  + rest / schedule                 (``add_rest_features``)

It reuses the Sprint-1..3 machinery wholesale rather than forking a pipeline:
``winprob.features`` for the mean/std standardization, ``winprob.model`` for the
guarded intercept-free L2 fit and the leakage-guarded working frame,
``winprob.evaluate`` for Brier / log-loss / calibration and the game-clustered
bootstrap primitives, ``winprob.ablation`` for the paired adjacent-tier difference
CI, and ``winprob.pregame`` for the pre-game table and the covered-games join.

*Selection is leakage-safe end to end.* Each tier picks its L2 ``lambda`` by
game-clustered VALIDATION log-loss (never test). The Empirical-Bayes shrinkage
``SHRINK_K`` that the form features depend on is ALSO chosen on validation only —
via the P2 tier, the tier that introduces form — and the single chosen value is
pinned in the output. The 2025 test season enters no fit and no selection; it is
scored exactly once at the end.

*Honest nulls.* ``gate_form_beats_prior_strength`` reports whether P2's held-out
Brier improvement over P1 clears game-clustered bootstrap noise;
``gate_pregame_calibrated`` checks P3 is itself calibrated. Beating the market is
NOT a gate — ``fraction_of_gap_closed`` is REPORTED with a game-clustered CI, so a
"the market is still sharper" result is a valid finding, not a failure.

Pure numpy/pandas; every function returns a new object and never mutates its
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from winprob import ablation, evaluate, features, model, pregame, pregame_features

TARGET_COLUMN = model.TARGET_COLUMN

# --------------------------------------------------------------------------
# Ladder feature contracts. Each tier strictly nests in the next; the added
# column is named in ``TIER_ADDS`` for the report.
# --------------------------------------------------------------------------

LADDER_TIERS: tuple[str, ...] = ("P0", "P1", "P2", "P3")

# Derived pre-game columns the ladder frame carries (built by ``build_ladder_frame``).
PRIOR_STRENGTH_DIFF = "prior_strength_diff"
FORM_STRENGTH_DIFF = "form_strength_diff"      # from add_current_season_form
REST_DIFF = "rest_diff"                        # from add_rest_features
HOME_B2B = "home_back_to_back"
AWAY_B2B = "away_back_to_back"

TIER_P0_FEATURES: tuple[str, ...] = ("intercept",)
TIER_P1_FEATURES: tuple[str, ...] = TIER_P0_FEATURES + (PRIOR_STRENGTH_DIFF,)
TIER_P2_FEATURES: tuple[str, ...] = TIER_P1_FEATURES + (FORM_STRENGTH_DIFF,)
TIER_P3_FEATURES: tuple[str, ...] = TIER_P2_FEATURES + (REST_DIFF, HOME_B2B, AWAY_B2B)

TIER_FEATURES: dict[str, tuple[str, ...]] = {
    "P0": TIER_P0_FEATURES,
    "P1": TIER_P1_FEATURES,
    "P2": TIER_P2_FEATURES,
    "P3": TIER_P3_FEATURES,
}
TIER_ADDS: dict[str, str] = {
    "P0": "intercept + home court",
    "P1": "prior-season team strength",
    "P2": "current-season form (EB-shrunk)",
    "P3": "rest / schedule",
}

# Continuous columns are the only ones mean/std standardized; the intercept and the
# back-to-back indicators pass through the scaling as an identity.
LADDER_CONTINUOUS_FEATURES: frozenset[str] = frozenset(
    {PRIOR_STRENGTH_DIFF, FORM_STRENGTH_DIFF, REST_DIFF}
)

# Empirical-Bayes shrinkage candidates for the current-season form blend, chosen
# on VALIDATION only via the P2 tier. The number of games at which the current
# season and the prior receive equal weight.
SHRINK_K_GRID: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

TEST_SPLIT = "test"

STRUCTURAL_GATE_NAMES: tuple[str, ...] = (
    "gate_predictions_in_open_interval",
    "gate_test_season_untouched",
)


# --------------------------------------------------------------------------
# Game-grain ladder frame: identity + outcome + pre-game feature columns.
# --------------------------------------------------------------------------

def build_ladder_frame(df: pd.DataFrame, shrink_k: float) -> pd.DataFrame:
    """One pre-tip row per game carrying every ladder feature at a given ``shrink_k``.

    Reuses ``pregame.build_pregame_table`` (identity, split, outcome, prior-season
    strengths), then attaches leakage-safe current-season form
    (``add_current_season_form``) and rest/schedule (``add_rest_features``) columns
    computed from the same-season history in ``season_game_results``. The
    prior-season and form signals are reduced to home-minus-away DIFFERENTIALS so
    the ladder nests cleanly (P1 adds one prior column, P2 one form column). Returns
    a NEW frame; the input is never mutated.
    """
    table = pregame.build_pregame_table(df)
    history = pregame_features.season_game_results(df)
    with_form = pregame_features.add_current_season_form(table, history, shrink_k)
    out = pregame_features.add_rest_features(with_form, history)
    out = out.copy()
    out[PRIOR_STRENGTH_DIFF] = (
        out[pregame_features.HOME_PRIOR_STRENGTH]
        - out[pregame_features.AWAY_PRIOR_STRENGTH]
    )
    return out


def _raw_ladder_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw (pre-standardization) ladder columns keyed by feature name."""
    n = len(frame)
    cols: dict[str, np.ndarray] = {"intercept": features.constant_column(n)}
    for name in (PRIOR_STRENGTH_DIFF, FORM_STRENGTH_DIFF, REST_DIFF, HOME_B2B, AWAY_B2B):
        if name in frame.columns:
            cols[name] = np.asarray(frame[name].to_numpy(), dtype=np.float64)
    return cols


def _tier_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    """Raw design matrix with columns in ``feature_names`` order for one tier."""
    raw = _raw_ladder_columns(frame)
    columns: list[np.ndarray] = []
    for name in feature_names:
        if name not in raw:
            raise ValueError(f"unknown ladder feature: {name!r}")
        columns.append(raw[name])
    return np.column_stack(columns)


def _continuous_mask(feature_names: tuple[str, ...]) -> np.ndarray:
    """Boolean mask marking which named columns are standardized continuous ones."""
    return np.array(
        [name in LADDER_CONTINUOUS_FEATURES for name in feature_names], dtype=bool
    )


# --------------------------------------------------------------------------
# Per-tier fit: lambda selection on validation, refit on train+validation.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderFit:
    """Everything needed to apply a fitted ladder tier to a fresh (e.g. test) frame."""

    name: str
    feature_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    beta: np.ndarray
    chosen_lambda: float
    validation_log_loss: list[float]
    chosen_index: int
    splits_used: frozenset[str]
    holdout_rows_excluded: int


def fit_ladder_tier(
    frame: pd.DataFrame,
    name: str,
    feature_names: tuple[str, ...],
    grid: tuple[float, ...] = model.LAMBDA_GRID,
) -> LadderFit:
    """Fit one ladder tier leakage-safe: select lambda on validation, refit on train+val.

    Mirrors ``ablation.fit_tier`` at the GAME grain. Drops holdout rows via the
    shared ``model.working_frame`` guard, standardizes the tier's continuous columns
    on the train+validation frame, fits each grid lambda on TRAIN rows, scores
    game-clustered log-loss on VALIDATION rows, and refits at the grid argmin on
    train+validation combined. The 2025 test season never enters here.
    """
    if len(grid) == 0:
        raise ValueError("lambda grid must be non-empty")

    work = model.working_frame(frame)
    fit_frame = work.frame
    raw = _tier_matrix(fit_frame, feature_names)
    mask = _continuous_mask(feature_names)
    X, means, stds = features.standardize_columns(raw, mask)
    pen = model.penalty_mask(list(feature_names))

    y = fit_frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = fit_frame["game_id"].to_numpy()
    is_train = (fit_frame["split"] == "train").to_numpy()
    is_val = (fit_frame["split"] == "validation").to_numpy()

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

    return LadderFit(
        name=name,
        feature_names=tuple(feature_names),
        means=np.asarray(means, dtype=np.float64),
        stds=np.asarray(stds, dtype=np.float64),
        beta=beta_final,
        chosen_lambda=chosen_lambda,
        validation_log_loss=validation_log_loss,
        chosen_index=chosen_index,
        splits_used=work.splits_used,
        holdout_rows_excluded=work.holdout_rows_excluded,
    )


def predict_ladder(fit: LadderFit, frame: pd.DataFrame) -> np.ndarray:
    """Guarded win probabilities for ``frame`` under a fitted tier's saved scaling."""
    raw = _tier_matrix(frame, fit.feature_names)
    X = (raw - fit.means) / fit.stds
    return model.predict_proba(X, fit.beta)


# --------------------------------------------------------------------------
# SHRINK_K selection — validation only, via the P2 (form-introducing) tier.
# --------------------------------------------------------------------------

def select_shrink_k(
    df: pd.DataFrame,
    grid: tuple[float, ...] = SHRINK_K_GRID,
    lambda_grid: tuple[float, ...] = model.LAMBDA_GRID,
) -> dict:
    """Choose the Empirical-Bayes ``shrink_k`` on VALIDATION log-loss, never on test.

    For each candidate ``k``, builds the ladder frame at ``k`` and fits the P2 tier
    (the tier that introduces the form signal). P2's own lambda is selected on
    validation, so the tier's best validation log-loss is
    ``validation_log_loss[chosen_index]``; the ``k`` minimizing that is chosen. The
    test season is dropped by ``working_frame`` before any of this runs, so
    selection is leakage-safe. Returns the chosen ``k`` and the per-``k`` scores.
    """
    if len(grid) == 0:
        raise ValueError("shrink_k grid must be non-empty")

    scores: list[dict] = []
    for k in grid:
        frame = build_ladder_frame(df, float(k))
        fit = fit_ladder_tier(frame, "P2", TIER_P2_FEATURES, lambda_grid)
        scores.append(
            {
                "shrink_k": float(k),
                "validation_log_loss": float(fit.validation_log_loss[fit.chosen_index]),
                "chosen_lambda": fit.chosen_lambda,
            }
        )

    best = min(range(len(scores)), key=lambda i: scores[i]["validation_log_loss"])
    return {
        "chosen_k": scores[best]["shrink_k"],
        "grid": [float(k) for k in grid],
        "by_k": scores,
    }


# --------------------------------------------------------------------------
# Gap-close vs market + orthogonality on the covered games.
# --------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    """Guarded logit ``log(p / (1 - p))`` clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=np.float64), model.PROB_EPS, 1.0 - model.PROB_EPS)
    return np.log(p / (1.0 - p))


def _bootstrap_game_ci(
    game_ids: np.ndarray,
    statistic,
    n_boot: int = evaluate.N_BOOTSTRAP,
    seed: int = evaluate.BOOTSTRAP_SEED,
    alpha: float = evaluate.BOOTSTRAP_ALPHA,
) -> dict[str, float]:
    """Percentile CI of a scalar ``statistic(idx)`` under a game-clustered resample.

    Draws whole games with replacement each iteration (via the ``evaluate``
    primitives) and evaluates ``statistic`` on the resampled row indices. Non-finite
    draws (e.g. a degenerate denominator) are dropped before the percentile so one
    unlucky resample cannot poison the interval.
    """
    groups = evaluate.build_game_groups(game_ids)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = evaluate.draw_game_rows(groups, rng)
        val = statistic(idx)
        if np.isfinite(val):
            vals.append(float(val))
    return evaluate._percentile_ci(vals, alpha)


def _fraction_of_gap_closed(
    y: np.ndarray, p_baseline: np.ndarray, p_new: np.ndarray, p_market: np.ndarray
) -> float:
    """``(baseline_gap - new_gap) / baseline_gap`` in Brier terms; nan if degenerate.

    ``baseline_gap`` is the frozen tier-E model's Brier minus the market's;
    ``new_gap`` is P3's minus the market's. A positive fraction means P3 closed
    that share of the model-to-market gap.
    """
    baseline_gap = evaluate.brier_score(y, p_baseline) - evaluate.brier_score(y, p_market)
    new_gap = evaluate.brier_score(y, p_new) - evaluate.brier_score(y, p_market)
    if abs(baseline_gap) < model.PROB_EPS:
        return float("nan")
    return (baseline_gap - new_gap) / baseline_gap


def orthogonality(
    y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray, game_ids: np.ndarray
) -> dict:
    """Does the model add signal ORTHOGONAL to the market?

    Fits the unpenalized logistic ``outcome ~ intercept + market_logit +
    model_logit`` and reports the MODEL-logit coefficient with a game-clustered
    bootstrap CI (plus the market coefficient for context). A model coefficient
    whose CI excludes zero means the model carries pre-game information the market
    line does not already price.
    """
    y = np.asarray(y, dtype=np.float64)
    market_logit = _logit(p_market)
    model_logit = _logit(p_model)
    design = np.column_stack([np.ones_like(y), market_logit, model_logit])
    pen = np.zeros(3)
    beta = model.fit_l2_logistic(design, y, lam=0.0, pen_mask=pen)

    def _model_coef(idx: np.ndarray) -> float:
        b = model.fit_l2_logistic(design[idx], y[idx], lam=0.0, pen_mask=pen)
        return float(b[2])

    return {
        "market_coefficient": float(beta[1]),
        "model_coefficient": float(beta[2]),
        "model_coefficient_ci": _bootstrap_game_ci(game_ids, _model_coef),
    }


def gap_close_vs_market(
    covered: pd.DataFrame, p3_by_game: pd.Series
) -> dict:
    """P3 vs market on the covered games: the new gap, the gap closed, correlation.

    ``covered`` is ``pregame.covered_games_frame`` output (one row per covered test
    game with the outcome, the frozen tier-E model probability ``p_model``, and the
    vig-free ``market_home_prob``). ``p3_by_game`` maps ``game_id`` to the P3
    pre-game probability. Reports P3's Brier/log-loss/calibration, the
    baseline/new Brier gaps to market, ``fraction_of_gap_closed`` with a
    game-clustered CI, the P3<->market Pearson correlation, and the orthogonality
    check.
    """
    joined = covered.assign(p3=covered["game_id"].map(p3_by_game))
    if joined["p3"].isna().any():
        raise ValueError("covered game missing a P3 prediction (game_id mismatch)")

    y = joined[TARGET_COLUMN].to_numpy().astype(np.float64)
    p_baseline = joined["p_model"].to_numpy(dtype=np.float64)
    p_market = joined["market_home_prob"].to_numpy(dtype=np.float64)
    p3 = joined["p3"].to_numpy(dtype=np.float64)
    game_ids = joined["game_id"].to_numpy()

    baseline_brier = evaluate.brier_score(y, p_baseline)
    market_brier = evaluate.brier_score(y, p_market)
    p3_brier = evaluate.brier_score(y, p3)
    baseline_gap = baseline_brier - market_brier
    new_gap = p3_brier - market_brier

    def _fraction(idx: np.ndarray) -> float:
        return _fraction_of_gap_closed(y[idx], p_baseline[idx], p3[idx], p_market[idx])

    return {
        "n_games": int(len(joined)),
        "p3": pregame.score_forecast(y, p3, game_ids),
        "market_brier": market_brier,
        "baseline_model_brier": baseline_brier,
        "p3_brier": p3_brier,
        "baseline_gap": float(baseline_gap),
        "new_gap": float(new_gap),
        "fraction_of_gap_closed": _fraction_of_gap_closed(y, p_baseline, p3, p_market),
        "fraction_of_gap_closed_ci": _bootstrap_game_ci(game_ids, _fraction),
        "correlation_p3_market": float(np.corrcoef(p3, p_market)[0, 1]),
        "orthogonality": orthogonality(y, p3, p_market, game_ids),
    }


# --------------------------------------------------------------------------
# Gates + verdict.
# --------------------------------------------------------------------------

def compute_ladder_gates(metrics: dict) -> dict[str, bool]:
    """Encode the pre-game exit criteria as explicit booleans.

    Structural gates guard integrity (every prediction strictly inside (0, 1); the
    test season entered no fit or selection). ``gate_form_beats_prior_strength``
    reports whether P2's held-out Brier improvement over P1 clears bootstrap noise
    (its difference CI upper bound below zero — negative means P2 better).
    ``gate_pregame_calibrated`` checks the top tier P3 is itself calibrated. Beating
    the market is deliberately NOT a gate.
    """
    tiers = metrics["tiers"]
    predictions_ok = all(
        t["predictions_min"] > 0.0 and t["predictions_max"] < 1.0
        for t in tiers.values()
    )
    untouched_ok = all(
        set(t["splits_used"]) == {"train", "validation"} for t in tiers.values()
    )

    form_diff = metrics["paired_diff"]["P2_minus_P1"]
    form_beats = form_diff["brier"]["hi"] < 0.0

    cal = tiers["P3"]["calibration"]
    calibrated = (
        abs(cal["intercept"]) < evaluate.CALIB_INTERCEPT_TOL
        and evaluate.CALIB_SLOPE_LO <= cal["slope"] <= evaluate.CALIB_SLOPE_HI
    )

    return {
        "gate_predictions_in_open_interval": bool(predictions_ok),
        "gate_test_season_untouched": bool(untouched_ok),
        "gate_form_beats_prior_strength": bool(form_beats),
        "gate_pregame_calibrated": bool(calibrated),
    }


def structural_gates_pass(metrics: dict) -> bool:
    """True iff every structural gate holds — the single source of truth for exit."""
    gates = metrics["gates"]
    return all(bool(gates[name]) for name in STRUCTURAL_GATE_NAMES)


def ladder_verdict(metrics: dict) -> dict:
    """The honest 'did form help, and how much of the market gap closed' statement."""
    gates = metrics["gates"]
    gap = metrics["gap_close"]
    frac = gap["fraction_of_gap_closed"]
    ci = gap["fraction_of_gap_closed_ci"]
    ortho = gap["orthogonality"]["model_coefficient_ci"]
    adds_signal = ortho["lo"] > 0.0
    summary = (
        f"current-season form {'beats' if gates['gate_form_beats_prior_strength'] else 'does not beat'} "
        f"prior strength out of sample; P3 closes {frac:+.1%} of the tier-E-to-market "
        f"Brier gap (CI [{ci['lo']:+.1%}, {ci['hi']:+.1%}]); P3 "
        f"{'adds' if adds_signal else 'does not add'} signal orthogonal to the market; "
        f"the model {'is' if gates['gate_pregame_calibrated'] else 'is NOT'} calibrated pre-game"
    )
    return {
        "form_beats_prior_strength": bool(gates["gate_form_beats_prior_strength"]),
        "pregame_calibrated": bool(gates["gate_pregame_calibrated"]),
        "fraction_of_gap_closed": float(frac),
        "p3_adds_orthogonal_signal": bool(adds_signal),
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Assembly: fit the ladder, score on test, measure the market gap.
# --------------------------------------------------------------------------

def evaluate_ladder(df: pd.DataFrame, covered: pd.DataFrame) -> dict:
    """Fit the P0..P3 ladder, score it out of sample, and measure the market gap.

    Selects ``shrink_k`` on validation, builds the ladder frame at the chosen value,
    fits every tier leakage-safe (train+validation only), scores each on the 2025
    test split, runs the paired game-clustered bootstrap over adjacent tiers, and
    measures P3's gap-close versus the market on the ``covered`` games. Pure with
    respect to both inputs.
    """
    selection = select_shrink_k(df)
    chosen_k = selection["chosen_k"]
    frame = build_ladder_frame(df, chosen_k)

    test = frame.loc[frame["split"] == TEST_SPLIT].reset_index(drop=True)
    if len(test) == 0:
        raise ValueError("no rows with split == 'test' to evaluate on")
    y = test[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = test["game_id"].to_numpy()

    fits = {t: fit_ladder_tier(frame, t, TIER_FEATURES[t]) for t in LADDER_TIERS}
    preds = {t: predict_ladder(fits[t], test) for t in LADDER_TIERS}

    tiers: dict[str, dict] = {}
    for t in LADDER_TIERS:
        p = preds[t]
        intercept, slope = evaluate.fit_calibration(y, p)
        tiers[t] = {
            "adds": TIER_ADDS[t],
            "features": list(TIER_FEATURES[t]),
            "n_features": len(TIER_FEATURES[t]),
            "chosen_lambda": fits[t].chosen_lambda,
            "brier": evaluate.brier_score(y, p),
            "log_loss": evaluate.mean_log_loss(y, p),
            "calibration": {"intercept": intercept, "slope": slope},
            "predictions_min": float(p.min()),
            "predictions_max": float(p.max()),
            "splits_used": sorted(fits[t].splits_used),
            "holdout_rows_excluded": fits[t].holdout_rows_excluded,
        }

    adjacent_pairs = [("P1", "P0"), ("P2", "P1"), ("P3", "P2")]
    paired = ablation.paired_diff_ci(y, preds, game_ids, adjacent_pairs)

    p3_by_game = pd.Series(preds["P3"], index=pd.Index(game_ids, name="game_id"))
    gap_close = gap_close_vs_market(covered, p3_by_game)

    metrics: dict = {
        "n_test_games": int(pd.Series(game_ids).nunique()),
        "home_win_rate_test": float(y.mean()),
        "shrink_k_selection": selection,
        "chosen_k": chosen_k,
        "tiers": tiers,
        "paired_diff": paired,
        "gap_close": gap_close,
        "bootstrap": {
            "n_boot": int(evaluate.N_BOOTSTRAP),
            "seed": int(evaluate.BOOTSTRAP_SEED),
            "alpha": float(evaluate.BOOTSTRAP_ALPHA),
        },
    }
    metrics["gates"] = compute_ladder_gates(metrics)
    metrics["structural_gates_pass"] = structural_gates_pass(metrics)
    metrics["verdict"] = ladder_verdict(metrics)
    return metrics
