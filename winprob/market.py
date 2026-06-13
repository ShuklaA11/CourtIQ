"""Sprint-3 Phase-5 model-vs-market comparison at the pre-game state.

The win-probability model emits P(home win) at every possession; a market closing
line is a single pre-tip price. Comparing a full in-game trajectory to one
pre-game number would compare at mismatched information sets, so the only honest
alignment is the model's OPENING-STATE probability versus the vig-free market
probability. And it must use the rating-aware tier-E logistic: the sparse
score+time model predicts the base home-win rate for every game before tip and
cannot be told apart from a coin, whereas tier-E's opening number varies by
matchup through prior-season team strength — so this doubles as the test of where
RAPM earns its keep, pre-game.

*Data.* MGM closing moneylines via a public, ungated Kaggle dataset, covering the
2025-26 regular season through the All-Star break (2026-02-12). That window holds
~800 of the 1,258 test games — no late regular season and no playoffs — so every
claim here is scoped to the covered games and stated as such. The odds CSV's
sha256 is pinned in the audit so the comparison reproduces from a fixed artifact.

*Honesty.* The market prices injuries, rest, travel, and matchup edges a
season-pooled RAPM cannot, so it is expected to be SHARPER pre-game. The gate here
checks only that the MODEL is well-calibrated pre-game (a check it can pass on its
own terms); the market-sharpness gap is REPORTED with a paired game-clustered
bootstrap interval, not gated — "the market is sharper but our model is calibrated"
is the honest, publishable result, not a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import ablation, design, evaluate, model

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
ODDS_CSV_NAME = "nba_closing_odds_mgm.csv"
METRICS_JSON_NAME = "market_metrics.json"
AUDIT_JSON_NAME = "market_audit.json"

TEST_SPLIT = "test"
TARGET_COLUMN = model.TARGET_COLUMN
PREGAME_FEATURES = ablation.TIER_E_FEATURES

ODDS_SOURCE_URL = (
    "https://www.kaggle.com/datasets/caseydurfee/mgm-grand-nba-betting-data"
)
ODDS_SOURCE_NOTE = (
    "MGM closing moneylines; 2025-26 coverage runs through the All-Star break "
    "(2026-02-12), i.e. regular season only, no playoffs."
)

# Deterministic map from the dataset's team labels to stable NBA team ids, so the
# odds join needs no runtime nba_api call. All 30 franchises; ids are constants.
MGM_TEAM_TO_ID: dict[str, int] = {
    "Atlanta": 1610612737,
    "Boston": 1610612738,
    "Brooklyn": 1610612751,
    "Charlotte": 1610612766,
    "Chicago": 1610612741,
    "Cleveland": 1610612739,
    "Dallas": 1610612742,
    "Denver": 1610612743,
    "Detroit": 1610612765,
    "Golden State": 1610612744,
    "Houston": 1610612745,
    "Indiana": 1610612754,
    "LA Clippers": 1610612746,
    "LA Lakers": 1610612747,
    "Memphis": 1610612763,
    "Miami": 1610612748,
    "Milwaukee": 1610612749,
    "Minnesota": 1610612750,
    "New Orleans": 1610612740,
    "New York": 1610612752,
    "Oklahoma City": 1610612760,
    "Orlando": 1610612753,
    "Philadelphia": 1610612755,
    "Phoenix": 1610612756,
    "Portland": 1610612757,
    "Sacramento": 1610612758,
    "San Antonio": 1610612759,
    "Toronto": 1610612761,
    "Utah": 1610612762,
    "Washington": 1610612764,
}


# --------------------------------------------------------------------------
# Vig removal + odds loading.
# --------------------------------------------------------------------------

def vig_free_home_prob(dec_home: np.ndarray, dec_away: np.ndarray) -> np.ndarray:
    """Two-way vig-free home win probability from decimal odds.

    Each side's implied probability is ``1 / decimal_odds``; their sum exceeds 1
    by the bookmaker's overround (the vig). Normalizing the home implied
    probability by that sum removes the vig under the standard proportional
    assumption, yielding a probability that with its away complement sums to 1.
    """
    imp_home = 1.0 / np.asarray(dec_home, dtype=np.float64)
    imp_away = 1.0 / np.asarray(dec_away, dtype=np.float64)
    return imp_home / (imp_home + imp_away)


def load_odds(path: Path) -> pd.DataFrame:
    """Load MGM closing odds; map team labels to ids and remove the vig.

    Returns one row per game with ``date`` (a python ``date``), ``home_id`` /
    ``away_id`` (NBA team ids), the vig-free ``market_home_prob``, and the
    ``money_home_won`` outcome. Fails fast if any team label is unmapped, so a
    silent mis-join can never happen.
    """
    odds = pd.read_csv(path)
    labels = set(odds["home_team"]) | set(odds["away_team"])
    unknown = labels - set(MGM_TEAM_TO_ID)
    if unknown:
        raise ValueError(f"unmapped team labels in odds file: {sorted(unknown)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(odds["game_date"].str[:10]).dt.date,
        "home_id": odds["home_team"].map(MGM_TEAM_TO_ID).astype(int),
        "away_id": odds["away_team"].map(MGM_TEAM_TO_ID).astype(int),
        "market_home_prob": vig_free_home_prob(
            odds["money_home_decimal_odds"].to_numpy(),
            odds["money_away_decimal_odds"].to_numpy(),
        ),
        "money_home_won": odds["money_home_won"],
    })
    return out


# --------------------------------------------------------------------------
# Opening-state extraction.
# --------------------------------------------------------------------------

def opening_state_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The earliest possession of each game — its pre-tip state.

    Sorted by ``(period, possession_number)`` and reduced to the first row per
    game, so the model is scored at 0-0 with the full clock, where its prediction
    reflects team strength rather than any in-game score.
    """
    ordered = df.sort_values(["game_id", "period", "possession_number"])
    return ordered.groupby("game_id", sort=False, as_index=False).first()


# --------------------------------------------------------------------------
# Comparison metrics.
# --------------------------------------------------------------------------

def comparison_metrics(
    y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray, game_ids: np.ndarray
) -> dict:
    """Score model and market vs the outcome; pair the difference by bootstrap.

    Each forecast gets Brier, log loss, and a logistic-recalibration
    (intercept/slope) against the realized outcome; the paired game-clustered
    bootstrap (one row per game here) yields the ``market - model`` difference
    interval — negative means the market scored lower (sharper). Also reports the
    Pearson correlation between the two pre-game probabilities.
    """
    y = np.asarray(y, dtype=np.float64)
    preds = {"model": p_model, "market": p_market}
    paired = ablation.paired_diff_ci(y, preds, game_ids, [("market", "model")])

    out: dict = {"paired_diff": paired, "correlation": float(np.corrcoef(p_model, p_market)[0, 1])}
    for name, p in preds.items():
        intercept, slope = evaluate.fit_calibration(y, p)
        out[name] = {
            "brier": evaluate.brier_score(y, p),
            "log_loss": evaluate.mean_log_loss(y, p),
            "calibration": {"intercept": intercept, "slope": slope},
            "predictions_min": float(np.min(p)),
            "predictions_max": float(np.max(p)),
        }
    return out


# --------------------------------------------------------------------------
# Gates + verdict.
# --------------------------------------------------------------------------

def compute_market_gates(metrics: dict) -> dict[str, bool]:
    """Encode the Phase-5 market check: is the MODEL calibrated pre-game?

    Beating the market is not the bar — the market is expected to be sharper. The
    honest gate is that the model's opening-state probabilities are themselves
    calibrated (intercept near zero, slope near one) and strictly inside (0, 1).
    """
    cal = metrics["model"]["calibration"]
    calibrated = (
        abs(cal["intercept"]) < evaluate.CALIB_INTERCEPT_TOL
        and evaluate.CALIB_SLOPE_LO <= cal["slope"] <= evaluate.CALIB_SLOPE_HI
    )
    predictions_ok = (
        metrics["model"]["predictions_min"] > 0.0
        and metrics["model"]["predictions_max"] < 1.0
        and metrics["market"]["predictions_min"] > 0.0
        and metrics["market"]["predictions_max"] < 1.0
    )
    return {
        "gate_model_pregame_calibrated": bool(calibrated),
        "gate_predictions_in_open_interval": bool(predictions_ok),
    }


def market_verdict(metrics: dict) -> dict:
    """The honest 'market is sharper, is the model still calibrated' statement."""
    diff = metrics["paired_diff"]["market_minus_model"]
    market_sharper = diff["brier"]["hi"] < 0.0  # market Brier strictly below model
    calibrated = metrics["gates"]["gate_model_pregame_calibrated"]
    gap = metrics["model"]["brier"] - metrics["market"]["brier"]
    if market_sharper:
        summary = (
            f"the market is sharper pre-game (Brier gap {gap:+.4f}, CI excludes 0) — "
            "expected, since it prices injuries/rest/matchups the season-pooled RAPM "
            f"cannot; the model {'is' if calibrated else 'is NOT'} calibrated pre-game"
        )
    else:
        summary = (
            f"the model matches the market within noise pre-game (Brier gap {gap:+.4f}, "
            f"CI includes 0); the model {'is' if calibrated else 'is NOT'} calibrated"
        )
    return {
        "market_sharper": bool(market_sharper),
        "model_pregame_calibrated": bool(calibrated),
        "brier_gap_model_minus_market": float(gap),
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def evaluate_market(df: pd.DataFrame, odds: pd.DataFrame) -> dict:
    """Fit the tier-E model, score its pre-game probs against the market.

    Prepares team strength, fits tier E leakage-safe (train+validation only),
    takes each test game's opening state, predicts the pre-game home-win
    probability, joins the vig-free market probability on (date, home, away), and
    computes the comparison metrics, gates, and verdict on the covered games.
    """
    prepared = ablation.add_team_strength_columns(df)
    fit = ablation.fit_tier(prepared, "E", PREGAME_FEATURES, model.LAMBDA_GRID)

    test = prepared.loc[prepared["split"] == TEST_SPLIT]
    opening = opening_state_rows(test).reset_index(drop=True)
    opening["p_model"] = ablation.predict_tier(fit, opening)
    opening["date"] = pd.to_datetime(opening["game_date"]).dt.date

    joined = opening.merge(
        odds[["date", "home_id", "away_id", "market_home_prob"]],
        left_on=["date", "home_team_id", "away_team_id"],
        right_on=["date", "home_id", "away_id"],
        how="inner",
    )
    if len(joined) == 0:
        raise ValueError("no test games joined to the odds file (date/team mismatch)")

    y = joined[TARGET_COLUMN].to_numpy().astype(np.float64)
    p_model = joined["p_model"].to_numpy()
    p_market = joined["market_home_prob"].to_numpy()
    game_ids = joined["game_id"].to_numpy()

    metrics = comparison_metrics(y, p_model, p_market, game_ids)
    metrics["n_games"] = int(len(joined))
    metrics["n_test_games_total"] = int(test["game_id"].nunique())
    metrics["home_win_rate_covered"] = float(y.mean())
    metrics["coverage_note"] = ODDS_SOURCE_NOTE
    metrics["chosen_lambda"] = fit.chosen_lambda
    metrics["splits_used_for_fit"] = sorted(fit.splits_used)
    metrics["bootstrap"] = {
        "n_boot": int(evaluate.N_BOOTSTRAP),
        "seed": int(evaluate.BOOTSTRAP_SEED),
        "alpha": float(evaluate.BOOTSTRAP_ALPHA),
    }
    metrics["gates"] = compute_market_gates(metrics)
    metrics["verdict"] = market_verdict(metrics)
    return metrics


# --------------------------------------------------------------------------
# Serialization + provenance audit.
# --------------------------------------------------------------------------

def audit_payload(metrics: dict, parquet_path: Path, odds_path: Path) -> dict:
    """Provenance document pinning the mart and the odds CSV by sha256."""
    return {
        "metrics_hash": design.canonical_hash(metrics),
        "dataset_parquet_sha256": design.file_hash(parquet_path),
        "odds_csv_sha256": design.file_hash(odds_path),
        "odds_source_url": ODDS_SOURCE_URL,
        "odds_source_note": ODDS_SOURCE_NOTE,
        "split_definition": design.SPLIT_DEFINITION,
        "split_hash": design.canonical_hash(design.SPLIT_DEFINITION),
        "n_games": metrics["n_games"],
        "n_test_games_total": metrics["n_test_games_total"],
        "model_brier": metrics["model"]["brier"],
        "market_brier": metrics["market"]["brier"],
        "paired_market_minus_model": metrics["paired_diff"]["market_minus_model"],
        "gates": metrics["gates"],
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart + odds CSV, run the comparison, write metrics + audit JSON."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    odds_path = data_dir / ODDS_CSV_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run ./game_states.sh first"
        )
    if not odds_path.exists():
        raise FileNotFoundError(
            f"missing odds file at {odds_path}; download it from {ODDS_SOURCE_URL} "
            f"(anonymous) and save all_odds.csv as {odds_path}"
        )
    df = pd.read_parquet(parquet_path)
    odds = load_odds(odds_path)
    metrics = evaluate_market(df, odds)
    audit = audit_payload(metrics, parquet_path, odds_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    diff = metrics["paired_diff"]["market_minus_model"]
    print(
        f"winprob market comparison: {metrics['n_games']:,} covered test games "
        f"of {metrics['n_test_games_total']:,} "
        f"(home win rate {metrics['home_win_rate_covered']:.3f})"
    )
    print(f"  coverage: {metrics['coverage_note']}")
    print("\n  forecast   Brier      log_loss   calib(intercept/slope)")
    for name in ("model", "market"):
        m = metrics[name]
        print(
            f"  {name:9} {m['brier']:9.5f} {m['log_loss']:10.5f}   "
            f"{m['calibration']['intercept']:+.3f} / {m['calibration']['slope']:.3f}"
        )
    print(
        f"\n  market - model paired  Brier {diff['brier']['point']:+.5f} "
        f"[{diff['brier']['lo']:+.5f}, {diff['brier']['hi']:+.5f}]   "
        f"log_loss {diff['log_loss']['point']:+.5f} "
        f"[{diff['log_loss']['lo']:+.5f}, {diff['log_loss']['hi']:+.5f}]  "
        f"(negative = market sharper)"
    )
    print(f"  model<->market correlation: {metrics['correlation']:.3f}")

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  verdict: {metrics['verdict']['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the tier-E model's pre-game probabilities to the market"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not metrics["gates"]["gate_predictions_in_open_interval"]:
        raise SystemExit("structural gate failure: predictions not all in (0, 1)")


if __name__ == "__main__":
    main()
