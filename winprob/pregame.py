"""Sprint-4 pre-game evaluation harness + Sprint-3 market baseline reproduction.

Sprint 4 measures every new pre-game idea against a fixed stick: the tier-E
logistic's OPENING-STATE probability versus the vig-free market line, scored on
the covered test games. This module pins that stick. It builds nothing new — it
stands on the Sprint 1-3 infrastructure and only re-expresses it at the game
grain:

* ``build_pregame_table`` collapses the possession mart to one pre-tip row per
  game (via ``ablation.add_team_strength_columns`` then
  ``market.opening_state_rows``), the clean table downstream pre-game features
  attach to.
* ``score_forecast`` is the single scoring contract — Brier, log loss, and a
  logistic-recalibration (intercept/slope) — computed entirely through
  ``winprob.evaluate`` so no metric is ever re-derived here.
* ``reproduce_baseline`` re-fits tier E leakage-safe on train+validation, predicts
  each test game's opening-state home-win probability, and joins the vig-free
  market probability EXACTLY as ``winprob.market.run`` does (reusing
  ``market.load_odds`` — the odds CSV is never re-parsed here). It returns the
  model's and the market's Brier/log-loss/calibration on the covered games, the
  frozen baseline the rest of Sprint 4 is compared against.

Pure numpy/pandas; every function returns a new object and never mutates its
input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import ablation, design, evaluate, market, model

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
ODDS_CSV_NAME = market.ODDS_CSV_NAME
METRICS_JSON_NAME = "pregame_metrics.json"
AUDIT_JSON_NAME = "pregame_audit.json"

TARGET_COLUMN = model.TARGET_COLUMN
TEST_SPLIT = market.TEST_SPLIT

# The pre-game table's exact columns, in fixed order: game identity, split label,
# the outcome, and the two prior-season team-strength ratings that make the
# opening line matchup-specific.
PREGAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
    "split",
    TARGET_COLUMN,
    "home_team_strength",
    "away_team_strength",
)


# --------------------------------------------------------------------------
# Game-grain pre-game table.
# --------------------------------------------------------------------------

def build_pregame_table(df: pd.DataFrame) -> pd.DataFrame:
    """One pre-tip row per game: identity, outcome, and team strengths.

    Attaches the prior-season team-strength ratings with
    ``ablation.add_team_strength_columns`` (which copies its input), then reduces
    to the earliest possession of each game with ``market.opening_state_rows``.
    The result carries exactly one row per ``game_id`` and only the columns in
    ``PREGAME_COLUMNS``. Returns a NEW frame; the input is never mutated.
    """
    prepared = ablation.add_team_strength_columns(df)
    opening = market.opening_state_rows(prepared)
    table = opening.loc[:, list(PREGAME_COLUMNS)].reset_index(drop=True)
    if table["game_id"].duplicated().any():
        raise ValueError("pre-game table has duplicate game_id rows")
    return table


# --------------------------------------------------------------------------
# Scoring contract.
# --------------------------------------------------------------------------

def score_forecast(
    y: np.ndarray, p: np.ndarray, game_ids: np.ndarray
) -> dict:
    """Score a probability forecast against the realized outcome.

    Returns Brier, mean log loss, a logistic-recalibration ``{intercept, slope}``,
    and the forecast's min/max — all computed through ``winprob.evaluate`` so the
    metric definitions live in exactly one place. ``game_ids`` pins the resampling
    unit for callers and is validated for length here; the scalar scores
    themselves are row-wise. Pure: nothing is mutated.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    game_ids = np.asarray(game_ids)
    if not (len(y) == len(p) == len(game_ids)):
        raise ValueError(
            f"length mismatch: y={len(y)}, p={len(p)}, game_ids={len(game_ids)}"
        )
    if len(y) == 0:
        raise ValueError("cannot score an empty forecast")

    intercept, slope = evaluate.fit_calibration(y, p)
    return {
        "brier": evaluate.brier_score(y, p),
        "log_loss": evaluate.mean_log_loss(y, p),
        "calibration": {"intercept": intercept, "slope": slope},
        "predictions_min": float(np.min(p)),
        "predictions_max": float(np.max(p)),
    }


# --------------------------------------------------------------------------
# Fixed pre-game baseline: model vs market on the covered test games.
# --------------------------------------------------------------------------

def covered_games_frame(df: pd.DataFrame, odds: Path) -> pd.DataFrame:
    """The covered test games: outcome, tier-E model prob, and vig-free market prob.

    Fits tier E leakage-safe on train+validation, predicts the opening-state
    home-win probability (``p_model``) for every TEST game, and joins the vig-free
    market probability on ``(date, home_team_id, away_team_id)`` exactly as
    ``market.run`` does — ``odds`` is the MGM closing-odds CSV path, loaded (and
    de-vigged) through ``market.load_odds`` so the parsing is never forked here.
    Returns one row per covered game; the shared join both ``reproduce_baseline``
    and the pre-game ladder's gap-close measure against. Raises if nothing joins.
    """
    prepared = ablation.add_team_strength_columns(df)
    fit = ablation.fit_tier(
        prepared, "E", ablation.TIER_E_FEATURES, model.LAMBDA_GRID
    )

    test = prepared.loc[prepared["split"] == TEST_SPLIT]
    opening = market.opening_state_rows(test).reset_index(drop=True)
    opening = opening.assign(
        p_model=ablation.predict_tier(fit, opening),
        date=pd.to_datetime(opening["game_date"]).dt.date,
    )

    vig_free = market.load_odds(Path(odds))
    joined = opening.merge(
        vig_free[["date", "home_id", "away_id", "market_home_prob"]],
        left_on=["date", "home_team_id", "away_team_id"],
        right_on=["date", "home_id", "away_id"],
        how="inner",
    )
    if len(joined) == 0:
        raise ValueError("no test games joined to the odds file (date/team mismatch)")
    return joined


def reproduce_baseline(df: pd.DataFrame, odds: Path) -> dict:
    """Re-fit the tier-E pre-game baseline and score it against the market.

    Reuses ``covered_games_frame`` for the leakage-safe tier-E fit and the vig-free
    market join, then returns the model's and the market's ``score_forecast`` on the
    covered games plus ``n_games``. This is the frozen stick the rest of Sprint 4
    measures against.
    """
    joined = covered_games_frame(df, odds)
    y = joined[TARGET_COLUMN].to_numpy().astype(np.float64)
    game_ids = joined["game_id"].to_numpy()
    return {
        "model": score_forecast(y, joined["p_model"].to_numpy(), game_ids),
        "market": score_forecast(y, joined["market_home_prob"].to_numpy(), game_ids),
        "n_games": int(len(joined)),
    }


# --------------------------------------------------------------------------
# Pre-game ladder entry point + provenance audit.
# --------------------------------------------------------------------------

def audit_payload(metrics: dict, parquet_path: Path, odds_path: Path) -> dict:
    """Provenance document pinning the mart and the odds CSV by sha256."""
    return {
        "metrics_hash": design.canonical_hash(metrics),
        "dataset_parquet_sha256": design.file_hash(parquet_path),
        "odds_csv_sha256": design.file_hash(odds_path),
        "split_definition": design.SPLIT_DEFINITION,
        "split_hash": design.canonical_hash(design.SPLIT_DEFINITION),
        "chosen_k": metrics["chosen_k"],
        "n_test_games": metrics["n_test_games"],
        "tier_brier": {t: metrics["tiers"][t]["brier"] for t in metrics["tiers"]},
        "paired_P2_minus_P1": metrics["paired_diff"]["P2_minus_P1"],
        "fraction_of_gap_closed": metrics["gap_close"]["fraction_of_gap_closed"],
        "fraction_of_gap_closed_ci": metrics["gap_close"]["fraction_of_gap_closed_ci"],
        "gates": metrics["gates"],
        "structural_gates_pass": metrics["structural_gates_pass"],
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart + odds CSV, run the pre-game ladder, write metrics + audit JSON."""
    from winprob import pregame_ladder

    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    odds_path = data_dir / ODDS_CSV_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run ./game_states.sh first"
        )
    if not odds_path.exists():
        raise FileNotFoundError(
            f"missing odds file at {odds_path}; see market.sh for how to fetch it"
        )
    df = pd.read_parquet(parquet_path)
    covered = covered_games_frame(df, odds_path)
    metrics = pregame_ladder.evaluate_ladder(df, covered)
    audit = audit_payload(metrics, parquet_path, odds_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    print(
        f"winprob pre-game ladder: {metrics['n_test_games']:,} test games "
        f"(home win rate {metrics['home_win_rate_test']:.3f}, "
        f"shrink_k={metrics['chosen_k']:g})"
    )
    print("\n  tier  adds                              Brier     log_loss")
    for t, tier in metrics["tiers"].items():
        print(
            f"  {t}    {tier['adds']:<32} {tier['brier']:9.5f} {tier['log_loss']:10.5f}"
        )

    fp = metrics["paired_diff"]["P2_minus_P1"]["brier"]
    print(
        f"\n  P2 - P1 paired Brier {fp['point']:+.5f} [{fp['lo']:+.5f}, {fp['hi']:+.5f}] "
        f"(negative = form beats prior strength)"
    )
    gap = metrics["gap_close"]
    ci = gap["fraction_of_gap_closed_ci"]
    print(
        f"  gap closed vs market: {gap['fraction_of_gap_closed']:+.1%} "
        f"[{ci['lo']:+.1%}, {ci['hi']:+.1%}]  "
        f"(P3 Brier {gap['p3_brier']:.5f} vs market {gap['market_brier']:.5f})"
    )
    ortho = gap["orthogonality"]
    oc = ortho["model_coefficient_ci"]
    print(
        f"  orthogonality: model coef {ortho['model_coefficient']:+.3f} "
        f"[{oc['lo']:+.3f}, {oc['hi']:+.3f}]"
    )

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  verdict: {metrics['verdict']['summary']}")


def main() -> None:
    from winprob import pregame_ladder

    parser = argparse.ArgumentParser(
        description="Run the pre-game ablation ladder and its gap-close-vs-market"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not pregame_ladder.structural_gates_pass(metrics):
        raise SystemExit("structural gate failure: pre-game integrity check did not pass")


if __name__ == "__main__":
    main()
