"""Sprint-6 Polymarket pre-game benchmark — model-vs-prediction-market comparison.

Retests the Sprint-4 gap-close on a fuller, firmer sample. Where Sprint 3-5 scored
the model against a single-book sportsbook on 769 regular-season games, this scores
it against **Polymarket** — a real-money prediction market — on the ~96%-covered
test season INCLUDING the playoffs, from the frozen ``polymarket_closing.parquet``
snapshot.

The comparison builds nothing new; it stands on the Sprint 3-4 machinery and only
re-points the "market" at Polymarket:

* ``covered_games_frame`` re-fits the tier-E possession model leakage-safe
  (``ablation``), takes each test game's opening state, and joins the vig-free
  Polymarket home probability ON ``game_id`` (the snapshot carries it, so no
  date/team join is needed).
* the P3 pre-game ladder is fit exactly as Sprint 4 does (``pregame_ladder``), and
  its per-game probability is the "model" in the head-to-head.
* ``market.comparison_metrics`` scores P3 vs Polymarket (Brier, log loss,
  calibration, the paired game-clustered difference CI, correlation), and
  ``pregame_ladder.gap_close_vs_market`` re-measures the fraction of the
  tier-E→market Brier gap P3 closes — now against Polymarket.

*Honest philosophy.* Structural gates guard integrity (predictions in (0, 1); every
snapshot price strictly pre-tip and within the 24h window; test-season only) and
drive the exit code. The MODEL gate is P3's own pre-game calibration. **Beating
Polymarket is NOT a gate** — the market-sharpness gap and coverage count are
reported with CIs, exactly as in Sprint 3-4.

Pure numpy/pandas; every function returns a new object and never mutates its inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from winprob import (
    ablation,
    design,
    evaluate,
    market,
    model,
    polymarket,
    pregame_ladder,
)

DEFAULT_DATA_DIR = Path("data/winprob")
PARQUET_NAME = "fct_game_states.parquet"
SNAPSHOT_NAME = "polymarket_closing.parquet"
METRICS_JSON_NAME = "polymarket_metrics.json"
AUDIT_JSON_NAME = "polymarket_audit.json"

TEST_SPLIT = market.TEST_SPLIT
TARGET_COLUMN = model.TARGET_COLUMN

# Structural gates guard integrity and drive the exit code; the model gate (P3
# pre-game calibration) is reported but never fails the run, mirroring Sprint 4.
STRUCTURAL_GATE_NAMES: tuple[str, ...] = (
    "gate_predictions_in_open_interval",
    "gate_all_prices_strictly_pretip",
    "gate_test_season_only",
)

SOURCE_NOTE = (
    "Polymarket vig-free closing probabilities (real-money prediction market); "
    "last pre-tip prices-history tick per token, ~96% test-game coverage incl. playoffs"
)


# --------------------------------------------------------------------------
# Covered games: tier-E model prob + vig-free Polymarket prob, joined on game_id.
# --------------------------------------------------------------------------

def covered_games_frame(df: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Covered test games with the tier-E model prob and the Polymarket prob.

    Fits tier E leakage-safe on train+validation, predicts each TEST game's
    opening-state home-win probability (``p_model``) exactly as ``market.run`` /
    ``pregame.covered_games_frame`` do, then joins the vig-free Polymarket home
    probability on ``game_id`` (the snapshot's key), keeping one row per covered
    game. Raises if nothing joins. Returns a NEW frame.
    """
    prepared = ablation.add_team_strength_columns(df)
    fit = ablation.fit_tier(prepared, "E", ablation.TIER_E_FEATURES, model.LAMBDA_GRID)

    test = prepared.loc[prepared["split"] == TEST_SPLIT]
    opening = market.opening_state_rows(test).reset_index(drop=True)
    opening = opening.assign(p_model=ablation.predict_tier(fit, opening))

    snap = snapshot.copy()
    snap["game_id"] = snap["game_id"].astype(str)
    opening = opening.assign(game_id=opening["game_id"].astype(str))
    joined = opening.merge(
        snap[["game_id", "market_home_prob", "moneyline_volume",
              "last_tick_seconds_before_tip"]],
        on="game_id",
        how="inner",
    )
    if len(joined) == 0:
        raise ValueError("no test games joined to the Polymarket snapshot (game_id mismatch)")
    return joined


def p3_predictions(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Fit the Sprint-4 P3 ladder leakage-safe; return its per-game test probability.

    Selects ``shrink_k`` on validation, builds the ladder frame at the chosen value,
    fits the P3 tier on train+validation only, and predicts the 2025 test games.
    Returns the ``select_shrink_k`` selection record and a ``game_id -> P3 prob``
    Series (the "model" in the head-to-head). The test season enters no fit.
    """
    selection = pregame_ladder.select_shrink_k(df)
    frame = pregame_ladder.build_ladder_frame(df, selection["chosen_k"])
    test = frame.loc[frame["split"] == TEST_SPLIT].reset_index(drop=True)
    fit3 = pregame_ladder.fit_ladder_tier(
        frame, "P3", pregame_ladder.TIER_P3_FEATURES
    )
    p3 = pregame_ladder.predict_ladder(fit3, test)
    p3_by_game = pd.Series(
        p3, index=pd.Index(test["game_id"].astype(str).to_numpy(), name="game_id")
    )
    return selection, p3_by_game


# --------------------------------------------------------------------------
# Gates + verdict.
# --------------------------------------------------------------------------

def compute_polymarket_gates(metrics: dict, covered: pd.DataFrame) -> dict[str, bool]:
    """Encode the Sprint-6 gates: structural integrity + P3 pre-game calibration.

    Reuses ``market.compute_market_gates`` for the "predictions in (0, 1)" and "model
    (P3) calibrated" checks (the comparison's ``model`` key is P3, ``market`` is
    Polymarket), and adds two snapshot-integrity gates: every recorded last tick is
    strictly pre-tip and within the 24h window, and every covered game is in the test
    split. Beating Polymarket is deliberately NOT a gate.
    """
    base = market.compute_market_gates(metrics["comparison"])
    seconds = covered["last_tick_seconds_before_tip"].to_numpy()
    prices_pretip = bool(
        np.all(seconds > 0) and np.all(seconds <= polymarket.COVERAGE_MAX_SECONDS)
    )
    test_only = bool((covered["split"] == TEST_SPLIT).all())
    return {
        "gate_predictions_in_open_interval": bool(base["gate_predictions_in_open_interval"]),
        "gate_all_prices_strictly_pretip": prices_pretip,
        "gate_test_season_only": test_only,
        "gate_model_pregame_calibrated": bool(base["gate_model_pregame_calibrated"]),
    }


def structural_gates_pass(metrics: dict) -> bool:
    """True iff every structural gate holds — the single source of truth for exit."""
    gates = metrics["gates"]
    return all(bool(gates[name]) for name in STRUCTURAL_GATE_NAMES)


def polymarket_verdict(metrics: dict) -> dict:
    """The honest 'the prediction market is sharper, the model stays calibrated' line."""
    diff = metrics["comparison"]["paired_diff"]["market_minus_model"]
    market_sharper = diff["brier"]["hi"] < 0.0
    gap = metrics["gap_close"]
    frac = gap["fraction_of_gap_closed"]
    ci = gap["fraction_of_gap_closed_ci"]
    calibrated = metrics["gates"]["gate_model_pregame_calibrated"]
    summary = (
        f"on {metrics['n_covered']} of {metrics['n_test_games_total']} test games "
        f"(incl. playoffs), the prediction market is "
        f"{'sharper' if market_sharper else 'within noise of the model'} pre-game; "
        f"P3 closes {frac:+.1%} of the tier-E→Polymarket Brier gap "
        f"(CI [{ci['lo']:+.1%}, {ci['hi']:+.1%}]); the model "
        f"{'is' if calibrated else 'is NOT'} calibrated pre-game"
    )
    return {
        "market_sharper": bool(market_sharper),
        "model_pregame_calibrated": bool(calibrated),
        "fraction_of_gap_closed": float(frac),
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def evaluate_polymarket(df: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    """Score P3 vs Polymarket on the covered test games and measure the gap-close.

    Builds the covered frame (tier-E ``p_model`` + Polymarket prob), fits the P3
    ladder, computes the P3-vs-Polymarket head-to-head via
    ``market.comparison_metrics`` and the fraction-of-gap-closed via
    ``pregame_ladder.gap_close_vs_market``, and assembles coverage stats, gates, and
    the verdict. Pure with respect to both inputs.
    """
    covered = covered_games_frame(df, snapshot)
    selection, p3_by_game = p3_predictions(df)

    joined = covered.assign(p3=covered["game_id"].map(p3_by_game))
    if joined["p3"].isna().any():
        raise ValueError("covered game missing a P3 prediction (game_id mismatch)")

    y = joined[TARGET_COLUMN].to_numpy().astype(np.float64)
    p3 = joined["p3"].to_numpy(dtype=np.float64)
    p_market = joined["market_home_prob"].to_numpy(dtype=np.float64)
    game_ids = joined["game_id"].to_numpy()

    comparison = market.comparison_metrics(y, p3, p_market, game_ids)
    gap_close = pregame_ladder.gap_close_vs_market(covered, p3_by_game)

    n_test_total = int(df.loc[df["split"] == TEST_SPLIT, "game_id"].nunique())
    thin = int((snapshot["moneyline_volume"] < polymarket.THIN_VOLUME_USD).sum())

    metrics: dict = {
        "n_covered": len(covered),
        "n_test_games_total": n_test_total,
        "coverage_fraction": float(len(covered) / n_test_total) if n_test_total else 0.0,
        "thin_market_covered_games": thin,
        "home_win_rate_covered": float(y.mean()),
        "coverage_note": SOURCE_NOTE,
        "shrink_k_selection": selection,
        "chosen_k": selection["chosen_k"],
        "comparison": comparison,
        "gap_close": gap_close,
        "bootstrap": {
            "n_boot": int(evaluate.N_BOOTSTRAP),
            "seed": int(evaluate.BOOTSTRAP_SEED),
            "alpha": float(evaluate.BOOTSTRAP_ALPHA),
        },
    }
    metrics["gates"] = compute_polymarket_gates(metrics, joined)
    metrics["structural_gates_pass"] = structural_gates_pass(metrics)
    metrics["verdict"] = polymarket_verdict(metrics)
    return metrics


# --------------------------------------------------------------------------
# Serialization + provenance audit.
# --------------------------------------------------------------------------

def audit_payload(
    metrics: dict, parquet_path: Path, snapshot_path: Path
) -> dict:
    """Provenance document pinning the mart and the Polymarket snapshot by sha256."""
    return {
        "metrics_hash": design.canonical_hash(metrics),
        "dataset_parquet_sha256": design.file_hash(parquet_path),
        "snapshot_sha256": design.file_hash(snapshot_path),
        "split_definition": design.SPLIT_DEFINITION,
        "split_hash": design.canonical_hash(design.SPLIT_DEFINITION),
        "source_note": SOURCE_NOTE,
        "n_covered": metrics["n_covered"],
        "n_test_games_total": metrics["n_test_games_total"],
        "coverage_fraction": metrics["coverage_fraction"],
        "chosen_k": metrics["chosen_k"],
        "p3_brier": metrics["comparison"]["model"]["brier"],
        "market_brier": metrics["comparison"]["market"]["brier"],
        "paired_market_minus_model": metrics["comparison"]["paired_diff"]["market_minus_model"],
        "fraction_of_gap_closed": metrics["gap_close"]["fraction_of_gap_closed"],
        "fraction_of_gap_closed_ci": metrics["gap_close"]["fraction_of_gap_closed_ci"],
        "gates": metrics["gates"],
        "structural_gates_pass": metrics["structural_gates_pass"],
        "verdict": metrics["verdict"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load the mart + snapshot, run the comparison, write metrics + audit JSON."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / PARQUET_NAME
    snapshot_path = data_dir / SNAPSHOT_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run ./game_states.sh first"
        )
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"missing Polymarket snapshot at {snapshot_path}; "
            "run `python -m winprob.polymarket_pull` first (see polymarket.sh)"
        )
    df = pd.read_parquet(parquet_path)
    snapshot = pd.read_parquet(snapshot_path)
    metrics = evaluate_polymarket(df, snapshot)
    audit = audit_payload(metrics, parquet_path, snapshot_path)
    _write_json(data_dir / METRICS_JSON_NAME, metrics)
    _write_json(data_dir / AUDIT_JSON_NAME, audit)
    return metrics


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    comp = metrics["comparison"]
    diff = comp["paired_diff"]["market_minus_model"]
    print(
        f"polymarket comparison: {metrics['n_covered']:,} covered test games of "
        f"{metrics['n_test_games_total']:,} ({metrics['coverage_fraction']:.1%}; "
        f"{metrics['thin_market_covered_games']} thin), "
        f"home win rate {metrics['home_win_rate_covered']:.3f}"
    )
    print("\n  forecast     Brier      log_loss   calib(intercept/slope)")
    for label, key in (("P3 model", "model"), ("Polymarket", "market")):
        m = comp[key]
        print(
            f"  {label:11} {m['brier']:9.5f} {m['log_loss']:10.5f}   "
            f"{m['calibration']['intercept']:+.3f} / {m['calibration']['slope']:.3f}"
        )
    print(
        f"\n  market - P3 paired  Brier {diff['brier']['point']:+.5f} "
        f"[{diff['brier']['lo']:+.5f}, {diff['brier']['hi']:+.5f}]   "
        f"log_loss {diff['log_loss']['point']:+.5f} "
        f"[{diff['log_loss']['lo']:+.5f}, {diff['log_loss']['hi']:+.5f}]  "
        f"(negative = market sharper)"
    )
    print(f"  P3<->Polymarket correlation: {comp['correlation']:.3f}")

    gap = metrics["gap_close"]
    ci = gap["fraction_of_gap_closed_ci"]
    print(
        f"\n  gap closed vs Polymarket: {gap['fraction_of_gap_closed']:+.1%} "
        f"[{ci['lo']:+.1%}, {ci['hi']:+.1%}]  "
        f"(P3 Brier {gap['p3_brier']:.5f} vs market {gap['market_brier']:.5f}, "
        f"tier-E {gap['baseline_model_brier']:.5f})"
    )

    print("\n  gates:")
    for name, passed in metrics["gates"].items():
        kind = "structural" if name in STRUCTURAL_GATE_NAMES else "model"
        print(f"    [{'PASS' if passed else 'FAIL'}] ({kind}) {name}")
    print(f"\n  verdict: {metrics['verdict']['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the P3 pre-game model to Polymarket on the covered test games"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = run(Path(args.data_dir))
    _print_summary(metrics)
    if not structural_gates_pass(metrics):
        raise SystemExit("structural gate failure: Polymarket comparison integrity check failed")


if __name__ == "__main__":
    main()
