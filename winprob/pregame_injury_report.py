"""CLI + human-readable summary for the P4 injury ladder (``winprob.pregame_injury``).

The library module ``winprob.pregame_injury`` stays pure (no print, no argparse):
``run(data_dir)`` fits the P0..P4 ladder, re-measures the market gap, and writes
``injury_metrics.json`` + ``injury_audit.json``. This thin module is the CLI face —
it runs the pipeline, prints the one-screen summary, and exits NON-ZERO only on a
STRUCTURAL gate failure (a prediction outside (0, 1) or the test season leaking into
a fit), mirroring ``winprob.pregame``'s reporting contract. The scientific gates
(does availability beat form? does the model beat the market?) are REPORTED, never a
hard failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from winprob import pregame_injury, pregame_ladder


def _summary_lines(metrics: dict) -> list[str]:
    """The one-screen P4 summary as a list of lines (pure; no I/O)."""
    gap = metrics["gap_close"]
    ci = gap["fraction_of_gap_closed_ci"]
    diff = metrics["paired_diff"]["P4_minus_P3"]["brier"]
    lines = [
        (
            f"winprob P4 injury ladder: {metrics['n_test_games']:,} test games "
            f"(home win rate {metrics['home_win_rate_test']:.3f}, "
            f"shrink_k={metrics['chosen_k']:g})"
        ),
        (
            f"  P3 Brier {metrics['tiers']['P3']['brier']:.5f}  ->  "
            f"P4 Brier {metrics['tiers']['P4']['brier']:.5f}"
        ),
        (
            f"  P4 - P3 paired Brier {diff['point']:+.5f} "
            f"[{diff['lo']:+.5f}, {diff['hi']:+.5f}] "
            "(negative = availability beats form)"
        ),
        (
            f"  gap closed vs market: P4 {gap['fraction_of_gap_closed']:+.1%} "
            f"[{ci['lo']:+.1%}, {ci['hi']:+.1%}] vs form "
            f"{metrics['form_gap_close']['fraction_of_gap_closed']:+.1%} "
            f"({metrics['extra_gap_closed']:+.1%} extra); corr(P4, line) "
            f"{gap['correlation_p4_market']:+.3f}"
        ),
    ]
    lines += [f"  [{'PASS' if v else 'FAIL'}] {k}" for k, v in metrics["gates"].items()]
    lines.append(f"  verdict: {metrics['verdict']['summary']}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P4 availability ladder tier and re-measure the market gap"
    )
    parser.add_argument("--data-dir", default=str(pregame_injury.DEFAULT_DATA_DIR))
    args = parser.parse_args()
    metrics = pregame_injury.run(Path(args.data_dir))
    print("\n".join(_summary_lines(metrics)))
    if not pregame_ladder.structural_gates_pass(metrics):
        raise SystemExit("structural gate failure: pre-game integrity check did not pass")


if __name__ == "__main__":
    main()
