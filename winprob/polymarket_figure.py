"""Sprint-6 Polymarket-gap figure — hand-rolled SVG, zero plotting dependencies.

Re-renders the Sprint-4 gap decomposition against the **prediction market** instead
of the sportsbook: on the covered test games it plots the tier-E possession baseline
and the P3 pre-game ladder as each forecast's excess Brier over the vig-free
Polymarket line. Where Sprint 4's picture was drawn on 769 regular-season games, this
one is drawn on the ~1,255-game, playoff-inclusive Polymarket sample, so the same
"pre-game features close about half the gap" story appears on a firmer footing.

Reuses ``winprob.pregame_figure``'s SVG builders so there is exactly one bar-renderer
in the codebase. Every function is pure and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from winprob import pregame_figure

DEFAULT_DATA_DIR = Path("data/winprob")
DEFAULT_FIGURE_DIR = Path("figures")
METRICS_JSON_NAME = "polymarket_metrics.json"
FIGURE_NAME = "polymarket_gap.svg"

_BASELINE = pregame_figure._BASELINE
_MODEL = pregame_figure._MODEL


def build_bars(metrics: dict) -> tuple[list[dict], float]:
    """Covered-game bars (tier-E baseline, P3 model) and the Polymarket floor.

    Reads only the covered-games section (``gap_close``) so every bar is on the same
    Polymarket-covered sample as the market line — the only honest comparison. The
    ``gap_close`` block carries the same fields the Sprint-4 figure reads, so this
    mirrors ``pregame_figure.build_bars`` exactly. Returns ``(bars, market_brier)``.
    """
    gap = metrics["gap_close"]
    bars = [
        {"label": "tier-E baseline", "brier": float(gap["baseline_model_brier"]),
         "color": _BASELINE},
        {"label": "P3 pre-game model", "brier": float(gap["p3_brier"]),
         "color": _MODEL},
    ]
    return bars, float(gap["market_brier"])


def render(metrics: dict) -> str:
    """Render the Polymarket-gap figure from a polymarket metrics dict (pure)."""
    bars, market_brier = build_bars(metrics)
    gap = metrics["gap_close"]
    frac = gap["fraction_of_gap_closed"]
    n = gap["n_games"]
    title = "Closing the pre-game gap to the prediction market"
    subtitle = (
        f"covered-game Brier, {n:,} Polymarket games (incl. playoffs)  ·  "
        f"P3 closes {frac:+.1%} of the tier-E to market gap"
    )
    return pregame_figure.gap_bars_svg(bars, market_brier, title=title, subtitle=subtitle)


def run(
    data_dir: Path = DEFAULT_DATA_DIR, figure_dir: Path = DEFAULT_FIGURE_DIR
) -> Path:
    """Read ``polymarket_metrics.json`` and write the Polymarket-gap SVG."""
    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)
    metrics_path = data_dir / METRICS_JSON_NAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing {metrics_path}; run ./polymarket.sh "
            "(python -m winprob.polymarket_compare) first"
        )
    metrics = json.loads(metrics_path.read_text())
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / FIGURE_NAME
    out_path.write_text(render(metrics) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the Sprint-6 Polymarket-gap SVG figure"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()
    path = run(Path(args.data_dir), Path(args.figure_dir))
    print(f"wrote polymarket_gap: {path}")


if __name__ == "__main__":
    main()
