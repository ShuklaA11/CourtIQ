"""Sprint-5 injury-gap figure — hand-rolled SVG, zero plotting dependencies.

Extends the Sprint-4 gap decomposition with the availability tier: on the covered
test games it plots the tier-E baseline, the P3 form ladder, and the P4 tier that
adds the pre-tip inactive-list signal, each as its excess Brier over the vig-free
market line. The Sprint-4 picture showed the pre-game features closing ~half the
gap; this one shows the P4 bar landing on top of P3 — the visual statement of the
null (availability ~= form). Reuses ``winprob.pregame_figure``'s SVG builders so
there is exactly one bar-renderer in the codebase.

Every function is pure and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from winprob import pregame_figure

DEFAULT_DATA_DIR = Path("data/winprob")
DEFAULT_FIGURE_DIR = Path("figures")
METRICS_JSON_NAME = "injury_metrics.json"
FIGURE_NAME = "injury_gap.svg"

_BASELINE = pregame_figure._BASELINE
_MODEL = pregame_figure._MODEL
_AVAIL = "#2e8b57"  # sea green: the P4 availability tier, distinct from the P3 blue


def build_bars(metrics: dict) -> tuple[list[dict], float]:
    """Covered-game bars (tier-E baseline, P3 form, P4 availability) and the floor.

    Reads only the covered-games sections (``form_gap_close`` for P3, ``gap_close``
    for P4) so every bar is on the same 769-game sample as the market — the honest
    comparison. Returns ``(bars, market_brier)``.
    """
    p3 = metrics["form_gap_close"]
    p4 = metrics["gap_close"]
    bars = [
        {"label": "tier-E baseline", "brier": float(p4["baseline_model_brier"]),
         "color": _BASELINE},
        {"label": "P3 form ladder", "brier": float(p3["p3_brier"]), "color": _MODEL},
        {"label": "P4 + availability", "brier": float(p4["p4_brier"]), "color": _AVAIL},
    ]
    return bars, float(p4["market_brier"])


def render(metrics: dict) -> str:
    """Render the injury-gap figure from an injury metrics dict (pure)."""
    bars, market_brier = build_bars(metrics)
    p4 = metrics["gap_close"]
    frac = p4["fraction_of_gap_closed"]
    n = p4["n_games"]
    title = "The injury edge: availability does not beat form"
    subtitle = (
        f"covered-game Brier, {n:,} games with a closing line  ·  "
        f"P4 closes {frac:+.1%} of the tier-E to market gap (P3 form: "
        f"{metrics['verdict']['form_fraction_of_gap_closed']:+.1%})"
    )
    return pregame_figure.gap_bars_svg(bars, market_brier, title=title, subtitle=subtitle)


def run(
    data_dir: Path = DEFAULT_DATA_DIR, figure_dir: Path = DEFAULT_FIGURE_DIR
) -> Path:
    """Read ``injury_metrics.json`` and write the injury-gap SVG."""
    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)
    metrics_path = data_dir / METRICS_JSON_NAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing {metrics_path}; run ./injury.sh "
            "(python -m winprob.pregame_injury_report) first"
        )
    metrics = json.loads(metrics_path.read_text())
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / FIGURE_NAME
    out_path.write_text(render(metrics) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the Sprint-5 injury-gap SVG figure"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()
    path = run(Path(args.data_dir), Path(args.figure_dir))
    print(f"wrote injury_gap: {path}")


if __name__ == "__main__":
    main()
