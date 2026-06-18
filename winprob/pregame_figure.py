"""Sprint-4 pre-game gap figure — hand-rolled SVG, zero plotting dependencies.

The Sprint-4 finding is a *gap decomposition*: on the covered test games the
tier-E possession model sits some distance above the vig-free market's Brier, and
the pre-game ladder (P3) closes part of that distance. This module renders that
one picture — the same hand-rolled-SVG approach as ``winprob.figures`` (SVG is
just XML text, so a bar chart needs no matplotlib), kept in its own module so the
Sprint-3 figure code is untouched.

The chart is deliberately anchored at the *market line*: each bar's height is the
forecast's EXCESS Brier over the market (its gap to the sharp reference), so the
zero of the plot is a meaningful quantity — perfect agreement with the market —
rather than an arbitrary truncation. The tier-E bar is the full gap; the P3 bar is
what remains after the pre-game features do their work; the fraction between them
is the reported ``fraction_of_gap_closed``. Absolute Brier values are printed on
each bar so the numbers still tie out to ``pregame_metrics.json``.

Every rendering function is pure (a function of the numbers), returns a new
string, and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATA_DIR = Path("data/winprob")
DEFAULT_FIGURE_DIR = Path("figures")
METRICS_JSON_NAME = "pregame_metrics.json"
FIGURE_NAME = "pregame_gap.svg"

# Palette matches winprob.figures: legible on light or dark, carries its own light
# background so it renders standalone when embedded.
_BG = "#ffffff"
_INK = "#1a1a1a"
_MUTED = "#8a8a8a"
_MARKET = "#c0392b"
_BASELINE = "#8a8a8a"
_MODEL = "#1f77b4"


# --------------------------------------------------------------------------
# Tiny SVG element builders (self-contained; mirror winprob.figures' style).
# --------------------------------------------------------------------------

def _r(v: float) -> str:
    """Round a coordinate to 2 dp for compact, stable output."""
    return f"{float(v):.2f}"


def _esc(s: str) -> str:
    """Escape the XML special characters that can appear in a label."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attrs(**kwargs) -> str:
    """Render an attribute string, mapping ``class_`` -> ``class`` and skipping None."""
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        name = "class" if key == "class_" else key.replace("_", "-")
        parts.append(f'{name}="{value}"')
    return " ".join(parts)


def _line(x1, y1, x2, y2, stroke, width=1.0, dash=None, class_=None) -> str:
    a = _attrs(x1=_r(x1), y1=_r(y1), x2=_r(x2), y2=_r(y2), stroke=stroke,
               stroke_width=width, stroke_dasharray=dash, class_=class_)
    return f"<line {a} />"


def _rect(x, y, w, h, fill, class_=None) -> str:
    a = _attrs(x=_r(x), y=_r(y), width=_r(w), height=_r(h), fill=fill, class_=class_)
    return f"<rect {a} />"


def _text(x, y, s, fill=_INK, size=12, anchor="start", weight=None, class_=None) -> str:
    a = _attrs(x=_r(x), y=_r(y), fill=fill, font_size=size, text_anchor=anchor,
               font_weight=weight, font_family="sans-serif", class_=class_)
    return f"<text {a}>{_esc(s)}</text>"


def _document(width: float, height: float, body: list[str], title: str) -> str:
    """Wrap body elements in an SVG root with a light background."""
    inner = "\n  ".join([_rect(0, 0, width, height, _BG)] + body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_r(width)} {_r(height)}" '
        f'width="{_r(width)}" height="{_r(height)}" role="img" aria-label="{_esc(title)}">\n  '
        f'{inner}\n</svg>'
    )


# --------------------------------------------------------------------------
# The gap-decomposition bar chart.
# --------------------------------------------------------------------------

def gap_bars_svg(
    bars: list[dict],
    market_brier: float,
    title: str,
    subtitle: str,
    width: float = 560.0,
    height: float = 360.0,
    margin: float = 56.0,
) -> str:
    """Render covered-game Brier as gap-to-market bars, with the market line at zero.

    ``bars`` is an ordered list of ``{"label", "brier", "color"}`` for the non-market
    forecasts (tier-E baseline, then P3). ``market_brier`` is the reference floor.
    Each bar rises from the market line to its own Brier, so its height encodes the
    gap to the market; the absolute Brier is printed on the bar. Pure: nothing is
    mutated.
    """
    if not bars:
        raise ValueError("need at least one forecast bar to render")

    x0, y_bottom = margin, height - margin
    plot_w = width - 2 * margin
    plot_top = margin + 24.0
    plot_h = y_bottom - plot_top

    top_brier = max(b["brier"] for b in bars)
    span = top_brier - market_brier
    if span <= 0:
        raise ValueError("every forecast is at least as sharp as the market; no gap to plot")
    y_max = top_brier + 0.18 * span  # headroom above the tallest bar

    def py(brier: float) -> float:
        # market_brier -> bottom, y_max -> top.
        frac = (brier - market_brier) / (y_max - market_brier)
        return y_bottom - frac * plot_h

    body: list[str] = []

    # Y-axis ticks at the market line and each bar's Brier (honest absolute scale).
    tick_values = sorted({market_brier, *(b["brier"] for b in bars)})
    for v in tick_values:
        body.append(_line(x0 - 4, py(v), x0, py(v), _MUTED))
        body.append(_text(x0 - 8, py(v) + 4, f"{v:.4f}", fill=_MUTED, size=11, anchor="end"))

    # Bars: one per non-market forecast, rising from the market floor.
    n = len(bars)
    slot = plot_w / n
    bar_w = min(slot * 0.5, 96.0)
    for i, bar in enumerate(bars):
        cx = x0 + slot * (i + 0.5)
        top = py(bar["brier"])
        body.append(_rect(cx - bar_w / 2, top, bar_w, y_bottom - top, bar["color"],
                          class_="gap-bar"))
        body.append(_text(cx, top - 8, f"{bar['brier']:.4f}", fill=_INK, size=12,
                          anchor="middle", weight="bold"))
        gap = bar["brier"] - market_brier
        body.append(_text(cx, top - 22, f"+{gap:.4f} vs market", fill=_MUTED, size=10,
                          anchor="middle"))
        body.append(_text(cx, y_bottom + 16, bar["label"], fill=_INK, size=11,
                          anchor="middle"))

    # Market reference line at the floor, spanning the plot.
    body.append(_line(x0, py(market_brier), x0 + plot_w, py(market_brier), _MARKET,
                      width=1.6, dash="5 3", class_="market-line"))
    body.append(_text(x0 + plot_w, py(market_brier) - 6,
                      f"market (vig-free)  {market_brier:.4f}", fill=_MARKET, size=11,
                      anchor="end"))

    # Titles.
    body.append(_text(width / 2, 22, title, fill=_INK, size=14, anchor="middle",
                      weight="bold"))
    body.append(_text(width / 2, 40, subtitle, fill=_MUTED, size=12, anchor="middle"))
    rot = f"rotate(-90 16 {_r(height / 2)})"
    ylabel = _attrs(x=16, y=_r(height / 2), fill=_MUTED, font_size=12,
                    text_anchor="middle", font_family="sans-serif", transform=rot)
    body.append(f"<text {ylabel}>covered-game Brier (lower is sharper)</text>")

    return _document(width, height, body, title)


def build_bars(metrics: dict) -> tuple[list[dict], float]:
    """Extract the gap-decomposition bars and the market floor from the metrics dict.

    Reads only the covered-games section (``gap_close``) so every bar is on the same
    769-game sample as the market — the only honest comparison. Returns
    ``(bars, market_brier)``.
    """
    gap = metrics["gap_close"]
    bars = [
        {"label": "tier-E baseline", "brier": float(gap["baseline_model_brier"]),
         "color": _BASELINE},
        {"label": "P3 pre-game ladder", "brier": float(gap["p3_brier"]),
         "color": _MODEL},
    ]
    return bars, float(gap["market_brier"])


# --------------------------------------------------------------------------
# Runner.
# --------------------------------------------------------------------------

def render(metrics: dict) -> str:
    """Render the gap-decomposition figure from a pregame metrics dict (pure)."""
    bars, market_brier = build_bars(metrics)
    gap = metrics["gap_close"]
    frac = gap["fraction_of_gap_closed"]
    n = gap["n_games"]
    title = "Closing the pre-game gap to the market"
    subtitle = (
        f"covered-game Brier, {n:,} games with a closing line  ·  "
        f"P3 closes {frac:+.1%} of the tier-E to market gap"
    )
    return gap_bars_svg(bars, market_brier, title=title, subtitle=subtitle)


def run(
    data_dir: Path = DEFAULT_DATA_DIR, figure_dir: Path = DEFAULT_FIGURE_DIR
) -> Path:
    """Read ``pregame_metrics.json`` and write the gap-decomposition SVG."""
    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)
    metrics_path = data_dir / METRICS_JSON_NAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing {metrics_path}; run ./pregame.sh (python -m winprob.pregame) first"
        )
    metrics = json.loads(metrics_path.read_text())
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / FIGURE_NAME
    out_path.write_text(render(metrics) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the Sprint-4 pre-game gap-decomposition SVG figure"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()
    path = run(Path(args.data_dir), Path(args.figure_dir))
    print(f"wrote pregame_gap: {path}")


if __name__ == "__main__":
    main()
