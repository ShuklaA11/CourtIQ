"""Tests for the Sprint-4 pre-game gap-decomposition figure (winprob.pregame_figure).

The real figure is emitted by `python -m winprob.pregame_figure` (wired into
`./pregame.sh`); these tests pin the contract on synthetic metrics. The SVG
rendering is pure — a function of the covered-game Brier numbers — so it is checked
directly and parsed with the stdlib XML parser to prove it is well-formed. The
figure is anchored at the market line: each bar's height must encode its gap to the
market, so a larger gap must draw a taller bar. That geometric invariant (not the
exact pixels) is what the tests assert.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from winprob import pregame_figure


def _parse(svg: str) -> ET.Element:
    """Parse the SVG string; raises if it is not well-formed XML."""
    return ET.fromstring(svg)


def _by_class(root: ET.Element, cls: str) -> list[ET.Element]:
    """All elements carrying the given `class` attribute, namespace-agnostic."""
    return [e for e in root.iter() if e.attrib.get("class") == cls]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _bars() -> list[dict]:
    return [
        {"label": "tier-E baseline", "brier": 0.22990, "color": "#8a8a8a"},
        {"label": "P3 pre-game ladder", "brier": 0.22234, "color": "#1f77b4"},
    ]


def test_gap_bars_svg_is_well_formed():
    svg = pregame_figure.gap_bars_svg(_bars(), market_brier=0.21466,
                                      title="t", subtitle="s")
    assert _local(_parse(svg).tag) == "svg"


def test_one_bar_per_forecast_and_a_single_market_line():
    root = _parse(pregame_figure.gap_bars_svg(_bars(), 0.21466, title="t", subtitle="s"))
    assert len(_by_class(root, "gap-bar")) == 2
    assert len(_by_class(root, "market-line")) == 1


def test_bar_height_grows_with_the_gap_to_market():
    # The tier-E baseline is further from the market than P3, so its bar is taller.
    root = _parse(pregame_figure.gap_bars_svg(_bars(), 0.21466, title="t", subtitle="s"))
    baseline, p3 = _by_class(root, "gap-bar")  # rendered in list order
    assert float(baseline.attrib["height"]) > float(p3.attrib["height"]) > 0.0


def test_bar_height_is_proportional_to_gap():
    # Height ratio must equal the gap ratio: the chart cannot exaggerate the closure.
    bars = _bars()
    market = 0.21466
    root = _parse(pregame_figure.gap_bars_svg(bars, market, title="t", subtitle="s"))
    baseline, p3 = _by_class(root, "gap-bar")
    gap_ratio = (bars[1]["brier"] - market) / (bars[0]["brier"] - market)
    height_ratio = float(p3.attrib["height"]) / float(baseline.attrib["height"])
    assert height_ratio == pytest.approx(gap_ratio, abs=1e-3)


def test_build_bars_reads_only_the_covered_games_section():
    metrics = {"gap_close": {
        "baseline_model_brier": 0.23, "p3_brier": 0.222, "market_brier": 0.214,
        "fraction_of_gap_closed": 0.5, "n_games": 769,
    }}
    bars, market = pregame_figure.build_bars(metrics)
    assert [b["label"] for b in bars] == ["tier-E baseline", "P3 pre-game ladder"]
    assert [b["brier"] for b in bars] == [0.23, 0.222]
    assert market == 0.214


def test_render_end_to_end_from_metrics_dict():
    metrics = {"gap_close": {
        "baseline_model_brier": 0.22990, "p3_brier": 0.22234, "market_brier": 0.21466,
        "fraction_of_gap_closed": 0.496, "n_games": 769,
    }}
    root = _parse(pregame_figure.render(metrics))
    assert _local(root.tag) == "svg"
    assert len(_by_class(root, "gap-bar")) == 2


def test_raises_when_no_forecast_beats_or_trails_the_market():
    # A forecast exactly at the market has zero gap -> nothing to plot -> fail loudly.
    with pytest.raises(ValueError):
        pregame_figure.gap_bars_svg(
            [{"label": "x", "brier": 0.21466, "color": "#000"}],
            market_brier=0.21466, title="t", subtitle="s",
        )
