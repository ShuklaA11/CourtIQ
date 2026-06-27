"""Tests for the Sprint-6 Polymarket-gap figure (winprob.polymarket_figure).

The real figure is emitted by `python -m winprob.polymarket_figure` (wired into
`./polymarket.sh`); these tests pin the module-specific contract on synthetic
metrics. The shared bar renderer lives in `winprob.pregame_figure` and is tested
there, so here we only check that `build_bars` reads the covered-games section of a
polymarket metrics dict and that `render` produces a well-formed two-bar SVG.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from winprob import polymarket_figure


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _by_class(root: ET.Element, cls: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.attrib.get("class") == cls]


def _metrics() -> dict:
    return {"gap_close": {
        "baseline_model_brier": 0.22453, "p3_brier": 0.20987, "market_brier": 0.19692,
        "fraction_of_gap_closed": 0.531, "n_games": 1255,
    }}


def test_build_bars_reads_only_the_covered_games_section():
    bars, market = polymarket_figure.build_bars(_metrics())
    assert [b["label"] for b in bars] == ["tier-E baseline", "P3 pre-game model"]
    assert [b["brier"] for b in bars] == [0.22453, 0.20987]
    assert market == 0.19692


def test_render_end_to_end_is_well_formed_two_bar_svg():
    root = ET.fromstring(polymarket_figure.render(_metrics()))
    assert _local(root.tag) == "svg"
    assert len(_by_class(root, "gap-bar")) == 2
    assert len(_by_class(root, "market-line")) == 1


def test_render_labels_the_chart_for_the_prediction_market():
    svg = polymarket_figure.render(_metrics())
    assert "prediction market" in svg
    assert "+53.1%" in svg  # fraction of the gap closed, from the metrics dict
