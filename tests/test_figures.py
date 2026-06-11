"""Tests for the Sprint-3 Phase-5 hand-rolled SVG figures.

The real figures are emitted by `python -m winprob.figures` (wired into
`./report.sh`); these tests pin the contract on synthetic inputs. The SVG
rendering is pure (a function of a reliability table or a list of trajectories),
so it is checked directly, and the trajectory extraction takes an injected
`predict_fn` so the plotting logic needs no heavy model fixture. Every rendered
document is parsed with the stdlib XML parser to prove it is well-formed — a
figure that does not parse is a broken figure, caught here rather than in a
browser.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from winprob import figures


def _parse(svg: str) -> ET.Element:
    """Parse the SVG string; raises if it is not well-formed XML."""
    return ET.fromstring(svg)


def _by_class(root: ET.Element, cls: str) -> list[ET.Element]:
    """All elements carrying the given `class` attribute, namespace-agnostic."""
    return [e for e in root.iter() if e.attrib.get("class") == cls]


def _local(tag: str) -> str:
    """Strip the XML namespace from a tag, e.g. '{ns}svg' -> 'svg'."""
    return tag.rsplit("}", 1)[-1]


# --------------------------------------------------------------------------
# Reliability diagram.
# --------------------------------------------------------------------------

def _reliability_table(pairs: list[tuple[float, float, int]]) -> list[dict]:
    """Build a reliability table from (mean_predicted, empirical, n) triples."""
    table = []
    for i, (mp, emp, n) in enumerate(pairs):
        table.append({
            "bin_lower": i / len(pairs),
            "bin_upper": (i + 1) / len(pairs),
            "n": n,
            "mean_predicted": mp if n else None,
            "empirical": emp if n else None,
        })
    return table


def test_reliability_diagram_is_well_formed_svg():
    table = _reliability_table([(0.1, 0.12, 100), (0.5, 0.48, 200), (0.9, 0.88, 150)])
    svg = figures.reliability_diagram_svg(table, title="Test reliability")
    root = _parse(svg)
    assert _local(root.tag) == "svg"


def test_reliability_diagram_has_diagonal_and_one_point_per_nonempty_bin():
    table = _reliability_table([(0.1, 0.12, 100), (0.5, 0.48, 200), (0.9, 0.0, 0)])
    svg = figures.reliability_diagram_svg(table, title="t")
    root = _parse(svg)
    assert len(_by_class(root, "diagonal")) == 1
    # Two non-empty bins -> two plotted points (the empty bin is skipped).
    assert len(_by_class(root, "bin-point")) == 2


def _point_line_distance(px, py, x1, y1, x2, y2) -> float:
    """Perpendicular distance from (px,py) to the line through the two endpoints."""
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def test_reliability_perfect_calibration_points_sit_on_diagonal():
    # When mean_predicted == empirical, each plotted point lies ON the y=x
    # diagonal — verified geometrically against the diagonal element's own coords.
    table = _reliability_table([(0.2, 0.2, 50), (0.8, 0.8, 50)])
    root = _parse(figures.reliability_diagram_svg(table, title="t"))
    diag = _by_class(root, "diagonal")[0]
    x1, y1, x2, y2 = (float(diag.attrib[k]) for k in ("x1", "y1", "x2", "y2"))
    points = _by_class(root, "bin-point")
    assert len(points) == 2
    for pt in points:
        d = _point_line_distance(
            float(pt.attrib["cx"]), float(pt.attrib["cy"]), x1, y1, x2, y2
        )
        assert d < 1e-6


def test_reliability_mapping_is_monotone():
    # Higher empirical -> higher on the page (smaller cy); higher mean_predicted
    # -> further right (larger cx). Guards against a flipped axis.
    table = _reliability_table([(0.2, 0.2, 50), (0.8, 0.8, 50)])
    root = _parse(figures.reliability_diagram_svg(table, title="t"))
    low, high = _by_class(root, "bin-point")
    assert float(high.attrib["cx"]) > float(low.attrib["cx"])
    assert float(high.attrib["cy"]) < float(low.attrib["cy"])


# --------------------------------------------------------------------------
# Trajectory figure.
# --------------------------------------------------------------------------

def _traj(game_id: str, category: str, ps: list[float], home_win: bool) -> dict:
    n = len(ps)
    return {
        "game_id": game_id,
        "category": category,
        "home_win": home_win,
        "points": [{"x": i / (n - 1), "p": p} for i, p in enumerate(ps)],
    }


def test_trajectory_svg_is_well_formed():
    trajs = [_traj("g1", "close_late", [0.5, 0.6, 0.55, 0.7], True)]
    svg = figures.trajectory_svg(trajs, title="Trajectories")
    assert _local(_parse(svg).tag) == "svg"


def test_trajectory_svg_has_one_polyline_per_game_and_a_reference_line():
    trajs = [
        _traj("g1", "close_late", [0.5, 0.6, 0.7], True),
        _traj("g2", "blowout", [0.5, 0.3, 0.1], False),
    ]
    root = _parse(figures.trajectory_svg(trajs, title="t"))
    assert len(_by_class(root, "trajectory")) == 2
    assert len(_by_class(root, "reference-line")) == 1
    # One endpoint marker per trajectory.
    assert len(_by_class(root, "endpoint")) == 2


# --------------------------------------------------------------------------
# Game selection + trajectory extraction.
# --------------------------------------------------------------------------

def _game(gid, margins, n=None) -> pd.DataFrame:
    """One synthetic 2025 game with a given margin path; final state sets outcome."""
    n = n or len(margins)
    return pd.DataFrame({
        "game_id": gid,
        "season": 2025,
        "split": "test",
        "period": np.clip(np.arange(n) // (n // 4 + 1) + 1, 1, 4),
        "possession_number": np.arange(n),
        "elapsed_game_seconds": np.linspace(0.0, 2880.0, n),
        "regulation_seconds_remaining": np.linspace(2880.0, 0.0, n),
        "home_score_differential": np.asarray(margins, dtype=float),
        "home_has_possession": (np.arange(n) % 2 == 0),
        "home_win": bool(margins[-1] > 0),
    })


def _selection_frame() -> pd.DataFrame:
    blowout = _game("g_blowout", list(np.linspace(0, 25, 30)))          # +25 final
    close = _game("g_close", list(np.linspace(0, 2, 30)))               # +2 final, late
    leadchange = _game("g_leadchange", [0, 8, 6, -7, -4, 9, 3])         # both led >5
    return pd.concat([blowout, close, leadchange], ignore_index=True)


def test_select_trajectory_games_covers_distinct_categories():
    selected = figures.select_trajectory_games(_selection_frame())
    cats = {s["category"] for s in selected}
    gids = {s["game_id"] for s in selected}
    assert "close_late" in cats and "blowout" in cats and "lead_change" in cats
    assert len(gids) == len(selected)  # distinct games


def test_build_trajectories_produces_monotone_points_in_range():
    df = _selection_frame()
    selected = figures.select_trajectory_games(df)

    def predict_fn(sub: pd.DataFrame) -> np.ndarray:
        # A trivial in-(0,1) forecaster: logistic of the margin.
        return 1.0 / (1.0 + np.exp(-0.2 * sub["home_score_differential"].to_numpy()))

    trajs = figures.build_trajectories(df, predict_fn, selected)
    assert len(trajs) == len(selected)
    for t in trajs:
        xs = [pt["x"] for pt in t["points"]]
        ps = [pt["p"] for pt in t["points"]]
        assert xs == sorted(xs)               # game progress is monotone
        assert xs[0] == 0.0 and xs[-1] == 1.0  # normalized to [0, 1]
        assert all(0.0 < p < 1.0 for p in ps)
