"""Sprint-3 Phase-5 figures — hand-rolled SVG, zero plotting dependencies.

The project is deliberately lean (no matplotlib); a reliability diagram and a
handful of win-probability trajectories are simple enough to emit as SVG (which is
just XML text). Text figures diff readably, reproduce from the repo with no
dependency, and render inline on GitHub — a better fit for this codebase than a
binary PNG blob and a heavyweight import.

Two figures:

*Reliability diagram.* Predicted probability (x) versus empirical outcome rate (y)
over the model's calibration bins, with the ``y = x`` perfect-calibration diagonal
and one point per non-empty bin (radius scaled by how many states fall in it).
Points hugging the diagonal mean the probabilities are honest, not merely ranked.

*Win-probability trajectories.* P(home win) across a few representative 2025 games
— a close late game, a blowout, a lead-change — showing how the forecast tracks
the game. The 0.5 reference line separates "home favored" from "away favored"; the
endpoint marker is filled when the home team actually won.

The SVG rendering is pure (a function of a reliability table / a list of
trajectories). Trajectory extraction takes an injected ``predict_fn`` so the
plotting is decoupled from any particular model — the runner passes the serialized
Phase-2 logistic, the tests pass a trivial forecaster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data/winprob")
DEFAULT_FIGURE_DIR = Path("figures")
PARQUET_NAME = "fct_game_states.parquet"
MODEL_JSON_NAME = "logistic_model.json"
METRICS_JSON_NAME = "winprob_metrics.json"

# Palette chosen to read on both light and dark backgrounds; the figure carries
# its own light background so it is always legible when embedded.
_BG = "#ffffff"
_INK = "#1a1a1a"
_MUTED = "#8a8a8a"
_DIAGONAL = "#c0392b"
_POINT = "#1f77b4"
_CATEGORY_COLORS = {
    "close_late": "#1f77b4",
    "blowout": "#ff7f0e",
    "lead_change": "#2ca02c",
}
_FALLBACK_COLOR = "#6a5acd"


# --------------------------------------------------------------------------
# Tiny SVG element builders.
# --------------------------------------------------------------------------

def _attrs(**kwargs) -> str:
    """Render an attribute string, mapping `class_` -> `class` and skipping None."""
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


def _circle(cx, cy, r, fill, stroke=None, width=1.0, class_=None) -> str:
    a = _attrs(cx=_r(cx), cy=_r(cy), r=_r(r), fill=fill, stroke=stroke,
               stroke_width=width, class_=class_)
    return f"<circle {a} />"


def _polyline(points: list[tuple[float, float]], stroke, width=1.5, class_=None) -> str:
    pts = " ".join(f"{_r(x)},{_r(y)}" for x, y in points)
    a = _attrs(points=pts, fill="none", stroke=stroke, stroke_width=width, class_=class_)
    return f"<polyline {a} />"


def _text(x, y, s, fill=_INK, size=12, anchor="start", class_=None) -> str:
    a = _attrs(x=_r(x), y=_r(y), fill=fill, font_size=size, text_anchor=anchor,
               font_family="sans-serif", class_=class_)
    return f"<text {a}>{_esc(s)}</text>"


def _rect(x, y, w, h, fill, class_=None) -> str:
    a = _attrs(x=_r(x), y=_r(y), width=_r(w), height=_r(h), fill=fill, class_=class_)
    return f"<rect {a} />"


def _r(v: float) -> str:
    """Round a coordinate to 2 dp for compact, stable output."""
    return f"{float(v):.2f}"


def _esc(s: str) -> str:
    """Escape the XML special characters that can appear in a label."""
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _document(width: float, height: float, body: list[str], title: str) -> str:
    """Wrap body elements in an SVG root with a light background."""
    inner = "\n  ".join([_rect(0, 0, width, height, _BG)] + body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_r(width)} {_r(height)}" '
        f'width="{_r(width)}" height="{_r(height)}" role="img" aria-label="{_esc(title)}">\n  '
        f'{inner}\n</svg>'
    )


# --------------------------------------------------------------------------
# Reliability diagram.
# --------------------------------------------------------------------------

def reliability_diagram_svg(
    reliability_table: list[dict],
    title: str,
    width: float = 360.0,
    height: float = 360.0,
    margin: float = 44.0,
) -> str:
    """Render a calibration reliability diagram from a reliability table."""
    x0, y0 = margin, height - margin
    span = min(width - 2 * margin, height - 2 * margin)

    def px(v: float) -> float:
        return x0 + v * span

    def py(v: float) -> float:
        return y0 - v * span

    body: list[str] = []
    # Plot frame + axis ticks at 0, 0.5, 1.
    body.append(_rect(x0, y0 - span, span, span, "none"))
    for t in (0.0, 0.5, 1.0):
        body.append(_line(px(t), y0, px(t), y0 + 4, _MUTED))
        body.append(_text(px(t), y0 + 18, f"{t:.1f}", fill=_MUTED, anchor="middle"))
        body.append(_line(x0 - 4, py(t), x0, py(t), _MUTED))
        body.append(_text(x0 - 8, py(t) + 4, f"{t:.1f}", fill=_MUTED, anchor="end"))

    # Perfect-calibration diagonal.
    body.append(_line(px(0), py(0), px(1), py(1), _DIAGONAL, width=1.0,
                      dash="4 3", class_="diagonal"))

    # Reliability curve + points for non-empty bins.
    populated = [r for r in reliability_table if r.get("n", 0) > 0]
    curve = [(px(r["mean_predicted"]), py(r["empirical"])) for r in populated]
    if len(curve) >= 2:
        body.append(_polyline(curve, _POINT, width=1.5, class_="reliability-curve"))
    max_n = max((r["n"] for r in populated), default=1)
    for r in populated:
        radius = 3.0 + 5.0 * (r["n"] / max_n) ** 0.5
        body.append(_circle(px(r["mean_predicted"]), py(r["empirical"]), radius,
                            _POINT, stroke=_BG, width=1.0, class_="bin-point"))

    # Labels.
    body.append(_text(width / 2, 22, title, fill=_INK, size=14, anchor="middle"))
    body.append(_text(width / 2, height - 6, "predicted probability",
                      fill=_MUTED, anchor="middle"))
    rot = f"rotate(-90 14 {_r(height / 2)})"
    ylabel_attrs = _attrs(x=14, y=_r(height / 2), fill=_MUTED, font_size=12,
                          text_anchor="middle", font_family="sans-serif", transform=rot)
    body.append(f"<text {ylabel_attrs}>empirical rate</text>")
    return _document(width, height, body, title)


# --------------------------------------------------------------------------
# Win-probability trajectories.
# --------------------------------------------------------------------------

def trajectory_svg(
    trajectories: list[dict],
    title: str,
    width: float = 640.0,
    height: float = 320.0,
    margin: float = 46.0,
) -> str:
    """Render P(home win) trajectories over game progress for several games."""
    x0, y0 = margin, height - margin
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def px(x: float) -> float:
        return x0 + x * plot_w

    def py(p: float) -> float:
        return y0 - p * plot_h

    body: list[str] = []
    body.append(_rect(x0, y0 - plot_h, plot_w, plot_h, "none"))
    for t in (0.0, 0.5, 1.0):
        body.append(_line(x0 - 4, py(t), x0, py(t), _MUTED))
        body.append(_text(x0 - 8, py(t) + 4, f"{t:.1f}", fill=_MUTED, anchor="end"))
    # 0.5 coin-flip reference.
    body.append(_line(px(0), py(0.5), px(1), py(0.5), _MUTED, width=1.0,
                      dash="4 3", class_="reference-line"))

    for i, traj in enumerate(trajectories):
        color = _CATEGORY_COLORS.get(traj.get("category"), _FALLBACK_COLOR)
        pts = [(px(pt["x"]), py(pt["p"])) for pt in traj["points"]]
        body.append(_polyline(pts, color, width=1.6, class_="trajectory"))
        end_x, end_y = pts[-1]
        fill = color if traj.get("home_win") else _BG
        body.append(_circle(end_x, end_y, 4.0, fill, stroke=color, width=1.5,
                            class_="endpoint"))
        # Legend row.
        ly = 30 + i * 16
        body.append(_line(width - 150, ly - 4, width - 132, ly - 4, color, width=2.0))
        body.append(_text(width - 128, ly, traj.get("category", traj["game_id"]),
                          fill=_INK, size=11))

    body.append(_text(width / 2, 22, title, fill=_INK, size=14, anchor="middle"))
    body.append(_text(width / 2, height - 6, "game progress ->",
                      fill=_MUTED, anchor="middle"))
    return _document(width, height, body, title)


# --------------------------------------------------------------------------
# Representative game selection + trajectory extraction.
# --------------------------------------------------------------------------

def _game_summaries(test: pd.DataFrame) -> pd.DataFrame:
    """Per-game final margin and lead extremes, ordered by game_id."""
    grouped = test.sort_values(["game_id", "period", "possession_number"]).groupby(
        "game_id", sort=True
    )
    return pd.DataFrame({
        "final_margin": grouped["home_score_differential"].last(),
        "max_margin": grouped["home_score_differential"].max(),
        "min_margin": grouped["home_score_differential"].min(),
        "n_states": grouped.size(),
    })


def select_trajectory_games(
    df: pd.DataFrame, close_final: float = 3.0, blowout_final: float = 20.0,
    lead: float = 5.0,
) -> list[dict]:
    """Pick one close-late, one blowout, and one lead-change 2025 game.

    Deterministic: within each category the lowest ``game_id`` is chosen, and a
    game already picked for an earlier category is not reused. Categories with no
    qualifying game are omitted rather than forced.
    """
    test = df.loc[df["split"] == "test"]
    summ = _game_summaries(test)
    chosen: list[dict] = []
    used: set[str] = set()

    def take(mask, category: str) -> None:
        candidates = [g for g in summ.index[mask] if g not in used]
        if candidates:
            gid = sorted(candidates)[0]
            used.add(gid)
            chosen.append({"game_id": gid, "category": category})

    take(summ["final_margin"].abs() <= close_final, "close_late")
    take(summ["final_margin"].abs() >= blowout_final, "blowout")
    take((summ["max_margin"] >= lead) & (summ["min_margin"] <= -lead), "lead_change")
    return chosen


def build_trajectories(df: pd.DataFrame, predict_fn, selected: list[dict]) -> list[dict]:
    """Score each selected game state-by-state into an (x, p) trajectory.

    ``x`` is normalized elapsed game time in [0, 1] (monotone through overtime);
    ``p`` is ``predict_fn(sub)`` for the game's states in possession order. Pure
    with respect to ``df``.
    """
    out: list[dict] = []
    for item in selected:
        gid = item["game_id"]
        sub = df.loc[df["game_id"] == gid].sort_values(
            ["period", "possession_number"]
        ).reset_index(drop=True)
        elapsed = sub["elapsed_game_seconds"].to_numpy(dtype=float)
        span = elapsed[-1] - elapsed[0]
        xs = (elapsed - elapsed[0]) / span if span > 0 else np.linspace(0.0, 1.0, len(sub))
        ps = np.asarray(predict_fn(sub), dtype=float)
        out.append({
            "game_id": gid,
            "category": item["category"],
            "home_win": bool(sub["home_win"].iloc[0]),
            "points": [{"x": float(x), "p": float(p)} for x, p in zip(xs, ps)],
        })
    return out


# --------------------------------------------------------------------------
# Runner.
# --------------------------------------------------------------------------

def run(
    data_dir: Path = DEFAULT_DATA_DIR, figure_dir: Path = DEFAULT_FIGURE_DIR
) -> dict[str, Path]:
    """Render both figures from the pinned mart, model, and metrics artifacts."""
    from winprob import evaluate  # local import keeps figures dependency-light

    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((data_dir / METRICS_JSON_NAME).read_text())
    reliability_table = metrics["calibration"]["reliability_table"]
    reliability_svg = reliability_diagram_svg(
        reliability_table, title="Win-probability reliability (held-out 2025)"
    )

    df = pd.read_parquet(data_dir / PARQUET_NAME)
    model_doc = json.loads((data_dir / MODEL_JSON_NAME).read_text())
    selected = select_trajectory_games(df)
    trajs = build_trajectories(
        df, lambda sub: evaluate.predict_with_model(sub, model_doc), selected
    )
    traj_svg = trajectory_svg(trajs, title="Win-probability trajectories (2025 test games)")

    reliability_path = figure_dir / "reliability.svg"
    traj_path = figure_dir / "trajectories.svg"
    reliability_path.write_text(reliability_svg + "\n")
    traj_path.write_text(traj_svg + "\n")
    return {"reliability": reliability_path, "trajectories": traj_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Phase-5 SVG figures")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()
    paths = run(Path(args.data_dir), Path(args.figure_dir))
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
