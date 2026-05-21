"""Pure linear-algebra primitives for the RAPM ridge baseline.

These are the warehouse-free building blocks the Phase-2 ridge fit stands on:
a selective-penalty normal-equation solver, the helpers that shape its penalty
diagonal and intercept column, and the game-margin retrodiction aggregation the
fit is scored against. Nothing here touches DuckDB, `data/rapm/` artifacts, or
sklearn — arrays in, arrays out — so every function is independently importable
and unit-testable on tiny synthetic inputs.

Why a *selective* penalty. Ridge shrinks coefficients toward zero, which is what
we want for player-season columns (fringe players should regress to the mean)
but not for the structural terms — the home-court effect and the global
intercept carry real, unshrunk signal. Rather than special-casing them, the
solver takes a full per-column penalty vector `d` and forms
`(XᵀX + diag(d)) beta = Xᵀy`; setting `d_j = 0` leaves column `j` at its
ordinary-least-squares value while `d_j = lambda` shrinks the rest.

Why the normal equations + Cholesky. The design is tall and sparse (millions of
possessions, thousands of columns); `XᵀX` is a small dense p×p matrix that fits
in memory, and once the ridge penalty makes it symmetric positive definite,
`cho_factor` / `cho_solve` is the fast, numerically stable way to solve it.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


def penalty_diagonal(penalized: np.ndarray, lam: float) -> np.ndarray:
    """Build the per-column ridge penalty vector `d`.

    `penalized` is a boolean mask aligned to the columns of the design; penalized
    columns (player-seasons, replacement) get `lam`, unpenalized structural
    columns (home, intercept) get `0.0` so they solve at their OLS value. Pure:
    returns a fresh array and never mutates the mask.
    """
    mask = np.asarray(penalized, dtype=bool)
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return np.where(mask, float(lam), 0.0)


def append_intercept(X):
    """Append a single all-ones column to `X`, returning the augmented matrix.

    The intercept is a global offset (mean points-per-possession); it must stay
    *unpenalized*, which the caller enforces by marking its column `False` in the
    penalty mask — this helper only adds the column. Sparse in -> sparse (CSR)
    out; dense in -> dense out. Pure: the input matrix is not modified.
    """
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    if sparse.issparse(X):
        return sparse.hstack([X, sparse.csr_matrix(ones)], format="csr")
    return np.hstack([np.asarray(X, dtype=np.float64), ones])


def solve_ridge(X, y: np.ndarray, penalty_diag: np.ndarray) -> np.ndarray:
    """Solve the selective-penalty ridge normal equations for `beta`.

    Forms the dense normal matrix `A = XᵀX + diag(penalty_diag)` and the
    right-hand side `b = Xᵀy`, then solves `A beta = b` by Cholesky factorization
    (`cho_factor` / `cho_solve`). `A` must be symmetric positive definite, which
    holds when either every column is penalized or the unpenalized columns are
    linearly independent within the design (full column rank).

    `X` may be a scipy CSR matrix or a dense array of shape (n, p); `y` is length
    n; `penalty_diag` is length p (see `penalty_diagonal`). Returns `beta` of
    length p in points-per-possession units. Pure: no inputs are mutated and no
    artifacts are read or written.
    """
    y = np.asarray(y, dtype=np.float64)
    d = np.asarray(penalty_diag, dtype=np.float64)

    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    if d.shape != (p,):
        raise ValueError(f"penalty_diag must have shape ({p},), got {d.shape}")

    # XᵀX is small (p×p) even when X has millions of rows; densify so we can
    # Cholesky-factor it. `@` on CSR yields a sparse product, hence .toarray().
    gram = X.T @ X
    gram = gram.toarray() if sparse.issparse(gram) else np.asarray(gram, dtype=np.float64)

    # New array (no in-place +=): keeps `gram` and the caller's inputs untouched.
    normal = gram + np.diag(d)

    rhs = X.T @ y
    rhs = np.asarray(rhs, dtype=np.float64).ravel()

    factor = cho_factor(normal, lower=True)
    return cho_solve(factor, rhs)


def game_margins(
    values: np.ndarray,
    game_id: np.ndarray,
    offense_is_home: np.ndarray,
) -> dict:
    """Aggregate per-possession offense points into per-game predicted margins.

    For each game g, the margin is the home team's points minus the away team's:
    summing `values` (predicted offense points-per-possession `yhat_i`) over the
    possessions where the offense is the home team, minus the sum over
    possessions where the offense is the away team —

        Mhat_g = sum_{i in g, offense home} yhat_i  -  sum_{i in g, offense away} yhat_i

    Because every possession's offense is exactly one team, this nets the two
    teams' scored points into the game margin. Passing the *observed* response
    `y` instead of `yhat` yields the actual margins, so the same function scores
    both sides of the retrodiction. Returns a dict keyed by game id (in
    first-seen order) so predicted and actual margins align on the same keys.
    Pure: inputs are not mutated.
    """
    values = np.asarray(values, dtype=np.float64)
    game_id = np.asarray(game_id)
    offense_is_home = np.asarray(offense_is_home, dtype=bool)

    n = len(values)
    if game_id.shape != (n,) or offense_is_home.shape != (n,):
        raise ValueError(
            "values, game_id, offense_is_home must be the same length; got "
            f"{values.shape}, {game_id.shape}, {offense_is_home.shape}"
        )

    # +yhat when the offense is home, -yhat when the offense is away; grouping by
    # game and summing these signed contributions gives the home-minus-away margin.
    signed = np.where(offense_is_home, values, -values)

    margins: dict = {}
    for g, s in zip(game_id.tolist(), signed.tolist()):
        margins[g] = margins.get(g, 0.0) + s
    return margins


def margin_rmse(predicted: dict, actual: dict) -> float:
    """Root-mean-square error between predicted and actual per-game margins.

    Both arguments are `game_margins` outputs (game id -> margin). The error is
    taken over the games present in both dicts, so a predicted-only or
    actual-only game is ignored rather than silently treated as zero. Pure.
    """
    keys = sorted(set(predicted) & set(actual))
    if not keys:
        raise ValueError("no shared games between predicted and actual margins")
    diffs = np.array([predicted[g] - actual[g] for g in keys], dtype=np.float64)
    return float(np.sqrt(np.mean(diffs**2)))


# ==========================================================================
# Phase-2 pipeline: load `data/rapm/` artifacts, select lambda by grouped CV,
# refit, and score three retrodiction methods on the held-out test fold.
#
# Everything below *uses* the pure primitives above and never reaches into
# their internals — arrays flow in through `solve_ridge` / `game_margins` /
# `margin_rmse`, results flow out. The only impurities live in `load_artifacts`
# (reads `data/rapm/`), the optional warehouse helpers, and `main`.
# ==========================================================================

DEFAULT_DATA_DIR = Path("data/rapm")
DEFAULT_WAREHOUSE = "warehouse/courtiq.duckdb"

# fold.npy holds ints 0..4; whole games share a fold. Fold 4 is the held-out
# test set; folds 0-3 are the training corpus and double as the inner CV folds.
TEST_FOLD = 4
INNER_FOLDS = (0, 1, 2, 3)

# beta is in points-per-possession; ratings are conventionally per-100.
RATING_SCALE = 100.0

# Lambda search grid. RAPM at the possession grain with thousands of penalized
# player-season columns wants substantial shrinkage, so the grid is centered on
# large lambdas. The chosen optimum MUST land in the interior — if it pins to an
# endpoint the grid is too narrow and `main` warns to widen it (see the failure
# playbook in this feature's spec).
LAMBDA_GRID = np.logspace(1.0, 6.0, 16)


@dataclass(frozen=True)
class Artifacts:
    """The materialized `data/rapm/` design, one field per on-disk artifact."""

    X: sparse.csr_matrix          # (n, p) 0/1 design, NO intercept column yet
    y: np.ndarray                 # (n,) offense points per possession
    columns: list[dict]           # length p, parsed columns.jsonl rows
    fold: np.ndarray              # (n,) int 0..4, whole games share a fold
    row_game_id: np.ndarray       # (n,) game id per possession (object)
    row_season: np.ndarray        # (n,) season per possession
    row_offense_is_home: np.ndarray  # (n,) bool: offense team is the home team
    manifest: dict                # corpus_hash, n_possessions, n_columns, ...


def load_artifacts(data_dir=DEFAULT_DATA_DIR) -> Artifacts:
    """Load the design produced by `rapm.design` and validate row/column shapes.

    Fails fast (ValueError) if any artifact's length disagrees with X's shape,
    so a stale or partially rebuilt `data/rapm/` is caught before it silently
    misaligns predictions and game ids.
    """
    d = Path(data_dir)
    X = sparse.load_npz(d / "X.npz").tocsr()
    y = np.load(d / "y.npy")
    columns = [
        json.loads(line)
        for line in (d / "columns.jsonl").read_text().splitlines()
        if line.strip()
    ]
    fold = np.load(d / "fold.npy")
    row_game_id = np.load(d / "row_game_id.npy", allow_pickle=True)
    row_season = np.load(d / "row_season.npy")
    row_offense_is_home = np.load(d / "row_offense_is_home.npy")
    manifest = json.loads((d / "manifest.json").read_text())

    n, p = X.shape
    lengths = {
        "y": len(y),
        "fold": len(fold),
        "row_game_id": len(row_game_id),
        "row_season": len(row_season),
        "row_offense_is_home": len(row_offense_is_home),
    }
    bad = {k: v for k, v in lengths.items() if v != n}
    if bad:
        raise ValueError(f"artifacts misaligned with X rows (n={n}): {bad}")
    if len(columns) != p:
        raise ValueError(f"columns.jsonl has {len(columns)} rows, X has {p} columns")

    return Artifacts(
        X=X,
        y=np.asarray(y, dtype=np.float64),
        columns=columns,
        fold=np.asarray(fold),
        row_game_id=row_game_id,
        row_season=row_season,
        row_offense_is_home=np.asarray(row_offense_is_home, dtype=bool),
        manifest=manifest,
    )


def penalized_mask(columns: list[dict]) -> np.ndarray:
    """Boolean penalty mask over the (intercept-free) columns.

    The `penalized` flag was decided at design time — True for player-season and
    replacement columns, False for the structural home term. Reused as-is so the
    penalty policy has a single source of truth.
    """
    return np.array([bool(c["penalized"]) for c in columns], dtype=bool)


def ridge_predict(X, beta: np.ndarray) -> np.ndarray:
    """Per-possession prediction `yhat_i = x_i . beta` (X may be sparse)."""
    return np.asarray(X @ np.asarray(beta, dtype=np.float64)).ravel()


def retrodiction_rmse(
    yhat: np.ndarray,
    y: np.ndarray,
    game_id: np.ndarray,
    offense_is_home: np.ndarray,
) -> float:
    """Game-margin retrodiction RMSE — the one metric every method is scored by.

    Aggregates predicted per-possession offense points into per-game home-minus-
    away margins, does the same for the observed `y`, and RMSEs over the shared
    games. Identical for ridge, raw plus-minus, and team-net so the comparison is
    apples-to-apples; a thin wrapper over the pure `game_margins`/`margin_rmse`.
    """
    predicted = game_margins(yhat, game_id, offense_is_home)
    actual = game_margins(y, game_id, offense_is_home)
    return margin_rmse(predicted, actual)


def is_interior_optimum(best_idx: int, grid_len: int) -> bool:
    """True when the argmin sits strictly inside the grid (not on an endpoint).

    Extracted and tested on its own because an endpoint optimum is the first
    thing to check when the ridge-beats-rawPM gate fails: it means the lambda
    grid is too narrow and the real minimum is off the edge.
    """
    return 0 < best_idx < grid_len - 1


def cv_select_lambda(
    X,
    y: np.ndarray,
    fold: np.ndarray,
    game_id: np.ndarray,
    offense_is_home: np.ndarray,
    penalty_mask: np.ndarray,
    grid: np.ndarray = LAMBDA_GRID,
    inner_folds: tuple[int, ...] = INNER_FOLDS,
) -> tuple[float, list[float], int]:
    """Grouped-CV lambda selection *within* the training folds.

    For each lambda and each inner fold `f`, fit on the other inner folds
    (training rows with fold not in {f, TEST_FOLD}) and score game-margin
    retrodiction RMSE on the games in fold `f`; average across the inner folds.
    Returns (best_lambda, mean-RMSE curve aligned to `grid`, best index). The
    test fold never enters any fit or score here.

    `X` already carries the intercept column and `penalty_mask` is aligned to it
    (intercept + home unpenalized); each candidate reuses the pure `solve_ridge`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    fold = np.asarray(fold)
    curve: list[float] = []
    # Precompute per-fold row masks once; reused across every lambda.
    eval_masks = {f: fold == f for f in inner_folds}
    train_masks = {
        f: np.isin(fold, [k for k in inner_folds if k != f]) for f in inner_folds
    }

    for lam in grid:
        diag = penalty_diagonal(penalty_mask, lam)
        fold_rmses = []
        for f in inner_folds:
            tr, ev = train_masks[f], eval_masks[f]
            beta = solve_ridge(X[tr], y[tr], diag)
            yhat = ridge_predict(X[ev], beta)
            fold_rmses.append(
                retrodiction_rmse(yhat, y[ev], game_id[ev], offense_is_home[ev])
            )
        curve.append(float(np.mean(fold_rmses)))

    best_idx = int(np.argmin(curve))
    return float(grid[best_idx]), curve, best_idx


def raw_plus_minus_predict(
    X,
    y: np.ndarray,
    columns: list[dict],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    offense_is_home: np.ndarray,
) -> np.ndarray:
    """Unadjusted raw plus-minus baseline — THE GATE ridge must beat.

    Fit on training rows: mu = mean(y); for each offense (O) column the effect is
    mean(y over training possessions containing that player) - mu, and likewise
    for each defense (D) column; the home effect h = mean(y | home on offense) -
    mean(y | away on offense). Prediction for possession i is

        yhat_i = mu + sum_{p in off_i} a_p + sum_{q in def_i} d_q + (+h/2 home | -h/2 away)

    The design's O/D columns already encode per-possession membership, so the
    player sums are just `X_eval @ effects` (the home column's effect is zeroed
    and the home term added explicitly). Columns unseen in training contribute 0.
    No co-floor adjustment — this is deliberately the naive baseline.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    offense_is_home = np.asarray(offense_is_home, dtype=bool)

    Xtr = X[train_mask]
    ytr = y[train_mask]
    mu = float(np.mean(ytr))

    # Indicator (1 per active column, collapsing any pooled-replacement doubles):
    # a possession "contains" a column iff it has a nonzero entry there.
    indicator = Xtr.copy()
    indicator.data = np.ones_like(indicator.data)
    counts = np.asarray(indicator.sum(axis=0)).ravel()
    ysum = np.asarray(indicator.T @ ytr).ravel()

    sides = [c["side"] for c in columns]
    is_od = np.array([s in ("O", "D") for s in sides], dtype=bool)
    seen = counts > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        col_mean = ysum / np.where(counts > 0, counts, 1.0)
    effect = np.zeros(len(columns), dtype=np.float64)
    keep = is_od & seen
    effect[keep] = col_mean[keep] - mu

    home_tr = train_mask & offense_is_home
    away_tr = train_mask & ~offense_is_home
    h = float(np.mean(y[home_tr]) - np.mean(y[away_tr]))

    # Prediction uses the raw design (not the indicator): two pooled-replacement
    # players in one possession legitimately sum their shared effect twice.
    Xev = X[eval_mask]
    base = ridge_predict(Xev, effect)
    home_term = np.where(offense_is_home[eval_mask], h / 2.0, -h / 2.0)
    return mu + base + home_term


def team_net_predict(
    offense_team_id: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> np.ndarray:
    """Team net-rating baseline (report only): predict a team's training off-ppp.

    For every possession, yhat = the offense team's mean training offensive
    points-per-possession; teams unseen in training fall back to the global
    training mean.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    tr_teams = np.asarray(offense_team_id)[train_mask]
    tr_y = np.asarray(y)[train_mask]
    global_ppp = float(np.mean(tr_y))
    ppp = {
        t: float(np.mean(tr_y[tr_teams == t])) for t in np.unique(tr_teams)
    }
    return np.array(
        [ppp.get(t, global_ppp) for t in np.asarray(offense_team_id)[eval_mask]],
        dtype=np.float64,
    )


def extract_ratings(columns: list[dict], beta: np.ndarray, X) -> list[dict]:
    """Per (player, season) ratings from a fitted beta, in per-100 units.

    Pairs each player-season's O and D columns: off_rating = 100*beta[O],
    def_rating = 100*beta[D], net_rating = off - def. `n_possessions` is the
    player-season's total floor time in the corpus (rows active in its O column
    plus rows active in its D column). Replacement columns are structural and not
    emitted as ratings.
    """
    beta = np.asarray(beta, dtype=np.float64)
    active = np.asarray((X != 0).sum(axis=0)).ravel()  # per-column floor count

    pairs: dict[tuple, dict] = {}
    for i, c in enumerate(columns):
        if c["kind"] != "player":
            continue
        pairs.setdefault((c["player_id"], c["season"]), {})[c["side"]] = i

    ratings: list[dict] = []
    for (player_id, season), sides in pairs.items():
        o, dcol = sides.get("O"), sides.get("D")
        if o is None or dcol is None:
            continue  # a player-season needs both an offense and a defense column
        off = RATING_SCALE * beta[o]
        deff = RATING_SCALE * beta[dcol]
        ratings.append(
            {
                "player_id": player_id,
                "season": season,
                "off_rating": off,
                "def_rating": deff,
                "net_rating": off - deff,
                "n_possessions": int(active[o] + active[dcol]),
            }
        )
    return ratings


def _load_offense_team_id(
    warehouse: str, expected_offense_is_home: np.ndarray
) -> np.ndarray | None:
    """Optionally pull per-possession offense_team_id from the warehouse.

    Team ids are not in the `data/rapm/` artifacts, so the report-only team-net
    baseline needs the warehouse. Row order must match the design build's
    `order by game_id, period, possession_number`; we re-derive offense_is_home
    from the same join and require it to match the artifact array exactly before
    trusting the alignment. Returns None (never raises) if the warehouse is
    absent, the query fails, or the alignment check fails.
    """
    try:
        import duckdb

        con = duckdb.connect(warehouse, read_only=True)
        try:
            res = con.execute(
                """
                select
                    p.offense_team_id,
                    (p.offense_team_id = g.home_team_id) as offense_is_home
                from fct_possessions p
                join dim_games g using (game_id)
                order by p.game_id, p.period, p.possession_number
                """
            ).fetchnumpy()
        finally:
            con.close()
    except Exception as exc:  # warehouse missing / schema drift — degrade quietly
        print(f"  (team-net skipped: warehouse unavailable — {exc})")
        return None

    oid = np.asarray(res["offense_team_id"])
    oih = np.asarray(res["offense_is_home"], dtype=bool)
    expected = np.asarray(expected_offense_is_home, dtype=bool)
    if len(oid) != len(expected) or not np.array_equal(oih, expected):
        print("  (team-net skipped: warehouse rows do not align with artifacts)")
        return None
    return oid


def _load_player_names(warehouse: str) -> dict[int, str]:
    """Optional person_id -> name map for face-validity printing. Never raises."""
    try:
        import duckdb

        con = duckdb.connect(warehouse, read_only=True)
        try:
            rows = con.execute(
                "select distinct person_id, player_name from stg_box_player"
            ).fetchall()
        finally:
            con.close()
        return {int(pid): name for pid, name in rows if pid is not None}
    except Exception:
        return {}


def _print_report(
    metrics: dict, ratings: list[dict], names: dict[int, str]
) -> None:
    """Print the retrodiction table and top/bottom-10 net-rating player-seasons."""
    print("\nRetrodiction (game-margin RMSE on test fold 4, lower is better)")
    print(f"  {'ridge':<12} {metrics['test_rmse_ridge']:.4f}")
    print(f"  {'raw +/-':<12} {metrics['test_rmse_rawpm']:.4f}  (the gate)")
    teamnet = metrics["test_rmse_teamnet"]
    print(f"  {'team-net':<12} " + (f"{teamnet:.4f}" if teamnet is not None else "n/a"))

    gate_ok = metrics["test_rmse_ridge"] < metrics["test_rmse_rawpm"]
    verdict = "PASS: ridge beats raw +/-" if gate_ok else "FAIL: ridge does NOT beat raw +/-"
    print(f"  -> {verdict}")

    def label(r: dict) -> str:
        who = names.get(int(r["player_id"]), str(r["player_id"]))
        return f"{who} ({r['season']})"

    ranked = sorted(ratings, key=lambda r: r["net_rating"], reverse=True)
    print("\nTop 10 net rating (per 100)")
    for r in ranked[:10]:
        print(f"  {r['net_rating']:+7.2f}  {label(r):<28} n={r['n_possessions']}")
    print("Bottom 10 net rating (per 100)")
    for r in ranked[-10:]:
        print(f"  {r['net_rating']:+7.2f}  {label(r):<28} n={r['n_possessions']}")


def run_pipeline(data_dir=DEFAULT_DATA_DIR, warehouse: str = DEFAULT_WAREHOUSE) -> dict:
    """Drive the full evaluation and write the two output artifacts.

    Loads the design, selects lambda by grouped CV within the training folds,
    refits on all training rows (folds 0-3) at lambda*, scores ridge / raw
    plus-minus / team-net on test fold 4, writes ridge_ratings.parquet and
    ridge_metrics.json, and returns the metrics dict.
    """
    import pandas as pd

    art = load_artifacts(data_dir)
    print(
        f"Loaded design: {art.X.shape[0]:,} possessions x {art.X.shape[1]:,} columns "
        f"(corpus {art.manifest.get('corpus_hash')})"
    )

    # Intercept: one unpenalized ones-column appended; extend the mask with False.
    Xi = append_intercept(art.X)
    mask_i = np.append(penalized_mask(art.columns), False)

    test_mask = art.fold == TEST_FOLD
    train_mask = np.isin(art.fold, INNER_FOLDS)

    print(f"Selecting lambda over {len(LAMBDA_GRID)} candidates by 4-fold grouped CV ...")
    best_lambda, curve, best_idx = cv_select_lambda(
        Xi, art.y, art.fold, art.row_game_id, art.row_offense_is_home, mask_i
    )
    if not is_interior_optimum(best_idx, len(LAMBDA_GRID)):
        warnings.warn(
            f"CV optimum at grid endpoint (lambda={best_lambda:g}, index {best_idx}); "
            "widen LAMBDA_GRID so the minimum is interior.",
            stacklevel=2,
        )
    print(f"  lambda* = {best_lambda:g}  (CV margin RMSE {curve[best_idx]:.4f})")

    # Refit on ALL training rows at lambda*, then score once on the test fold.
    beta = solve_ridge(Xi[train_mask], art.y[train_mask], penalty_diagonal(mask_i, best_lambda))

    def score(yhat: np.ndarray) -> float:
        return retrodiction_rmse(
            yhat, art.y[test_mask], art.row_game_id[test_mask],
            art.row_offense_is_home[test_mask],
        )

    test_rmse_ridge = score(ridge_predict(Xi[test_mask], beta))
    test_rmse_rawpm = score(
        raw_plus_minus_predict(
            art.X, art.y, art.columns, train_mask, test_mask, art.row_offense_is_home
        )
    )

    test_rmse_teamnet = None
    offense_team_id = _load_offense_team_id(warehouse, art.row_offense_is_home)
    if offense_team_id is not None:
        test_rmse_teamnet = score(
            team_net_predict(offense_team_id, art.y, train_mask, test_mask)
        )

    ratings = extract_ratings(art.columns, beta, art.X)
    n_test_games = len(set(art.row_game_id[test_mask].tolist()))

    metrics = {
        "chosen_lambda": best_lambda,
        "lambda_grid": LAMBDA_GRID.tolist(),
        "cv_rmse_curve": curve,
        "test_rmse_ridge": test_rmse_ridge,
        "test_rmse_rawpm": test_rmse_rawpm,
        "test_rmse_teamnet": test_rmse_teamnet,
        "n_test_games": n_test_games,
        "corpus_hash": art.manifest.get("corpus_hash"),
    }

    out = Path(data_dir)
    pd.DataFrame(
        ratings,
        columns=[
            "player_id", "season", "off_rating", "def_rating",
            "net_rating", "n_possessions",
        ],
    ).to_parquet(out / "ridge_ratings.parquet", index=False)
    (out / "ridge_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}/ridge_ratings.parquet and ridge_metrics.json")

    _print_report(metrics, ratings, _load_player_names(warehouse))
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit the RAPM ridge baseline and run the retrodiction gate."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir, warehouse=args.warehouse)


if __name__ == "__main__":
    main()
