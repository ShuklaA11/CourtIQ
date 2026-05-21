"""Exact Gaussian-posterior RAPM: ridge reinterpreted as a conjugate model.

The Phase-2 ridge fit is not just *a* point estimate — it is the posterior mean
(MAP) of a fully conjugate Gaussian model, and that model has a closed-form
posterior we can read off without any MCMC or variational approximation. The
generative story:

    y_i = x_i . beta + eps_i,        eps_i ~ N(0, sigma^2)
    beta_j ~ N(0, tau^2)             on PENALIZED columns (player-seasons)
    beta_j ~ flat                    on UNPENALIZED columns (home, intercept)

Completing the square gives the posterior exactly:

    A     = X^T X + lambda * D       (D = 1 on penalized cols, 0 otherwise)
    mu    = A^{-1} X^T y             (== the ridge solution at lambda = sigma^2/tau^2)
    Sigma = sigma^2 * A^{-1}

So a single Cholesky factor of A yields *everything*: the mean (`cho_solve` on
X^T y), every marginal SD (`sqrt(sigma^2 * diag(A^{-1}))`), any game's predictive
variance (a quadratic form `w^T A^{-1} w`), and each SBC re-solve (one more
`cho_solve`). We factor A ONCE and reuse it.

`sigma^2` is the genuine training residual variance and `tau^2 = sigma^2/lambda`
falls out of the ridge identity — neither is tuned to hit a coverage target. The
pure core below is arrays-in / arrays-out and warehouse-free; `run()` and
`main()` own all disk and DuckDB I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from rapm.ridge import (
    DEFAULT_DATA_DIR,
    DEFAULT_WAREHOUSE,
    INNER_FOLDS,
    RATING_SCALE,
    TEST_FOLD,
    _load_player_names,
    append_intercept,
    game_margins,
    load_artifacts,
    margin_rmse,
    penalized_mask,
    penalty_diagonal,
)

# 90% two-sided normal quantile. Fixed by the spec (not 1.6448...) so intervals
# and coverage checks agree to the reported digits.
Z90 = 1.645

# Number of SBC replications. The pooled statistic averages over K datasets x
# thousands of penalized coefficients, so even K=40 gives a very tight estimate.
DEFAULT_SBC_DATASETS = 40
SBC_BASE_SEED = 20240724

# Recency half-life (days) for the report-only time-decay variant.
RECENCY_HALF_LIFE_DAYS = 60.0


# ==========================================================================
# Pure core — no disk, no warehouse. Arrays / Cholesky factors in and out.
# ==========================================================================


def build_normal_matrix(X, penalty_diag: np.ndarray) -> np.ndarray:
    """Assemble the posterior precision `A = X^T X + diag(penalty_diag)`.

    `X` may be sparse CSR or dense (n, p); `penalty_diag` is length p with the
    ridge penalty `lambda` on penalized columns and `0.0` on the unpenalized
    structural columns (home, intercept) — the SAME `D` the Phase-2 ridge uses,
    so `A`'s inverse is exactly the (unscaled) posterior covariance. Pure: a
    fresh dense array is returned and no input is mutated.
    """
    d = np.asarray(penalty_diag, dtype=np.float64)
    if d.shape != (X.shape[1],):
        raise ValueError(f"penalty_diag must be length {X.shape[1]}, got {d.shape}")
    gram = X.T @ X
    gram = gram.toarray() if sparse.issparse(gram) else np.asarray(gram, dtype=np.float64)
    return gram + np.diag(d)


def posterior_mean(factor: tuple, X, y: np.ndarray) -> np.ndarray:
    """Posterior mean `mu = A^{-1} X^T y` via one back-substitution.

    `factor` is the `cho_factor(A)` tuple; reusing it makes this a cheap solve.
    Equals `rapm.ridge.solve_ridge(X, y, penalty_diag)` at the same lambda by
    construction — the ridge point estimate IS this posterior mean. Pure.
    """
    rhs = np.asarray(X.T @ np.asarray(y, dtype=np.float64)).ravel()
    return cho_solve(factor, rhs)


def residual_variance(X, y: np.ndarray, mu: np.ndarray) -> float:
    """Noise variance `sigma^2 = ||y - X mu||^2 / n` (training residual variance).

    The genuine MLE-style residual variance on the training rows — reported, and
    never inflated to widen intervals toward a coverage target. Uses `n` (not
    `n - p`) as the spec prescribes. Pure.
    """
    resid = np.asarray(y, dtype=np.float64) - np.asarray(X @ mu).ravel()
    return float(resid @ resid / len(resid))


def inverse_from_factor(factor: tuple, p: int) -> np.ndarray:
    """Dense `A^{-1}` from its Cholesky factor (solve against the identity).

    `sigma^2 * A^{-1}` is the full posterior covariance; its diagonal gives every
    coefficient SD and its O/D blocks give net-rating SDs (which need the O-D
    covariance term). Pure.
    """
    return cho_solve(factor, np.eye(p))


def predictive_intervals(
    factor: tuple,
    weights: np.ndarray,
    mu: np.ndarray,
    sigma2: float,
    n_possessions: np.ndarray,
    z: float = Z90,
    noise_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior-predictive margin, variance, and interval for game weight rows.

    Each row `w_g` of `weights` is a game's signed possession sum, so the margin
    is `w_g . mu` and its predictive variance decomposes into parameter
    uncertainty `w_g^T Sigma w_g = sigma^2 (w_g^T A^{-1} w_g)` plus per-possession
    observation noise over the game's `n_g` possessions. The parameter term is
    ALWAYS scaled by the posterior `sigma^2` (since `Sigma = sigma^2 A^{-1}`); the
    noise term uses `noise_var` when supplied and otherwise falls back to the same
    `sigma^2` — this lets the caller pass the effective per-possession *margin*
    noise (see `effective_margin_noise`) without disturbing the posterior scale.
    One `cho_solve` handles all games at once (RHS = weights^T). Returns
    (means, var, lo, hi). Pure.
    """
    noise = sigma2 if noise_var is None else float(noise_var)
    means = np.asarray(weights @ mu).ravel()
    solved = cho_solve(factor, weights.T)  # A^{-1} w_g for every game g
    quad = np.einsum("ij,ij->j", weights.T, solved)  # w_g^T A^{-1} w_g
    var = sigma2 * quad + noise * np.asarray(n_possessions, dtype=np.float64)
    half = z * np.sqrt(np.maximum(var, 0.0))
    return means, var, means - half, means + half


def effective_margin_noise(
    factor: tuple,
    weights: np.ndarray,
    mu: np.ndarray,
    sigma2: float,
    actual_margins: np.ndarray,
    n_possessions: np.ndarray,
) -> float:
    """Method-of-moments per-possession noise that survives the signed margin sum.

    The generative model treats possessions as iid `N(0, sigma^2)`, so it predicts
    a game-margin noise of `sigma^2 * n_g`. Empirically that OVER-disperses the
    home-minus-away margin: within-game possessions are positively autocorrelated
    in raw points (shared pace / game state), so under the alternating signs the
    noise partly cancels and the margin varies less than `sigma^2 * n_g`. This
    estimates the noise that ACTUALLY survives — from the given (training) games,
    subtract the parameter-uncertainty mass `sum_g sigma^2 w_g^T A^{-1} w_g` from
    the total margin residual sum of squares and divide by total possessions. It
    is a plain variance-component estimate fit on the supplied games only (the
    caller passes TRAINING games, never the held-out fold) and is never nudged
    toward a coverage target. Pure.
    """
    resid = np.asarray(actual_margins, dtype=np.float64) - np.asarray(weights @ mu).ravel()
    solved = cho_solve(factor, weights.T)
    quad = np.einsum("ij,ij->j", weights.T, solved)  # w_g^T A^{-1} w_g
    noise_ss = float(resid @ resid) - sigma2 * float(quad.sum())
    return max(noise_ss, 0.0) / float(np.sum(n_possessions))


def game_weight_matrix(
    X, game_ids: np.ndarray, offense_is_home: np.ndarray
) -> tuple[np.ndarray, list, np.ndarray]:
    """Signed per-game possession sums `w_g = sum_{i in g} s_i x_i`.

    `s_i = +1` when the home team is on offense, `-1` otherwise, so `w_g . beta`
    is the home-minus-away expected margin (mirrors `rapm.ridge.game_margins`).
    Rows of the returned dense matrix follow first-seen game order; also returns
    that game-id order and each game's possession count `n_g`. Pure.
    """
    signed = np.where(np.asarray(offense_is_home, dtype=bool), 1.0, -1.0)
    signed_X = sparse.diags(signed) @ X
    order: list = []
    index: dict = {}
    for g in game_ids.tolist():
        if g not in index:
            index[g] = len(order)
            order.append(g)
    rows = np.array([index[g] for g in game_ids.tolist()])
    selector = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, np.arange(len(rows)))),
        shape=(len(order), X.shape[0]),
    )
    weights = np.asarray((selector @ signed_X).todense())
    n_g = np.asarray(selector.sum(axis=1)).ravel()
    return weights, order, n_g


def coverage_fraction(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of `actual` values falling within their `[lo, hi]` intervals."""
    a = np.asarray(actual, dtype=np.float64)
    return float(np.mean((a >= lo) & (a <= hi)))


def sbc_trial(
    factor: tuple,
    X,
    mu: np.ndarray,
    penalized_idx: np.ndarray,
    sigma2: float,
    tau2: float,
    ainv_diag: np.ndarray,
    seed: int,
    z: float = Z90,
) -> np.ndarray:
    """One simulation-based-calibration replication over penalized coefficients.

    Plant a truth from the prior (`beta*_penalized ~ N(0, tau^2)`, unpenalized
    fixed at the fitted `mu`), simulate `y_sim = X beta* + N(0, sigma^2)`, re-solve
    the posterior mean `mu_sim` with the SAME factor, and test whether each
    planted `beta*_j` lands in its 90% credible interval
    `mu_sim_j +/- z * sqrt(sigma^2 * [A^{-1}]_jj)`. Returns a boolean cover array
    over the penalized coefficients. Pure given `seed`.
    """
    rng = np.random.default_rng(seed)
    beta_star = np.array(mu, dtype=np.float64)
    beta_star[penalized_idx] = rng.normal(0.0, np.sqrt(tau2), size=len(penalized_idx))
    y_sim = np.asarray(X @ beta_star).ravel() + rng.normal(
        0.0, np.sqrt(sigma2), size=X.shape[0]
    )
    mu_sim = cho_solve(factor, np.asarray(X.T @ y_sim).ravel())
    half = z * np.sqrt(sigma2 * ainv_diag[penalized_idx])
    truth = beta_star[penalized_idx]
    return (truth >= mu_sim[penalized_idx] - half) & (truth <= mu_sim[penalized_idx] + half)


def sbc_coverage(
    factor: tuple,
    X,
    mu: np.ndarray,
    penalized_idx: np.ndarray,
    sigma2: float,
    tau2: float,
    ainv_diag: np.ndarray,
    n_datasets: int = DEFAULT_SBC_DATASETS,
    base_seed: int = SBC_BASE_SEED,
    z: float = Z90,
) -> float:
    """Pooled planted-truth coverage across `n_datasets` SBC replications.

    Each dataset varies only by `base_seed + index` for reproducibility. Coverage
    is pooled over all datasets and all penalized coefficients; for the exact
    Gaussian posterior it must sit at ~0.90. Pure given the seeds.
    """
    covered = [
        sbc_trial(factor, X, mu, penalized_idx, sigma2, tau2, ainv_diag, base_seed + i, z)
        for i in range(n_datasets)
    ]
    return float(np.mean(np.concatenate(covered)))


def net_rating_variance(ainv: np.ndarray, o: int, d: int, sigma2: float) -> float:
    """Posterior variance of `net = off - def` for one player-season.

    `Var(beta_o - beta_d) = sigma^2 (A^{-1}_oo + A^{-1}_dd - 2 A^{-1}_od)`; the
    cross term is why net SDs need the full inverse, not just its diagonal. Pure.
    """
    return float(sigma2 * (ainv[o, o] + ainv[d, d] - 2.0 * ainv[o, d]))


def posterior_ratings(
    columns: list[dict],
    mu: np.ndarray,
    ainv: np.ndarray,
    sigma2: float,
    X,
    z: float = Z90,
) -> list[dict]:
    """Per player-season ratings with posterior SDs and net credible intervals.

    Pairs each player-season's O and D columns (same pairing as the ridge
    baseline), scales to per-100 units, and attaches `off_sd`/`def_sd`/`net_sd`
    from the posterior covariance plus a 90% net credible interval. `X` is the
    intercept-free design used only for the floor-time possession count. Pure.
    """
    active = np.asarray((X != 0).sum(axis=0)).ravel()
    pairs: dict[tuple, dict] = {}
    for i, c in enumerate(columns):
        if c["kind"] != "player":
            continue
        pairs.setdefault((c["player_id"], c["season"]), {})[c["side"]] = i

    ratings: list[dict] = []
    for (player_id, season), sides in pairs.items():
        o, d = sides.get("O"), sides.get("D")
        if o is None or d is None:
            continue
        off, deff = RATING_SCALE * mu[o], RATING_SCALE * mu[d]
        off_sd = RATING_SCALE * np.sqrt(max(sigma2 * ainv[o, o], 0.0))
        def_sd = RATING_SCALE * np.sqrt(max(sigma2 * ainv[d, d], 0.0))
        net_sd = RATING_SCALE * np.sqrt(max(net_rating_variance(ainv, o, d, sigma2), 0.0))
        net = off - deff
        ratings.append(
            {
                "player_id": player_id,
                "season": season,
                "off_rating": off,
                "def_rating": deff,
                "net_rating": net,
                "off_sd": off_sd,
                "def_sd": def_sd,
                "net_sd": net_sd,
                "net_ci_low": net - z * net_sd,
                "net_ci_high": net + z * net_sd,
                "n_possessions": int(active[o] + active[d]),
            }
        )
    return ratings


# ==========================================================================
# Fit assembly (pure): one factor, reused everywhere downstream.
# ==========================================================================


@dataclass(frozen=True)
class PosteriorFit:
    """Everything the exact posterior needs, computed once from the training rows."""

    lam: float
    sigma2: float
    tau2: float
    factor: tuple
    mu: np.ndarray
    ainv: np.ndarray
    Xi: sparse.csr_matrix          # intercept-appended full design (all rows)
    mask_i: np.ndarray             # penalty mask aligned to Xi columns
    train_mask: np.ndarray
    test_mask: np.ndarray


def fit_posterior(art, lam: float) -> PosteriorFit:
    """Factor `A` on the training rows at the reused lambda and derive the posterior.

    Appends the unpenalized intercept exactly as Phase 2 does so `D` (and hence
    the mean) match the ridge fit, factors `A = X_tr^T X_tr + lambda D` once, and
    computes `mu`, `sigma^2`, `tau^2 = sigma^2/lambda`, and the dense `A^{-1}`.
    Pure: reads nothing from disk.
    """
    Xi = append_intercept(art.X)
    mask_i = np.append(penalized_mask(art.columns), False)
    train_mask = np.isin(art.fold, INNER_FOLDS)
    test_mask = art.fold == TEST_FOLD

    Xtr, ytr = Xi[train_mask], art.y[train_mask]
    A = build_normal_matrix(Xtr, penalty_diagonal(mask_i, lam))
    factor = cho_factor(A, lower=True)
    mu = posterior_mean(factor, Xtr, ytr)
    sigma2 = residual_variance(Xtr, ytr, mu)
    tau2 = sigma2 / lam
    ainv = inverse_from_factor(factor, A.shape[0])
    return PosteriorFit(
        lam=lam, sigma2=sigma2, tau2=tau2, factor=factor, mu=mu, ainv=ainv,
        Xi=Xi, mask_i=mask_i, train_mask=train_mask, test_mask=test_mask,
    )


def _game_arrays(
    art, Xi, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signed weight rows, possession counts, actual margins) for games in `mask`.

    Shared by the training calibration and the test-fold evaluation so both build
    game rows identically. The per-game actual margins come from
    `rapm.ridge.game_margins`, aligned to the weight-row order. Pure.
    """
    gid, oh = art.row_game_id[mask], art.row_offense_is_home[mask]
    weights, order, n_g = game_weight_matrix(Xi[mask], gid, oh)
    actual = game_margins(art.y[mask], gid, oh)
    actual_vec = np.array([actual[g] for g in order], dtype=np.float64)
    return weights, n_g, actual_vec


def predictive_coverage_90(art, fit: PosteriorFit) -> tuple[float, int, float]:
    """GATE 1: 90% predictive coverage of held-out fold-4 game margins.

    Calibrates the effective per-possession margin noise on the TRAINING games
    (`effective_margin_noise`) — the iid-possession `sigma^2 * n_g` term
    over-disperses the signed margin, so the honest noise that survives is
    estimated from held-out-of-test games — then forms each test game's predictive
    interval with that noise (the parameter term keeps the posterior `sigma^2`) and
    measures the fraction of ACTUAL margins inside. The training estimate never
    sees fold 4 and is never tuned to the coverage target. Returns
    (coverage, n_test_games, sigma2_margin).
    """
    w_tr, n_tr, act_tr = _game_arrays(art, fit.Xi, fit.train_mask)
    sigma2_margin = effective_margin_noise(
        fit.factor, w_tr, fit.mu, fit.sigma2, act_tr, n_tr
    )

    w_te, n_te, act_te = _game_arrays(art, fit.Xi, fit.test_mask)
    _, _, lo, hi = predictive_intervals(
        fit.factor, w_te, fit.mu, fit.sigma2, n_te, noise_var=sigma2_margin
    )
    return coverage_fraction(lo, hi, act_te), len(act_te), sigma2_margin


def oos_margin_rmse(art, fit: PosteriorFit) -> float:
    """GATE 3: posterior-mean game-margin RMSE on test fold 4.

    Since `mu` equals the ridge solution, this reproduces the ridge test RMSE; it
    reuses `rapm.ridge.game_margins`/`margin_rmse` on the test rows.
    """
    tm = fit.test_mask
    yhat = np.asarray(fit.Xi[tm] @ fit.mu).ravel()
    predicted = game_margins(yhat, art.row_game_id[tm], art.row_offense_is_home[tm])
    actual = game_margins(art.y[tm], art.row_game_id[tm], art.row_offense_is_home[tm])
    return margin_rmse(predicted, actual)


# ==========================================================================
# Report-only recency-weighted variant (warehouse-dependent, never gated).
# ==========================================================================


def decay_weights(
    row_ordinals: np.ndarray, row_season: np.ndarray, half_life: float
) -> np.ndarray:
    """Within-season exponential time-decay weights `0.5^(days_before_end/half_life)`.

    `row_ordinals` are per-possession game-date ordinals (NaN if unknown); each
    season's reference is its own latest game, so recency is measured within
    season. Missing dates fall back to weight 1.0. Pure.
    """
    weights = np.ones(len(row_ordinals), dtype=np.float64)
    for s in np.unique(row_season):
        m = row_season == s
        ords = row_ordinals[m]
        if np.all(np.isnan(ords)):
            continue
        days = np.nanmax(ords) - ords
        weights[m] = np.where(np.isnan(days), 1.0, 0.5 ** (days / half_life))
    return weights


def weighted_posterior_mean(
    X, y: np.ndarray, w: np.ndarray, penalty_diag: np.ndarray
) -> np.ndarray:
    """Weighted ridge mean `mu_w = (X^T W X + lambda D)^{-1} X^T W y`, `W = diag(w)`."""
    Dw = sparse.diags(np.asarray(w, dtype=np.float64))
    gram = (X.T @ Dw @ X)
    gram = gram.toarray() if sparse.issparse(gram) else np.asarray(gram)
    A = gram + np.diag(np.asarray(penalty_diag, dtype=np.float64))
    rhs = np.asarray(X.T @ (np.asarray(w) * np.asarray(y, dtype=np.float64))).ravel()
    return cho_solve(cho_factor(A, lower=True), rhs)


def _net_ratings_map(columns: list[dict], mu: np.ndarray) -> dict[tuple, float]:
    """(player_id, season) -> net rating (per-100) from a coefficient vector."""
    pairs: dict[tuple, dict] = {}
    for i, c in enumerate(columns):
        if c["kind"] != "player":
            continue
        pairs.setdefault((c["player_id"], c["season"]), {})[c["side"]] = i
    out: dict[tuple, float] = {}
    for key, sides in pairs.items():
        o, d = sides.get("O"), sides.get("D")
        if o is not None and d is not None:
            out[key] = RATING_SCALE * (mu[o] - mu[d])
    return out


def _row_ordinals(warehouse: str, row_game_id: np.ndarray) -> np.ndarray | None:
    """Per-possession game-date ordinals from the warehouse, or None if unavailable."""
    try:
        import duckdb

        con = duckdb.connect(warehouse, read_only=True)
        try:
            rows = con.execute("select game_id, game_date from dim_games").fetchall()
        finally:
            con.close()
    except Exception as exc:  # warehouse missing / schema drift
        print(f"  (recency variant skipped: warehouse unavailable — {exc})")
        return None
    dmap = {str(g): dt for g, dt in rows if dt is not None}
    ords = np.array(
        [dmap[str(g)].toordinal() if str(g) in dmap else np.nan for g in row_game_id],
        dtype=np.float64,
    )
    if np.all(np.isnan(ords)):
        print("  (recency variant skipped: no game_date coverage)")
        return None
    return ords


def report_recency_variant(art, fit: PosteriorFit, warehouse: str) -> None:
    """Refit with within-season time decay and report top-10 net-rating movement."""
    ords = _row_ordinals(warehouse, art.row_game_id)
    if ords is None:
        return
    w = decay_weights(ords, art.row_season, RECENCY_HALF_LIFE_DAYS)
    tr = fit.train_mask
    mu_w = weighted_posterior_mean(
        fit.Xi[tr], art.y[tr], w[tr], penalty_diagonal(fit.mask_i, fit.lam)
    )
    base = _net_ratings_map(art.columns, fit.mu)
    wtd = _net_ratings_map(art.columns, mu_w)
    top_base = {k for k, _ in sorted(base.items(), key=lambda kv: kv[1], reverse=True)[:10]}
    top_wtd = {k for k, _ in sorted(wtd.items(), key=lambda kv: kv[1], reverse=True)[:10]}
    shared = len(top_base & top_wtd)
    print(
        f"\nRecency-weighted variant (half-life {RECENCY_HALF_LIFE_DAYS:g}d): "
        f"{shared}/10 of the top-10 net ratings unchanged, {10 - shared} swapped "
        f"({'material' if shared < 8 else 'minor'} reshuffle)."
    )


# ==========================================================================
# Reporting + orchestration (impure: disk + warehouse I/O).
# ==========================================================================


def _print_player_table(ratings: list[dict], warehouse: str) -> None:
    """Print stars + lowest-possession role players with net +/- 90% CI widths."""
    names = _load_player_names(warehouse)
    by_key = {(r["player_id"], r["season"]): r for r in ratings}
    watch_ids = {1641705, 1628983, 1627936}  # Wembanyama, SGA, Caruso
    stars = [r for r in ratings if r["player_id"] in watch_ids]
    seen = {(r["player_id"], r["season"]) for r in stars}
    lowest = sorted(
        (r for r in ratings if (r["player_id"], r["season"]) not in seen),
        key=lambda r: r["n_possessions"],
    )[:3]
    print("\nNet rating +/- 90% CI (interval widens for low-possession/entangled)")
    print(f"  {'player':<26} {'season':>6} {'net':>7} {'90% CI':>18} {'n_poss':>8}")
    for r in stars + lowest:
        who = names.get(int(r["player_id"]), str(r["player_id"]))
        ci = f"[{r['net_ci_low']:+6.1f},{r['net_ci_high']:+6.1f}]"
        print(
            f"  {who[:26]:<26} {r['season']:>6} {r['net_rating']:+7.2f} "
            f"{ci:>18} {r['n_possessions']:>8}"
        )


def run(
    data_dir=DEFAULT_DATA_DIR, warehouse: str = DEFAULT_WAREHOUSE,
    n_sbc: int = DEFAULT_SBC_DATASETS,
) -> dict:
    """Fit the exact posterior, run all three gates, and write the two artifacts."""
    import pandas as pd

    art = load_artifacts(data_dir)
    ridge_metrics = json.loads((Path(data_dir) / "ridge_metrics.json").read_text())
    lam = float(ridge_metrics["chosen_lambda"])
    ridge_rmse = float(ridge_metrics["test_rmse_ridge"])
    print(
        f"Loaded design: {art.X.shape[0]:,} possessions x {art.X.shape[1]:,} columns; "
        f"reusing lambda={lam:g} (corpus {art.manifest.get('corpus_hash')})"
    )

    fit = fit_posterior(art, lam)
    print(f"  sigma2={fit.sigma2:.4f}  tau2={fit.tau2:.6f}  lambda={lam:g}")

    cov90, n_test_games, sigma2_margin = predictive_coverage_90(art, fit)
    print(f"  sigma2_margin={sigma2_margin:.4f} (effective per-possession margin "
          f"noise, train-calibrated; sigma2={fit.sigma2:.4f} over-disperses margins)")
    penalized_idx = np.flatnonzero(fit.mask_i)
    ainv_diag = np.diag(fit.ainv)
    sbc90 = sbc_coverage(
        fit.factor, fit.Xi[fit.train_mask], fit.mu, penalized_idx,
        fit.sigma2, fit.tau2, ainv_diag, n_datasets=n_sbc,
    )
    oos_rmse = oos_margin_rmse(art, fit)

    g1 = "PASS" if 0.85 <= cov90 <= 0.95 else "FAIL"
    print(f"\nGATE 1  predictive_coverage_90 = {cov90:.4f}  "
          f"({g1} in [0.85, 0.95], over {n_test_games} test games)")
    if cov90 > 0.95:
        print("        diagnosis: over-coverage — even the train-calibrated margin "
              "noise leaves intervals too wide; reporting the true value, not tuned.")
    print(f"GATE 2  sbc_coverage_90        = {sbc90:.4f}  "
          f"({'PASS' if 0.88 <= sbc90 <= 0.92 else 'FAIL'} in [0.88, 0.92], "
          f"over {n_sbc} datasets)")
    print(f"GATE 3  oos_rmse               = {oos_rmse:.6f} vs ridge {ridge_rmse:.6f}  "
          f"({'PASS' if oos_rmse <= ridge_rmse + 1e-6 else 'FAIL'})")

    ratings = posterior_ratings(art.columns, fit.mu, fit.ainv, fit.sigma2, art.X)
    metrics = {
        "lambda": lam,
        "sigma2": fit.sigma2,
        "tau2": fit.tau2,
        "sigma2_margin": sigma2_margin,
        "predictive_coverage_90": cov90,
        "n_test_games": n_test_games,
        "sbc_coverage_90": sbc90,
        "n_sbc_datasets": n_sbc,
        "oos_rmse": oos_rmse,
        "ridge_rmse": ridge_rmse,
        "corpus_hash": art.manifest.get("corpus_hash"),
    }

    out = Path(data_dir)
    pd.DataFrame(
        ratings,
        columns=[
            "player_id", "season", "off_rating", "def_rating", "net_rating",
            "off_sd", "def_sd", "net_sd", "net_ci_low", "net_ci_high", "n_possessions",
        ],
    ).to_parquet(out / "bayes_ratings.parquet", index=False)
    (out / "bayes_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {out}/bayes_ratings.parquet and bayes_metrics.json")

    _print_player_table(ratings, warehouse)
    report_recency_variant(art, fit, warehouse)
    return metrics


def gates_pass(metrics: dict) -> bool:
    """True iff all three hard gates hold for a metrics dict.

    Predictive coverage in [0.85, 0.95], SBC coverage in [0.88, 0.92], and OOS
    RMSE no worse than ridge. Pure — the single source of truth for the exit code.
    """
    return (
        0.85 <= metrics["predictive_coverage_90"] <= 0.95
        and 0.88 <= metrics["sbc_coverage_90"] <= 0.92
        and metrics["oos_rmse"] <= metrics["ridge_rmse"] + 1e-6
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit the exact Gaussian-posterior RAPM and run its coverage gates."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    parser.add_argument("--n-sbc", type=int, default=DEFAULT_SBC_DATASETS)
    args = parser.parse_args()
    metrics = run(data_dir=args.data_dir, warehouse=args.warehouse, n_sbc=args.n_sbc)
    if not gates_pass(metrics):
        raise SystemExit("gate failure: one or more metrics fell outside contract bounds")


if __name__ == "__main__":
    main()
