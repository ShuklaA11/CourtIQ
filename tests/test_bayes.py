"""Unit tests for the exact Gaussian-posterior RAPM core (rapm.bayes).

All pure and synthetic: no warehouse, no data/rapm/ reads. Each test pins one
piece of the conjugate identity against an independent computation — the mean
against the ridge solver, the covariance against a dense numpy inverse, the
predictive-variance decomposition against a direct formula, SBC coverage against
its 0.90 target, and interval width against possession count.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from rapm.bayes import (
    Z90,
    build_normal_matrix,
    effective_margin_noise,
    inverse_from_factor,
    posterior_mean,
    predictive_intervals,
    residual_variance,
    sbc_coverage,
)
from rapm.ridge import penalty_diagonal, solve_ridge


def _factor(X, penalty_diag):
    """Build A and its Cholesky factor for a design and penalty vector."""
    A = build_normal_matrix(X, penalty_diag)
    return A, cho_factor(A, lower=True)


# --------------------------------------------------------------------------
# (1) posterior mean == rapm.ridge.solve_ridge at the same lambda
# --------------------------------------------------------------------------

def test_posterior_mean_equals_solve_ridge():
    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    mask = np.array([True, True, True, False, False])
    d = penalty_diagonal(mask, lam=3.0)

    _, factor = _factor(X, d)
    mu = posterior_mean(factor, X, y)
    beta_ridge = solve_ridge(X, y, d)

    assert np.allclose(mu, beta_ridge, atol=1e-10)


def test_posterior_mean_matches_ridge_on_sparse_design():
    rng = np.random.default_rng(1)
    dense = rng.standard_normal((30, 4))
    dense[dense < 0.3] = 0.0
    X = sparse.csr_matrix(dense)
    y = rng.standard_normal(30)
    d = penalty_diagonal(np.ones(4, dtype=bool), lam=1.2)

    _, factor = _factor(X, d)
    assert np.allclose(posterior_mean(factor, X, y), solve_ridge(X, y, d), atol=1e-10)


# --------------------------------------------------------------------------
# (2) posterior covariance == sigma2 * inv(X^T X + lambda D), dense cross-check
# --------------------------------------------------------------------------

def test_posterior_covariance_matches_dense_inverse():
    rng = np.random.default_rng(2)
    n, p = 25, 4
    X = rng.standard_normal((n, p))
    mask = np.array([True, True, False, False])
    lam = 2.5
    d = penalty_diagonal(mask, lam)
    sigma2 = 1.7

    A, factor = _factor(X, d)
    cov = sigma2 * inverse_from_factor(factor, p)

    # Independent dense inverse of X^T X + lambda D — no shared code path.
    A_ref = X.T @ X + np.diag(d)
    cov_ref = sigma2 * np.linalg.inv(A_ref)

    assert np.allclose(A, A_ref, atol=1e-10)
    assert np.allclose(cov, cov_ref, atol=1e-8)


# --------------------------------------------------------------------------
# (3) predictive-variance decomposition on a tiny hand-built game
# --------------------------------------------------------------------------

def test_predictive_variance_decomposition_matches_direct():
    rng = np.random.default_rng(3)
    n, p = 20, 3
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    d = penalty_diagonal(np.array([True, True, False]), lam=1.0)
    A, factor = _factor(X, d)
    mu = posterior_mean(factor, X, y)
    sigma2 = 0.8

    # One game: two possessions with signs +1 and -1 -> w = x_a - x_b, n_g = 2.
    x_a = np.array([1.0, 0.0, 1.0])
    x_b = np.array([0.0, 2.0, 1.0])
    w = x_a - x_b
    n_g = 2

    means, var, lo, hi = predictive_intervals(
        factor, w.reshape(1, -1), mu, sigma2, np.array([n_g])
    )

    # Direct: mean = w.mu; var = sigma2 * (w^T A^{-1} w) + sigma2 * n_g.
    ainv = np.linalg.inv(A)
    var_direct = sigma2 * (w @ ainv @ w) + sigma2 * n_g
    assert np.isclose(means[0], w @ mu, atol=1e-10)
    assert np.isclose(var[0], var_direct, atol=1e-8)
    assert np.isclose(lo[0], means[0] - Z90 * np.sqrt(var_direct), atol=1e-10)
    assert np.isclose(hi[0], means[0] + Z90 * np.sqrt(var_direct), atol=1e-10)


# --------------------------------------------------------------------------
# (3b) noise_var overrides only the observation-noise term, not the posterior
# --------------------------------------------------------------------------

def test_predictive_intervals_noise_var_overrides_noise_term():
    rng = np.random.default_rng(30)
    n, p = 18, 3
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    d = penalty_diagonal(np.array([True, True, False]), lam=1.0)
    A, factor = _factor(X, d)
    mu = posterior_mean(factor, X, y)
    sigma2, noise_var = 0.8, 0.3

    w = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]])
    n_g = np.array([2, 3])

    _, var, _, _ = predictive_intervals(factor, w, mu, sigma2, n_g, noise_var=noise_var)

    ainv = np.linalg.inv(A)
    quad = np.array([wi @ ainv @ wi for wi in w])
    var_direct = sigma2 * quad + noise_var * n_g  # param term keeps sigma2
    assert np.allclose(var, var_direct, atol=1e-10)
    # And noise_var actually changes the answer relative to the sigma2 default.
    _, var_default, _, _ = predictive_intervals(factor, w, mu, sigma2, n_g)
    assert not np.allclose(var, var_default)


# --------------------------------------------------------------------------
# (3c) effective margin noise is the variance-component estimate
# --------------------------------------------------------------------------

def test_effective_margin_noise_matches_variance_component():
    rng = np.random.default_rng(31)
    n, p = 24, 3
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    d = penalty_diagonal(np.array([True, True, False]), lam=1.5)
    _, factor = _factor(X, d)
    mu = posterior_mean(factor, X, y)
    sigma2 = 0.9

    # Two games with known signed weight rows, possession counts, and actual margins.
    w = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    n_g = np.array([3, 4, 2])
    actual = np.array([1.2, -0.7, 0.4])

    s2m = effective_margin_noise(factor, w, mu, sigma2, actual, n_g)

    resid = actual - w @ mu
    ainv = np.linalg.inv(build_normal_matrix(X, d))
    quad = np.array([wi @ ainv @ wi for wi in w])
    expected = (float(resid @ resid) - sigma2 * quad.sum()) / n_g.sum()
    assert np.isclose(s2m, max(expected, 0.0), atol=1e-10)


# --------------------------------------------------------------------------
# (4) SBC sanity — planted-truth 90% coverage lands near 0.90
# --------------------------------------------------------------------------

def test_sbc_planted_truth_coverage_near_090():
    rng = np.random.default_rng(4)
    n, p = 300, 30  # many penalized coefficients for a tight pooled estimate
    X = rng.standard_normal((n, p))
    lam = 8.0
    d = penalty_diagonal(np.ones(p, dtype=bool), lam)
    A, factor = _factor(X, d)
    # Any consistent (sigma2, tau2=sigma2/lam) pair calibrates; the value cancels.
    sigma2 = 1.3
    tau2 = sigma2 / lam
    mu = np.zeros(p)  # unpenalized set to fitted mean; here all penalized anyway
    ainv_diag = np.diag(inverse_from_factor(factor, p))
    penalized_idx = np.arange(p)

    cov = sbc_coverage(
        factor, X, mu, penalized_idx, sigma2, tau2, ainv_diag,
        n_datasets=200, base_seed=123,
    )
    assert 0.87 <= cov <= 0.93


# --------------------------------------------------------------------------
# (5) credible-interval width larger for a low-possession coefficient
# --------------------------------------------------------------------------

def test_low_possession_coefficient_has_wider_interval():
    # Column 0 ("high possession") appears in every row; column 1 ("low
    # possession") appears in just a few. More Fisher information -> smaller
    # posterior variance -> narrower credible interval.
    n = 60
    high = np.ones(n)
    low = np.zeros(n)
    low[:4] = 1.0
    X = np.column_stack([high, low])
    d = penalty_diagonal(np.array([True, True]), lam=1.0)
    _, factor = _factor(X, d)
    sigma2 = 1.0

    cov_diag = sigma2 * np.diag(inverse_from_factor(factor, 2))
    sd_high, sd_low = np.sqrt(cov_diag)
    assert sd_low > sd_high
    assert Z90 * sd_low > Z90 * sd_high  # interval half-width strictly wider


# --------------------------------------------------------------------------
# residual_variance — sanity on a known-noise design
# --------------------------------------------------------------------------

def test_residual_variance_is_mean_squared_residual():
    rng = np.random.default_rng(5)
    n, p = 50, 3
    X = rng.standard_normal((n, p))
    mu = np.array([1.0, -2.0, 0.5])
    resid = 0.3 * rng.standard_normal(n)
    y = X @ mu + resid
    s2 = residual_variance(X, y, mu)
    assert np.isclose(s2, float(resid @ resid / n), atol=1e-12)
