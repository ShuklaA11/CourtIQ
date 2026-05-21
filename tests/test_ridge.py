"""Unit tests for the pure RAPM ridge primitives (rapm.ridge).

All synthetic and warehouse-free: no DuckDB, no data/rapm/ reads. Covers the
selective-penalty solver, its penalty/intercept helpers, and the game-margin
retrodiction aggregation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from rapm.ridge import (
    Artifacts,
    append_intercept,
    cv_select_lambda,
    extract_ratings,
    game_margins,
    is_interior_optimum,
    load_artifacts,
    margin_rmse,
    penalized_mask,
    penalty_diagonal,
    raw_plus_minus_predict,
    retrodiction_rmse,
    ridge_predict,
    solve_ridge,
    team_net_predict,
)


# --------------------------------------------------------------------------
# penalty_diagonal
# --------------------------------------------------------------------------

def test_penalty_diagonal_penalizes_masked_columns_only():
    mask = np.array([True, True, False, False])  # e.g. two players, home, intercept
    d = penalty_diagonal(mask, lam=5.0)
    assert d.tolist() == [5.0, 5.0, 0.0, 0.0]


def test_penalty_diagonal_does_not_mutate_mask():
    mask = np.array([True, False])
    penalty_diagonal(mask, lam=2.0)
    assert mask.tolist() == [True, False]


def test_penalty_diagonal_rejects_negative_lambda():
    with pytest.raises(ValueError):
        penalty_diagonal(np.array([True]), lam=-1.0)


# --------------------------------------------------------------------------
# append_intercept
# --------------------------------------------------------------------------

def test_append_intercept_dense_adds_ones_column():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    aug = append_intercept(X)
    assert aug.shape == (2, 3)
    assert aug[:, -1].tolist() == [1.0, 1.0]
    # Original columns preserved, input untouched.
    assert aug[:, :2].tolist() == X.tolist()
    assert X.shape == (2, 2)


def test_append_intercept_sparse_stays_sparse():
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    aug = append_intercept(X)
    assert sparse.issparse(aug)
    assert aug.shape == (3, 3)
    assert np.asarray(aug[:, -1].todense()).ravel().tolist() == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# solve_ridge — recovery, OLS equivalence, selective shrinkage
# --------------------------------------------------------------------------

def test_solver_recovers_known_beta_at_small_lambda():
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ beta_true + 1e-3 * rng.standard_normal(n)

    d = penalty_diagonal(np.ones(p, dtype=bool), lam=1e-6)  # near-zero penalty
    beta = solve_ridge(X, y, d)

    assert np.allclose(beta, beta_true, atol=1e-2)


def test_lambda_zero_matches_ols_normal_equation():
    rng = np.random.default_rng(1)
    n, p = 50, 4
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    beta = solve_ridge(X, y, penalty_diag=np.zeros(p))

    # Independent OLS solve via lstsq — no ridge, no shared code path.
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(beta, beta_ols, atol=1e-8)


def test_orthogonal_design_shrinks_penalized_not_unpenalized():
    # Orthonormal columns (from QR): with ||x_j||^2 = 1 and no cross-terms, the
    # ridge solution is beta_j = (x_j^T y) / (1 + d_j), so d_j = 0 leaves a column
    # exactly at its lambda=0 value while d_j > 0 shrinks it toward zero.
    rng = np.random.default_rng(2)
    n, p = 6, 3
    Q, _ = np.linalg.qr(rng.standard_normal((n, p)))  # Q has orthonormal columns
    y = rng.standard_normal(n)

    beta0 = solve_ridge(Q, y, penalty_diag=np.zeros(p))

    # Penalize column 0, leave columns 1 and 2 unpenalized ("home"/"intercept").
    mask = np.array([True, False, False])
    beta = solve_ridge(Q, y, penalty_diagonal(mask, lam=4.0))

    # Unpenalized columns are unchanged relative to the lambda=0 solution.
    assert np.isclose(beta[1], beta0[1], atol=1e-10)
    assert np.isclose(beta[2], beta0[2], atol=1e-10)
    # The penalized column is shrunk strictly toward zero.
    assert abs(beta[0]) < abs(beta0[0])
    assert np.isclose(beta[0], beta0[0] / (1.0 + 4.0), atol=1e-10)


def test_solver_accepts_sparse_csr_matching_dense():
    rng = np.random.default_rng(3)
    n, p = 40, 4
    dense = rng.standard_normal((n, p))
    dense[dense < 0.5] = 0.0  # make it genuinely sparse
    y = rng.standard_normal(n)
    d = penalty_diagonal(np.ones(p, dtype=bool), lam=0.7)

    beta_sparse = solve_ridge(sparse.csr_matrix(dense), y, d)
    beta_dense = solve_ridge(dense, y, d)
    assert np.allclose(beta_sparse, beta_dense, atol=1e-10)


def test_solver_rejects_wrong_length_penalty():
    X = np.eye(3)
    with pytest.raises(ValueError):
        solve_ridge(X, np.zeros(3), penalty_diag=np.zeros(2))


# --------------------------------------------------------------------------
# game_margins / margin_rmse — retrodiction aggregation
# --------------------------------------------------------------------------

def test_game_margins_hand_computed_multi_game():
    # Game A rows 0-2: home offense scores 2 and 3, away offense scores 1
    #   -> margin = (2 + 3) - 1 = 4
    # Game B rows 3-4: away offense scores 4, home offense scores 2
    #   -> margin = 2 - 4 = -2
    yhat = np.array([2.0, 1.0, 3.0, 4.0, 2.0])
    game_id = np.array(["A", "A", "A", "B", "B"])
    offense_is_home = np.array([True, False, True, False, True])

    margins = game_margins(yhat, game_id, offense_is_home)
    assert margins == {"A": 4.0, "B": -2.0}


def test_game_margins_reused_for_actuals_and_rmse():
    game_id = np.array(["A", "A", "B", "B"])
    offense_is_home = np.array([True, False, True, False])
    yhat = np.array([3.0, 1.0, 2.0, 2.0])   # A: 3-1=2,  B: 2-2=0
    y = np.array([2.0, 1.0, 4.0, 1.0])      # A: 2-1=1,  B: 4-1=3

    predicted = game_margins(yhat, game_id, offense_is_home)
    actual = game_margins(y, game_id, offense_is_home)
    assert predicted == {"A": 2.0, "B": 0.0}
    assert actual == {"A": 1.0, "B": 3.0}

    # errors: A -> (2-1)=1, B -> (0-3)=-3; rmse = sqrt((1 + 9)/2) = sqrt(5)
    assert np.isclose(margin_rmse(predicted, actual), np.sqrt(5.0))


def test_margin_rmse_zero_when_identical():
    game_id = np.array(["G1", "G1", "G2"])
    offense_is_home = np.array([True, False, True])
    v = np.array([2.0, 1.0, 5.0])
    m = game_margins(v, game_id, offense_is_home)
    assert margin_rmse(m, m) == 0.0


def test_margin_rmse_requires_shared_games():
    with pytest.raises(ValueError):
        margin_rmse({"A": 1.0}, {"B": 2.0})


def test_game_margins_rejects_length_mismatch():
    with pytest.raises(ValueError):
        game_margins(np.array([1.0, 2.0]), np.array(["A"]), np.array([True, False]))


# ==========================================================================
# Phase-2 pipeline layer (artifact-driven fit + retrodiction gate)
# ==========================================================================

# Two player-seasons (ids 1 and 2, season 2023) plus a home column. Columns:
#   0: player 1 offense   1: player 1 defense
#   2: player 2 offense   3: player 2 defense
#   4: home (unpenalized)
_COLUMNS = [
    {"index": 0, "kind": "player", "side": "O", "player_id": 1, "season": 2023, "penalized": True},
    {"index": 1, "kind": "player", "side": "D", "player_id": 1, "season": 2023, "penalized": True},
    {"index": 2, "kind": "player", "side": "O", "player_id": 2, "season": 2023, "penalized": True},
    {"index": 3, "kind": "player", "side": "D", "player_id": 2, "season": 2023, "penalized": True},
    {"index": 4, "kind": "home", "side": None, "player_id": None, "season": None, "penalized": False},
]


# --------------------------------------------------------------------------
# penalized_mask / ridge_predict / retrodiction_rmse
# --------------------------------------------------------------------------

def test_penalized_mask_reads_column_flags():
    mask = penalized_mask(_COLUMNS)
    assert mask.tolist() == [True, True, True, True, False]


def test_ridge_predict_is_dot_product():
    X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    beta = np.array([2.0, 3.0, 0.5])
    assert ridge_predict(X, beta).tolist() == [3.0, 3.5]


def test_ridge_predict_accepts_sparse():
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0]]))
    beta = np.array([4.0, 10.0])
    assert ridge_predict(X, beta).tolist() == [4.0, 14.0]


def test_retrodiction_rmse_matches_manual():
    # Same construction as the pure-primitive test, through the wrapper.
    game_id = np.array(["A", "A", "B", "B"])
    offense_is_home = np.array([True, False, True, False])
    yhat = np.array([3.0, 1.0, 2.0, 2.0])   # A: 2,  B: 0
    y = np.array([2.0, 1.0, 4.0, 1.0])      # A: 1,  B: 3
    assert np.isclose(
        retrodiction_rmse(yhat, y, game_id, offense_is_home), np.sqrt(5.0)
    )


# --------------------------------------------------------------------------
# is_interior_optimum — the gate-failure diagnostic
# --------------------------------------------------------------------------

def test_interior_optimum_true_only_inside_grid():
    assert is_interior_optimum(3, 7) is True
    assert is_interior_optimum(0, 7) is False   # left endpoint
    assert is_interior_optimum(6, 7) is False   # right endpoint
    assert is_interior_optimum(0, 1) is False   # degenerate single point


# --------------------------------------------------------------------------
# raw_plus_minus_predict — hand-computed baseline
# --------------------------------------------------------------------------

def test_raw_plus_minus_hand_computed():
    # Training rows:
    #   r0: p1 offense, p2 defense, home    y=3
    #   r1: p2 offense, p1 defense, away    y=1
    # Eval row r2 mirrors r0 (p1 off, p2 def, home); its y is unused.
    #   mu = 2; a1 = 3-2 = 1; d2 = 3-2 = 1; h = mean(home y) - mean(away y) = 3-1 = 2
    #   yhat = mu + a1 + d2 + h/2 = 2 + 1 + 1 + 1 = 5
    dense = np.array(
        [
            [1, 0, 0, 1, 1],   # r0: p1 O, p2 D, home
            [0, 1, 1, 0, 0],   # r1: p2 O, p1 D, away
            [1, 0, 0, 1, 1],   # r2: p1 O, p2 D, home
        ],
        dtype=np.float64,
    )
    X = sparse.csr_matrix(dense)
    y = np.array([3.0, 1.0, 99.0])
    train_mask = np.array([True, True, False])
    eval_mask = np.array([False, False, True])
    offense_is_home = np.array([True, False, True])

    yhat = raw_plus_minus_predict(X, y, _COLUMNS, train_mask, eval_mask, offense_is_home)
    assert np.isclose(yhat[0], 5.0)


def test_raw_plus_minus_unseen_column_contributes_zero():
    # Player 2 never appears in training -> its columns contribute 0, leaving
    # only mu + a1 + (no d) + home term.
    dense = np.array(
        [
            [1, 0, 0, 0, 1],   # r0: p1 O only, home  y=4
            [1, 0, 0, 0, 0],   # r1: p1 O only, away  y=2
            [0, 0, 1, 0, 1],   # r2 (eval): p2 O only, home
        ],
        dtype=np.float64,
    )
    X = sparse.csr_matrix(dense)
    y = np.array([4.0, 2.0, 0.0])
    train_mask = np.array([True, True, False])
    eval_mask = np.array([False, False, True])
    offense_is_home = np.array([True, False, True])
    # mu = 3; p2 unseen in training -> a2 = 0; h = 4 - 2 = 2; yhat = 3 + 0 + 1 = 4
    yhat = raw_plus_minus_predict(X, y, _COLUMNS, train_mask, eval_mask, offense_is_home)
    assert np.isclose(yhat[0], 4.0)


# --------------------------------------------------------------------------
# team_net_predict
# --------------------------------------------------------------------------

def test_team_net_predict_uses_training_ppp():
    offense_team_id = np.array([10, 20, 10, 20])
    y = np.array([2.0, 4.0, 3.0, 5.0])
    train_mask = np.array([True, True, False, False])
    eval_mask = np.array([False, False, True, True])
    # team 10 train ppp = 2, team 20 train ppp = 4
    yhat = team_net_predict(offense_team_id, y, train_mask, eval_mask)
    assert yhat.tolist() == [2.0, 4.0]


def test_team_net_predict_unseen_team_falls_back_to_global():
    offense_team_id = np.array([10, 10, 99])
    y = np.array([2.0, 4.0, 0.0])
    train_mask = np.array([True, True, False])
    eval_mask = np.array([False, False, True])
    # global train ppp = 3; team 99 unseen -> 3
    yhat = team_net_predict(offense_team_id, y, train_mask, eval_mask)
    assert yhat.tolist() == [3.0]


# --------------------------------------------------------------------------
# extract_ratings
# --------------------------------------------------------------------------

def test_extract_ratings_pairs_and_scales():
    beta = np.array([0.1, 0.05, 0.2, -0.1, 0.0])  # per O/D column, home last
    # Player 1 active in 3 O rows + 2 D rows -> n=5; player 2 in 1 + 1 -> n=2.
    dense = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    X = sparse.csr_matrix(dense)
    ratings = {(r["player_id"], r["season"]): r for r in extract_ratings(_COLUMNS, beta, X)}

    p1 = ratings[(1, 2023)]
    assert np.isclose(p1["off_rating"], 10.0)
    assert np.isclose(p1["def_rating"], 5.0)
    assert np.isclose(p1["net_rating"], 5.0)
    assert p1["n_possessions"] == 5

    p2 = ratings[(2, 2023)]
    assert np.isclose(p2["net_rating"], 20.0 - (-10.0))
    assert p2["n_possessions"] == 2


# --------------------------------------------------------------------------
# cv_select_lambda — structure + prefers low shrinkage on clean signal
# --------------------------------------------------------------------------

def _synthetic_cv_design(seed: int = 0):
    """Two inner folds of clean, strongly-signalled possessions.

    Each row is one player-season on offense (+ intercept appended by caller);
    y is a near-noiseless linear function of the design, so CV should prefer the
    smallest lambda. Games are whole-fold so grouped CV is leakage-free.
    """
    rng = np.random.default_rng(seed)
    n_per_fold, p = 60, 4
    Xrows, y, fold, game_id, offense_is_home = [], [], [], [], []
    beta_true = np.array([2.0, -1.0, 3.0, 0.5])
    for f in (0, 1):
        for g in range(6):  # 6 games per fold, 10 possessions each
            for _ in range(10):
                row = np.zeros(p)
                row[rng.integers(p)] = 1.0
                Xrows.append(row)
                y.append(float(row @ beta_true) + 1e-3 * rng.standard_normal())
                fold.append(f)
                game_id.append(f"f{f}g{g}")
                offense_is_home.append(bool(rng.integers(2)))
    X = append_intercept(np.array(Xrows))
    mask = np.append(np.ones(p, dtype=bool), False)  # penalize players, not intercept
    return (
        X,
        np.array(y),
        np.array(fold),
        np.array(game_id),
        np.array(offense_is_home),
        mask,
    )


def test_cv_select_lambda_structure_and_prefers_small_on_clean_signal():
    X, y, fold, game_id, oih, mask = _synthetic_cv_design()
    grid = np.array([0.01, 1.0, 100.0, 10000.0])
    best_lambda, curve, best_idx = cv_select_lambda(
        X, y, fold, game_id, oih, mask, grid=grid, inner_folds=(0, 1)
    )
    assert len(curve) == len(grid)
    assert best_lambda == grid[best_idx]
    # Near-noiseless linear signal -> least shrinkage wins.
    assert best_idx == 0
    # Heavier shrinkage strictly worsens retrodiction here.
    assert curve[0] < curve[-1]


# --------------------------------------------------------------------------
# load_artifacts — round-trip through on-disk format
# --------------------------------------------------------------------------

def test_load_artifacts_roundtrip(tmp_path):
    from scipy.sparse import save_npz

    dense = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    X = sparse.csr_matrix(dense)
    save_npz(tmp_path / "X.npz", X)
    np.save(tmp_path / "y.npy", np.array([1.0, 2.0]))
    np.save(tmp_path / "fold.npy", np.array([0, 4]))
    np.save(tmp_path / "row_game_id.npy", np.array(["g1", "g2"], dtype=object))
    np.save(tmp_path / "row_season.npy", np.array([2023, 2023]))
    np.save(tmp_path / "row_offense_is_home.npy", np.array([True, False]))
    cols = _COLUMNS[:3]
    (tmp_path / "columns.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cols)  # no trailing newline, matches design.py
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"corpus_hash": "abc123", "n_possessions": 2, "n_columns": 3})
    )

    art = load_artifacts(tmp_path)
    assert isinstance(art, Artifacts)
    assert art.X.shape == (2, 3)
    assert art.y.tolist() == [1.0, 2.0]
    assert art.fold.tolist() == [0, 4]
    assert art.row_game_id.tolist() == ["g1", "g2"]
    assert len(art.columns) == 3
    assert art.manifest["corpus_hash"] == "abc123"


def test_load_artifacts_rejects_misaligned_lengths(tmp_path):
    from scipy.sparse import save_npz

    X = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.float64))
    save_npz(tmp_path / "X.npz", X)
    np.save(tmp_path / "y.npy", np.array([1.0]))  # too short
    np.save(tmp_path / "fold.npy", np.array([0, 4]))
    np.save(tmp_path / "row_game_id.npy", np.array(["g1", "g2"], dtype=object))
    np.save(tmp_path / "row_season.npy", np.array([2023, 2023]))
    np.save(tmp_path / "row_offense_is_home.npy", np.array([True, False]))
    (tmp_path / "columns.jsonl").write_text(
        "\n".join(json.dumps(c) for c in _COLUMNS[:2])
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"corpus_hash": "z"}))

    with pytest.raises(ValueError):
        load_artifacts(tmp_path)
