"""Tests for the Sprint-3 Phase-5 model-vs-market comparison.

The heavy real-data run lives in `python -m winprob.market` (and `./market.sh`);
these tests pin the contract on synthetic inputs. The comparison is deliberately
at the PRE-GAME (opening) state: the market closing line is a single pre-tip
price, so the only honest alignment is the model's opening-state probability
versus the vig-free market probability. It uses the rating-aware tier-E logistic
because the sparse score+time model predicts the base rate for every game pre-tip
and cannot be compared to a team-specific line.

Pinned here:
1. Vig removal is the standard two-way normalization of implied probabilities.
2. Team labels map to NBA ids deterministically; an unknown label fails fast.
3. The opening state is the earliest possession of each game.
4. The comparison reports Brier/log-loss/calibration for BOTH model and market
   and a paired game-clustered CI on the difference; the gate checks the MODEL's
   pre-game calibration (a null vs the market is an honest, valid result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winprob import market


# --------------------------------------------------------------------------
# 1. Vig removal.
# --------------------------------------------------------------------------

def test_vig_free_home_prob_two_way_normalization():
    # dec 1.40 -> implied 0.7143; dec 3.00 -> implied 0.3333; overround 1.0476.
    p = market.vig_free_home_prob(np.array([1.40]), np.array([3.00]))
    assert p[0] == pytest.approx(0.7142857 / (0.7142857 + 0.3333333), abs=1e-6)


def test_vig_free_equal_odds_is_a_coin_flip():
    p = market.vig_free_home_prob(np.array([1.91]), np.array([1.91]))
    assert p[0] == pytest.approx(0.5)


def test_vig_free_probs_sum_to_one_with_complement():
    dh, da = np.array([1.5, 2.5]), np.array([2.6, 1.55])
    home = market.vig_free_home_prob(dh, da)
    away = market.vig_free_home_prob(da, dh)  # symmetry: swap roles
    assert np.allclose(home + away, 1.0)


# --------------------------------------------------------------------------
# 2. Team mapping + odds loading.
# --------------------------------------------------------------------------

def test_all_thirty_teams_are_mapped():
    assert len(market.MGM_TEAM_TO_ID) == 30
    assert len(set(market.MGM_TEAM_TO_ID.values())) == 30  # ids are distinct


def _write_odds_csv(path, rows):
    cols = ["game_date", "away_team", "home_team",
            "money_away_decimal_odds", "money_home_decimal_odds", "money_home_won"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_load_odds_maps_ids_and_computes_market_prob(tmp_path):
    csv = tmp_path / "odds.csv"
    _write_odds_csv(csv, [
        ["2025-10-21-10:00", "Golden State", "LA Lakers", 1.70, 2.18, False],
        ["2025-10-21-10:00", "Houston", "Oklahoma City", 3.00, 1.40, True],
    ])
    odds = market.load_odds(csv)
    lakers = odds[odds["home_id"] == market.MGM_TEAM_TO_ID["LA Lakers"]].iloc[0]
    assert lakers["away_id"] == market.MGM_TEAM_TO_ID["Golden State"]
    assert 0.0 < lakers["market_home_prob"] < 1.0
    # OKC heavily favored (1.40) -> market prob well above 0.5.
    okc = odds[odds["home_id"] == market.MGM_TEAM_TO_ID["Oklahoma City"]].iloc[0]
    assert okc["market_home_prob"] > 0.6


def test_load_odds_rejects_unknown_team(tmp_path):
    csv = tmp_path / "odds.csv"
    _write_odds_csv(csv, [["2025-10-21-10:00", "Fake City", "Boston", 2.0, 1.9, True]])
    with pytest.raises(ValueError):
        market.load_odds(csv)


# --------------------------------------------------------------------------
# 3. Opening-state extraction.
# --------------------------------------------------------------------------

def test_opening_state_is_earliest_possession_per_game():
    frame = pd.DataFrame({
        "game_id": ["g1", "g1", "g1", "g2", "g2"],
        "period": [1, 1, 2, 1, 1],
        "possession_number": [3, 1, 8, 5, 2],
        "home_score_differential": [2, 0, 6, -1, 0],
    })
    opening = market.opening_state_rows(frame)
    assert len(opening) == 2
    g1 = opening[opening["game_id"] == "g1"].iloc[0]
    assert g1["possession_number"] == 1 and g1["home_score_differential"] == 0


# --------------------------------------------------------------------------
# 4. Comparison metrics + gates + verdict.
# --------------------------------------------------------------------------

def test_comparison_metrics_scores_both_and_pairs_the_difference():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.uniform(size=n) < 0.55).astype(float)
    # Market slightly sharper than the model (closer to the truth).
    p_market = np.clip(0.7 * y + 0.15 + rng.normal(0, 0.05, n), 0.02, 0.98)
    p_model = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.10, n), 0.02, 0.98)
    game_ids = np.array([f"g{i}" for i in range(n)])
    m = market.comparison_metrics(y, p_model, p_market, game_ids)
    assert m["model"]["brier"] > 0 and m["market"]["brier"] > 0
    assert set(m["paired_diff"]["market_minus_model"]["brier"]) == {"lo", "hi", "point"}
    assert "intercept" in m["model"]["calibration"]
    assert "correlation" in m


def _stub_metrics(model_calibrated: bool, market_brier=0.20, model_brier=0.22) -> dict:
    intercept = 0.02 if model_calibrated else 0.9
    slope = 1.02 if model_calibrated else 0.3
    return {
        "n_games": 769,
        "model": {"brier": model_brier, "log_loss": 0.6,
                  "calibration": {"intercept": intercept, "slope": slope},
                  "predictions_min": 0.1, "predictions_max": 0.9},
        "market": {"brier": market_brier, "log_loss": 0.55,
                   "predictions_min": 0.05, "predictions_max": 0.95},
        "paired_diff": {"market_minus_model": {
            "brier": {"lo": -0.03, "hi": -0.01, "point": -0.02},
            "log_loss": {"lo": -0.06, "hi": -0.02, "point": -0.04}}},
        "correlation": 0.8,
    }


def test_gate_passes_when_model_is_pregame_calibrated():
    gates = market.compute_market_gates(_stub_metrics(model_calibrated=True))
    assert gates["gate_model_pregame_calibrated"] is True
    assert gates["gate_predictions_in_open_interval"] is True


def test_gate_fails_when_model_is_miscalibrated():
    gates = market.compute_market_gates(_stub_metrics(model_calibrated=False))
    assert gates["gate_model_pregame_calibrated"] is False


def test_verdict_reports_market_sharper_without_calling_it_a_failure():
    metrics = _stub_metrics(model_calibrated=True)  # market brier < model brier
    metrics["gates"] = market.compute_market_gates(metrics)
    verdict = market.market_verdict(metrics)
    assert verdict["market_sharper"] is True
    assert verdict["model_pregame_calibrated"] is True
    # A sharper market is an expected, honest result — not a model failure.
    assert "market" in verdict["summary"].lower()


# --------------------------------------------------------------------------
# 5. End-to-end on a synthetic frame + synthetic odds.
# --------------------------------------------------------------------------

def _game(gid, season, split, home_id, away_id, home_net, away_net, date, rng, n=40):
    edge = 1.0 if home_net > away_net else -1.0
    reg = np.linspace(2880.0, 0.0, n)
    margin = edge * np.linspace(0.0, 12.0, n) + rng.normal(0, 1, n)
    return pd.DataFrame({
        "game_id": gid, "season": season, "split": split,
        "game_date": pd.Timestamp(date),
        "period": np.clip((4 - np.floor(reg / 720.0)).astype(int), 1, 4),
        "possession_number": np.arange(n),
        "home_score_differential": margin,
        "regulation_seconds_remaining": reg,
        "home_has_possession": rng.integers(0, 2, n).astype(bool),
        "home_win": bool(margin[-1] + edge > 0),
        "home_team_id": home_id, "away_team_id": away_id,
        "home_lineup_net_rapm": float(home_net), "away_lineup_net_rapm": float(away_net),
        "lineup_net_rapm_differential": float(home_net - away_net),
        "home_rated_players": 5, "away_rated_players": 5,
        "rapm_source_season": float(season - 1),
    })


def _synthetic_market_frame():
    rng = np.random.default_rng(3)
    parts, c = [], [0]

    def add(season, split, h, a, date):
        parts.append(_game(f"g{c[0]}", season, split, h, a,
                           4.0 if h == 1 else -4.0, 4.0 if a == 1 else -4.0, date, rng))
        c[0] += 1

    for s in (2022, 2023):
        for _ in range(4):
            add(s, "train", 1, 2, f"{s}-11-01")
            add(s, "train", 2, 1, f"{s}-11-02")
    for _ in range(4):
        add(2024, "validation", 1, 2, "2024-11-01")
        add(2024, "validation", 2, 1, "2024-11-02")
    for i in range(6):
        add(2025, "test", 1, 2, f"2025-11-0{i+1}")
        add(2025, "test", 2, 1, f"2025-11-1{i}")
    return pd.concat(parts, ignore_index=True)


def test_evaluate_market_end_to_end():
    df = _synthetic_market_frame()
    test_games = df[df["split"] == "test"].groupby("game_id").agg(
        date=("game_date", "first"), home_id=("home_team_id", "first"),
        away_id=("away_team_id", "first"), home_win=("home_win", "first")).reset_index()
    # Synthetic "market": a sharp line that leans toward the actual outcome.
    odds = pd.DataFrame({
        "date": pd.to_datetime(test_games["date"]).dt.date,
        "home_id": test_games["home_id"], "away_id": test_games["away_id"],
        "market_home_prob": np.where(test_games["home_win"], 0.7, 0.3),
    })
    metrics = market.evaluate_market(df, odds)
    assert metrics["n_games"] == len(test_games)
    assert 0.0 < metrics["model"]["predictions_min"]
    assert metrics["model"]["predictions_max"] < 1.0
    assert "market_minus_model" in metrics["paired_diff"]
    assert set(metrics["gates"]) >= {"gate_model_pregame_calibrated",
                                     "gate_predictions_in_open_interval"}
