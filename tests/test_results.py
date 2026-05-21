"""Unit tests for the pure RAPM results diagnostics (rapm.results)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rapm.results import ci_width_by_tercile, leaderboard, season_to_season_stability


def test_ci_width_terciles_split_and_widths():
    # 9 player-seasons; width deliberately shrinks as possessions grow.
    n_poss = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    low = np.zeros(9)
    high = np.array([9.0, 8, 7, 6, 5, 4, 3, 2, 1])  # widest at low possessions
    out = ci_width_by_tercile(n_poss, low, high)
    assert [t["tercile"] for t in out] == ["low", "mid", "high"]
    assert all(t["n_player_seasons"] == 3 for t in out)
    # low-possession tercile is the widest, high the narrowest (honest uncertainty).
    assert out[0]["mean_ci_width"] > out[1]["mean_ci_width"] > out[2]["mean_ci_width"]


def test_season_to_season_pairs_and_correlation():
    # Player 1 present in 2021,2022,2023; player 2 only 2021 -> 2 consecutive pairs.
    pid = np.array([1, 1, 1, 2])
    season = np.array([2021, 2022, 2023, 2021])
    net = np.array([2.0, 3.0, 4.0, -1.0])
    out = season_to_season_stability(pid, season, net)
    assert out["n_pairs"] == 2  # (2021->2022) and (2022->2023) for player 1
    assert out["pearson_r"] == pytest.approx(1.0)  # perfectly increasing -> r=1


def test_season_to_season_insufficient_pairs_returns_none():
    pid = np.array([1, 2])
    season = np.array([2021, 2023])  # no consecutive-season pairs
    net = np.array([1.0, 2.0])
    out = season_to_season_stability(pid, season, net)
    assert out["pearson_r"] is None
    assert out["n_pairs"] == 0


def test_leaderboard_orders_by_net_desc():
    df = pd.DataFrame({
        "player_id": [10, 11, 12],
        "season": [2023, 2023, 2023],
        "net_rating": [1.0, 5.0, 3.0],
        "net_ci_low": [-1.0, 3.0, 1.0],
        "net_ci_high": [3.0, 7.0, 5.0],
        "n_possessions": [5000, 6000, 7000],
    })
    board = leaderboard(df, k=2)
    assert [r["player_id"] for r in board] == [11, 12]  # 5.0 then 3.0
