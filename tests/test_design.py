"""Unit tests for the RAPM design-matrix assembler (rapm.design).

Pure-function tests on tiny synthetic possession sets — no warehouse needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapm.design import (
    PLAYERS_PER_SIDE,
    Column,
    build_design,
    corpus_hash,
    split_games,
)


def _arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Turn a list of possession dicts into the columnar arrays build_design wants."""
    return {
        "game_id": np.array([r["game_id"] for r in rows]),
        "season": np.array([r["season"] for r in rows], dtype=np.int64),
        "offense_five": np.array([r["offense"] for r in rows], dtype=np.int64),
        "defense_five": np.array([r["defense"] for r in rows], dtype=np.int64),
        "points": np.array([r["points"] for r in rows], dtype=np.float64),
        "offense_is_home": np.array([r["home"] for r in rows], dtype=bool),
        "duration_seconds": np.array([r.get("dur", 15.0) for r in rows], dtype=np.float64),
    }


def _col_index(columns: list[Column], kind, side, player, season) -> int:
    for i, c in enumerate(columns):
        if (c.kind, c.side, c.player_id, c.season) == (kind, side, player, season):
            return i
    raise KeyError((kind, side, player, season))


# A base corpus: two teams (1-5 vs 6-10) alternating offense across three
# possessions in season 2023, plus two 2024 possessions where players 5 and 11
# fall below a 2-possession threshold.
BASE = [
    {"game_id": "G1", "season": 2023, "offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10], "points": 2, "home": True},
    {"game_id": "G1", "season": 2023, "offense": [6, 7, 8, 9, 10], "defense": [1, 2, 3, 4, 5], "points": 0, "home": False},
    {"game_id": "G2", "season": 2023, "offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10], "points": 3, "home": True},
    {"game_id": "G3", "season": 2024, "offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10], "points": 1, "home": False},
    {"game_id": "G4", "season": 2024, "offense": [1, 2, 3, 4, 11], "defense": [6, 7, 8, 9, 10], "points": 2, "home": True},
]


def test_every_row_has_ten_player_entries():
    d = build_design(min_possessions=2, **_arrays(BASE))
    player_cols = [i for i, c in enumerate(d.columns) if c.kind in ("player", "replacement")]
    block = d.X[:, player_cols]
    # Each possession contributes exactly 5 offensive + 5 defensive +1s.
    assert (np.asarray(block.sum(axis=1)).ravel() == 2 * PLAYERS_PER_SIDE).all()


def test_home_column_flags_only_home_offense_rows():
    d = build_design(min_possessions=2, **_arrays(BASE))
    home_col = _col_index(d.columns, "home", None, None, None)
    got = np.asarray(d.X[:, home_col].todense()).ravel()
    assert (got == np.array([1, 0, 1, 0, 1])).all()
    assert d.columns[home_col].penalized is False


def test_response_is_points():
    d = build_design(min_possessions=2, **_arrays(BASE))
    assert (d.y == np.array([2, 0, 3, 1, 2], dtype=float)).all()


def test_offense_and_defense_use_distinct_columns():
    d = build_design(min_possessions=2, **_arrays(BASE))
    o = _col_index(d.columns, "player", "O", 1, 2023)
    dfn = _col_index(d.columns, "player", "D", 1, 2023)
    assert o != dfn
    # Player 1 is on offense in rows 0 and 2, on defense in row 1.
    assert np.asarray(d.X[:, o].todense()).ravel().tolist() == [1, 0, 1, 0, 0]
    assert np.asarray(d.X[:, dfn].todense()).ravel().tolist() == [0, 1, 0, 0, 0]


def test_same_player_different_seasons_get_separate_columns():
    d = build_design(min_possessions=2, **_arrays(BASE))
    c2023 = _col_index(d.columns, "player", "O", 1, 2023)
    c2024 = _col_index(d.columns, "player", "O", 1, 2024)
    assert c2023 != c2024


def test_fringe_player_pooled_to_replacement():
    d = build_design(min_possessions=2, **_arrays(BASE))
    # Player 11 (one possession, 2024) has no own column; player 5 in 2024 (one
    # offensive possession) is also below threshold -> both pool to replacement.
    with pytest.raises(KeyError):
        _col_index(d.columns, "player", "O", 11, 2024)
    repl_o_2024 = _col_index(d.columns, "replacement", "O", None, 2024)
    # Row 4 (index 4) offense = [1,2,3,4,11]; player 11 lands in replacement-O.
    assert np.asarray(d.X[:, repl_o_2024].todense()).ravel()[4] == 1


def test_threshold_keeps_players_at_or_above_cutoff():
    d = build_design(min_possessions=2, **_arrays(BASE))
    # Players 1-5 in 2023 appear 3 times each (>=2) -> own columns.
    for p in range(1, 6):
        assert _col_index(d.columns, "player", "O", p, 2023) >= 0


def test_wrong_lineup_width_raises():
    bad = _arrays(BASE)
    bad["offense_five"] = bad["offense_five"][:, :4]  # only four players
    with pytest.raises(ValueError):
        build_design(min_possessions=2, **bad)


def test_split_games_no_game_straddles_folds():
    d = build_design(min_possessions=2, **_arrays(BASE))
    folds = split_games(d.row_game_id, d.row_season, n_folds=2, seed=7)
    # Rows 0 and 1 share game G1 -> must be in the same fold.
    assert folds[0] == folds[1]
    # No game id appears in more than one fold.
    by_game: dict[str, set[int]] = {}
    for g, f in zip(d.row_game_id, folds):
        by_game.setdefault(g, set()).add(int(f))
    assert all(len(fs) == 1 for fs in by_game.values())


def test_split_games_is_deterministic_in_seed():
    d = build_design(min_possessions=2, **_arrays(BASE))
    a = split_games(d.row_game_id, d.row_season, n_folds=2, seed=7)
    b = split_games(d.row_game_id, d.row_season, n_folds=2, seed=7)
    assert (a == b).all()


def test_corpus_hash_changes_with_corpus():
    gids = np.array(["G1", "G1", "G2"])
    h1 = corpus_hash(gids, 3)
    h2 = corpus_hash(np.array(["G1", "G1", "G2", "G3"]), 4)
    assert h1 != h2
    assert corpus_hash(gids, 3) == h1  # deterministic
