"""Tests for the leakage-safe pre-tip availability layer (`winprob.availability`).

These pin the contracts the rest of the pre-game stack depends on:

1. `classify_player_availability` is the single, deterministic rule — minutes on
   the floor win over any marker, empty / Coach's Decision are available, only an
   explicit marker is unavailable, and an unrecognized note defaults to available.
2. `parse_box_availability` reads BOTH teams of one V3 payload, keying on
   `personId` (== RAPM player_id) and `gameId`.
3. `build_game_availability` folds a raw box tree keeping ONLY mart games, and
   `rollup_unavailable` collapses to a per-(game, team) inactive list.

Synthetic, deterministic, no network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from winprob import availability as av

HOME_TEAM, AWAY_TEAM = 1610612749, 1610612744  # arbitrary team ids
GAME_ID = "0022100001"


# --------------------------------------------------------------------------
# classify_player_availability
# --------------------------------------------------------------------------

def test_minutes_played_wins_over_any_marker():
    # Arrange: a comment that would otherwise be UNAVAILABLE, but the player
    # actually logged minutes.
    # Act
    result = av.classify_player_availability("DNP - Injury/Illness", 12.5)

    # Assert: being on the floor is proof of availability.
    assert result is True


def test_empty_comment_is_available():
    # Arrange / Act
    result = av.classify_player_availability("", 0.0)

    # Assert
    assert result is True


def test_coach_decision_is_available():
    # Arrange / Act
    result = av.classify_player_availability("DNP - Coach's Decision", 0.0)

    # Assert: a healthy scratch could have played.
    assert result is True


@pytest.mark.parametrize(
    "comment",
    [
        "DNP - Injury/Illness",
        "DND - Injury/Illness",
        "NWT - Not With Team",
        "DND - Rest",
        "NWT - Personal",
        "DNP - League Suspension",
        "DND_LEAGUE_SUSPENSION",
        "NWT - Health and Safety Protocols",
        "DND-Return to Competition Reconditioning",
    ],
)
def test_markers_are_unavailable(comment):
    # Arrange / Act
    result = av.classify_player_availability(comment, 0.0)

    # Assert
    assert result is False


def test_marker_matching_is_case_insensitive():
    # Arrange / Act
    result = av.classify_player_availability("dnp - INJURY/illness", 0.0)

    # Assert
    assert result is False


def test_unrecognized_comment_defaults_to_available():
    # Arrange: a real feed comment that matches no marker (concussion protocol).
    # Act
    result = av.classify_player_availability("DND - Concussion Protocol", 0.0)

    # Assert: the layer only flags OUT on an explicit marker.
    assert result is True


# --------------------------------------------------------------------------
# parse_box_availability
# --------------------------------------------------------------------------

def _player(person_id, comment="", minutes=None):
    stats = {} if minutes is None else {"minutes": minutes}
    return {"personId": person_id, "comment": comment, "statistics": stats}


def _payload(home_players, away_players, game_id=GAME_ID):
    return {
        "boxScoreTraditional": {
            "gameId": game_id,
            "homeTeam": {"teamId": HOME_TEAM, "players": home_players},
            "awayTeam": {"teamId": AWAY_TEAM, "players": away_players},
        }
    }


def test_parse_box_reads_both_teams():
    # Arrange
    payload = _payload(
        home_players=[_player(101, minutes="30:00"), _player(102, "DNP - Injury/Illness")],
        away_players=[_player(201, minutes="10:00")],
    )

    # Act
    records = av.parse_box_availability(payload)

    # Assert
    assert len(records) == 3
    by_pid = {r.player_id: r for r in records}
    assert by_pid[101].team_id == HOME_TEAM and by_pid[101].available is True
    assert by_pid[102].team_id == HOME_TEAM and by_pid[102].available is False
    assert by_pid[201].team_id == AWAY_TEAM and by_pid[201].available is True
    assert all(r.game_id == GAME_ID for r in records)


def test_parse_box_missing_key_raises():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="boxScoreTraditional"):
        av.parse_box_availability({"meta": {}})


def test_parse_box_does_not_mutate_input():
    # Arrange
    payload = _payload([_player(101, minutes="30:00")], [_player(201, "DND - Rest")])
    snapshot = json.dumps(payload, sort_keys=True)

    # Act
    av.parse_box_availability(payload)

    # Assert
    assert json.dumps(payload, sort_keys=True) == snapshot


# --------------------------------------------------------------------------
# build_game_availability + rollup
# --------------------------------------------------------------------------

def _write_box(tmp_path, game_id, home_players, away_players):
    box_dir = tmp_path / "raw" / "box_traditional"
    box_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload(home_players, away_players, game_id=game_id)
    (box_dir / f"{game_id}.json").write_text(json.dumps(payload))
    return box_dir


def test_build_keeps_only_mart_games(tmp_path):
    # Arrange: two raw files, only one of which is in the mart.
    box_dir = _write_box(
        tmp_path, "0022100001",
        [_player(101, minutes="30:00"), _player(102, "DNP - Injury/Illness")],
        [_player(201, minutes="10:00")],
    )
    _write_box(tmp_path, "0022100099", [_player(999, minutes="5:00")], [_player(998)])

    # Act
    frame = av.build_game_availability(box_dir, mart_game_ids=["0022100001"])

    # Assert: the off-mart game is dropped entirely.
    assert list(frame.columns) == list(av.AVAILABILITY_COLUMNS)
    assert set(frame["game_id"].unique()) == {"0022100001"}
    assert len(frame) == 3
    assert 999 not in set(frame["player_id"])


def test_rollup_lists_unavailable_per_team(tmp_path):
    # Arrange
    box_dir = _write_box(
        tmp_path, "0022100001",
        [
            _player(101, minutes="30:00"),
            _player(103, "DND - Injury/Illness"),
            _player(102, "NWT - Personal"),
        ],
        [_player(201, minutes="10:00")],
    )
    frame = av.build_game_availability(box_dir, mart_game_ids=["0022100001"])

    # Act
    rollup = av.rollup_unavailable(frame)

    # Assert: one row per (game, team); home has two OUT (sorted), away has none.
    assert list(rollup.columns) == list(av.ROLLUP_COLUMNS)
    home = rollup[rollup["team_id"] == HOME_TEAM].iloc[0]
    away = rollup[rollup["team_id"] == AWAY_TEAM].iloc[0]
    assert home["unavailable_player_ids"] == [102, 103]
    assert home["n_unavailable"] == 2
    assert away["unavailable_player_ids"] == []
    assert away["n_unavailable"] == 0


def test_rollup_missing_column_raises():
    # Arrange
    frame = pd.DataFrame({"game_id": ["g"], "team_id": [1]})

    # Act / Assert
    with pytest.raises(ValueError, match="missing columns"):
        av.rollup_unavailable(frame)
