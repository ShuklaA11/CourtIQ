"""Tests for availability-adjusted pre-game team strength (`winprob.pregame_availability`).

These pin the contracts the pre-game model consumes:

1. `expected_team_strength` is the prior-possession-weighted mean net_rating over
   the FULL season roster; unrated players (weight 0) never move it.
2. `available_team_strength` drops the game's out list and renormalizes over the
   remaining available rated players; with nobody out it equals expected exactly.
3. `injury_hit = available - expected` is `<= 0` when a good, heavily-weighted
   player sits and `~0` when only a fringe (low-weight) player is out.
4. `add_availability_features` appends exactly the five additive columns, is
   immutable, and reads ratings from the PRIOR season (S -> S-1).

Parts 1-4 are synthetic and deterministic. A final real-data sanity check asserts
that big known absences produce a clearly negative injury_hit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from winprob import pregame_availability as pa

# Synthetic ids. Prior-season strong star, mid starter, and an unrated fringe.
STAR, STARTER, FRINGE, UNRATED = 101, 102, 103, 104
TEAM_A, TEAM_B = 1610610001, 1610610002
GAME_ID = "0022200001"
SEASON = 2023  # ratings therefore read from season 2022.

DATA_DIR = Path("data/winprob")


def _roster() -> pa.TeamRoster:
    # STAR: +10 net, 1000 poss; STARTER: 0 net, 500 poss; FRINGE: -5 net, 100 poss.
    return pa.TeamRoster(
        team_id=TEAM_A,
        season=SEASON,
        player_ids=(STAR, STARTER, FRINGE),
        weights=(1000.0, 500.0, 100.0),
        net_ratings=(10.0, 0.0, -5.0),
    )


# --------------------------------------------------------------------------
# expected_team_strength
# --------------------------------------------------------------------------

def test_expected_strength_is_possession_weighted_mean():
    # Arrange
    roster = _roster()

    # Act
    strength = pa.expected_team_strength(roster)

    # Assert: (1000*10 + 500*0 + 100*-5) / 1600 = 9500 / 1600.
    assert strength == pytest.approx(9500.0 / 1600.0)


def test_unrated_players_have_zero_weight_and_no_influence():
    # Arrange: add an unrated player (replacement prior: weight 0, net 0).
    with_unrated = pa.TeamRoster(
        team_id=TEAM_A,
        season=SEASON,
        player_ids=(STAR, STARTER, FRINGE, UNRATED),
        weights=(1000.0, 500.0, 100.0, 0.0),
        net_ratings=(10.0, 0.0, -5.0, 0.0),
    )

    # Act / Assert: identical to the roster without them.
    assert pa.expected_team_strength(with_unrated) == pytest.approx(
        pa.expected_team_strength(_roster())
    )


def test_empty_roster_falls_back_to_replacement_prior():
    empty = pa.TeamRoster(team_id=TEAM_A, season=SEASON, player_ids=(), weights=(), net_ratings=())
    assert pa.expected_team_strength(empty) == pa.REPLACEMENT_NET_RATING


# --------------------------------------------------------------------------
# available_team_strength + injury_hit
# --------------------------------------------------------------------------

def test_nobody_out_available_equals_expected():
    roster = _roster()
    assert pa.available_team_strength(roster, frozenset()) == pytest.approx(
        pa.expected_team_strength(roster)
    )
    assert pa.injury_hit(roster, frozenset()) == pytest.approx(0.0)


def test_star_out_drives_injury_hit_clearly_negative():
    # Arrange
    roster = _roster()

    # Act: the +10, 1000-possession star sits.
    hit = pa.injury_hit(roster, frozenset({STAR}))

    # Assert: available is (500*0 + 100*-5)/600 = -0.833; expected is 5.9375.
    assert pa.available_team_strength(roster, frozenset({STAR})) == pytest.approx(
        -500.0 / 600.0
    )
    assert hit < -5.0


def test_fringe_out_barely_moves_injury_hit():
    # Arrange: two strong starters and a genuinely fringe bench player — tiny
    # prior weight (20 possessions) and only mildly below average.
    roster = pa.TeamRoster(
        team_id=TEAM_A,
        season=SEASON,
        player_ids=(STAR, STARTER, FRINGE),
        weights=(1000.0, 1000.0, 20.0),
        net_ratings=(8.0, 6.0, 0.0),
    )

    # Act: only the low-weight bench player sits.
    hit = pa.injury_hit(roster, frozenset({FRINGE}))

    # Assert: a low-weight absence barely moves the team mean — a fringe out is ~0.
    assert abs(hit) < 0.1


def test_unrated_player_out_changes_nothing():
    # Arrange: unrated players carry weight 0, so their absence is a no-op.
    roster = pa.TeamRoster(
        team_id=TEAM_A,
        season=SEASON,
        player_ids=(STAR, STARTER, UNRATED),
        weights=(1000.0, 500.0, 0.0),
        net_ratings=(10.0, 0.0, 0.0),
    )

    # Act / Assert
    assert pa.injury_hit(roster, frozenset({UNRATED})) == pytest.approx(0.0)


def test_everyone_rated_out_falls_back_to_replacement():
    roster = _roster()
    available = pa.available_team_strength(roster, frozenset({STAR, STARTER, FRINGE}))
    assert available == pa.REPLACEMENT_NET_RATING


# --------------------------------------------------------------------------
# build_team_rosters — prior-season lookup and season membership.
# --------------------------------------------------------------------------

def _ratings_frame() -> pd.DataFrame:
    # Prior season (2022) ratings only; a 2023 game must read these.
    return pd.DataFrame(
        {
            "player_id": [STAR, STARTER, FRINGE],
            "season": [2022, 2022, 2022],
            "net_rating": [10.0, 0.0, -5.0],
            "n_possessions": [1000, 500, 100],
        }
    )


def _availability_frame() -> pd.DataFrame:
    # TEAM_A dresses STAR/STARTER/FRINGE/UNRATED; STAR is OUT this game.
    return pd.DataFrame(
        {
            "game_id": [GAME_ID] * 5,
            "team_id": [TEAM_A, TEAM_A, TEAM_A, TEAM_A, TEAM_B],
            "player_id": [STAR, STARTER, FRINGE, UNRATED, STARTER],
            "available": [False, True, True, True, True],
            "comment": ["DNP - Injury", "", "", "", ""],
        }
    )


def test_build_team_rosters_reads_prior_season_and_full_membership():
    # Arrange
    ratings = _ratings_frame()
    lookup = pa.build_prior_rating_lookup(ratings)

    # Act
    rosters = pa.build_team_rosters(
        _availability_frame(), {GAME_ID: SEASON}, lookup
    )

    # Assert: TEAM_A's 2023 roster is the FULL dressed list including the unrated,
    # weighted by PRIOR-season (2022) possessions; the unrated gets weight 0.
    roster = rosters[(TEAM_A, SEASON)]
    assert roster.player_ids == (STAR, STARTER, FRINGE, UNRATED)
    weight_by_id = dict(zip(roster.player_ids, roster.weights))
    assert weight_by_id[STAR] == 1000.0
    assert weight_by_id[UNRATED] == pa.REPLACEMENT_WEIGHT


# --------------------------------------------------------------------------
# add_availability_features — additive, immutable, prior-season.
# --------------------------------------------------------------------------

def _games_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "season": [SEASON],
            "home_team_id": [TEAM_A],
            "away_team_id": [TEAM_B],
        }
    )


def test_add_features_appends_exactly_the_additive_columns_immutably():
    # Arrange
    games = _games_frame()
    before = games.copy(deep=True)

    # Act
    out = pa.add_availability_features(games, _availability_frame(), _ratings_frame())

    # Assert: original untouched, output is a new frame with the five new columns.
    pd.testing.assert_frame_equal(games, before)
    assert out is not games
    for column in pa.AVAILABILITY_FEATURE_COLUMNS:
        assert column in out.columns
    assert list(games.columns) == [c for c in out.columns if c in games.columns]


def test_add_features_home_injury_hit_reflects_star_out():
    # Arrange: home TEAM_A has its star out; away TEAM_B has everyone available.
    games = _games_frame()

    # Act
    out = pa.add_availability_features(games, _availability_frame(), _ratings_frame())
    row = out.iloc[0]

    # Assert: home takes a clearly negative hit, away takes none, and the diff is
    # home minus away.
    assert row["home_injury_hit"] < -5.0
    assert row["away_injury_hit"] == pytest.approx(0.0)
    assert row["injury_hit_diff"] == pytest.approx(
        row["home_injury_hit"] - row["away_injury_hit"]
    )


def test_add_features_injury_hit_diff_is_home_minus_away():
    out = pa.add_availability_features(
        _games_frame(), _availability_frame(), _ratings_frame()
    )
    row = out.iloc[0]
    assert row["injury_hit_diff"] == pytest.approx(
        row["home_injury_hit"] - row["away_injury_hit"]
    )


# --------------------------------------------------------------------------
# Real-data sanity: big known absences produce a clearly negative injury_hit.
# --------------------------------------------------------------------------

def _read_or_skip(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"missing {path}; run the winprob pipeline to materialize it")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    mart = _read_or_skip(DATA_DIR / "fct_game_states.parquet")
    availability = _read_or_skip(DATA_DIR / "game_availability.parquet")
    ratings = _read_or_skip(Path("data/rapm/bayes_ratings.parquet"))
    games = (
        mart[["game_id", "season", "home_team_id", "away_team_id"]]
        .drop_duplicates("game_id")
        .reset_index(drop=True)
    )
    games["game_id"] = games["game_id"].astype(str)
    return pa.add_availability_features(games, availability, ratings)


def test_real_data_big_absences_are_clearly_negative(real_features):
    # The worst injury_hit over thousands of real games must reflect a genuine
    # star-out night, not a rounding wobble.
    assert real_features["home_injury_hit"].min() < -0.5
    assert real_features["away_injury_hit"].min() < -0.5


def test_real_data_injury_hit_is_bounded_and_mostly_zero(real_features):
    # Most nights nobody meaningful is out, so the median hit is exactly 0; and a
    # single team's mean can never swing by an absurd amount.
    assert real_features["home_injury_hit"].median() == pytest.approx(0.0)
    assert real_features["home_injury_hit"].abs().max() < 15.0
    assert real_features["away_injury_hit"].abs().max() < 15.0
