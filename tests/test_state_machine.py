"""Synthetic-log tests for the on-floor state machine and minutes reconciliation.

The core scenario is a full 12-minute period with several substitutions on both
teams, hand-computed so every player's on-floor seconds are known. The tests
assert (a) reconstructed seconds match, (b) they reconcile to a box score, and
(c) each team's on-floor set is exactly five between every event.
"""

from __future__ import annotations

import pytest

from recon.clock import (
    cumulative_elapsed,
    elapsed_in_period,
    parse_clock,
    period_length_seconds,
    period_start_cumulative,
)
from recon.lineups import (
    Action,
    LineupSizeError,
    OnFloorError,
    LINEUP_SIZE,
    lineup_id,
    reconcile_minutes,
    recon_period_starters,
    reconstruct_lineups,
)

HOME = 100
AWAY = 200
Q_LEN = 720  # regulation quarter length in seconds


def remaining_clock(elapsed: float, period_len: int = Q_LEN) -> str:
    """Build a remaining-time clock string for `elapsed` seconds into a period."""
    remaining = period_len - elapsed
    minutes, seconds = divmod(remaining, 60)
    return f"PT{int(minutes)}M{seconds:05.2f}S"


def sub(order: int, period: int, team: int, elapsed: float, out: int, into: int) -> Action:
    return Action(
        order=order,
        period=period,
        clock=remaining_clock(elapsed),
        team_id=team,
        action_type="Substitution",
        sub_out=out,
        sub_in=into,
    )


def stat(order: int, period: int, team: int, elapsed: float, person: int) -> Action:
    return Action(
        order=order,
        period=period,
        clock=remaining_clock(elapsed),
        team_id=team,
        action_type="Made Shot",
        person_id=person,
    )


# --- The canonical full-period scenario -----------------------------------

# Home starters 1..5, away starters 6..10.
# Home: at 360s player 5 -> 11; at 600s player 11 -> 5 (5 returns).
# Away: at 540s player 10 -> 12.
HOME_STARTERS = frozenset({1, 2, 3, 4, 5})
AWAY_STARTERS = frozenset({6, 7, 8, 9, 10})
PERIOD_STARTERS = {
    (1, HOME): HOME_STARTERS,
    (1, AWAY): AWAY_STARTERS,
}
FULL_PERIOD_ACTIONS = [
    stat(1, 1, HOME, 30, 1),
    stat(2, 1, AWAY, 45, 6),
    sub(3, 1, HOME, 360, out=5, into=11),
    stat(4, 1, AWAY, 500, 8),
    sub(5, 1, AWAY, 540, out=10, into=12),
    sub(6, 1, HOME, 600, out=11, into=5),
    stat(7, 1, HOME, 700, 5),
]
EXPECTED_SECONDS = {
    1: 720, 2: 720, 3: 720, 4: 720,  # home starters who never leave
    5: 480,   # 0-360 and 600-720
    11: 240,  # 360-600
    6: 720, 7: 720, 8: 720, 9: 720,  # away starters who never leave
    10: 540,  # 0-540
    12: 180,  # 540-720
}


def test_player_seconds_match_hand_computation():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    assert recon.player_seconds == pytest.approx(EXPECTED_SECONDS)


def test_reconstruction_reconciles_to_box_score():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    # Box score seconds equal the hand-computed truth -> the minutes gate passes.
    assert reconcile_minutes(recon.player_seconds, EXPECTED_SECONDS) == []


def test_reconcile_flags_a_mismatch():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    corrupt_box = dict(EXPECTED_SECONDS)
    corrupt_box[5] = 500  # claim 20 extra seconds for player 5
    diffs = reconcile_minutes(recon.player_seconds, corrupt_box)
    assert [d.person_id for d in diffs] == [5]
    assert diffs[0].reconstructed_seconds == pytest.approx(480)
    assert diffs[0].box_seconds == pytest.approx(500)
    assert diffs[0].delta_seconds == pytest.approx(-20)


def test_each_team_has_exactly_five_between_every_event():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    # Every stint (the interval between consecutive lineup-change events) must
    # show exactly five players on each side.
    for stint in recon.stints:
        assert len(set(stint.home_five)) == LINEUP_SIZE
        assert len(set(stint.away_five)) == LINEUP_SIZE
    # And every action is tagged with a valid five-on-five lineup.
    for tagged in recon.tagged_actions:
        assert len(set(tagged.home_five)) == LINEUP_SIZE
        assert len(set(tagged.away_five)) == LINEUP_SIZE


def test_stints_are_contiguous_and_cover_the_full_period():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    home_stints = sorted(
        (s for s in recon.stints if s.team_id == HOME),
        key=lambda s: s.start_seconds,
    )
    # No gaps, no overlaps, spanning the whole period.
    assert home_stints[0].start_seconds == 0
    assert home_stints[-1].end_seconds == Q_LEN
    for prev, nxt in zip(home_stints, home_stints[1:]):
        assert nxt.start_seconds == prev.end_seconds
    # The stints partition the period timeline, so their durations sum to its length.
    assert sum(s.duration_seconds for s in home_stints) == pytest.approx(Q_LEN)
    # And the players who appear for home across those stints account for 5 * length.
    home_players = {1, 2, 3, 4, 5, 11}
    home_player_seconds = sum(
        v for k, v in recon.player_seconds.items() if k in home_players
    )
    assert home_player_seconds == pytest.approx(LINEUP_SIZE * Q_LEN)


def test_two_rows_per_interval_with_team_specific_lineup_ids():
    recon = reconstruct_lineups("G", HOME, AWAY, PERIOD_STARTERS, FULL_PERIOD_ACTIONS)
    starting_home = next(
        s for s in recon.stints if s.team_id == HOME and s.start_seconds == 0
    )
    starting_away = next(
        s for s in recon.stints if s.team_id == AWAY and s.start_seconds == 0
    )
    assert starting_home.lineup_id == lineup_id(HOME_STARTERS)
    assert starting_away.lineup_id == lineup_id(AWAY_STARTERS)
    # Both rows describe the same on-floor matchup.
    assert starting_home.home_five == starting_away.home_five
    assert starting_home.away_five == starting_away.away_five


# --- Substitution-wave collapsing -----------------------------------------

def test_simultaneous_subs_collapse_to_one_boundary():
    # Two home swaps at the same game-time (360s): players 4,5 out; 11,12 in.
    actions = [
        sub(1, 1, HOME, 360, out=4, into=11),
        sub(2, 1, HOME, 360, out=5, into=12),
    ]
    starters = {(1, HOME): HOME_STARTERS, (1, AWAY): AWAY_STARTERS}
    recon = reconstruct_lineups("G", HOME, AWAY, starters, actions)
    home_stints = sorted(
        (s for s in recon.stints if s.team_id == HOME),
        key=lambda s: s.start_seconds,
    )
    # Exactly two stints (before/after the wave) — no zero-duration middle stint.
    assert [(s.start_seconds, s.end_seconds) for s in home_stints] == [(0, 360), (360, 720)]
    assert home_stints[1].home_five == (1, 2, 3, 11, 12)


# --- Invariant / error paths ----------------------------------------------

def test_subbing_out_a_player_not_on_floor_raises():
    actions = [sub(1, 1, HOME, 360, out=99, into=11)]  # 99 never started
    starters = {(1, HOME): HOME_STARTERS, (1, AWAY): AWAY_STARTERS}
    with pytest.raises(OnFloorError):
        reconstruct_lineups("G", HOME, AWAY, starters, actions)


def test_starting_five_of_wrong_size_raises():
    bad = {(1, HOME): frozenset({1, 2, 3, 4}), (1, AWAY): AWAY_STARTERS}
    with pytest.raises(LineupSizeError):
        reconstruct_lineups("G", HOME, AWAY, bad, [stat(1, 1, HOME, 10, 1)])


# --- recon_period_starters back-inference ---------------------------------

def test_recon_period_starters_recovers_starters_and_round_trips():
    # Back-inference can only see a starter who leaves a trace, so give the
    # otherwise-silent starters (2,3,4,7,9) an early stat. Extra stats don't move
    # minutes, so the round-trip still reproduces EXPECTED_SECONDS.
    round_trip_actions = FULL_PERIOD_ACTIONS + [
        stat(8, 1, HOME, 100, 2),
        stat(9, 1, HOME, 110, 3),
        stat(10, 1, HOME, 120, 4),
        stat(11, 1, AWAY, 130, 7),
        stat(12, 1, AWAY, 140, 9),
    ]
    inferred = recon_period_starters(HOME, AWAY, round_trip_actions)
    assert inferred[(1, HOME)] == HOME_STARTERS
    assert inferred[(1, AWAY)] == AWAY_STARTERS
    # Feeding the inferred starters back through the machine reproduces minutes.
    recon = reconstruct_lineups("G", HOME, AWAY, inferred, round_trip_actions)
    assert recon.player_seconds == pytest.approx(EXPECTED_SECONDS)


def test_recon_period_starters_excludes_bench_players():
    # Player 11 first appears as a sub-IN -> bench, not a starter.
    actions = [
        stat(1, 1, HOME, 30, 1),
        stat(2, 1, HOME, 40, 2),
        stat(3, 1, HOME, 50, 3),
        stat(4, 1, HOME, 60, 4),
        sub(5, 1, HOME, 300, out=5, into=11),
        stat(6, 1, HOME, 400, 11),
    ]
    inferred = recon_period_starters(HOME, AWAY, actions)
    assert inferred[(1, HOME)] == frozenset({1, 2, 3, 4, 5})
    assert 11 not in inferred[(1, HOME)]


# --- Overtime spanning ----------------------------------------------------

def test_overtime_period_uses_five_minute_length():
    ot_starters = {
        (5, HOME): HOME_STARTERS,
        (5, AWAY): AWAY_STARTERS,
    }
    actions = [
        Action(order=1, period=5, clock="PT2M30.00S", team_id=HOME,
                action_type="Substitution", sub_out=5, sub_in=11),
    ]
    recon = reconstruct_lineups("G", HOME, AWAY, ot_starters, actions)
    # OT is 300s; player 5 plays the first 150s, player 11 the last 150s.
    assert recon.player_seconds[5] == pytest.approx(150)
    assert recon.player_seconds[11] == pytest.approx(150)
    assert recon.player_seconds[1] == pytest.approx(300)


# --- clock.py unit tests --------------------------------------------------

def test_parse_clock_variants():
    assert parse_clock("PT11M34.00S") == pytest.approx(694.0)
    assert parse_clock("PT0M0.00S") == pytest.approx(0.0)
    assert parse_clock("PT34.00S") == pytest.approx(34.0)  # sub-minute, no M group


def test_parse_clock_rejects_garbage():
    with pytest.raises(ValueError):
        parse_clock("11:34")


def test_period_lengths_and_cumulative_offsets():
    assert period_length_seconds(1) == 720
    assert period_length_seconds(4) == 720
    assert period_length_seconds(5) == 300  # first overtime
    assert period_start_cumulative(1) == 0
    assert period_start_cumulative(2) == 720
    assert period_start_cumulative(5) == 4 * 720
    assert period_start_cumulative(6) == 4 * 720 + 300


def test_elapsed_and_cumulative():
    assert elapsed_in_period(1, "PT11M00.00S") == pytest.approx(60)
    assert cumulative_elapsed(2, "PT11M00.00S") == pytest.approx(720 + 60)


def test_clock_out_of_range_raises():
    with pytest.raises(ValueError):
        elapsed_in_period(1, "PT13M00.00S")  # 780s remaining > 720s period
