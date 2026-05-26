"""Segmentation tests on hand-built synthetic action logs.

Each test constructs a tiny, fully-specified sequence so the expected possession
boundaries are unambiguous, then asserts both the boundary structure and — where
scoring is involved — that the points reconcile. The four cases the segmenter is
most likely to get wrong get dedicated tests: and-1, three-shot foul, an
offensive rebound extending a possession, and a period-ending buzzer.
"""

from __future__ import annotations

from recon.possessions import (
    assert_reconciles,
    points_by_team,
    recon_possessions,
)

GAME_ID = "0022100001"
HOME = 1610612737  # Atlanta, arbitrary
AWAY = 1610612738  # Boston, arbitrary

HOME_FIVE = (1, 2, 3, 4, 5)
AWAY_FIVE = (6, 7, 8, 9, 10)
LINEUPS = {HOME: HOME_FIVE, AWAY: AWAY_FIVE}


def _action(event, team=None, period=1, seconds=0.0, points=0, **extra):
    """Build one action row with the shared two-team lineup tag attached."""
    row = {
        "event": event,
        "team_id": team,
        "period": period,
        "seconds_elapsed": seconds,
        "points": points,
        "lineups": LINEUPS,
    }
    row.update(extra)
    return row


def test_possession_captures_score_before_its_opening_action():
    actions = [
        _action(
            "made_shot", team=HOME, seconds=10.0, points=2,
            score_home_before=0, score_away_before=0,
        ),
        _action(
            "made_shot", team=AWAY, seconds=20.0, points=3,
            score_home_before=2, score_away_before=0,
        ),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert possessions[0].home_score_before == 0
    assert possessions[0].away_score_before == 0
    assert possessions[1].home_score_before == 2
    assert possessions[1].away_score_before == 0


# --------------------------------------------------------------------------- #
# And-1: made FG + shooting foul + one FT — a single possession, not two.
# --------------------------------------------------------------------------- #

def test_and1_is_one_possession_worth_three():
    actions = [
        _action("made_shot", team=HOME, seconds=10.0, points=2, is_and1=True),
        _action("foul", team=AWAY, seconds=10.0),  # the shooting foul
        _action("free_throw", team=HOME, seconds=25.0, points=1,
                made=True, ft_index=1, ft_total=1, ft_kind="regular"),
        # Opponent answers, so we can see the and-1 possession closed cleanly.
        _action("made_shot", team=AWAY, seconds=40.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    home_poss = [p for p in possessions if p.offense_team_id == HOME]
    assert len(home_poss) == 1, "and-1 must not open a second possession"
    assert home_poss[0].points == 3  # 2 (FG) + 1 (FT)
    assert home_poss[0].defense_team_id == AWAY
    assert home_poss[0].offense_five == HOME_FIVE
    assert home_poss[0].defense_five == AWAY_FIVE
    assert points_by_team(possessions) == {HOME: 3, AWAY: 2}


def test_and1_detected_structurally_without_flag():
    # No is_and1 flag; the segmenter must infer it from the following FT.
    actions = [
        _action("made_shot", team=HOME, seconds=10.0, points=3),
        _action("foul", team=AWAY, seconds=10.0),
        _action("free_throw", team=HOME, seconds=20.0, points=1,
                made=True, ft_index=1, ft_total=1, ft_kind="regular"),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 1
    assert possessions[0].points == 4  # a four-point play


# --------------------------------------------------------------------------- #
# Three-shot foul: fouled on a missed 3, three FTs, single possession.
# --------------------------------------------------------------------------- #

def test_three_shot_foul_is_one_possession():
    actions = [
        _action("missed_shot", team=HOME, seconds=8.0),  # fouled 3PT attempt
        _action("foul", team=AWAY, seconds=8.0),
        _action("free_throw", team=HOME, seconds=15.0, points=1,
                made=True, ft_index=1, ft_total=3, ft_kind="regular"),
        _action("free_throw", team=HOME, seconds=17.0, points=0,
                made=False, ft_index=2, ft_total=3, ft_kind="regular"),
        _action("free_throw", team=HOME, seconds=19.0, points=1,
                made=True, ft_index=3, ft_total=3, ft_kind="regular"),
        _action("missed_shot", team=AWAY, seconds=30.0),
        _action("rebound", team=HOME, seconds=31.0),  # defensive rebound by HOME
    ]

    possessions = recon_possessions(GAME_ID, actions)

    home_poss = [p for p in possessions if p.offense_team_id == HOME]
    # One possession for the trip; one more opened by HOME's defensive rebound.
    assert home_poss[0].points == 2, "two made FTs across a three-shot trip"
    assert home_poss[0].possession_number == 1
    # The trip closed on the made final FT, so AWAY's miss is a separate poss.
    assert any(p.offense_team_id == AWAY for p in possessions)
    assert points_by_team(possessions) == {HOME: 2, AWAY: 0}


def test_missed_final_ft_does_not_close_until_rebound():
    actions = [
        _action("free_throw", team=HOME, seconds=15.0, points=1,
                made=True, ft_index=1, ft_total=2, ft_kind="regular"),
        _action("free_throw", team=HOME, seconds=17.0, points=0,
                made=False, ft_index=2, ft_total=2, ft_kind="regular"),
        _action("rebound", team=HOME, seconds=18.0),   # offensive rebound
        _action("made_shot", team=HOME, seconds=22.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 1, "missed final FT + OREB keeps one possession"
    assert possessions[0].points == 3  # 1 FT + 2 putback


# --------------------------------------------------------------------------- #
# Offensive rebound extends a possession (does not flip it).
# --------------------------------------------------------------------------- #

def test_offensive_rebound_extends_possession():
    actions = [
        _action("missed_shot", team=HOME, seconds=5.0),
        _action("rebound", team=HOME, seconds=6.0),     # offensive rebound
        _action("missed_shot", team=HOME, seconds=9.0),
        _action("rebound", team=HOME, seconds=10.0),    # second offensive rebound
        _action("made_shot", team=HOME, seconds=14.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 1
    poss = possessions[0]
    assert poss.offense_team_id == HOME
    assert poss.points == 2
    assert poss.start_seconds == 5.0
    assert poss.end_seconds == 14.0  # spans both offensive rebounds


def test_defensive_rebound_flips_possession():
    actions = [
        _action("missed_shot", team=HOME, seconds=5.0),
        _action("rebound", team=AWAY, seconds=6.0),     # defensive rebound
        _action("made_shot", team=AWAY, seconds=12.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 2
    assert possessions[0].offense_team_id == HOME
    assert possessions[0].points == 0
    assert possessions[1].offense_team_id == AWAY
    assert possessions[1].points == 2
    # The flip's start is the rebound, not the ensuing shot.
    assert possessions[1].start_seconds == 6.0


# --------------------------------------------------------------------------- #
# End-of-quarter buzzer.
# --------------------------------------------------------------------------- #

def test_buzzer_beater_make_closes_and_period_end_is_no_op():
    actions = [
        _action("made_shot", team=HOME, period=1, seconds=718.0, points=2),
        _action("period_end", period=1, seconds=720.0),
        _action("made_shot", team=AWAY, period=2, seconds=5.0, points=3),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    # The make closed period 1; period_end adds no phantom possession.
    p1 = [p for p in possessions if p.period == 1]
    assert len(p1) == 1
    assert p1[0].points == 2
    # Numbering resets each period.
    p2 = [p for p in possessions if p.period == 2]
    assert p2[0].possession_number == 1
    assert p2[0].points == 3


def test_missed_buzzer_beater_closes_at_period_end():
    actions = [
        _action("missed_shot", team=HOME, period=1, seconds=719.0),
        _action("period_end", period=1, seconds=720.0),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 1
    assert possessions[0].points == 0
    assert possessions[0].end_seconds == 720.0  # closed by the buzzer


def test_period_boundary_closes_open_possession_without_explicit_marker():
    # No period_end row: the period change itself must close the possession.
    actions = [
        _action("missed_shot", team=HOME, period=1, seconds=700.0),
        _action("made_shot", team=AWAY, period=2, seconds=10.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 2
    assert possessions[0].period == 1 and possessions[0].points == 0
    assert possessions[1].period == 2 and possessions[1].points == 2


# --------------------------------------------------------------------------- #
# Turnovers and flagrant / technical free throws.
# --------------------------------------------------------------------------- #

def test_turnover_flips_possession():
    actions = [
        _action("turnover", team=HOME, seconds=6.0),
        _action("made_shot", team=AWAY, seconds=11.0, points=2),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert possessions[0].offense_team_id == HOME and possessions[0].points == 0
    assert possessions[1].offense_team_id == AWAY and possessions[1].points == 2


def test_flagrant_free_throws_retain_possession():
    # HOME is fouled flagrantly, shoots two, KEEPS the ball and scores again.
    actions = [
        _action("free_throw", team=HOME, seconds=10.0, points=1,
                made=True, ft_index=1, ft_total=2, ft_kind="flagrant"),
        _action("free_throw", team=HOME, seconds=12.0, points=1,
                made=True, ft_index=2, ft_total=2, ft_kind="flagrant"),
        _action("made_shot", team=HOME, seconds=18.0, points=2),  # retained ball
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert len(possessions) == 1, "flagrant FTs must not flip possession"
    assert possessions[0].points == 4  # 2 FTs + retained-ball bucket


def test_technical_free_throw_makes_no_phantom_possession():
    # A technical is whistled during AWAY's possession; HOME shoots one and the
    # ball goes back to AWAY. No extra possession row; points still reconcile.
    actions = [
        _action("made_shot", team=AWAY, seconds=5.0, points=2),   # AWAY poss #1
        _action("free_throw", team=HOME, seconds=8.0, points=1,   # technical
                made=True, ft_index=1, ft_total=1, ft_kind="technical"),
        _action("made_shot", team=AWAY, seconds=14.0, points=2),  # AWAY poss #2
        _action("made_shot", team=HOME, seconds=20.0, points=3),  # HOME poss
    ]

    possessions = recon_possessions(GAME_ID, actions)

    away_poss = [p for p in possessions if p.offense_team_id == AWAY]
    home_poss = [p for p in possessions if p.offense_team_id == HOME]
    assert len(away_poss) == 2, "technical must not add an AWAY possession"
    assert len(home_poss) == 1, "the technical folds into HOME's real possession"
    # 1 (technical) folded into HOME's next possession + 3 (FG) = 4.
    assert points_by_team(possessions) == {AWAY: 4, HOME: 4}


def test_technical_at_buzzer_still_reconciles():
    # Technical shot with no later possession for that team: it must still land.
    actions = [
        _action("made_shot", team=AWAY, seconds=700.0, points=2),
        _action("free_throw", team=HOME, seconds=718.0, points=1,
                made=True, ft_index=1, ft_total=1, ft_kind="technical"),
        _action("period_end", seconds=720.0),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    assert points_by_team(possessions).get(HOME, 0) == 1
    assert points_by_team(possessions)[AWAY] == 2


# --------------------------------------------------------------------------- #
# The hard gate: a longer mixed sequence reconciles to an exact final score.
# --------------------------------------------------------------------------- #

def test_full_sequence_reconciles_to_final_score():
    actions = [
        # HOME scores a regular two.
        _action("made_shot", team=HOME, seconds=12.0, points=2),
        # AWAY misses, HOME defensive rebound (flip to HOME).
        _action("missed_shot", team=AWAY, seconds=30.0),
        _action("rebound", team=HOME, seconds=31.0),
        # HOME and-1 three-point play.
        _action("made_shot", team=HOME, seconds=45.0, points=2),
        _action("foul", team=AWAY, seconds=45.0),
        _action("free_throw", team=HOME, seconds=55.0, points=1,
                made=True, ft_index=1, ft_total=1, ft_kind="regular"),
        # AWAY turnover, HOME misses, AWAY defensive rebound.
        _action("made_shot", team=AWAY, seconds=70.0, points=3),
        _action("turnover", team=HOME, seconds=85.0),
        # AWAY three-shot foul: makes two of three.
        _action("missed_shot", team=AWAY, seconds=95.0),
        _action("foul", team=HOME, seconds=95.0),
        _action("free_throw", team=AWAY, seconds=100.0, points=1,
                made=True, ft_index=1, ft_total=3, ft_kind="regular"),
        _action("free_throw", team=AWAY, seconds=102.0, points=1,
                made=True, ft_index=2, ft_total=3, ft_kind="regular"),
        _action("free_throw", team=AWAY, seconds=104.0, points=0,
                made=False, ft_index=3, ft_total=3, ft_kind="regular"),
        _action("rebound", team=HOME, seconds=105.0),  # defensive rebound
        _action("made_shot", team=HOME, seconds=115.0, points=2),
        _action("period_end", seconds=120.0),
    ]

    possessions = recon_possessions(GAME_ID, actions)

    # HOME: 2 + 3 (and-1) + 2 = 7 ; AWAY: 3 + 2 (two of three FTs) = 5.
    assert_reconciles(possessions, {HOME: 7, AWAY: 5})
    # Every possession is attributed to one of the two teams.
    for p in possessions:
        assert p.offense_team_id in (HOME, AWAY)
        assert p.defense_team_id in (HOME, AWAY)
        assert p.offense_team_id != p.defense_team_id


def test_assert_reconciles_raises_on_mismatch():
    actions = [_action("made_shot", team=HOME, seconds=5.0, points=2)]
    possessions = recon_possessions(GAME_ID, actions)

    try:
        assert_reconciles(possessions, {HOME: 3, AWAY: 0})
    except AssertionError as exc:
        assert "reconcile" in str(exc)
    else:  # pragma: no cover - guard against a silent pass
        raise AssertionError("expected a reconciliation failure")
