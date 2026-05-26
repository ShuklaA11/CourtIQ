"""Regression tests for the raw-V3 -> lineup-Action adapter.

These lock in the four data-shape fixes that the reconstruction depends on:
substitution direction (personId is the OUTGOING player), incoming-name
resolution (accents, suffixes, initials, shared surnames), array-order-as-key
(actionNumber is not chronological), and technical-foul/ejection events not
seeding period starters.
"""

from __future__ import annotations

import pytest

from recon.adapter import AdapterError, parse_lineup_actions, parse_rosters, to_action_log
from recon.lineups import SUBSTITUTION


def _box(home_players, away_players=None):
    """Minimal box_traditional payload for parse_rosters."""
    def team(tid, players):
        return {
            "teamId": tid,
            "players": [
                {"personId": pid, "nameI": namei, "familyName": fam, "firstName": first}
                for pid, namei, fam, first in players
            ],
        }
    return {"boxScoreTraditional": {
        "homeTeam": team(1, home_players),
        "awayTeam": team(2, away_players or []),
    }}


def _roster(*players):
    return parse_rosters(_box(list(players)))[1]


def _resolve(name, *players, exclude=frozenset()):
    from recon.adapter import _resolve_incoming
    return _resolve_incoming(name, _roster(*players), exclude=exclude)


# --- name resolution ------------------------------------------------------- #

def test_resolves_bare_surname():
    assert _resolve("Poeltl", (10, "J. Poeltl", "Poeltl", "Jakob")) == 10


def test_resolves_accented_surname_from_ascii_description():
    # Description spells "Jokic"; the box roster keeps "Jokić".
    assert _resolve("Jokic", (20, "N. Jokić", "Jokić", "Nikola")) == 20


def test_resolves_suffix_in_roster_but_not_description():
    # "Butler III" on the roster, plain "Butler" in the description.
    assert _resolve("Butler", (30, "J. Butler III", "Butler III", "Jimmy")) == 30


def test_resolves_suffix_in_description_but_not_roster():
    assert _resolve("Martin Jr.", (40, "K. Martin", "Martin", "Kenyon")) == 40


def test_multiletter_initial_disambiguates_shared_surname():
    # "Ja. Green" -> Jalen, not Josh, when both Greens dress.
    pid = _resolve(
        "Ja. Green",
        (50, "Ja. Green", "Green", "Jalen"),
        (51, "Jo. Green", "Green", "Josh"),
    )
    assert pid == 50


def test_exact_suffix_beats_stripped_when_both_present():
    # "Jackson Jr." must pick the Jr., not the plain Jackson.
    pid = _resolve(
        "Jackson Jr.",
        (60, "J. Jackson Jr.", "Jackson Jr.", "Jaren"),
        (61, "A. Jackson", "Jackson", "Andre"),
    )
    assert pid == 60


def test_ambiguous_shared_surname_raises():
    with pytest.raises(AdapterError, match="ambiguous"):
        _resolve("Williams", (70, "R. Williams", "Williams", "Robert"),
                 (71, "G. Williams", "Williams", "Grant"))


def test_exclude_disambiguates_shared_surname():
    # With one Williams already on the floor, the incoming one is the other.
    pid = _resolve("Williams",
                   (70, "R. Williams", "Williams", "Robert"),
                   (71, "G. Williams", "Williams", "Grant"),
                   exclude=frozenset({70}))
    assert pid == 71


def test_unknown_name_raises():
    with pytest.raises(AdapterError, match="not found"):
        _resolve("Nobody", (80, "A. Somebody", "Somebody", "Al"))


# --- action parsing -------------------------------------------------------- #

def _pbp(actions):
    return {"game": {"gameId": "x", "actions": actions}}


def _sub(action_number, person_id, desc, period=1, clock="PT06M00.00S"):
    return {"actionNumber": action_number, "period": period, "clock": clock,
            "teamId": 1, "personId": person_id, "actionType": "Substitution",
            "subType": "", "description": desc}


def test_action_log_carries_score_before_action_not_post_action_score():
    pbp = _pbp([
        {
            "actionNumber": 1, "period": 1, "clock": "PT11M00.00S",
            "teamId": 1, "actionType": "Made Shot", "shotValue": 2,
            "scoreHome": "2", "scoreAway": "0", "description": "make",
        },
        {
            "actionNumber": 2, "period": 1, "clock": "PT10M30.00S",
            "teamId": 2, "actionType": "Made Shot", "shotValue": 3,
            "scoreHome": "2", "scoreAway": "3", "description": "make",
        },
    ])

    rows = to_action_log(pbp)

    assert (rows[0]["score_home_before"], rows[0]["score_away_before"]) == (0, 0)
    assert (rows[1]["score_home_before"], rows[1]["score_away_before"]) == (2, 0)


def test_substitution_out_is_personid_in_from_description():
    box = _box([(10, "J. Poeltl", "Poeltl", "Jakob"), (11, "S. Barnes", "Barnes", "Scottie")])
    rosters = parse_rosters(box)
    actions = parse_lineup_actions(_pbp([_sub(99, 10, "SUB: Barnes FOR Poeltl")]), rosters)
    assert len(actions) == 1
    sub = actions[0]
    assert sub.action_type == SUBSTITUTION
    assert sub.sub_out == 10   # personId == outgoing
    assert sub.sub_in == 11    # from the description


def test_order_is_array_index_not_action_number():
    # actionNumber is deliberately descending; array order must win.
    box = _box([(10, "A. One", "One", "Al")])
    rosters = parse_rosters(box)
    pbp = _pbp([
        {"actionNumber": 900, "period": 1, "clock": "PT12M00.00S", "teamId": 1,
         "personId": 10, "actionType": "Made Shot"},
        {"actionNumber": 5, "period": 1, "clock": "PT11M00.00S", "teamId": 1,
         "personId": 10, "actionType": "Rebound"},
    ])
    orders = [a.order for a in parse_lineup_actions(pbp, rosters)]
    assert orders == [0, 1]


def test_technical_foul_and_ejection_do_not_seed_person():
    box = _box([(10, "A. One", "One", "Al")])
    rosters = parse_rosters(box)
    pbp = _pbp([
        {"actionNumber": 1, "period": 1, "clock": "PT06M00.00S", "teamId": 1,
         "personId": 10, "actionType": "Foul", "subType": "Technical",
         "description": "One T.FOUL"},
        {"actionNumber": 2, "period": 1, "clock": "PT05M00.00S", "teamId": 1,
         "personId": 10, "actionType": "Ejection", "description": "One ejected"},
    ])
    actions = parse_lineup_actions(pbp, rosters)
    assert all(a.person_id is None for a in actions)  # neither proves on-floor


def test_period_markers_are_dropped():
    box = _box([(10, "A. One", "One", "Al")])
    rosters = parse_rosters(box)
    pbp = _pbp([
        {"actionNumber": 1, "period": 1, "clock": "PT12M00.00S", "teamId": 0,
         "personId": 0, "actionType": "period", "subType": "start"},
        {"actionNumber": 2, "period": 1, "clock": "PT11M00.00S", "teamId": 1,
         "personId": 10, "actionType": "Made Shot"},
    ])
    actions = parse_lineup_actions(pbp, rosters)
    assert [a.action_type for a in actions] == ["Made Shot"]
