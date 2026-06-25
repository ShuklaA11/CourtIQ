"""Tests for the Sprint-6 Polymarket pre-game benchmark — pure parsing.

The heavy real-data pull lives in `python -m winprob.polymarket_pull` (and
`./polymarket.sh`); these tests pin the parsing contract on a small saved JSON
fixture with NO network. Polymarket is a real-money prediction market benchmark,
structurally different from the Sprint-3 sportsbook lines, so the honest question
is unchanged ("model vs a prediction market") and the leakage law is strict.

Pinned here:
1. Slug construction and candidate ordering (exact date + orientation first, then
   the flipped orientation and ±1 day for the known edge cases).
2. The moneyline is parsed from JSON-string fields; missing fields make a game
   uncoverable (None), never an error.
3. Orientation is self-correcting: the home token is picked by matching the slug's
   tricodes to team ids, so a flipped slug is oriented correctly and a wrong game
   is rejected.
4. The leakage law: the chosen price is the last tick with t <= tip; a post-tip
   tick is never chosen, and no pre-tip tick yields None.
5. Vig-free normalization of the two tokens' last prices.
6. The 24h window rule: a stale last tick (or a too-stale token) is uncovered.
7. The coverage rollup counts covered games, thin markets, and reasons.
"""

from __future__ import annotations

import datetime as dt

import pytest

from winprob import polymarket

# --------------------------------------------------------------------------
# Fixtures — a realistic event JSON with the JSON-string fields Polymarket sends.
# --------------------------------------------------------------------------

TRICODE_TO_ID = {1610612747: "lal", 1610612744: "gsw"}
HOME_ID, AWAY_ID = 1610612747, 1610612744  # Lakers host Warriors
TIP_TS = 1761098400  # 2025-10-22 02:00:00+00


def _event(slug: str = "nba-gsw-lal-2025-10-21") -> dict:
    """An events-API event whose moneyline mirrors a real Polymarket payload."""
    return {
        "slug": slug,
        "markets": [
            {
                "slug": slug,
                "outcomes": '["Warriors", "Lakers"]',       # [away, home]
                "clobTokenIds": '["AWAY_TOKEN", "HOME_TOKEN"]',  # [away, home]
                "gameStartTime": "2025-10-22 02:00:00+00",
                "volumeNum": 3_674_281.73,
            }
        ],
    }


# --------------------------------------------------------------------------
# 1. Slug construction + candidate ordering.
# --------------------------------------------------------------------------

def test_event_slug_lowercases_tricodes_and_orders_away_home():
    assert polymarket.event_slug("GSW", "LAL", "2025-10-21") == "nba-gsw-lal-2025-10-21"


def test_candidate_slugs_prefer_exact_date_and_orientation_first():
    cands = polymarket.candidate_slugs("LAL", "GSW", dt.date(2025, 10, 21))
    # Exact ET date, away-home orientation is the first guess.
    assert cands[0] == "nba-gsw-lal-2025-10-21"
    # The flipped orientation on the same date is tried before any ±1 day slug.
    assert cands[1] == "nba-lal-gsw-2025-10-21"
    assert "nba-gsw-lal-2025-10-20" in cands
    assert "nba-gsw-lal-2025-10-22" in cands
    assert len(cands) == len(set(cands))  # no duplicates


# --------------------------------------------------------------------------
# 2. Event-market parsing.
# --------------------------------------------------------------------------

def test_parse_event_market_reads_json_string_fields():
    market = polymarket.parse_event_market(_event())
    assert market is not None
    assert market.outcomes == ("Warriors", "Lakers")
    assert market.token_ids == ("AWAY_TOKEN", "HOME_TOKEN")
    assert int(market.game_start.timestamp()) == TIP_TS
    assert market.volume == pytest.approx(3_674_281.73)


def test_parse_event_market_missing_fields_is_uncoverable_not_error():
    event = _event()
    del event["markets"][0]["gameStartTime"]
    assert polymarket.parse_event_market(event) is None


def test_parse_event_market_ignores_non_moneyline_markets():
    event = _event()
    # A prop market whose slug differs from the event slug is not the moneyline.
    event["markets"].insert(0, {"slug": "nba-gsw-lal-2025-10-21-total", "outcomes": "[]"})
    market = polymarket.parse_event_market(event)
    assert market is not None
    assert market.token_ids == ("AWAY_TOKEN", "HOME_TOKEN")


# --------------------------------------------------------------------------
# 3. Orientation — self-correcting via team ids.
# --------------------------------------------------------------------------

def test_home_token_index_normal_slug_picks_second_position():
    market = polymarket.parse_event_market(_event("nba-gsw-lal-2025-10-21"))
    idx = polymarket.home_token_index(market, TRICODE_TO_ID, HOME_ID, AWAY_ID)
    assert idx == 1  # home (lal) is the second slug tricode -> home token index 1


def test_home_token_index_flipped_slug_is_corrected_by_team_id():
    # A flipped slug nba-<home>-<away> must still orient to the home token by id,
    # not by trusting the away-first position convention.
    market = polymarket.parse_event_market(_event("nba-lal-gsw-2025-10-21"))
    idx = polymarket.home_token_index(market, TRICODE_TO_ID, HOME_ID, AWAY_ID)
    assert idx == 0  # home (lal) is now the first slug tricode -> home token index 0


def test_home_token_index_rejects_a_wrong_game():
    # A slug whose tricodes are not this game's {home, away} is not a match.
    market = polymarket.parse_event_market(_event("nba-bos-mia-2025-10-21"))
    tri = {1610612738: "bos", 1610612748: "mia"}
    idx = polymarket.home_token_index(market, tri, HOME_ID, AWAY_ID)
    assert idx is None


# --------------------------------------------------------------------------
# 4. Pre-tip tick selection — the leakage law.
# --------------------------------------------------------------------------

def test_select_pretip_price_never_chooses_a_post_tip_tick():
    history = [
        {"t": TIP_TS - 600, "p": 0.44},
        {"t": TIP_TS - 60, "p": 0.45},   # last pre-tip tick
        {"t": TIP_TS + 7, "p": 0.99},    # post-tip — must be ignored
    ]
    price = polymarket.select_pretip_price(history, TIP_TS)
    assert price is not None
    assert price.price == 0.45
    assert price.seconds_before_tip == 60
    assert price.n_pretip_ticks == 2  # only the two pre-tip ticks are counted


def test_select_pretip_price_none_when_all_ticks_are_post_tip():
    history = [{"t": TIP_TS + 5, "p": 0.5}, {"t": TIP_TS + 600, "p": 0.6}]
    assert polymarket.select_pretip_price(history, TIP_TS) is None


def test_select_pretip_price_tick_exactly_at_tip_is_pre_tip():
    price = polymarket.select_pretip_price([{"t": TIP_TS, "p": 0.5}], TIP_TS)
    assert price is not None and price.seconds_before_tip == 0


# --------------------------------------------------------------------------
# 5. Vig-free normalization.
# --------------------------------------------------------------------------

def test_vig_free_home_prob_normalizes_two_token_prices():
    # Home 0.45, away 0.57 (overround-inflated) -> 0.45 / 1.02.
    assert polymarket.vig_free_home_prob(0.45, 0.57) == pytest.approx(0.45 / 1.02)


def test_vig_free_home_prob_rejects_non_positive_sum():
    with pytest.raises(ValueError, match="non-positive"):
        polymarket.vig_free_home_prob(0.0, 0.0)


# --------------------------------------------------------------------------
# 6. The 24h window rule in build_snapshot_row.
# --------------------------------------------------------------------------

def test_build_snapshot_row_covered_within_window():
    home = polymarket.PretipPrice(price=0.45, seconds_before_tip=595, n_pretip_ticks=181)
    away = polymarket.PretipPrice(price=0.57, seconds_before_tip=590, n_pretip_ticks=180)
    row = polymarket.build_snapshot_row(
        "0022500002", dt.date(2025, 10, 21), HOME_ID, AWAY_ID, home, away, 3_674_281.73
    )
    assert row is not None
    assert row["market_home_prob"] == pytest.approx(0.45 / 1.02)
    assert row["last_tick_seconds_before_tip"] == 595  # the staler of the two
    assert row["n_pretip_ticks"] == 180                # the smaller of the two counts
    assert row["game_id"] == "0022500002"


def test_build_snapshot_row_stale_last_tick_is_uncovered():
    # One token last traded > 24h before tip -> the game is not covered.
    home = polymarket.PretipPrice(price=0.45, seconds_before_tip=25 * 3600, n_pretip_ticks=3)
    away = polymarket.PretipPrice(price=0.57, seconds_before_tip=600, n_pretip_ticks=180)
    row = polymarket.build_snapshot_row(
        "x", dt.date(2025, 10, 21), HOME_ID, AWAY_ID, home, away, 100.0
    )
    assert row is None


# --------------------------------------------------------------------------
# 7. Coverage rollup.
# --------------------------------------------------------------------------

def test_coverage_stats_counts_covered_thin_and_reasons():
    statuses = [
        {"status": "covered", "moneyline_volume": 1_500_000.0},
        {"status": "covered", "moneyline_volume": 1_000.0},  # thin
        {"status": "event_not_found"},
        {"status": "stale_last_tick"},
    ]
    stats = polymarket.coverage_stats(statuses, n_test_games_total=1258)
    assert stats["n_covered"] == 2
    assert stats["thin_market_covered_games"] == 1
    assert stats["coverage_fraction"] == pytest.approx(2 / 1258)
    assert stats["status_counts"] == {
        "covered": 2,
        "event_not_found": 1,
        "stale_last_tick": 1,
    }
