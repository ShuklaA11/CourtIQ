"""Sprint-6 Polymarket pre-game benchmark — pure, socket-free parsing.

Sprint 3-5 benchmarked the model's pre-game probabilities against a single-book
sportsbook (MGM vig-free closing moneylines), which covered only 769 of the 1,258
test games and stopped at the All-Star break. Sprint 6 adds a second,
structurally-different sharp benchmark — **Polymarket**, a real-money prediction
market — with ~96% test-game coverage *including the playoffs*, to retest the
Sprint-4 gap-close on a fuller, firmer sample. The honest framing is unchanged:
this is "model vs a prediction market," complementary to "model vs a sportsbook,"
and the market is still expected to be sharper.

This module holds every function that turns Polymarket's JSON into a snapshot row
WITHOUT touching a socket, so the whole parsing contract is unit-testable from
saved fixtures. The networked pull lives in ``winprob.polymarket_pull``; the
comparison in ``winprob.polymarket_compare``.

The leakage law is enforced here, in ``select_pretip_price``: a game's price is the
last ``prices-history`` tick STRICTLY before that market's ``gameStartTime`` (tip),
and only if that tick is within ``COVERAGE_MAX_SECONDS`` (24h) of tip. A post-tip
tick can never be chosen. Orientation is self-correcting: rather than trust the
slug's away/home order, we map the returned event slug's two tricodes to team ids
and pick the home token by which side equals the mart's home team id.

Pure numpy/pandas-free helpers; every function returns a new object and never
mutates its input.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------

# The price for a game must be the last pre-tip tick within this many seconds of
# tip-off; a staler last tick (or none) means the game is uncovered, recorded as
# such rather than faked.
COVERAGE_MAX_SECONDS = 24 * 3600

# Moneyline dollar volume below this is flagged thin (reported, not excluded).
THIN_VOLUME_USD = 50_000.0

# Slug date offsets to try, in preference order: exact ET game date first, then
# ±1 day for the ~4% of slug edge cases where Polymarket's date is off by one.
SLUG_DATE_OFFSETS: tuple[int, ...] = (0, -1, 1)

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "home_team_id",
    "away_team_id",
    "market_home_prob",
    "n_pretip_ticks",
    "last_tick_seconds_before_tip",
    "moneyline_volume",
)


# --------------------------------------------------------------------------
# Team id <-> tricode map (local box scores, no network).
# --------------------------------------------------------------------------

def build_team_tricode_map(box_dir: Path) -> dict[int, str]:
    """Deterministic team-id -> lowercase-tricode map from V3 box-score JSON.

    Reads ``boxScoreTraditional.homeTeam.teamTricode`` / ``homeTeamId`` (and away)
    from the local traditional box scores until all 30 franchises are seen.
    Tricodes are stable across seasons, so any covering set of games suffices.
    Raises if fewer than 30 teams are found, so a silent mis-map cannot happen.
    """
    box_dir = Path(box_dir)
    mapping: dict[int, str] = {}
    for path in sorted(box_dir.glob("*.json")):
        box = json.loads(path.read_text())["boxScoreTraditional"]
        for side in ("homeTeam", "awayTeam"):
            team = box[side]
            mapping[int(team["teamId"])] = str(team["teamTricode"]).lower()
        if len(mapping) >= 30:
            break
    if len(mapping) < 30:
        raise ValueError(
            f"found only {len(mapping)} teams in {box_dir}; need all 30 for the slug map"
        )
    return mapping


# --------------------------------------------------------------------------
# Slug construction + candidates.
# --------------------------------------------------------------------------

def event_slug(away_tricode: str, home_tricode: str, game_date: str) -> str:
    """Polymarket event slug ``nba-<away>-<home>-<YYYY-MM-DD>`` (tricodes lowercased)."""
    return f"nba-{away_tricode.lower()}-{home_tricode.lower()}-{game_date}"


def candidate_slugs(
    home_tricode: str, away_tricode: str, game_date: dt.date
) -> list[str]:
    """Slugs to try for a game, in preference order.

    The canonical guess is ``nba-<away>-<home>-<date>`` on the mart's ET game date.
    We also try the FLIPPED orientation (guards a home/away disagreement with
    Polymarket) and ``±1`` day (guards Polymarket date edge cases), deduplicated
    with the exact guess first. Orientation is confirmed later against team ids, so
    a flipped candidate that happens to resolve is still oriented correctly.
    """
    slugs: list[str] = []
    for offset in SLUG_DATE_OFFSETS:
        date_str = (game_date + dt.timedelta(days=offset)).isoformat()
        for away, home in ((away_tricode, home_tricode), (home_tricode, away_tricode)):
            slug = event_slug(away, home, date_str)
            if slug not in slugs:
                slugs.append(slug)
    return slugs


# --------------------------------------------------------------------------
# Event market parsing.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EventMarket:
    """The moneyline market of a Polymarket NBA event, parsed from its JSON."""

    slug: str
    outcomes: tuple[str, str]        # (away_name, home_name) per Polymarket order
    token_ids: tuple[str, str]       # (away_token, home_token) per Polymarket order
    game_start: dt.datetime          # tip-off (UTC-aware)
    volume: float                    # moneyline dollar volume


def _parse_json_pair(value: object) -> tuple[str, str]:
    """Parse a JSON-string (or list) two-element field into a ``(a, b)`` tuple."""
    items = json.loads(value) if isinstance(value, str) else list(value)
    if len(items) != 2:
        raise ValueError(f"expected a 2-element list, got {items!r}")
    return str(items[0]), str(items[1])


def parse_event_market(event_json: dict) -> EventMarket | None:
    """Extract the game moneyline market from one events-API event object.

    The moneyline is the market whose ``slug`` equals the event ``slug``. Returns a
    parsed ``EventMarket`` or ``None`` if the event carries no such market or is
    missing the fields the comparison needs (a missing tip time, tokens, or
    outcomes makes the game uncoverable, not an error).
    """
    event_slug_value = event_json.get("slug")
    for market in event_json.get("markets", []):
        if market.get("slug") != event_slug_value:
            continue
        start_raw = market.get("gameStartTime")
        outcomes_raw = market.get("outcomes")
        tokens_raw = market.get("clobTokenIds")
        if not (start_raw and outcomes_raw and tokens_raw):
            return None
        try:
            outcomes = _parse_json_pair(outcomes_raw)
            token_ids = _parse_json_pair(tokens_raw)
            game_start = dt.datetime.fromisoformat(start_raw)
        except (ValueError, TypeError):
            return None
        if game_start.tzinfo is None:
            game_start = game_start.replace(tzinfo=dt.timezone.utc)
        volume = float(market.get("volumeNum") or 0.0)
        return EventMarket(
            slug=str(event_slug_value),
            outcomes=outcomes,
            token_ids=token_ids,
            game_start=game_start,
            volume=volume,
        )
    return None


# --------------------------------------------------------------------------
# Orientation — self-correcting via team ids, not slug order.
# --------------------------------------------------------------------------

def slug_tricodes(slug: str) -> tuple[str, str] | None:
    """The ``(first, second)`` tricodes of an ``nba-<t1>-<t2>-<date>`` slug.

    Polymarket's convention orders them ``(away, home)``, aligned with the event's
    ``outcomes`` / ``clobTokenIds``. Returns ``None`` for a slug that does not match
    the expected 4-part shape, so a malformed slug is handled, not trusted.
    """
    parts = slug.split("-")
    if len(parts) < 3 or parts[0] != "nba":
        return None
    return parts[1], parts[2]


def home_token_index(
    market: EventMarket, tricode_to_id: dict[int, str], home_id: int, away_id: int
) -> int | None:
    """Which token index (0 or 1) is the mart's home team — or ``None`` if mismatched.

    Maps the event slug's two tricodes to team ids and requires that id-set to equal
    ``{home_id, away_id}`` (rejects a wrong game that happens to resolve). The home
    token is whichever slug position maps to ``home_id`` — determined by the ids, so
    a flipped slug is oriented correctly rather than trusted by position.
    """
    tricodes = slug_tricodes(market.slug)
    if tricodes is None:
        return None
    id_by_tricode = {tri: tid for tid, tri in tricode_to_id.items()}
    try:
        first_id = id_by_tricode[tricodes[0]]
        second_id = id_by_tricode[tricodes[1]]
    except KeyError:
        return None
    if {first_id, second_id} != {home_id, away_id}:
        return None
    return 1 if second_id == home_id else 0


# --------------------------------------------------------------------------
# Pre-tip tick selection — the leakage law.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PretipPrice:
    """The last pre-tip price for one token: its value, staleness, and tick count."""

    price: float
    seconds_before_tip: int
    n_pretip_ticks: int


def select_pretip_price(history: list[dict], tip_ts: int) -> PretipPrice | None:
    """The last ``prices-history`` tick STRICTLY at or before tip-off.

    Keeps only ticks with ``t <= tip_ts`` — a post-tip tick can never be chosen —
    and returns the latest of them with its staleness (``tip_ts - t``) and the
    pre-tip tick count. Returns ``None`` if there is no pre-tip tick at all. The
    24h coverage window is applied by the caller (``build_snapshot_row``), so this
    stays a pure "last valid tick" primitive.
    """
    pretip = [tick for tick in history if int(tick["t"]) <= tip_ts]
    if not pretip:
        return None
    last = max(pretip, key=lambda tick: int(tick["t"]))
    return PretipPrice(
        price=float(last["p"]),
        seconds_before_tip=int(tip_ts) - int(last["t"]),
        n_pretip_ticks=len(pretip),
    )


# --------------------------------------------------------------------------
# Vig-free home probability + snapshot row.
# --------------------------------------------------------------------------

def vig_free_home_prob(p_home_last: float, p_away_last: float) -> float:
    """Two-token vig-free home win probability ``p_home / (p_home + p_away)``.

    Mirrors ``market.vig_free_home_prob`` for a prediction market: each token's last
    pre-tip price is a (slightly overround) implied probability, so normalizing the
    home price by the two-token sum removes the spread and yields a home probability
    whose away complement sums to 1.
    """
    total = float(p_home_last) + float(p_away_last)
    if total <= 0.0:
        raise ValueError(f"non-positive token price sum: {total}")
    return float(p_home_last) / total


def build_snapshot_row(
    game_id: str,
    game_date: dt.date,
    home_team_id: int,
    away_team_id: int,
    home_price: PretipPrice,
    away_price: PretipPrice,
    volume: float,
) -> dict | None:
    """One snapshot row from both tokens' pre-tip prices — or ``None`` if uncovered.

    A game is covered iff BOTH tokens have a pre-tip tick and the STALER of the two
    is within ``COVERAGE_MAX_SECONDS`` of tip (the conservative 24h rule). The
    recorded ``last_tick_seconds_before_tip`` is that worst-case gap and
    ``n_pretip_ticks`` the smaller of the two tokens' counts. Returns a row with the
    vig-free ``market_home_prob``, or ``None`` when the window rule fails.
    """
    worst_gap = max(home_price.seconds_before_tip, away_price.seconds_before_tip)
    if worst_gap > COVERAGE_MAX_SECONDS:
        return None
    return {
        "game_id": str(game_id),
        "game_date": game_date.isoformat(),
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "market_home_prob": vig_free_home_prob(home_price.price, away_price.price),
        "n_pretip_ticks": int(min(home_price.n_pretip_ticks, away_price.n_pretip_ticks)),
        "last_tick_seconds_before_tip": int(worst_gap),
        "moneyline_volume": float(volume),
    }


# --------------------------------------------------------------------------
# Coverage rollup.
# --------------------------------------------------------------------------

def coverage_stats(statuses: list[dict], n_test_games_total: int) -> dict:
    """Roll per-game pull outcomes into coverage counts + reason breakdown.

    ``statuses`` is one dict per attempted game with a ``status`` string
    (``covered`` / an uncovered reason) and, for covered games, ``moneyline_volume``
    and ``last_tick_seconds_before_tip``. Reports coverage count and fraction, the
    reason histogram, and the thin-market count (covered games below
    ``THIN_VOLUME_USD``). Pure: reads the list, builds a new dict.
    """
    by_reason: dict[str, int] = {}
    covered = 0
    thin = 0
    for item in statuses:
        reason = str(item["status"])
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if reason == "covered":
            covered += 1
            if float(item.get("moneyline_volume", 0.0)) < THIN_VOLUME_USD:
                thin += 1
    total = int(n_test_games_total)
    return {
        "n_test_games_total": total,
        "n_covered": covered,
        "n_attempted": len(statuses),
        "coverage_fraction": (covered / total) if total else 0.0,
        "thin_market_covered_games": thin,
        "thin_volume_usd": THIN_VOLUME_USD,
        "coverage_max_seconds": COVERAGE_MAX_SECONDS,
        "status_counts": dict(sorted(by_reason.items())),
    }
