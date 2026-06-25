"""Sprint-6 Polymarket pre-game benchmark — the resumable, polite networked pull.

Drives enumeration FROM THE MART (the Polymarket ``nba`` tag listing is unreliable):
for every 2025-26 test game it constructs the event slug, fetches the moneyline
event and both tokens' pre-tip ``prices-history``, orients home/away by team id, and
snapshots one row per COVERED game to ``polymarket_closing.parquet``. Coverage,
source URLs, and the (passed-in) pull timestamp go to ``polymarket_pull.json``, and
the snapshot's sha256 is pinned there so the comparison recomputes from a frozen
artifact.

Every HTTP call carries a browser ``User-Agent`` (the default urllib UA gets a 403)
and goes through ``_polite_get_json``, which paces and retries transient failures
exactly like ``ingest.client.polite_call``. The pull is resumable: each attempted
game is appended to a JSONL progress log, and a rerun skips games already logged, so
an interrupted pull continues rather than restarting. All parsing is delegated to the
socket-free ``winprob.polymarket`` module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from winprob import design, polymarket

DEFAULT_DATA_DIR = Path("data/winprob")
DEFAULT_BOX_DIR = Path("data/raw/box_traditional")
PARQUET_NAME = "fct_game_states.parquet"
SNAPSHOT_NAME = "polymarket_closing.parquet"
PULL_JSON_NAME = "polymarket_pull.json"
PROGRESS_NAME = "polymarket_progress.jsonl"

TEST_SPLIT = "test"
DEFAULT_SEASONS: tuple[int, ...] = (2025,)

# API endpoints (public, no auth). Recorded in the pull provenance.
EVENTS_URL = "https://gamma-api.polymarket.com/events"
PRICES_URL = "https://clob.polymarket.com/prices-history"
USER_AGENT = "Mozilla/5.0"

# Price-history query window: far enough back to see the last pre-tip tick and
# prove the 24h rule, with a small post-tip pad so a tick landing seconds after tip
# is fetched and then dropped by the pre-tip filter (never chosen).
PRICE_LOOKBACK_SECONDS = 72 * 3600
POST_TIP_PAD_SECONDS = 3600
PRICE_FIDELITY = 10

# Polite pacing + retry, adapted from ingest.client. The pull runs a small pool of
# workers (the concurrency cap IS the politeness bound), and a thread-safe global gap
# smooths the aggregate request rate. The clob prices-history endpoint returns
# transient HTTP 500s that clear on an immediate retry, so the backoff is SHORT and
# LINEAR (not the exponential 1->2->4->8s ramp that would stall throughput on a flaky
# 500) — exponential growth is for rate-limit (429) storms, not fast-clearing 500s.
MIN_GAP_SECONDS = 0.05      # global spacing between successive requests (all workers)
MAX_WORKERS = 8             # bounded concurrency = the politeness cap
MAX_RETRIES = 8             # a transient 500 can repeat a few times; keep trying
BASE_BACKOFF_SECONDS = 0.3  # short base: a retry usually succeeds right away
BACKOFF_CAP_SECONDS = 3.0   # linear growth capped low
REQUEST_TIMEOUT = 30

RETRYABLE = (requests.exceptions.RequestException, json.JSONDecodeError)

_gap_lock = threading.Lock()
_last_request_at = 0.0


# --------------------------------------------------------------------------
# Polite HTTP.
# --------------------------------------------------------------------------

def _respect_min_gap() -> None:
    """Sleep so successive requests are spaced >= ``MIN_GAP_SECONDS`` apart, globally.

    Thread-safe: the gap is enforced under a lock so a pool of workers throttles to a
    single aggregate request rate rather than each racing independently.
    """
    global _last_request_at
    with _gap_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_GAP_SECONDS:
            time.sleep(MIN_GAP_SECONDS - elapsed)
        _last_request_at = time.monotonic()


def _polite_get_json(url: str, params: dict) -> object:
    """GET ``url?params`` as JSON with the browser UA, paced + retried on transients.

    Retries connection/timeout errors, non-JSON (rate-limit HTML) responses, and the
    clob endpoint's transient 5xx with a short, linear, jittered backoff. Re-raises
    the last exception if every attempt fails.
    """
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _respect_min_gap()
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            backoff = min(BASE_BACKOFF_SECONDS * (attempt + 1), BACKOFF_CAP_SECONDS)
            time.sleep(random.uniform(0, backoff))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} attempts: {url}") from last_exc


def fetch_event(slug: str) -> dict | None:
    """Fetch the first events-API event for ``slug`` (``None`` if none exist)."""
    payload = _polite_get_json(EVENTS_URL, {"slug": slug})
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


def fetch_prices(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    """Fetch a token's ``prices-history`` ticks over ``[start_ts, end_ts]``.

    ``interval=max`` returns empty for resolved markets, so the window is always
    given as explicit ``startTs`` / ``endTs``. Returns the raw ``history`` list.
    """
    payload = _polite_get_json(
        PRICES_URL,
        {
            "market": token_id,
            "startTs": int(start_ts),
            "endTs": int(end_ts),
            "fidelity": PRICE_FIDELITY,
        },
    )
    if isinstance(payload, dict):
        return list(payload.get("history", []))
    return []


# --------------------------------------------------------------------------
# Per-game pull.
# --------------------------------------------------------------------------

def _resolve_event(
    game: dict, tricode_to_id: dict[int, str]
) -> tuple[polymarket.EventMarket, int] | str:
    """Resolve a game to its oriented moneyline market, or an uncovered-reason string.

    Tries each candidate slug; the first that yields a parseable moneyline whose slug
    tricodes map to the mart's ``{home, away}`` ids wins, with the home token index
    determined by team id. Distinguishes "no market ever parsed" (``event_not_found``)
    from "market parsed but never oriented to this game" (``orientation_mismatch``).
    """
    game_date = _as_date(game["game_date"])
    home_id, away_id = int(game["home_team_id"]), int(game["away_team_id"])
    home_tri, away_tri = tricode_to_id[home_id], tricode_to_id[away_id]

    saw_market = False
    for slug in polymarket.candidate_slugs(home_tri, away_tri, game_date):
        event = fetch_event(slug)
        if event is None:
            continue
        market = polymarket.parse_event_market(event)
        if market is None:
            continue
        saw_market = True
        home_idx = polymarket.home_token_index(market, tricode_to_id, home_id, away_id)
        if home_idx is not None:
            return market, home_idx
    return "orientation_mismatch" if saw_market else "event_not_found"


def pull_game(game: dict, tricode_to_id: dict[int, str]) -> dict:
    """Pull one game to a progress record: a covered row, or an uncovered reason.

    Returns a dict always carrying ``game_id`` and ``status``; a ``covered`` status
    also carries the full snapshot row plus the resolved ``slug`` for auditing. All
    parsing (orientation, pre-tip selection, the 24h rule, vig removal) is delegated
    to ``winprob.polymarket``; this function only sequences the network calls.
    """
    game_id = str(game["game_id"])
    game_date = _as_date(game["game_date"])
    home_id, away_id = int(game["home_team_id"]), int(game["away_team_id"])

    resolved = _resolve_event(game, tricode_to_id)
    if isinstance(resolved, str):
        return {"game_id": game_id, "status": resolved}
    market, home_idx = resolved

    tip_ts = int(market.game_start.timestamp())
    start_ts = tip_ts - PRICE_LOOKBACK_SECONDS
    end_ts = tip_ts + POST_TIP_PAD_SECONDS
    home_token = market.token_ids[home_idx]
    away_token = market.token_ids[1 - home_idx]

    home_price = polymarket.select_pretip_price(
        fetch_prices(home_token, start_ts, end_ts), tip_ts
    )
    away_price = polymarket.select_pretip_price(
        fetch_prices(away_token, start_ts, end_ts), tip_ts
    )
    if home_price is None or away_price is None:
        return {"game_id": game_id, "status": "no_pretip_ticks", "slug": market.slug}

    row = polymarket.build_snapshot_row(
        game_id, game_date, home_id, away_id, home_price, away_price, market.volume
    )
    if row is None:
        return {"game_id": game_id, "status": "stale_last_tick", "slug": market.slug}
    return {"status": "covered", "slug": market.slug, **row}


# --------------------------------------------------------------------------
# Resumable orchestration.
# --------------------------------------------------------------------------

def _as_date(value: object) -> dt.date:
    """Coerce a mart ``game_date`` (date or ISO string) to a ``date``."""
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def test_season_games(df: pd.DataFrame, seasons: tuple[int, ...]) -> list[dict]:
    """One record per game in the requested seasons' test split, ordered by id.

    Reduces the possession mart to distinct games carrying identity + date, the
    minimal per-game metadata the pull drives from. Pure: reads ``df``, returns a
    new list.
    """
    scoped = df.loc[(df["split"] == TEST_SPLIT) & (df["season"].isin(seasons))]
    games = (
        scoped[["game_id", "game_date", "home_team_id", "away_team_id"]]
        .drop_duplicates("game_id")
        .sort_values("game_id")
    )
    return games.to_dict("records")


def _load_progress(path: Path) -> dict[str, dict]:
    """Load the JSONL progress log into a ``game_id -> record`` map (last wins)."""
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            records[str(record["game_id"])] = record
    return records


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def finalize_snapshot(
    records: dict[str, dict],
    n_test_games_total: int,
    pulled_at: str,
    seasons: tuple[int, ...],
    snapshot_path: Path,
    pull_json_path: Path,
) -> dict:
    """Write the frozen snapshot parquet + pull provenance from progress records.

    Covered records become the ``polymarket_closing.parquet`` rows (sorted by
    ``game_id`` for byte-stable output); the full status list drives the coverage
    rollup. ``polymarket_pull.json`` records source URLs, the passed-in pull
    timestamp, coverage stats, and the snapshot's sha256 — the pin the comparison
    audit and the repro test trace back to.
    """
    statuses = [
        {
            "status": rec["status"],
            "moneyline_volume": rec.get("moneyline_volume", 0.0),
            "last_tick_seconds_before_tip": rec.get("last_tick_seconds_before_tip"),
        }
        for rec in records.values()
    ]
    covered = [rec for rec in records.values() if rec["status"] == "covered"]
    frame = pd.DataFrame(
        [{col: rec[col] for col in polymarket.SNAPSHOT_COLUMNS} for rec in covered],
        columns=list(polymarket.SNAPSHOT_COLUMNS),
    ).sort_values("game_id").reset_index(drop=True)
    frame.to_parquet(snapshot_path, index=False)

    stats = polymarket.coverage_stats(statuses, n_test_games_total)
    pull_doc = {
        "pulled_at": pulled_at,
        "seasons": sorted(int(s) for s in seasons),
        "coverage": stats,
        "snapshot_sha256": design.file_hash(snapshot_path),
        "snapshot_rows": len(frame),
        "source_urls": {"events": EVENTS_URL, "prices": PRICES_URL},
        "leakage_rule": (
            "price = last prices-history tick with t <= gameStartTime, within "
            f"{polymarket.COVERAGE_MAX_SECONDS}s of tip; post-tip ticks never chosen"
        ),
        "price_window": {
            "lookback_seconds": PRICE_LOOKBACK_SECONDS,
            "post_tip_pad_seconds": POST_TIP_PAD_SECONDS,
            "fidelity": PRICE_FIDELITY,
        },
    }
    _write_json(pull_json_path, pull_doc)
    return pull_doc


def run_pull(
    pulled_at: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    box_dir: Path = DEFAULT_BOX_DIR,
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    limit: int | None = None,
) -> dict:
    """Pull every requested test game, resuming from the progress log, then finalize.

    ``pulled_at`` is the caller-supplied ISO timestamp (never ``datetime.now()``
    inside the library). Builds the team-id map from local box scores, enumerates
    the mart's test games, pulls each one not already logged (appending to the JSONL
    progress log as it goes), and writes the snapshot parquet + pull provenance.
    """
    data_dir, box_dir = Path(data_dir), Path(box_dir)
    parquet_path = data_dir / PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {parquet_path}; run ./game_states.sh first"
        )
    df = pd.read_parquet(
        parquet_path, columns=["game_id", "season", "game_date", "split",
                               "home_team_id", "away_team_id"]
    )
    tricode_to_id = polymarket.build_team_tricode_map(box_dir)
    games = test_season_games(df, seasons)
    n_total = len(games)

    progress_path = data_dir / PROGRESS_NAME
    records = _load_progress(progress_path)
    pending = [g for g in games if str(g["game_id"]) not in records]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"polymarket pull: {n_total} test games; {len(records)} already logged, "
        f"{len(pending)} to pull ({MAX_WORKERS} workers)"
    )
    # Games are independent, so pull them across a bounded worker pool; the global
    # request gap keeps the aggregate rate polite. Each finished record is appended
    # to the progress log under a lock, so an interrupted run still resumes cleanly.
    write_lock = threading.Lock()
    done = 0
    with progress_path.open("a") as log, ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {
            pool.submit(pull_game, game, tricode_to_id): game for game in pending
        }
        for future in as_completed(futures):
            record = future.result()
            with write_lock:
                records[str(record["game_id"])] = record
                log.write(json.dumps(record, default=str) + "\n")
                log.flush()
                done += 1
                if done % 50 == 0 or done == len(pending):
                    covered = sum(1 for r in records.values() if r["status"] == "covered")
                    print(f"  {done}/{len(pending)} pulled ({covered} covered so far)")

    return finalize_snapshot(
        records,
        n_total,
        pulled_at,
        seasons,
        data_dir / SNAPSHOT_NAME,
        data_dir / PULL_JSON_NAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot Polymarket vig-free closing probabilities for test games"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--box-dir", default=str(DEFAULT_BOX_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="pull at most N pending games this run (for smoke tests)")
    args = parser.parse_args()
    pulled_at = dt.datetime.now(dt.timezone.utc).isoformat()
    pull_doc = run_pull(
        pulled_at, Path(args.data_dir), Path(args.box_dir), limit=args.limit
    )
    cov = pull_doc["coverage"]
    print(
        f"\nsnapshot: {cov['n_covered']}/{cov['n_test_games_total']} covered "
        f"({cov['coverage_fraction']:.1%}); "
        f"{cov['thin_market_covered_games']} thin (<${int(cov['thin_volume_usd']):,})"
    )
    print(f"  status counts: {cov['status_counts']}")
    print(f"  snapshot sha256: {pull_doc['snapshot_sha256']}")


if __name__ == "__main__":
    main()
