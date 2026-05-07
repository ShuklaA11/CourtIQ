"""Resumable ingest of NBA play-by-play + box scores to raw JSON on disk.

Design (see PLAN.md Sprint 0):
  - Cache raw JSON per game, keyed by game_id, BEFORE any parsing.
  - Resumability = "the filesystem is the checkpoint": skip a game if its file
    already exists. No separate manifest to fall out of sync.
  - Crash safety = write to <name>.tmp, then atomically rename to <name>.json,
    so a half-written file never wears the "done" name.

Run:  python -m ingest.pull            # all seasons, both season types
      python -m ingest.pull --limit 3  # smoke test: first 3 games only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoretraditionalv3,
    leaguegamefinder,
    playbyplayv3,
)

from ingest.client import polite_call

# Deliberately excludes 2019-20 and 2020-21: the bubble and 72-game COVID
# seasons have anomalous pace/rest/home-court structure (documented in README).
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]

DATA_ROOT = Path("data/raw")

# Each endpoint the pull fetches per game: (subdir, endpoint class).
# V3 endpoints: the NBA API deprecated V2 play-by-play (now returns empty JSON,
# nba_api issue #591) and advanced box scores; V3 is the consistent modern format.
GAME_ENDPOINTS = {
    "pbp": playbyplayv3.PlayByPlayV3,
    "box_traditional": boxscoretraditionalv3.BoxScoreTraditionalV3,
    "box_advanced": boxscoreadvancedv3.BoxScoreAdvancedV3,
}


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write payload to path via temp-file + atomic rename (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")  # same dir => rename is atomic
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())  # force bytes to disk before we trust the rename
    os.replace(tmp, path)      # atomic on a local filesystem


def discover_game_ids() -> list[tuple[str, str]]:
    """Return sorted unique (season, game_id) across all seasons/season types.

    LeagueGameFinder yields one row per team per game, so we dedupe game_ids.
    """
    seen: dict[str, str] = {}  # game_id -> season
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            finder = polite_call(
                leaguegamefinder.LeagueGameFinder,
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",  # 00 = NBA (excludes G-League/preseason)
            )
            rows = finder.get_normalized_dict()["LeagueGameFinderResults"]
            for row in rows:
                gid = row["GAME_ID"]
                seen.setdefault(gid, season)
            print(f"  {season} {season_type}: {len(rows)} team-rows, "
                  f"{len(seen)} unique games so far")
    return sorted(seen.items(), key=lambda kv: kv[1])  # (game_id, season) by season


def fetch_game(game_id: str) -> int:
    """Fetch every endpoint for one game, skipping any already cached.

    Returns the number of endpoints actually fetched (0 => fully cached).
    """
    fetched = 0
    for subdir, endpoint_cls in GAME_ENDPOINTS.items():
        out = DATA_ROOT / subdir / f"{game_id}.json"
        if out.exists():
            continue  # filesystem-as-checkpoint: already done
        payload = polite_call(endpoint_cls, game_id=game_id).get_dict()
        write_json_atomic(out, payload)
        fetched += 1
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable NBA raw ingest")
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch only the first N games (smoke test)")
    args = parser.parse_args()

    print("Discovering games...")
    games = discover_game_ids()
    if args.limit:
        games = games[: args.limit]
    print(f"Total games to ensure cached: {len(games)}\n")

    newly_fetched = 0
    for i, (game_id, season) in enumerate(games, 1):
        n = fetch_game(game_id)
        newly_fetched += n
        status = "fetched" if n else "cached"
        print(f"[{i}/{len(games)}] {season} {game_id}: {status} "
              f"({n} endpoints)")

    print(f"\nDone. Newly fetched {newly_fetched} endpoint files; "
          f"raw data under {DATA_ROOT}/")


if __name__ == "__main__":
    main()
