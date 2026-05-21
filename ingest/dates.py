"""Backfill game tip-off dates into a single raw file the warehouse can join.

The per-game raw pulls (`ingest.pull`) carry only the scrape timestamp, not the
date a game was actually played — so `dim_games.game_date` has been NULL, and any
recency weighting or chronological train/test split downstream has no dates to
work with.

LeagueGameFinder — the same endpoint `ingest.pull.discover_game_ids` already
calls to enumerate games — returns `GAME_DATE` per game. This module captures it
(the one field the discovery step throws away) into `data/raw/game_dates.json`,
one row per game, which `dim_games` left-joins on. ~10 calls total (5 seasons ×
2 season types), no per-game requests.

Run:  python -m ingest.dates          # pull dates (skips if already cached)
      python -m ingest.dates --force  # re-pull even if the file exists
"""

from __future__ import annotations

import argparse

from nba_api.stats.endpoints import leaguegamefinder

from ingest.client import polite_call
from ingest.pull import DATA_ROOT, SEASON_TYPES, SEASONS, write_json_atomic

OUT = DATA_ROOT / "game_dates.json"


def pull_game_dates() -> dict[str, str]:
    """Return {game_id -> 'YYYY-MM-DD'} across every ingested season/season type.

    LeagueGameFinder yields one row per team per game; we dedupe on game_id and
    keep the first date seen (both team-rows carry the same GAME_DATE).
    """
    dates: dict[str, str] = {}
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
                dates.setdefault(row["GAME_ID"], row["GAME_DATE"])
            print(f"  {season} {season_type}: {len(rows)} team-rows, "
                  f"{len(dates)} unique games so far")
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NBA game tip-off dates")
    parser.add_argument("--force", action="store_true",
                        help="re-pull even if game_dates.json already exists")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} already exists; nothing to do (use --force to re-pull).")
        return

    print("Pulling game dates from LeagueGameFinder...")
    dates = pull_game_dates()
    payload = [{"game_id": gid, "game_date": date} for gid, date in sorted(dates.items())]
    write_json_atomic(OUT, payload)
    print(f"\nWrote {len(payload)} game dates to {OUT}")


if __name__ == "__main__":
    main()
