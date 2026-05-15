"""Corpus minutes-reconciliation gate — the load-bearing lineup check.

For each cached game this parses the raw V3 payloads, back-infers each period's
starting five, walks the event log through the on-floor state machine, and diffs
the reconstructed per-player seconds against the traditional box score.

Two contractual gates over the scoped corpus:
  1. Exactly-5 invariant, 100% of games. reconstruct_lineups raises on any team
     that is not exactly five on the floor after an event; a raising game is
     quarantined (never aborts the run) and counts against the 100%.
  2. Residual gate, >=99% of player-games with |reconstructed - box| <= tolerance.

Run:
    python -m recon.reconcile --season 2023-24            # regular + playoffs
    python -m recon.reconcile --season 2023-24 --limit 50 # smoke test
    python -m recon.reconcile                             # whole corpus
Exit codes: 0 = PASS, 1 = FAIL (a gate missed), 2 = no corpus found.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from recon.adapter import (
    AdapterError,
    parse_box_seconds,
    parse_lineup_actions,
    parse_rosters,
    parse_team_ids,
)
from recon.lineups import (
    LineupSizeError,
    OnFloorError,
    recon_period_starters,
    reconcile_minutes,
    reconstruct_lineups,
)

DEFAULT_RAW_ROOT = Path("data/raw")
TOLERANCE_SECONDS = 60.0
RESIDUAL_PASS_FRACTION = 0.99

# Regular-season and playoff game_id prefixes per season. A game_id is
# 00 + season_type(2=regular,4=playoff) + YY(season year) + 5-digit number, so the
# season is identified by the 5-char prefix (e.g. "00223" = 2023-24 regular).
_SEASON_PREFIXES = {
    "2021-22": ("00221", "00421"),
    "2022-23": ("00222", "00422"),
    "2023-24": ("00223", "00423"),
    "2024-25": ("00224", "00424"),
    "2025-26": ("00225", "00425"),
}


def game_ids(raw_root: Path, season: str | None) -> list[str]:
    ids = sorted(p.stem for p in (raw_root / "pbp").glob("*.json"))
    if season:
        prefixes = _SEASON_PREFIXES[season]
        ids = [gid for gid in ids if gid.startswith(prefixes)]
    return ids


def _load(raw_root: Path, subdir: str, game_id: str) -> dict:
    with open(raw_root / subdir / f"{game_id}.json") as fh:
        return json.load(fh)


def reconcile_game(raw_root: Path, game_id: str, tolerance: float) -> dict:
    """Reconstruct one game and return its residuals, or a quarantine reason."""
    try:
        pbp = _load(raw_root, "pbp", game_id)
        box = _load(raw_root, "box_traditional", game_id)
        home_id, away_id = parse_team_ids(box)
        rosters = parse_rosters(box)
        actions = parse_lineup_actions(pbp, rosters)
        starters = recon_period_starters(home_id, away_id, actions)
        recon = reconstruct_lineups(game_id, home_id, away_id, starters, actions)
    except (AdapterError, LineupSizeError, OnFloorError, KeyError, ValueError) as exc:
        return {"game_id": game_id, "quarantine": f"{type(exc).__name__}: {exc}"}

    box_seconds = parse_box_seconds(box)
    diffs = {d.person_id: d for d in reconcile_minutes(recon.player_seconds, box_seconds)}
    player_ids = set(recon.player_seconds) | set(box_seconds)
    residuals = [abs(diffs[pid].delta_seconds) if pid in diffs else 0.0 for pid in player_ids]
    return {
        "game_id": game_id,
        "quarantine": None,
        "residuals": residuals,
        "over_tolerance": sum(1 for r in residuals if r > tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minutes reconciliation gate")
    parser.add_argument("--season", choices=sorted(_SEASON_PREFIXES), default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", type=int, default=8, help="how many quarantines to list")
    args = parser.parse_args()

    ids = game_ids(args.raw_root, args.season)
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print("No games found for scope.", file=sys.stderr)
        sys.exit(2)

    quarantined: list[tuple[str, str]] = []
    all_residuals: list[float] = []
    player_games = 0
    over_tolerance = 0
    for gid in ids:
        result = reconcile_game(args.raw_root, gid, args.tolerance)
        if result["quarantine"]:
            quarantined.append((gid, result["quarantine"]))
            continue
        all_residuals.extend(result["residuals"])
        player_games += len(result["residuals"])
        over_tolerance += result["over_tolerance"]

    reconstructed = len(ids) - len(quarantined)
    within = player_games - over_tolerance
    frac = (within / player_games) if player_games else 0.0
    pass_invariant = len(quarantined) == 0
    pass_residual = player_games > 0 and frac >= RESIDUAL_PASS_FRACTION

    print(f"\nScope: {args.season or 'ALL'} — {len(ids)} games")
    print(f"Reconstructed: {reconstructed}/{len(ids)}  |  Quarantined: {len(quarantined)}")
    if all_residuals:
        print(
            f"Minutes residual (s): median={statistics.median(all_residuals):.1f} "
            f"p95={_pctl(all_residuals, 95):.1f} max={max(all_residuals):.1f}"
        )
    print(
        f"Player-games within {args.tolerance:.0f}s: {within}/{player_games} "
        f"({frac * 100:.2f}%)  |  over: {over_tolerance}"
    )
    if quarantined:
        print(f"\nQuarantine ({len(quarantined)}):")
        for gid, reason in quarantined[: args.show]:
            print(f"  {gid}: {reason}")
        if len(quarantined) > args.show:
            print(f"  … and {len(quarantined) - args.show} more")

    passed = pass_invariant and pass_residual
    print(f"\n{'PASS' if passed else 'FAIL'} "
          f"(invariant={'ok' if pass_invariant else 'MISS'}, "
          f"residual={'ok' if pass_residual else 'MISS'})")
    sys.exit(0 if passed else 1)


def _pctl(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


if __name__ == "__main__":
    main()
