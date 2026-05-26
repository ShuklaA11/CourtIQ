"""Reconstruct lineups + possessions for every staged game and write them to DuckDB.

This is the middle step of the pipeline: it runs AFTER `dbt run --select staging`
(which materializes dim_games / stg_* into warehouse/courtiq.duckdb) and BEFORE
`dbt build --select marts` (which reads the recon_* tables this writes as sources).

For each game staging exposes it:
  * parses the raw V3 payloads (recon.adapter),
  * back-infers period starters and walks the on-floor state machine
    (recon.lineups) -> stints, per-action fives,
  * segments possessions (recon.possessions), injecting the reconstructed fives
    so every possession carries its offensive/defensive five.

Quarantine policy: a game whose lineups cannot be reconstructed exactly-five is
written to recon_quarantine with a reason and contributes NOTHING to the recon_*
fact tables — the marts and their gates only ever see clean games. This keeps the
minutes/points gates honest rather than forcing a guess onto data RAPM depends on.

Run (from the repo root):
    python -m recon.build
    python -m recon.build --db warehouse/courtiq.duckdb --raw-root data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from recon.adapter import (
    AdapterError,
    final_score_from_box,
    parse_lineup_actions,
    parse_rosters,
    parse_team_ids,
    team_tokens_from_box,
    to_action_log,
)
from recon.lineups import (
    LineupSizeError,
    OnFloorError,
    reconstruct_lineups,
    recon_period_starters,
)
from recon.possessions import assert_reconciles, recon_possessions

DEFAULT_DB = "warehouse/courtiq.duckdb"
DEFAULT_RAW_ROOT = "data/raw"

_QUARANTINE_ERRORS = (AdapterError, LineupSizeError, OnFloorError, KeyError, ValueError)


def _season(game_id: str) -> int:
    """Season start year decoded from the game_id (matches dim_games)."""
    return 2000 + int(game_id[3:5])


def _load(raw_root: Path, subdir: str, game_id: str) -> dict:
    with open(raw_root / subdir / f"{game_id}.json") as fh:
        return json.load(fh)


def reconstruct_game(raw_root: Path, game_id: str) -> dict:
    """Reconstruct one game's stints and possessions.

    Returns ``{"quarantine": reason}`` if the game cannot be reconstructed
    exactly-five, else ``{"stints": [...], "possessions": [...]}`` rows ready to
    insert.
    """
    try:
        pbp = _load(raw_root, "pbp", game_id)
        box = _load(raw_root, "box_traditional", game_id)
        home_id, away_id = parse_team_ids(box)
        rosters = parse_rosters(box)
        actions = parse_lineup_actions(pbp, rosters)
        starters = recon_period_starters(home_id, away_id, actions)
        recon = reconstruct_lineups(game_id, home_id, away_id, starters, actions)
    except _QUARANTINE_ERRORS as exc:
        return {"quarantine": f"{type(exc).__name__}: {exc}"}

    season = _season(game_id)
    stint_rows = [
        (
            game_id, season, s.period, s.team_id, s.lineup_id,
            list(s.home_five if s.team_id == home_id else s.away_five),
            list(s.home_five), list(s.away_five),
            s.start_seconds, s.end_seconds, s.duration_seconds,
        )
        for s in recon.stints
    ]

    # Inject the reconstructed fives into the possession walk: map each raw action
    # index to the (home_five, away_five) in effect, forward-filling across rows
    # the lineup walk doesn't tag (period markers, timeouts).
    by_order = {ta.action.order: (ta.home_five, ta.away_five) for ta in recon.tagged_actions}
    raw_actions = pbp["game"]["actions"]
    fills: list[tuple] = []
    last = (tuple(sorted(starters[(1, home_id)])), tuple(sorted(starters[(1, away_id)])))
    for index in range(len(raw_actions)):
        if index in by_order:
            last = by_order[index]
        fills.append(last)

    tokens = team_tokens_from_box(box)

    def lineups_at(index: int, _raw: dict) -> dict:
        home_five, away_five = fills[index]
        return {home_id: home_five, away_id: away_five}

    log = to_action_log(pbp, lineups_at=lineups_at, team_tokens=tokens)
    possessions = recon_possessions(game_id, log)
    # Defensive: the possession points must equal the official score. A failure
    # here is a reconstruction defect, not an OT-lineup gap, so quarantine it.
    try:
        assert_reconciles(possessions, final_score_from_box(box))
    except AssertionError as exc:
        return {"quarantine": f"PointsMismatch: {exc}"}

    # Drop degenerate boundary possessions with no resolved offense (a period-end
    # team rebound the token matcher couldn't attribute): they carry no lineup and
    # zero points, so they are segmenter noise, not real possessions. Points
    # reconciliation has already passed, so none of them hold points.
    poss_rows = [
        (
            game_id, season, p.period, p.possession_number,
            p.offense_team_id, p.defense_team_id,
            list(p.offense_five), list(p.defense_five),
            p.points, p.start_seconds, p.end_seconds,
            p.end_seconds - p.start_seconds,
            p.home_score_before, p.away_score_before,
        )
        for p in possessions
        if p.offense_team_id is not None
    ]
    return {"stints": stint_rows, "possessions": poss_rows}


_DDL = """
drop table if exists recon_possessions;
create table recon_possessions (
    game_id varchar, season integer, period integer, possession_number integer,
    offense_team_id bigint, defense_team_id bigint,
    offense_five integer[], defense_five integer[],
    points integer, start_seconds double, end_seconds double, duration_seconds double,
    home_score_before integer, away_score_before integer
);
drop table if exists recon_stints;
create table recon_stints (
    game_id varchar, season integer, period integer, team_id bigint, lineup_id varchar,
    five integer[], home_five integer[], away_five integer[],
    start_seconds double, end_seconds double, seconds double
);
drop table if exists recon_quarantine;
create table recon_quarantine (game_id varchar, reason varchar);
"""


def build(db_path: str = DEFAULT_DB, raw_root: str = DEFAULT_RAW_ROOT) -> dict:
    """Reconstruct every staged game and write recon_* tables into the DuckDB."""
    root = Path(raw_root)
    con = duckdb.connect(db_path)
    try:
        game_ids = [r[0] for r in con.execute(
            "select distinct game_id from dim_games order by game_id"
        ).fetchall()]
        con.execute(_DDL)

        stint_rows: list[tuple] = []
        poss_rows: list[tuple] = []
        quarantine: list[tuple] = []
        for game_id in game_ids:
            result = reconstruct_game(root, game_id)
            if "quarantine" in result:
                quarantine.append((game_id, result["quarantine"]))
                continue
            stint_rows.extend(result["stints"])
            poss_rows.extend(result["possessions"])

        if poss_rows:
            con.executemany(
                "insert into recon_possessions values "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", poss_rows)
        if stint_rows:
            con.executemany(
                "insert into recon_stints values (?,?,?,?,?,?,?,?,?,?,?)", stint_rows)
        if quarantine:
            con.executemany("insert into recon_quarantine values (?,?)", quarantine)

        summary = {
            "games": len(game_ids),
            "reconstructed": len(game_ids) - len(quarantine),
            "quarantined": len(quarantine),
            "possessions": len(poss_rows),
            "stints": len(stint_rows),
        }
        return summary
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct lineups + possessions to DuckDB")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    args = parser.parse_args()
    summary = build(args.db, args.raw_root)
    print(
        f"recon.build: {summary['reconstructed']}/{summary['games']} games "
        f"({summary['quarantined']} quarantined) -> "
        f"{summary['possessions']} possessions, {summary['stints']} stints"
    )


if __name__ == "__main__":
    main()
