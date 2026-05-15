#!/usr/bin/env python3
"""Coverage check for the staging layer.

Prints ``PASS`` (exit 0) only when all three staging sources cover the exact
same set of game_ids AND that set has the expected size (1312 for the default
2023-24 scope). Any mismatch -- an extra or missing game_id in any direction,
or a wrong total -- prints ``FAIL`` with the offending game_ids and exits 1.

The staging models are the source of truth for coverage; this script reports it
so a regression (e.g. a partial pbp pull) is caught by a single green/red signal
rather than eyeballing row counts.

Usage:
    python warehouse/checks/coverage_check.py
    python warehouse/checks/coverage_check.py --db warehouse/courtiq.duckdb --expected 1312
"""
from __future__ import annotations

import argparse
import sys

import duckdb

# The three staging sources whose game_id coverage must be identical.
SOURCES = ("stg_pbp_actions", "stg_box_player", "stg_box_team_advanced")
DEFAULT_DB = "courtiq.duckdb"
DEFAULT_EXPECTED = 1312
# Cap on how many offending ids to echo so a wholesale mismatch stays readable.
MAX_LISTED = 20


def _distinct_game_ids(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"select distinct game_id from {table}").fetchall()
    return {r[0] for r in rows}


def _preview(ids: set[str]) -> str:
    listed = sorted(ids)[:MAX_LISTED]
    suffix = "" if len(ids) <= MAX_LISTED else f" (+{len(ids) - MAX_LISTED} more)"
    return ", ".join(listed) + suffix


def check_coverage(db_path: str, expected: int) -> bool:
    """Return True when coverage is identical across sources and sized `expected`."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        coverage = {table: _distinct_game_ids(con, table) for table in SOURCES}
    finally:
        con.close()

    reference = coverage[SOURCES[0]]
    problems: list[str] = []

    # Every source must equal the first source's set (differences empty both ways).
    for table in SOURCES[1:]:
        this = coverage[table]
        missing = reference - this          # in reference, absent from this source
        extra = this - reference            # in this source, absent from reference
        if missing:
            problems.append(
                f"{table} is MISSING {len(missing)} game_id(s) present in "
                f"{SOURCES[0]}: {_preview(missing)}"
            )
        if extra:
            problems.append(
                f"{table} has {len(extra)} EXTRA game_id(s) not in "
                f"{SOURCES[0]}: {_preview(extra)}"
            )

    # Size check on the (identical) coverage set.
    actual = len(reference)
    if actual != expected:
        problems.append(f"expected {expected} game_ids, found {actual}")

    if problems:
        print("FAIL")
        for line in problems:
            print(f"  - {line}")
        return False

    print(f"PASS: all {len(SOURCES)} staging sources cover the same {actual} game_ids")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Path to the DuckDB file (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=DEFAULT_EXPECTED,
        help=f"Expected game_id count for the scope (default: {DEFAULT_EXPECTED}).",
    )
    args = parser.parse_args()
    return 0 if check_coverage(args.db, args.expected) else 1


if __name__ == "__main__":
    sys.exit(main())
