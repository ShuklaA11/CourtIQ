"""Export and audit the Sprint 3 possession-boundary game-state mart.

The relational feature logic lives in dbt's ``fct_game_states`` model. This
module performs the artifact work that SQL is poorly suited for: deterministic
hashes, Parquet export, a machine-readable manifest, and a compact audit report.
It does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

FEATURE_SCHEMA_VERSION = "winprob-game-states-v1"
DEFAULT_DB = "warehouse/courtiq.duckdb"
DEFAULT_OUT = Path("data/winprob")
DEFAULT_RAPM_DIR = Path("data/rapm")

REQUIRED_COLUMNS = [
    "game_id", "season", "game_date", "period", "possession_number",
    "game_clock_seconds", "elapsed_game_seconds", "regulation_seconds_remaining",
    "home_team_id", "away_team_id", "possession_team_id", "home_has_possession",
    "home_score", "away_score", "home_score_differential", "home_five",
    "away_five", "home_win", "rapm_source_season", "home_lineup_off_rapm",
    "home_lineup_def_rapm", "home_lineup_net_rapm", "away_lineup_off_rapm",
    "away_lineup_def_rapm", "away_lineup_net_rapm",
    "lineup_net_rapm_differential", "rapm_coverage_status", "split",
]

TARGET_COLUMN = "home_win"
NON_PREDICTOR_METADATA = {
    "game_id", "season", "game_date", "period", "possession_number",
    "rapm_source_season", "rapm_coverage_status", "split",
    "home_rated_players", "away_rated_players", "replacement_player_appearances",
    "feed_home_score_before", "feed_away_score_before",
}
FORBIDDEN_PREDICTOR_COLUMNS = {
    "points", "possession_points", "home_final_score", "away_final_score",
    "future_score", "next_score", "possession_result",
}

SPLIT_DEFINITION = {
    "audit_only": [2021],
    "train": [2022, 2023],
    "validation": [2024],
    "test": [2025],
}


def split_for_season(season: int) -> str:
    """Return the explicit forward-chaining partition for a season."""
    for split, seasons in SPLIT_DEFINITION.items():
        if int(season) in seasons:
            return split
    raise ValueError(f"season {season} is outside the pinned split definition")


def canonical_hash(value: object) -> str:
    """SHA-256 of canonical JSON; stable across mapping insertion order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_content_hash(con) -> str:
    """Logical SHA-256 of the sorted mart, independent of Parquet encoding.

    DuckDB's Parquet bytes can change after an otherwise identical table
    rebuild because row-group encoding is a physical detail. Hashing canonical
    row values proves deterministic feature content across rebuilds.
    """
    digest = hashlib.sha256()
    cursor = con.execute("""
        select * from fct_game_states
        order by game_id, period, possession_number
    """)
    while True:
        batch = cursor.fetchmany(10_000)
        if not batch:
            break
        for row in batch:
            digest.update(json.dumps(
                row, separators=(",", ":"), default=str, ensure_ascii=False
            ).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def coverage_summary(rows: Iterable[tuple]) -> dict[str, dict]:
    """Convert grouped (season, status, rows, missing appearances) into audit JSON."""
    out: dict[str, dict] = {}
    for season, status, n_rows, missing in rows:
        item = out.setdefault(
            str(int(season)),
            {"rows": 0, "full_rows": 0, "replacement_player_appearances": 0,
             "status_rows": {}},
        )
        n_rows = int(n_rows)
        item["rows"] += n_rows
        item["status_rows"][str(status)] = n_rows
        item["replacement_player_appearances"] += int(missing or 0)
        if status == "full":
            item["full_rows"] += n_rows
    for item in out.values():
        item["full_row_rate"] = item["full_rows"] / item["rows"] if item["rows"] else 0.0
        item["replacement_appearance_rate"] = (
            item["replacement_player_appearances"] / (10 * item["rows"])
            if item["rows"] else 0.0
        )
    return out


def _dict_rows(cursor) -> list[dict]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _gate_results(con) -> dict[str, dict]:
    """Run independent machine-checkable gates over the materialized mart."""
    queries = {
        "unique_possession_boundary": """
            select count(*) from (
              select game_id, period, possession_number
              from fct_game_states group by 1,2,3 having count(*) <> 1
            )
        """,
        "lineups_exactly_five": """
            select count(*) from fct_game_states
            where len(home_five) <> 5 or len(away_five) <> 5
        """,
        "complete_possession_coverage": """
            select count(*) from (
              select coalesce(p.game_id,s.game_id)
              from fct_possessions p full outer join fct_game_states s
                using(game_id,period,possession_number)
              where p.game_id is null or s.game_id is null
            )
        """,
        "possession_flag_consistent": """
            select count(*) from fct_game_states
            where home_has_possession <> (possession_team_id = home_team_id)
        """,
        "score_and_time_monotonic": """
            select count(*) from (
              select *,
                lag(home_score) over w as ph,
                lag(away_score) over w as pa,
                lag(elapsed_game_seconds) over w as pe
              from fct_game_states
              window w as (partition by game_id order by period, possession_number)
            ) where home_score < ph or away_score < pa or elapsed_game_seconds < pe
        """,
        "clock_fields_consistent": """
            select count(*) from fct_game_states
            where game_clock_seconds < 0
               or (period <= 4 and game_clock_seconds > 720)
               or (period > 4 and game_clock_seconds > 300)
               or regulation_seconds_remaining <>
                  greatest(0.0, 2880.0 - elapsed_game_seconds)
        """,
        "pre_possession_score_semantics": """
            select count(*) from (
              select s.home_score, s.away_score,
                coalesce(sum(case when p.offense_team_id=g.home_team_id then p.points else 0 end)
                  over(partition by p.game_id order by p.period,p.possession_number
                    rows between unbounded preceding and 1 preceding),0) expected_home,
                coalesce(sum(case when p.offense_team_id=g.away_team_id then p.points else 0 end)
                  over(partition by p.game_id order by p.period,p.possession_number
                    rows between unbounded preceding and 1 preceding),0) expected_away
              from fct_game_states s
              join fct_possessions p using(game_id,period,possession_number)
              join dim_games g using(game_id)
            ) where home_score<>expected_home or away_score<>expected_away
        """,
        "final_score_and_winner_reconcile": """
            select count(*) from (
              select s.game_id,
                sum(case when p.offense_team_id=g.home_team_id then p.points else 0 end) rh,
                sum(case when p.offense_team_id=g.away_team_id then p.points else 0 end) ra,
                max(g.home_final_score) oh, max(g.away_final_score) oa,
                max(s.home_win::integer) hw
              from fct_game_states s
              join fct_possessions p using(game_id,period,possession_number)
              join dim_games g using(game_id)
              group by s.game_id
            ) where rh<>oh or ra<>oa or hw<>(oh>oa)::integer
        """,
        "rapm_strictly_prior_or_fallback": """
            select count(*) from fct_game_states
            where (rapm_coverage_status <> 'cold_start'
                   and (rapm_source_season is null or rapm_source_season <> season - 1))
               or rapm_coverage_status is null
        """,
        "game_level_splits": """
            select count(*) from (
              select game_id from fct_game_states group by game_id
              having count(distinct split) <> 1
            )
        """,
    }
    results = {}
    for name, sql in queries.items():
        failures = int(con.execute(sql).fetchone()[0])
        results[name] = {"passed": failures == 0, "failures": failures}
    return results


def _trajectories(con) -> list[dict]:
    """Three representative six-state trajectories, including a close late game."""
    close_game = con.execute("""
        select s.game_id
        from fct_game_states s join dim_games g using(game_id)
        where s.split = 'test' and s.regulation_seconds_remaining <= 120
        group by s.game_id, g.home_final_score, g.away_final_score
        having abs(g.home_final_score - g.away_final_score) <= 3
        order by s.game_id limit 1
    """).fetchone()[0]
    ordinary = [
        r[0] for r in con.execute("""
            select distinct game_id from fct_game_states
            where split in ('train','validation') order by game_id limit 2
        """).fetchall()
    ]
    output = []
    for game_id in ordinary + [close_game]:
        rows = _dict_rows(con.execute("""
            with numbered as (
              select game_id, period, possession_number, game_clock_seconds,
                     home_score, away_score, home_has_possession,
                     lineup_net_rapm_differential,
                     row_number() over (
                       partition by game_id order by period, possession_number
                     ) rn,
                     count(*) over (partition by game_id) n
              from fct_game_states where game_id = ?
            )
            select game_id, period, possession_number, game_clock_seconds,
                   home_score, away_score, home_has_possession,
                   lineup_net_rapm_differential
            from numbered
            where rn in (
              1,
              greatest(1, round(n * 0.2)),
              greatest(1, round(n * 0.4)),
              greatest(1, round(n * 0.6)),
              greatest(1, round(n * 0.8)),
              n
            )
            order by rn
        """, [game_id]))
        output.append({
            "game_id": game_id,
            "kind": "close_late_test" if game_id == close_game else "representative",
            "states": rows,
        })
    return output


def build(
    db_path: str = DEFAULT_DB,
    out_dir: Path = DEFAULT_OUT,
    rapm_dir: Path = DEFAULT_RAPM_DIR,
) -> tuple[dict, dict]:
    """Export fct_game_states and write deterministic manifest + audit JSON."""
    import duckdb

    out_dir = Path(out_dir)
    rapm_dir = Path(rapm_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Stable scan/row-group order makes the Parquet bytes reproducible too,
        # not only the logical content hash.
        con.execute("set threads=1")
        actual_columns = [
            row[0] for row in con.execute("describe fct_game_states").fetchall()
        ]
        missing = sorted(set(REQUIRED_COLUMNS) - set(actual_columns))
        if missing:
            raise RuntimeError(f"fct_game_states missing required columns: {missing}")
        forbidden = sorted(FORBIDDEN_PREDICTOR_COLUMNS & set(actual_columns))
        if forbidden:
            raise RuntimeError(f"future/current outcome fields leaked into mart: {forbidden}")
        predictor_columns = [
            c for c in actual_columns
            if c != TARGET_COLUMN and c not in NON_PREDICTOR_METADATA
        ]

        counts = _dict_rows(con.execute("""
            select season, split, count(*) as rows, count(distinct game_id) as games
            from fct_game_states group by 1,2 order by 1
        """))
        total_rows, total_games = con.execute(
            "select count(*), count(distinct game_id) from fct_game_states"
        ).fetchone()
        coverage_rows = con.execute("""
            select season, rapm_coverage_status, count(*) as rows,
                   sum(replacement_player_appearances) as replacement_appearances
            from fct_game_states group by 1,2 order by 1,2
        """).fetchall()
        coverage = coverage_summary(coverage_rows)
        null_counts = {}
        for column in REQUIRED_COLUMNS:
            null_counts[column] = int(con.execute(
                f'select count(*) from fct_game_states where "{column}" is null'
            ).fetchone()[0])
        null_rates = {
            column: count / int(total_rows) for column, count in null_counts.items()
        }
        feed_score_disagreement_rows = int(con.execute("""
            select count(*) from fct_game_states
            where home_score <> feed_home_score_before
               or away_score <> feed_away_score_before
        """).fetchone()[0])
        feed_score_decrease_rows = int(con.execute("""
            select count(*) from (
              select *,
                lag(feed_home_score_before) over w as ph,
                lag(feed_away_score_before) over w as pa
              from fct_game_states
              window w as (partition by game_id order by period, possession_number)
            ) where feed_home_score_before < ph or feed_away_score_before < pa
        """).fetchone()[0])

        gates = _gate_results(con)
        if not all(item["passed"] for item in gates.values()):
            raise RuntimeError(f"game-state gates failed: {gates}")

        dataset_hash = table_content_hash(con)
        tmp = out_dir / "fct_game_states.parquet.tmp"
        final = out_dir / "fct_game_states.parquet"
        con.execute("""
            copy (
              select * from fct_game_states
              order by game_id, period, possession_number
            ) to ? (format parquet, compression zstd)
        """, [str(tmp)])
        os.replace(tmp, final)
        trajectories = _trajectories(con)
    finally:
        con.close()

    rapm_manifest = json.loads((rapm_dir / "manifest.json").read_text())
    rapm_metrics = json.loads((rapm_dir / "bayes_metrics.json").read_text())
    split_hash = canonical_hash(SPLIT_DEFINITION)
    schema_hash = canonical_hash({
        "version": FEATURE_SCHEMA_VERSION,
        "required_columns": REQUIRED_COLUMNS,
    })
    rapm_model_hash = canonical_hash({
        "ratings_sha256": file_hash(rapm_dir / "bayes_ratings.parquet"),
        "metrics": rapm_metrics,
    })
    parquet_hash = file_hash(final)

    manifest = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": schema_hash,
        "source_possession_corpus_hash": rapm_manifest["corpus_hash"],
        "rapm_model_hash": rapm_model_hash,
        "split_definition": SPLIT_DEFINITION,
        "split_hash": split_hash,
        "row_count": int(total_rows),
        "game_count": int(total_games),
        "counts_by_season_and_split": counts,
        "rapm_coverage_by_season": coverage,
        "target_column": TARGET_COLUMN,
        "predictor_columns": predictor_columns,
        "dataset_sha256": dataset_hash,
    }
    audit = {
        "manifest_hash": canonical_hash(manifest),
        "parquet_sha256": parquet_hash,
        "total_rows": int(total_rows),
        "total_games": int(total_games),
        "counts_by_season_and_split": counts,
        "null_counts": null_counts,
        "null_rates": null_rates,
        "rapm_coverage_by_season": coverage,
        "cold_start_player_appearances": int(
            10 * sum(
                item["rows"] for item in coverage.values()
                if "cold_start" in item["status_rows"]
            )
        ),
        "replacement_player_appearances": int(sum(
            item["replacement_player_appearances"] for item in coverage.values()
            if "cold_start" not in item["status_rows"]
        )),
        "score_reconciliation_pass_rate": 1.0,
        "lineup_validity_pass_rate": 1.0,
        "split_leakage_games": gates["game_level_splits"]["failures"],
        "feed_score_disagreement_rows": feed_score_disagreement_rows,
        "feed_score_decrease_rows": feed_score_decrease_rows,
        "gates": gates,
        "representative_trajectories": trajectories,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n"
    )
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and audit fct_game_states")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--rapm-dir", default=str(DEFAULT_RAPM_DIR))
    args = parser.parse_args()
    manifest, audit = build(args.db, Path(args.out), Path(args.rapm_dir))
    print(
        f"fct_game_states: {manifest['row_count']:,} rows / "
        f"{manifest['game_count']:,} games -> {args.out}"
    )
    print(f"dataset sha256: {manifest['dataset_sha256']}")
    for season, item in manifest["rapm_coverage_by_season"].items():
        print(
            f"  {season}: full rows {item['full_row_rate']:.1%}; "
            f"replacement appearances {item['replacement_appearance_rate']:.1%}"
        )
    print(
        f"gates: {sum(x['passed'] for x in audit['gates'].values())}/"
        f"{len(audit['gates'])} passed"
    )


if __name__ == "__main__":
    main()
