#!/usr/bin/env bash
# Sprint 1 reconstruction pipeline, end to end.
#
#   0. python -m ingest.dates     -> ensure data/raw/game_dates.json exists
#   1. dbt run --select staging   -> materialize dim_games / stg_* into DuckDB
#   2. python -m recon.build      -> reconstruct lineups + possessions, write recon_*
#   3. dbt build --select marts   -> build fct_possessions + val_* and RUN THE GATES
#
# Step 0 is a no-op once cached (LeagueGameFinder dates, ~10 calls); dim_games
# joins the result, so staging fails fast if it is missing.
#
# `dbt build` runs models and their singular tests together, so a failing gate
# (a lineup not five, points that don't reconcile, systemic minutes drift) exits
# non-zero and stops the line.
#
# Scope defaults to 2023-24 (the game_id_patterns var in dbt_project.yml). Fan out
# to every season by passing a patterns list as the first argument, e.g.
#   ./run.sh '["00221%","00421%","00222%","00422%","00223%","00423%","00224%","00424%","00225%","00425%"]'
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$PWD"
# Absolute so they still resolve inside the `cd warehouse` subshells below.
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
DBT="${DBT:-$REPO_ROOT/.venv/bin/dbt}"
PATTERNS="${1:-}"

VARS_ARG=()
if [[ -n "$PATTERNS" ]]; then
    VARS_ARG=(--vars "{game_id_patterns: $PATTERNS}")
fi

echo "==> [0/4] python -m ingest.dates (ensure game_dates.json is cached)"
"$VENV_PY" -m ingest.dates

echo "==> [1/4] dbt run --select staging"
(cd warehouse && "$DBT" run --profiles-dir . --select staging "${VARS_ARG[@]+"${VARS_ARG[@]}"}")

echo "==> [2/4] python -m recon.build"
"$VENV_PY" -m recon.build --db warehouse/courtiq.duckdb --raw-root data/raw

echo "==> [3/4] dbt build --select marts (runs the validation gates)"
(cd warehouse && "$DBT" build --profiles-dir . --select marts "${VARS_ARG[@]+"${VARS_ARG[@]}"}")

echo "==> val_summary:"
(cd warehouse && "$DBT" show --profiles-dir . --limit 1 --inline "select * from {{ ref('val_summary') }}" "${VARS_ARG[@]+"${VARS_ARG[@]}"}")
