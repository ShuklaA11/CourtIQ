#!/usr/bin/env bash
# Sprint 3 Phase 1: build, gate, export, and audit possession-boundary states.
set -euo pipefail

cd "$(dirname "$0")"
ALL_SEASONS='["00221%","00421%","00222%","00422%","00223%","00423%","00224%","00424%","00225%","00425%"]'

if [[ ! -f data/rapm/bayes_ratings.parquet ]]; then
    echo "missing data/rapm/bayes_ratings.parquet; run python -m rapm.bayes first" >&2
    exit 1
fi

(
    cd warehouse
    ../.venv/bin/dbt build --profiles-dir . --select +fct_game_states \
        --vars "{enable_winprob: true, game_id_patterns: $ALL_SEASONS, expected_game_count: 6572}"
)
.venv/bin/python -m winprob.design
