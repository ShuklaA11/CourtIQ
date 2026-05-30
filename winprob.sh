#!/usr/bin/env bash
# Sprint 3 Phase 2: fit + out-of-sample evaluate the logistic win-probability model.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/fct_game_states.parquet ]]; then
    echo "missing data/winprob/fct_game_states.parquet; run ./game_states.sh first" >&2
    exit 1
fi

.venv/bin/python -m winprob.model
.venv/bin/python -m winprob.evaluate
