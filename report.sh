#!/usr/bin/env bash
# Sprint 3 Phase 5: consolidate Phase 2-4 results and enforce the reproducibility
# gate (all published numbers trace to one pinned corpus/feature/quality/split/
# model tuple, and the on-disk mart + model hash to their pinned values).
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/challenger_metrics.json ]]; then
    echo "missing Phase 2-4 artifacts; run ./winprob.sh, ./ablation.sh, and ./challenger.sh first" >&2
    exit 1
fi

.venv/bin/python -m winprob.report
.venv/bin/python -m winprob.figures
