#!/usr/bin/env bash
# Sprint 4: run the pre-game ablation ladder + gap-close-vs-market, then render the
# gap-decomposition figure.
#
# `python -m winprob.pregame` fits the P0..P3 pre-game ladder leakage-safe, scores
# it on the untouched 2025 test season, measures how much of the tier-E-to-market
# Brier gap the pre-game features close, and writes data/winprob/pregame_metrics.json
# + pregame_audit.json. It exits NON-ZERO only on a STRUCTURAL gate failure (a
# prediction outside (0, 1), or the test season leaking into a fit) — the honest
# scientific gates (does form beat prior strength? does the model beat the market?)
# are REPORTED, never a hard failure.
#
# The odds file is the same public MGM Kaggle dataset market.sh uses; see that
# script's header for the one-line, login-free download.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/fct_game_states.parquet ]]; then
    echo "missing data/winprob/fct_game_states.parquet; run ./game_states.sh first" >&2
    exit 1
fi
if [[ ! -f data/winprob/nba_closing_odds_mgm.csv ]]; then
    echo "missing data/winprob/nba_closing_odds_mgm.csv; see market.sh for how to fetch it" >&2
    exit 1
fi

.venv/bin/python -m winprob.pregame
.venv/bin/python -m winprob.pregame_figure
