#!/usr/bin/env bash
# Sprint 5: build the pre-tip availability layer, run the P4 injury ladder tier +
# gap-close-vs-market, then render the injury-gap figure.
#
# `python -m winprob.availability` folds the raw V3 box scores into a leakage-safe
# per-(game, team, player) inactive-list signal (data/winprob/game_availability.parquet).
# `python -m winprob.pregame_injury_report` fits the P4 tier (P3 form ladder +
# availability-adjusted prior-season RAPM strength) leakage-safe, scores it on the
# untouched 2025 test season, re-measures how much of the tier-E-to-market Brier gap
# it closes, and writes data/winprob/injury_metrics.json + injury_audit.json. It
# exits NON-ZERO only on a STRUCTURAL gate failure (a prediction outside (0, 1), or
# the test season leaking into a fit) — the scientific gate (does knowing who is out
# beat current-season form?) is REPORTED, never a hard failure.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/fct_game_states.parquet ]]; then
    echo "missing data/winprob/fct_game_states.parquet; run ./game_states.sh first" >&2
    exit 1
fi
if [[ ! -d data/raw/box_traditional ]]; then
    echo "missing data/raw/box_traditional; run the ingest (python -m ingest.pull) first" >&2
    exit 1
fi
if [[ ! -f data/rapm/bayes_ratings.parquet ]]; then
    echo "missing data/rapm/bayes_ratings.parquet; run the RAPM pipeline first" >&2
    exit 1
fi
if [[ ! -f data/winprob/nba_closing_odds_mgm.csv ]]; then
    echo "missing data/winprob/nba_closing_odds_mgm.csv; see market.sh for how to fetch it" >&2
    exit 1
fi

.venv/bin/python -m winprob.availability
.venv/bin/python -m winprob.pregame_injury_report
.venv/bin/python -m winprob.injury_figure
