#!/usr/bin/env bash
# Sprint 3 Phase 5: compare the tier-E model's pre-game win probabilities to the
# market (MGM closing moneylines) on the covered 2025-26 test games.
#
# The odds file is a public, ungated Kaggle dataset. If it is missing, download it
# (no login required) and save all_odds.csv to data/winprob/nba_closing_odds_mgm.csv:
#   curl -sSL -A "Mozilla/5.0" -o /tmp/mgm.zip \
#     https://www.kaggle.com/api/v1/datasets/download/caseydurfee/mgm-grand-nba-betting-data
#   unzip -p /tmp/mgm.zip all_odds.csv > data/winprob/nba_closing_odds_mgm.csv
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/fct_game_states.parquet ]]; then
    echo "missing data/winprob/fct_game_states.parquet; run ./game_states.sh first" >&2
    exit 1
fi
if [[ ! -f data/winprob/nba_closing_odds_mgm.csv ]]; then
    echo "missing data/winprob/nba_closing_odds_mgm.csv; see the header of this script to fetch it" >&2
    exit 1
fi

.venv/bin/python -m winprob.market
