#!/usr/bin/env bash
# Sprint 6: snapshot Polymarket vig-free closing probabilities for the test season,
# then compare the P3 pre-game model against them.
#
# `python -m winprob.polymarket_pull` drives enumeration FROM THE MART: for every
# 2025-26 test game it builds the Polymarket event slug, fetches the moneyline event
# and both tokens' pre-tip prices, orients home/away by team id, and snapshots one
# row per COVERED game to data/winprob/polymarket_closing.parquet (with coverage +
# provenance in polymarket_pull.json). The pull is resumable and polite (browser
# User-Agent, paced, retried); it is skipped here if the snapshot already exists.
#
# `python -m winprob.polymarket_compare` re-fits the tier-E possession model and the
# P3 pre-game ladder leakage-safe, scores P3 vs the Polymarket probability on the
# covered games (Brier / log loss / calibration / paired game-clustered CI /
# correlation), re-measures the Sprint-4 fraction-of-gap-closed on this fuller,
# playoff-inclusive sample, and writes polymarket_metrics.json + polymarket_audit.json.
# It exits NON-ZERO only on a STRUCTURAL gate failure (a prediction outside (0, 1), a
# non-pre-tip price, or a non-test game). Beating the prediction market is REPORTED,
# never a hard failure — the market is expected to be sharper, exactly as in Sprint 3-4.
# Finally `winprob.polymarket_figure` renders the covered-game gap SVG to figures/.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/winprob/fct_game_states.parquet ]]; then
    echo "missing data/winprob/fct_game_states.parquet; run ./game_states.sh first" >&2
    exit 1
fi
if [[ ! -d data/raw/box_traditional ]]; then
    echo "missing data/raw/box_traditional; run the ingest first (see ingest/)" >&2
    exit 1
fi

if [[ ! -f data/winprob/polymarket_closing.parquet ]]; then
    echo "no snapshot yet; pulling Polymarket closing probabilities (resumable)..."
    .venv/bin/python -m winprob.polymarket_pull
fi

.venv/bin/python -m winprob.polymarket_compare
.venv/bin/python -m winprob.polymarket_figure
