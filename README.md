# CourtIQ

Bayesian player-impact and win-probability modeling on NBA play-by-play, built
for statistical rigor: regularized adjusted plus-minus (RAPM) with honest
uncertainty, and a calibrated in-game win-probability model validated against
held-out seasons and betting-market lines.

## Status

Under construction. Current phase: **Sprint 0 — raw ingest** (complete).

## Results

_Pending — this section will lead with the validation table (out-of-sample RAPM
error by minutes tercile, posterior interval coverage) and calibration curves._

## Pipeline

| Stage | What it does |
|---|---|
| **Ingest** (`ingest/`) | Resumable, rate-limited pull of 5 seasons of play-by-play + box scores to raw JSON. |
| Reconstruction (dbt) | Possessions and on-floor lineups from the event log; validated against official box scores. |
| RAPM | Hierarchical Bayesian adjusted plus-minus vs. a ridge baseline, out-of-sample tested. |
| Win probability | Calibrated P(home win) from game state + lineup strength; leakage-controlled. |

Seasons: 2021-22 through 2025-26. The 2019-20 and 2020-21 seasons are excluded
deliberately — the bubble and 72-game COVID seasons have anomalous pace, rest,
and home-court structure.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ingest.pull            # full pull (~5h, resumable)
python -m ingest.pull --limit 3  # smoke test
```
