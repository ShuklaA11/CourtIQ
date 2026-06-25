# CourtIQ

**Bayesian player-impact ratings and a calibrated NBA win-probability model, built end
to end from raw play-by-play — with leakage-safe evaluation, honest uncertainty, and
every number reproducible from a pinned provenance hash.**

## Why this exists

Most public NBA analytics breaks in one of four ways, and CourtIQ is built to fix each:

- **Raw plus-minus mis-attributes credit.** A bench player riding a star's minutes looks
  elite. CourtIQ uses **regularized adjusted plus-minus (RAPM)** to isolate a player's
  own on-court impact — and it *beats* team net-rating at predicting held-out game
  margins, while raw plus-minus scores *worse* than guessing the mean.
- **Point estimates hide what they don't know.** A fringe player with 200 possessions
  gets the same confident number as a 10,000-possession star. CourtIQ's RAPM is the MAP
  of a **conjugate Bayesian model**, so every rating carries a credible interval that
  *widens* when the data is thin — validated by simulation-based calibration.
- **Win-probability models leak the future.** CourtIQ enforces **forward-chaining
  splits** (train 2022–24, score on an untouched 2025 test season), fits everything
  leakage-safe, and quantifies uncertainty with a **game-clustered bootstrap** — never a
  row-wise one that would fake significance on autocorrelated possessions.
- **Results don't reproduce.** Every published figure traces to one **sha256 provenance
  tuple** (corpus · split · features · model · parquet); a single gate re-hashes the
  on-disk artifacts and fails if anything drifted.

The model is then stress-tested against two sharp external benchmarks — a **sportsbook**
(MGM closing lines) and a **prediction market** (Polymarket) — pre-game, honestly:
beating them is *reported, never gated*.

## Architecture

```
NBA Stats API  ──ingest/──▶  raw JSON        5 seasons of play-by-play + box scores
 (rate-limited,             (2021-22 …          (resumable pull)
  resumable)                 2025-26)
                                │
                    recon/ + dbt (DuckDB)        possessions & on-floor lineups
                                │                 reconstructed from the event log,
                                ▼                 gated against official box scores
                         RAPM  (rapm/)            offense/defense per player-season
                    ridge point est. = MAP        ridge = MAP of a conjugate Bayesian
                    of a conjugate Bayesian        posterior → honest credible intervals
                                │
                                ▼
                 fct_game_states mart  (winprob/design.py)
                 score · clock · possession · lineup RAPM, one row per possession
                                │
                                ▼
             win-probability model  (intercept-free L2 logistic, from scratch)
                                │
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                           ▼
   RAPM ablation          GBM challenger            pre-game benchmarks
 (does the on-court     (is the signal linear?    vs MGM sportsbook · vs Polymarket
  five add signal?)      hist. gradient boosting)   prediction market (gap-close)
```

Everything downstream of the raw pull is **pure NumPy/pandas** — the logistic model, the
gradient-boosted challenger, RAPM, and the bootstrap are all hand-rolled, no third-party
ML dependency.

## What it demonstrates

Bayesian inference (conjugate posteriors, SBC) · regularized regression from scratch ·
leakage-safe out-of-sample evaluation · probability **calibration** & reliability
analysis · **game-clustered bootstrap** inference · gradient boosting from scratch ·
data engineering (dbt/DuckDB reconstruction gated against ground truth) · resilient API
ingestion (rate-limiting, retry, resumability, concurrency) · end-to-end
**reproducibility** via content hashing.

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Ingest | Python · `requests` | Resumable, rate-limited pull of 5 seasons of PBP + box scores |
| Reconstruction | **dbt** · **DuckDB** | Possessions & on-floor lineups from the event log, gated vs. box scores |
| RAPM | NumPy | Ridge point estimate = MAP of a conjugate Bayesian model; full posterior for CIs |
| Win-prob model | NumPy | Intercept-free L2 logistic, guarded sigmoid, leakage-safe λ-selection |
| Challenger | NumPy | Histogram **Newton gradient boosting**, hand-rolled — no ML dependency |
| Evaluation | NumPy · pandas | Brier / log-loss, calibration, phase breakdowns, paired **game-clustered** bootstrap CIs |
| Benchmarks | `requests` | MGM closing odds (public dataset) + Polymarket API (Gamma events + CLOB prices) |
| Reproducibility | `hashlib` (sha256) | Pinned corpus/split/feature/model/parquet tuple; `results.json` re-hash gate |

## Key results

All held-out on the untouched **2025 test season** (or OOS game margins for RAPM). Full
tables, methodology, and caveats in **[WRITEUP.md](WRITEUP.md)**.

| Question | Result |
|---|---|
| Does RAPM beat simpler ratings at predicting margins? | **Yes** — 13.63 RMSE vs. 15.06 team net-rating, 15.44 mean-floor (**−11.7%**); raw +/- is 30.33, *worse* than the floor |
| Is the Bayesian uncertainty honest? | **Yes** — 90% predictive coverage 0.88, SBC parameter coverage 0.90; CI width shrinks with possessions |
| Is the win-probability model calibrated? | **Yes** — Brier **0.156**, log loss **0.466**; every reliability bin within 0.031 of empirical |
| Does the specific on-court five add signal beyond team strength? | **No** (honest null) — team strength adds Brier −0.005 (CI clears 0), lineup RAPM's D−C CI straddles 0 |
| Is the signal non-linear? | **No** — a from-scratch GBM on identical features does *not* beat the logistic (paired CI straddles 0) |
| How does it fare vs. the market pre-game? | Markets **sharper** (as expected); model **calibrated**, correlates 0.73 (sportsbook) / **0.863** (Polymarket) |
| How much of the model→market gap does a box score close? | **+53.1%** vs. Polymarket (CI **[+37.4%, +68.5%]**, 1,255/1,258 games incl. playoffs) |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ingest.pull --limit 3   # smoke test (full pull is ~5h, resumable)
```

Each stage is a self-contained, gated script that writes artifacts to `data/` and exits
non-zero only on a structural failure:

| Script | Produces |
|---|---|
| `./game_states.sh` | Builds + audits the possession-boundary win-probability mart |
| `./winprob.sh` | Fits the logistic model, scores + gates it out-of-sample |
| `./ablation.sh` | Nested A→E RAPM lineup ablation with game-clustered CIs |
| `./challenger.sh` | Gradient-boosted challenger vs. the logistic |
| `./market.sh` | Pre-game model vs. MGM sportsbook closing lines |
| `./pregame.sh` | P0→P3 pre-game ladder; fraction of the market gap closed |
| `./injury.sh` | Pre-tip availability tier (the injury-edge null) |
| `./polymarket.sh` | Pre-game model vs. the Polymarket prediction market |
| `./report.sh` | Renders SVG figures + the reproducibility gate (`results.json`) |

Seasons span 2021-22 → 2025-26; 2019-20 and 2020-21 are excluded deliberately (the bubble
and 72-game COVID seasons have anomalous pace, rest, and home-court structure).

## Project structure

```
courtiq/
├── ingest/        Resilient NBA Stats API pull (rate-limit, retry, resume)
├── recon/         Possession & lineup reconstruction helpers
├── warehouse/     dbt project + DuckDB warehouse (possessions, lineups, marts)
├── rapm/          RAPM: design · ridge · Bayesian posterior · results
├── winprob/       Win-prob mart, logistic model, ablation, GBM, market benchmarks
├── figures/       Hand-rolled SVG figures (no plotting dependency)
├── tests/         356 tests (unit · integration · reproducibility gates)
├── *.sh           One gated runner per stage
├── README.md      This file
└── WRITEUP.md     Detailed results, methodology, and per-sprint findings
```

## Reproducibility

Every published result traces to one immutable provenance tuple, and a final gate
re-hashes the on-disk mart and model to prove the artifacts each number was computed
against are the ones still present:

```
corpus f3494b21 · split eb69be5d · feature-schema 80d9e8f0 · model 30a4972b · parquet 685233a9
```

`./report.sh` walks the mart→model→ablation→challenger chain, asserts one consistent
tuple, and emits `results.json` with an `all_results_reproduce` boolean. Tampering any
single link — a downstream hash, a quality gate, an on-disk file — flips the gate and
pinpoints the break.

---

**Deep dive → [WRITEUP.md](WRITEUP.md)** — full result tables, methodology, honest
caveats, and the sprint-by-sprint log.
