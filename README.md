# CourtIQ

Bayesian player-impact and win-probability modeling on NBA play-by-play, built
for statistical rigor: regularized adjusted plus-minus (RAPM) with honest
uncertainty, and a calibrated in-game win-probability model validated against
held-out seasons and betting-market lines.

## Status

**Sprint 2 — RAPM (complete).** Ridge baseline and exact Gaussian-posterior
Offense/Defense RAPM, fit on 1,273,794 reconstructed possessions (5 seasons,
2,601 player-seasons) and validated out-of-sample. Win probability is next.

## Results

Ratings are Offense/Defense RAPM at the possession grain, one offensive and
defensive coefficient per **player-season** (200-possession floor; fringe players
pooled to replacement). The ridge point estimate is the MAP of a conjugate
Gaussian model, so the Bayesian model is the *same* fit with a full posterior —
the point estimates are identical and the posterior adds honest uncertainty.
All numbers below are held-out (test games never seen in training or λ-selection)
and reproducible from corpus `f3494b21`.

**Out-of-sample retrodiction** — predict held-out game margins, 1,283 games
(RMSE, lower is better):

| Method | Margin RMSE |
|---|---|
| **Ridge / Bayesian RAPM** | **13.63** |
| Team net-rating | 15.06 |
| Predict the mean (floor; margin σ = 15.43) | 15.44 |
| Raw plus-minus (unadjusted) | 30.33 |

RAPM is the only method that meaningfully beats the constant-mean floor (−11.7%),
and it beats team-strength too. Single-game margins are variance-dominated, so
this is a strong retrodiction; raw plus-minus is *worse* than the floor because
it mis-attributes teammates' and opponents' quality — exactly what the adjustment
fixes.

**Calibration** — the posterior's uncertainty is honest, not decorative.
Predictive 90% intervals for held-out margins, and simulation-based calibration
(SBC) recovering known planted effects:

| Nominal | 50% | 80% | 90% | 95% |
|---|---|---|---|---|
| Empirical coverage | 0.51 | 0.79 | 0.88 | 0.93 |

SBC parameter coverage at 90% is **0.90** over 40 synthetic datasets, confirming
the covariance/interval math. The mild shortfall at 90–95% is the expected,
reported consequence of modeling discrete possession points as Gaussian
(intra-game correlation the iid term can't capture) — not tuned away.

**Uncertainty tracks data volume.** Mean 90% credible-interval width (net rating,
per 100) by possession tercile:

| Tercile | mean possessions | mean 90% CI width |
|---|---|---|
| Low | 1,151 | 10.96 |
| Mid | 4,436 | 9.40 |
| High | 9,077 | 8.67 |

This is the payoff of the Bayesian layer: low-possession players shrink toward
zero with wide intervals that cross it, so the confident-looking extreme ratings
a ridge point estimate assigns to fringe players are correctly flagged as
uncertain.

**Leaderboard** (top net rating per 100, with 90% credible interval):

| Net | 90% CI | Player (season) | Poss |
|---:|---|---|---:|
| +9.11 | [+4.8, +13.5] | V. Wembanyama (2025-26) | 10,300 |
| +7.41 | [+3.2, +11.6] | S. Gilgeous-Alexander (2024-25) | 14,198 |
| +7.15 | [+2.8, +11.5] | C. Holmgren (2025-26) | 9,745 |
| +6.90 | [+2.8, +11.0] | S. Curry (2021-22) | 12,255 |
| +6.63 | [+2.4, +10.9] | G. Antetokounmpo (2024-25) | 9,837 |
| +6.43 | [+2.1, +10.7] | J. Tatum (2021-22) | 13,456 |
| +5.97 | [+1.5, +10.5] | J. Embiid (2021-22) | 10,348 |
| +5.72 | [+1.2, +10.2] | N. Jokić (2024-25) | 12,493 |

Every interval clears zero. Two 3-and-D role players (Finney-Smith, Caldwell-Pope)
also rank high — the signature of the collinearity between stars and the
teammates who share their minutes; the credible intervals are how much to trust
each estimate.

**Honest caveats.** (1) A within-season recency-weighted variant (60-day
half-life) reshuffles 6 of the top 10 — recency weighting materially moves
ratings, so the headline table is the unweighted fit. (2) Season-to-season net
ratings correlate r = 0.32 (n = 1,663 player pairs): real predictive signal, far
from deterministic (aging, role and roster change). (3) 2.2% of games are
quarantined (OT lineups the play-by-play can't reconstruct), dropped whole and
roughly evenly across teams — a small, near-unbiased gap.

_Regenerate: `python -m rapm.design && python -m rapm.ridge && python -m rapm.bayes && python -m rapm.results`._

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
