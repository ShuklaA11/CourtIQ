"""Sprint 2 results & validation diagnostics for the RAPM models.

Assembles the evidence that fills the README Results section, all reproducible
from the pinned corpus (data/rapm/):
  - retrodiction ladder (ridge vs baselines, from the Phase-2 metrics)
  - predictive calibration curve across nominal levels (not just 90%)
  - credible-interval width by possession tercile (uncertainty vs data volume)
  - season-to-season stability of net ratings (are they predictive, secondarily)
  - quarantine missingness bias (are dropped games team-skewed?)
  - the leaderboard with honest 90% credible intervals

Every number is recomputed from the artifacts, so `python -m rapm.results` after
a corpus rebuild refreshes the whole section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from rapm.bayes import _game_arrays, effective_margin_noise, fit_posterior, predictive_intervals
from rapm.ridge import DEFAULT_DATA_DIR, DEFAULT_WAREHOUSE, load_artifacts


def coverage_curve(fit, art, levels: tuple[float, ...]) -> dict[str, float]:
    """Empirical predictive coverage of held-out margins at several nominal levels.

    Calibrates the effective margin noise on the training games (as the Phase-3
    gate does), then, for each nominal level, widens the interval to that level's
    z and measures the fraction of ACTUAL test margins inside. A well-calibrated
    model tracks the diagonal (empirical ~ nominal); systematic shortfall is the
    honest discreteness under-coverage.
    """
    w_tr, n_tr, act_tr = _game_arrays(art, fit.Xi, fit.train_mask)
    noise = effective_margin_noise(fit.factor, w_tr, fit.mu, fit.sigma2, act_tr, n_tr)
    w_te, n_te, act_te = _game_arrays(art, fit.Xi, fit.test_mask)
    curve: dict[str, float] = {}
    for level in levels:
        z = float(norm.ppf(0.5 + level / 2))
        _, _, lo, hi = predictive_intervals(
            fit.factor, w_te, fit.mu, fit.sigma2, n_te, z=z, noise_var=noise
        )
        curve[f"{level:.2f}"] = float(np.mean((act_te >= lo) & (act_te <= hi)))
    return curve


def ci_width_by_tercile(
    n_possessions: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray
) -> list[dict]:
    """Mean 90% credible-interval width within possession terciles.

    Sorts player-seasons by possessions and splits into three equal groups. The
    honest-uncertainty signature is monotone: low-possession terciles carry wider
    intervals. Pure.
    """
    n = np.asarray(n_possessions, dtype=np.float64)
    width = np.asarray(ci_high, dtype=np.float64) - np.asarray(ci_low, dtype=np.float64)
    order = np.argsort(n)
    thirds = np.array_split(order, 3)
    labels = ["low", "mid", "high"]
    out = []
    for label, idx in zip(labels, thirds):
        out.append({
            "tercile": label,
            "n_player_seasons": int(len(idx)),
            "mean_possessions": float(n[idx].mean()),
            "mean_ci_width": float(width[idx].mean()),
        })
    return out


def season_to_season_stability(
    player_id: np.ndarray, season: np.ndarray, net_rating: np.ndarray
) -> dict:
    """Correlation of a player's net rating between consecutive seasons.

    A secondary, harder check than retrodiction: do ratings carry predictive
    signal across seasons (confounded by roster/aging changes)? Builds all
    (net in season s, net in season s+1) pairs for players present in both and
    returns Pearson r plus the pair count. Pure.
    """
    by_key = {(int(p), int(s)): float(v) for p, s, v in zip(player_id, season, net_rating)}
    cur, nxt = [], []
    for (p, s), v in by_key.items():
        if (p, s + 1) in by_key:
            cur.append(v)
            nxt.append(by_key[(p, s + 1)])
    if len(cur) < 2:
        return {"pearson_r": None, "n_pairs": len(cur)}
    r = float(np.corrcoef(cur, nxt)[0, 1])
    return {"pearson_r": r, "n_pairs": len(cur)}


def quarantine_bias(warehouse: str) -> dict | None:
    """Are quarantined (mostly OT-unreconstructable) games team-skewed?

    Compares each team's share of quarantined games to its share of all games; a
    large ratio would mean the dropped set biases certain teams' ratings. Returns
    the most over-represented teams. Warehouse-dependent; returns None if absent.
    """
    try:
        import duckdb
    except ImportError:
        return None
    if not Path(warehouse).exists():
        return None
    con = duckdb.connect(warehouse, read_only=True)
    try:
        rows = con.execute(
            """
            with q as (select distinct game_id from recon_quarantine),
            team_games as (
                select home_team_id as team, game_id from dim_games
                union all
                select away_team_id as team, game_id from dim_games
            ),
            per_team as (
                select team,
                    count(*) as games,
                    count(*) filter (where game_id in (select game_id from q)) as quarantined
                from team_games group by team
            )
            select team, games, quarantined,
                quarantined::double / games as q_rate
            from per_team order by q_rate desc limit 5
            """
        ).fetchall()
        overall = con.execute(
            "select count(distinct game_id) from recon_quarantine"
        ).fetchone()[0]
        total = con.execute("select count(*) from dim_games").fetchone()[0]
    finally:
        con.close()
    return {
        "overall_quarantine_rate": overall / total if total else None,
        "most_affected_teams": [
            {"team": int(t), "games": int(g), "quarantined": int(q), "q_rate": float(qr)}
            for t, g, q, qr in rows
        ],
    }


def leaderboard(ratings_df, k: int) -> list[dict]:
    """Top-k player-seasons by net rating with their 90% credible interval. Pure."""
    top = ratings_df.sort_values("net_rating", ascending=False).head(k)
    return top[
        ["player_id", "season", "net_rating", "net_ci_low", "net_ci_high", "n_possessions"]
    ].to_dict("records")


def run(data_dir=DEFAULT_DATA_DIR, warehouse: str = DEFAULT_WAREHOUSE) -> dict:
    """Recompute every diagnostic and write data/rapm/results.json."""
    import pandas as pd

    data = Path(data_dir)
    ridge_metrics = json.loads((data / "ridge_metrics.json").read_text())
    bayes_metrics = json.loads((data / "bayes_metrics.json").read_text())
    ratings = pd.read_parquet(data / "bayes_ratings.parquet")

    art = load_artifacts(data_dir)
    fit = fit_posterior(art, float(ridge_metrics["chosen_lambda"]))

    curve = coverage_curve(fit, art, levels=(0.50, 0.80, 0.90, 0.95))
    terciles = ci_width_by_tercile(
        ratings["n_possessions"].to_numpy(),
        ratings["net_ci_low"].to_numpy(),
        ratings["net_ci_high"].to_numpy(),
    )
    stability = season_to_season_stability(
        ratings["player_id"].to_numpy(),
        ratings["season"].to_numpy(),
        ratings["net_rating"].to_numpy(),
    )
    bias = quarantine_bias(warehouse)
    board = leaderboard(ratings, k=15)

    results = {
        "corpus_hash": bayes_metrics["corpus_hash"],
        "retrodiction_rmse": {
            "ridge": ridge_metrics["test_rmse_ridge"],
            "raw_plus_minus": ridge_metrics["test_rmse_rawpm"],
            "team_net_rating": ridge_metrics["test_rmse_teamnet"],
            "n_test_games": ridge_metrics["n_test_games"],
        },
        "calibration": {
            "predictive_coverage_curve": curve,
            "sbc_coverage_90": bayes_metrics["sbc_coverage_90"],
            "n_sbc_datasets": bayes_metrics["n_sbc_datasets"],
        },
        "ci_width_by_tercile": terciles,
        "season_to_season_stability": stability,
        "quarantine_bias": bias,
        "leaderboard_top15": board,
    }
    (data / "results.json").write_text(json.dumps(results, indent=2, default=str))

    _print(results)
    return results


def _print(r: dict) -> None:
    rd = r["retrodiction_rmse"]
    print(f"Retrodiction RMSE (test fold, {rd['n_test_games']} games): "
          f"ridge {rd['ridge']:.2f} | team-net {rd['team_net_rating']:.2f} | "
          f"raw+/- {rd['raw_plus_minus']:.2f}")
    print("Predictive coverage curve (nominal -> empirical):")
    for lvl, cov in r["calibration"]["predictive_coverage_curve"].items():
        print(f"  {float(lvl):.0%} -> {cov:.3f}")
    print(f"SBC coverage @90%: {r['calibration']['sbc_coverage_90']:.3f} "
          f"over {r['calibration']['n_sbc_datasets']} datasets")
    print("CI width by possession tercile:")
    for t in r["ci_width_by_tercile"]:
        print(f"  {t['tercile']:>4}  mean_poss={t['mean_possessions']:>8.0f}  "
              f"mean_90%_width={t['mean_ci_width']:.2f}")
    s = r["season_to_season_stability"]
    print(f"Season-to-season net-rating stability: r={s['pearson_r']} (n={s['n_pairs']} pairs)")
    if r["quarantine_bias"]:
        print(f"Quarantine rate overall: {r['quarantine_bias']['overall_quarantine_rate']:.3f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAPM Sprint 2 results diagnostics")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()
    run(args.data_dir, args.warehouse)


if __name__ == "__main__":
    main()
