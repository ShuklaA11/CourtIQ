"""Tests for the Sprint-4 pre-game evaluation harness (`winprob.pregame`).

The heavy real-data run reuses `winprob.market`; these tests pin the harness
contract on synthetic inputs — deterministic, no network. The harness fixes the
measuring stick the rest of Sprint 4 is compared against, so what is pinned here
is exactly that contract:

1. `build_pregame_table` collapses the possession mart to one pre-tip row per
   game, keeps the fixed column set, carries the team strengths, and never
   mutates its input.
2. `score_forecast` computes Brier / log-loss / calibration through
   `winprob.evaluate` and nothing else, and fails fast on bad shapes.
3. `reproduce_baseline` re-fits tier E leakage-safe, predicts each test game's
   opening-state probability, and joins the vig-free market line exactly as
   `winprob.market.run` does (via `market.load_odds`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from winprob import evaluate, pregame, pregame_ladder

# Two real MGM-mapped franchises so the odds join resolves through
# `market.load_odds` (which maps team labels to these ids).
BOS_ID = 1610612738
LAL_ID = 1610612747
BOS_LABEL = "Boston"
LAL_LABEL = "LA Lakers"
TEAM_NET = {BOS_ID: 4.0, LAL_ID: -4.0}  # Boston strong, Lakers weak.


def _game(gid, season, split, home_id, away_id, date, rng, n=40):
    """One synthetic game as a possession-grain frame, tier-E ready."""
    home_net, away_net = TEAM_NET[home_id], TEAM_NET[away_id]
    edge = 1.0 if home_net > away_net else -1.0
    reg = np.linspace(2880.0, 0.0, n)
    margin = edge * np.linspace(0.0, 12.0, n) + rng.normal(0, 1, n)
    return pd.DataFrame({
        "game_id": gid, "season": season, "split": split,
        "game_date": pd.Timestamp(date),
        "period": np.clip((4 - np.floor(reg / 720.0)).astype(int), 1, 4),
        "possession_number": np.arange(n),
        "home_score_differential": margin,
        "regulation_seconds_remaining": reg,
        "home_has_possession": rng.integers(0, 2, n).astype(bool),
        "home_win": bool(margin[-1] + edge > 0),
        "home_team_id": home_id, "away_team_id": away_id,
        "home_lineup_net_rapm": home_net, "away_lineup_net_rapm": away_net,
        "lineup_net_rapm_differential": home_net - away_net,
        "home_rated_players": 5, "away_rated_players": 5,
        "rapm_source_season": float(season - 1),
    })


def _synthetic_frame():
    """Train/validation/test splits with both home/away orientations."""
    rng = np.random.default_rng(7)
    parts, c = [], [0]

    def add(season, split, home_id, away_id, date):
        parts.append(_game(f"g{c[0]}", season, split, home_id, away_id, date, rng))
        c[0] += 1

    for s in (2022, 2023):
        for i in range(4):
            add(s, "train", BOS_ID, LAL_ID, f"{s}-11-0{i + 1}")
            add(s, "train", LAL_ID, BOS_ID, f"{s}-12-0{i + 1}")
    for i in range(4):
        add(2024, "validation", BOS_ID, LAL_ID, f"2024-11-0{i + 1}")
        add(2024, "validation", LAL_ID, BOS_ID, f"2024-12-0{i + 1}")
    for i in range(4):
        add(2025, "test", BOS_ID, LAL_ID, f"2025-11-0{i + 1}")
        add(2025, "test", LAL_ID, BOS_ID, f"2025-12-0{i + 1}")
    return pd.concat(parts, ignore_index=True)


def _write_odds_csv(path, frame):
    """MGM-format odds CSV covering every test game, market leaning to outcome."""
    label = {BOS_ID: BOS_LABEL, LAL_ID: LAL_LABEL}
    test = frame[frame["split"] == "test"].groupby("game_id").agg(
        date=("game_date", "first"), home_id=("home_team_id", "first"),
        away_id=("away_team_id", "first"), home_win=("home_win", "first"),
    ).reset_index()
    rows = []
    for _, g in test.iterrows():
        # A sharp line: short home odds when the home team actually won.
        home_dec = 1.40 if g["home_win"] else 3.00
        away_dec = 3.00 if g["home_win"] else 1.40
        rows.append([
            pd.Timestamp(g["date"]).strftime("%Y-%m-%d-10:00"),
            label[g["away_id"]], label[g["home_id"]],
            away_dec, home_dec, bool(g["home_win"]),
        ])
    cols = ["game_date", "away_team", "home_team",
            "money_away_decimal_odds", "money_home_decimal_odds", "money_home_won"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return len(test)


# --------------------------------------------------------------------------
# 1. build_pregame_table.
# --------------------------------------------------------------------------

def test_build_pregame_table_one_row_per_game_with_fixed_columns():
    # Arrange
    df = _synthetic_frame()

    # Act
    table = pregame.build_pregame_table(df)

    # Assert
    assert list(table.columns) == list(pregame.PREGAME_COLUMNS)
    assert len(table) == df["game_id"].nunique()
    assert not table["game_id"].duplicated().any()


def test_build_pregame_table_carries_team_strength():
    # Arrange
    df = _synthetic_frame()

    # Act
    table = pregame.build_pregame_table(df)

    # Assert — a Boston-home game has Boston's (positive) strength on the home side.
    bos_home = table[table["home_team_id"] == BOS_ID].iloc[0]
    assert bos_home["home_team_strength"] > bos_home["away_team_strength"]


def test_build_pregame_table_does_not_mutate_input():
    # Arrange
    df = _synthetic_frame()
    before_cols = list(df.columns)
    before_rows = len(df)

    # Act
    pregame.build_pregame_table(df)

    # Assert
    assert list(df.columns) == before_cols
    assert len(df) == before_rows


# --------------------------------------------------------------------------
# 2. score_forecast.
# --------------------------------------------------------------------------

def test_score_forecast_matches_evaluate_primitives():
    # Arrange
    rng = np.random.default_rng(0)
    n = 100
    y = (rng.uniform(size=n) < 0.5).astype(float)
    p = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.1, n), 0.02, 0.98)
    game_ids = np.array([f"g{i}" for i in range(n)])

    # Act
    scored = pregame.score_forecast(y, p, game_ids)

    # Assert — every metric is exactly the evaluate primitive, not a re-derivation.
    intercept, slope = evaluate.fit_calibration(y, p)
    assert scored["brier"] == pytest.approx(evaluate.brier_score(y, p))
    assert scored["log_loss"] == pytest.approx(evaluate.mean_log_loss(y, p))
    assert scored["calibration"] == {"intercept": intercept, "slope": slope}
    assert scored["predictions_min"] == pytest.approx(float(np.min(p)))
    assert scored["predictions_max"] == pytest.approx(float(np.max(p)))


def test_score_forecast_rejects_length_mismatch():
    # Arrange
    y = np.array([1.0, 0.0])
    p = np.array([0.6, 0.4, 0.5])
    game_ids = np.array(["a", "b"])

    # Act / Assert
    with pytest.raises(ValueError):
        pregame.score_forecast(y, p, game_ids)


def test_score_forecast_rejects_empty_forecast():
    # Arrange
    empty = np.array([])

    # Act / Assert
    with pytest.raises(ValueError):
        pregame.score_forecast(empty, empty, empty)


# --------------------------------------------------------------------------
# 3. reproduce_baseline.
# --------------------------------------------------------------------------

def test_reproduce_baseline_scores_model_and_market(tmp_path):
    # Arrange
    df = _synthetic_frame()
    odds_csv = tmp_path / "odds.csv"
    n_test = _write_odds_csv(odds_csv, df)

    # Act
    baseline = pregame.reproduce_baseline(df, odds_csv)

    # Assert — both forecasts scored on the covered games; predictions in (0, 1).
    assert baseline["n_games"] == n_test
    for side in ("model", "market"):
        assert set(baseline[side]) == {
            "brier", "log_loss", "calibration",
            "predictions_min", "predictions_max",
        }
        assert 0.0 < baseline[side]["predictions_min"]
        assert baseline[side]["predictions_max"] < 1.0


def test_reproduce_baseline_raises_when_no_games_join(tmp_path):
    # Arrange — odds file whose dates never match any test game.
    df = _synthetic_frame()
    odds_csv = tmp_path / "odds.csv"
    _write_odds_csv(odds_csv, df)
    odds = pd.read_csv(odds_csv)
    odds["game_date"] = "2099-01-01-10:00"
    odds.to_csv(odds_csv, index=False)

    # Act / Assert
    with pytest.raises(ValueError):
        pregame.reproduce_baseline(df, odds_csv)


# --------------------------------------------------------------------------
# 4. Pre-game ablation ladder (`winprob.pregame_ladder`).
#
# The ladder needs a fuller possession mart than the harness fixtures above: the
# game-state feature columns AND final scores (for the current-season form /
# rest history). Four real MGM-mapped franchises across four seasons, with
# strength-driven-but-noisy outcomes so P1 improves on P0 without degenerating.
# --------------------------------------------------------------------------

MIA_ID = 1610612748
DEN_ID = 1610612743
LADDER_LABEL = {BOS_ID: "Boston", LAL_ID: "LA Lakers", MIA_ID: "Miami", DEN_ID: "Denver"}
LADDER_STRENGTH = {BOS_ID: 5.0, LAL_ID: 1.5, MIA_ID: -1.5, DEN_ID: -5.0}
LADDER_SPLITS = {2022: "train", 2023: "train", 2024: "validation", 2025: "test"}


def _ladder_game(gid, season, split, home, away, date, rng, n=50):
    """One synthetic game with the full game-state schema AND final scores."""
    hs, aw = LADDER_STRENGTH[home], LADDER_STRENGTH[away]
    final_home = 105.0 + (hs - aw) + 3.0 + rng.normal(0, 7)
    final_away = 105.0 + (aw - hs) + rng.normal(0, 7)
    home_score = np.round(np.linspace(0.0, final_home, n))
    away_score = np.round(np.linspace(0.0, final_away, n))
    margin = home_score - away_score
    reg = np.linspace(2880.0, 0.0, n)
    return pd.DataFrame({
        "game_id": gid, "season": season, "split": split,
        "game_date": pd.Timestamp(date),
        "period": np.clip((4 - np.floor(reg / 720.0)).astype(int), 1, 4),
        "possession_number": np.arange(n),
        "home_team_id": home, "away_team_id": away,
        "home_score": home_score, "away_score": away_score,
        "home_score_differential": margin,
        "regulation_seconds_remaining": reg,
        "home_has_possession": rng.integers(0, 2, n).astype(bool),
        "home_win": bool(home_score[-1] > away_score[-1]),
        "home_lineup_net_rapm": hs, "away_lineup_net_rapm": aw,
        "lineup_net_rapm_differential": hs - aw,
        "home_rated_players": 5, "away_rated_players": 5,
        "rapm_source_season": float(season - 1),
    })


def _ladder_frame():
    """A round-robin schedule per season so current-season form accrues by game."""
    rng = np.random.default_rng(11)
    ids = list(LADDER_STRENGTH)
    parts, c = [], [0]
    for season, split in LADDER_SPLITS.items():
        day = 1
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                parts.append(
                    _ladder_game(f"g{c[0]}", season, split, ids[i], ids[j],
                                 f"{season}-11-{day:02d}", rng)
                )
                c[0] += 1
                day += 1
    return pd.concat(parts, ignore_index=True)


def _write_ladder_odds(path, frame):
    """MGM-format odds CSV covering every test game, priced toward the outcome."""
    test = frame[frame["split"] == "test"].groupby("game_id").agg(
        date=("game_date", "first"), home_id=("home_team_id", "first"),
        away_id=("away_team_id", "first"), home_win=("home_win", "first"),
    ).reset_index()
    rows = []
    for _, g in test.iterrows():
        home_dec = 1.60 if g["home_win"] else 2.40
        away_dec = 2.40 if g["home_win"] else 1.60
        rows.append([
            pd.Timestamp(g["date"]).strftime("%Y-%m-%d-10:00"),
            LADDER_LABEL[g["away_id"]], LADDER_LABEL[g["home_id"]],
            away_dec, home_dec, bool(g["home_win"]),
        ])
    cols = ["game_date", "away_team", "home_team",
            "money_away_decimal_odds", "money_home_decimal_odds", "money_home_won"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return len(test)


# --- build_ladder_frame -----------------------------------------------------

def test_build_ladder_frame_emits_nested_feature_columns():
    # Arrange
    df = _ladder_frame()

    # Act
    frame = pregame_ladder.build_ladder_frame(df, shrink_k=10.0)

    # Assert — one pre-tip row per game with every ladder feature column present.
    assert len(frame) == df["game_id"].nunique()
    for col in (pregame_ladder.PRIOR_STRENGTH_DIFF, pregame_ladder.FORM_STRENGTH_DIFF,
                pregame_ladder.REST_DIFF, pregame_ladder.HOME_B2B, pregame_ladder.AWAY_B2B):
        assert col in frame.columns


def test_build_ladder_frame_does_not_mutate_input():
    # Arrange
    df = _ladder_frame()
    before_cols, before_rows = list(df.columns), len(df)

    # Act
    pregame_ladder.build_ladder_frame(df, shrink_k=10.0)

    # Assert
    assert list(df.columns) == before_cols
    assert len(df) == before_rows


# --- fit_ladder_tier --------------------------------------------------------

def test_fit_ladder_tier_never_reads_test_or_audit_rows():
    # Arrange
    df = _ladder_frame()
    frame = pregame_ladder.build_ladder_frame(df, shrink_k=10.0)

    # Act
    fit = pregame_ladder.fit_ladder_tier(frame, "P1", pregame_ladder.TIER_P1_FEATURES)

    # Assert — the fit's working frame is train+validation only.
    assert fit.splits_used == {"train", "validation"}
    assert fit.holdout_rows_excluded == (frame["split"] == "test").sum()


def test_predict_ladder_probabilities_strictly_inside_unit_interval():
    # Arrange
    df = _ladder_frame()
    frame = pregame_ladder.build_ladder_frame(df, shrink_k=10.0)
    fit = pregame_ladder.fit_ladder_tier(frame, "P3", pregame_ladder.TIER_P3_FEATURES)

    # Act
    p = pregame_ladder.predict_ladder(fit, frame)

    # Assert
    assert np.all(p > 0.0) and np.all(p < 1.0)


# --- select_shrink_k --------------------------------------------------------

def test_select_shrink_k_chooses_from_the_grid_on_validation():
    # Arrange
    df = _ladder_frame()

    # Act
    selection = pregame_ladder.select_shrink_k(df)

    # Assert — the pinned k is one of the grid values, scored per candidate.
    assert selection["chosen_k"] in pregame_ladder.SHRINK_K_GRID
    assert [s["shrink_k"] for s in selection["by_k"]] == list(pregame_ladder.SHRINK_K_GRID)


# --- evaluate_ladder --------------------------------------------------------

def test_evaluate_ladder_nested_tiers_and_structural_gates(tmp_path):
    # Arrange
    df = _ladder_frame()
    odds_csv = tmp_path / "odds.csv"
    _write_ladder_odds(odds_csv, df)
    covered = pregame.covered_games_frame(df, odds_csv)

    # Act
    metrics = pregame_ladder.evaluate_ladder(df, covered)

    # Assert — the four nested tiers, all predictions in (0, 1), structural gates hold.
    assert list(metrics["tiers"]) == list(pregame_ladder.LADDER_TIERS)
    n_features = [metrics["tiers"][t]["n_features"] for t in pregame_ladder.LADDER_TIERS]
    assert n_features == sorted(n_features) and n_features[0] == 1
    for tier in metrics["tiers"].values():
        assert tier["predictions_min"] > 0.0 and tier["predictions_max"] < 1.0
    assert metrics["gates"]["gate_predictions_in_open_interval"]
    assert metrics["gates"]["gate_test_season_untouched"]
    assert metrics["structural_gates_pass"]
    assert metrics["chosen_k"] in pregame_ladder.SHRINK_K_GRID


def test_evaluate_ladder_paired_diffs_and_gap_close_structure(tmp_path):
    # Arrange
    df = _ladder_frame()
    odds_csv = tmp_path / "odds.csv"
    n_test = _write_ladder_odds(odds_csv, df)
    covered = pregame.covered_games_frame(df, odds_csv)

    # Act
    metrics = pregame_ladder.evaluate_ladder(df, covered)

    # Assert — adjacent-tier paired diffs and the gap-close block are all present.
    for key in ("P1_minus_P0", "P2_minus_P1", "P3_minus_P2"):
        assert set(metrics["paired_diff"][key]) == {"brier", "log_loss"}
    gap = metrics["gap_close"]
    assert gap["n_games"] == n_test
    # fraction_of_gap_closed reconstructs (model - P3) / (model - market) on Brier.
    denom = gap["baseline_model_brier"] - gap["market_brier"]
    expected = (gap["baseline_model_brier"] - gap["p3_brier"]) / denom
    assert gap["fraction_of_gap_closed"] == pytest.approx(expected)
    assert set(gap["fraction_of_gap_closed_ci"]) == {"lo", "hi", "point"}
    assert -1.0 <= gap["correlation_p3_market"] <= 1.0


def test_evaluate_ladder_orthogonality_reports_model_coefficient_ci(tmp_path):
    # Arrange
    df = _ladder_frame()
    odds_csv = tmp_path / "odds.csv"
    _write_ladder_odds(odds_csv, df)
    covered = pregame.covered_games_frame(df, odds_csv)

    # Act
    ortho = pregame_ladder.evaluate_ladder(df, covered)["gap_close"]["orthogonality"]

    # Assert — the 2-feature logistic reports the model coefficient with a CI.
    assert "model_coefficient" in ortho
    assert set(ortho["model_coefficient_ci"]) == {"lo", "hi", "point"}
    assert ortho["model_coefficient_ci"]["lo"] <= ortho["model_coefficient_ci"]["hi"]


def test_form_beats_prior_strength_gate_tracks_the_p2_minus_p1_ci(tmp_path):
    # Arrange
    df = _ladder_frame()
    odds_csv = tmp_path / "odds.csv"
    _write_ladder_odds(odds_csv, df)
    covered = pregame.covered_games_frame(df, odds_csv)

    # Act
    metrics = pregame_ladder.evaluate_ladder(df, covered)

    # Assert — the gate is exactly "P2's Brier improvement CI upper bound below 0".
    hi = metrics["paired_diff"]["P2_minus_P1"]["brier"]["hi"]
    assert metrics["gates"]["gate_form_beats_prior_strength"] == (hi < 0.0)


# --- run: JSON artifacts + provenance --------------------------------------

def test_run_writes_metrics_and_audit_with_provenance(tmp_path):
    # Arrange — a data dir with the mart parquet and the odds CSV.
    from winprob import design

    df = _ladder_frame()
    data_dir = tmp_path / "winprob"
    data_dir.mkdir()
    parquet_path = data_dir / pregame.PARQUET_NAME
    odds_path = data_dir / pregame.ODDS_CSV_NAME
    df.to_parquet(parquet_path)
    _write_ladder_odds(odds_path, df)

    # Act
    metrics = pregame.run(data_dir)

    # Assert — both JSON artifacts exist; the audit pins the mart, odds, and split.
    assert (data_dir / pregame.METRICS_JSON_NAME).exists()
    audit_path = data_dir / pregame.AUDIT_JSON_NAME
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text())
    assert audit["dataset_parquet_sha256"] == design.file_hash(parquet_path)
    assert audit["odds_csv_sha256"] == design.file_hash(odds_path)
    assert audit["split_hash"] == design.canonical_hash(design.SPLIT_DEFINITION)
    assert audit["chosen_k"] == metrics["chosen_k"]
    assert audit["structural_gates_pass"] is True


def test_run_raises_when_the_mart_is_missing(tmp_path):
    # Arrange — an empty data dir.
    data_dir = tmp_path / "winprob"
    data_dir.mkdir()

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        pregame.run(data_dir)
