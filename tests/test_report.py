"""Tests for the Sprint-3 Phase-5 results consolidation + reproducibility gate.

The heavy real-data consolidation lives in `python -m winprob.report` (and
`./report.sh`); these tests pin the contract on small synthetic artifact dicts so
they stay fast and deterministic. Phase 5's final gate — "all published results
reproduce from pinned corpus, feature, quality, split, and model manifests" — is
a HASH CHAIN, not prose: every published number is a pure function of one
immutable provenance tuple, and if any downstream artifact was computed against a
different mart, split, or model, its hash breaks the chain. These tests lock that:

1. A self-consistent set of manifests/audits reproduces (`all_results_reproduce`).
2. Tampering ANY single link — a downstream parquet hash, the split hash, the
   model feature-schema hash, a mart quality gate, or an on-disk file hash — flips
   the gate to False and pinpoints the broken check.
3. Consolidation pulls the headline held-out numbers straight from the pinned
   metrics, and missing artifacts fail fast with an actionable error.
"""

from __future__ import annotations

import json

import pytest

from winprob import report

# Fixed fake hashes; every artifact in a self-consistent bundle shares this tuple.
PARQ = "685233a9deadbeef"
SPLIT = "eb69be5dcafef00d"
FSCHEMA = "80d9e8f0feedface"
MODELHASH = "30a4972b0badc0de"
MODELJSON = "09d72fde1337d00d"
CORPUS = "f3494b21c0ffee00"
RAPM = "600324f6ba5eba11"


def _artifacts() -> dict[str, dict]:
    """A fully self-consistent artifact bundle that must reproduce."""
    return {
        "mart_manifest": {
            "source_possession_corpus_hash": CORPUS,
            "rapm_model_hash": RAPM,
            "split_hash": SPLIT,
            "game_count": 6430,
            "row_count": 1273794,
        },
        "mart_audit": {
            "parquet_sha256": PARQ,
            "gates": {
                "clock_fields_consistent": {"failures": 0, "passed": True},
                "unique_possession_boundary": {"failures": 0, "passed": True},
            },
        },
        "model_manifest": {
            "feature_schema_hash": FSCHEMA,
            "model_hash": MODELHASH,
            "dataset_sha256": PARQ,
            "split_hash": SPLIT,
        },
        "model_metrics": {
            "model": {"brier": 0.16372, "log_loss": 0.48426},
            "calibration": {"intercept": -0.0726, "slope": 1.0313},
            "n_test_rows": 251140,
            "n_test_games": 1258,
            "home_win_rate_test": 0.554,
            "baselines": {
                "base_rate": {"brier": 0.24706, "log_loss": 0.68725},
                "score_time": {"brier": 0.16388, "log_loss": 0.48459},
                "score_time_possession": {"brier": 0.16367, "log_loss": 0.48403},
            },
        },
        "model_audit": {
            "dataset_parquet_sha256": PARQ,
            "model_dataset_sha256": PARQ,
            "model_feature_schema_hash": FSCHEMA,
            "model_json_sha256": MODELJSON,
            "all_gates_pass": True,
        },
        "ablation_metrics": {
            "tiers": {
                t: {"brier": b, "log_loss": 0.47, "features": ["x"]}
                for t, b in zip("ABCDE", [0.16388, 0.16367, 0.15865, 0.15861, 0.15619])
            },
            "paired_diff": {"D_minus_C": {"brier": {"lo": -1, "hi": 1, "point": 0}}},
            "rolling": [{"fold": "A", "test_seasons": [2024],
                         "d_minus_c_brier_point": -0.00062}],
            "verdict": {"lineup_strength_helped": False},
        },
        "ablation_audit": {
            "dataset_parquet_sha256": PARQ,
            "split_hash": SPLIT,
            "structural_gates_pass": True,
        },
        "challenger_metrics": {
            "gbm": {"brier": 0.15713, "log_loss": 0.46743,
                    "config": {"learning_rate": 0.05, "max_depth": 2, "n_trees": 152}},
            "logistic": {"brier": 0.15619, "log_loss": 0.46574},
            "paired_diff": {"gbm_minus_logistic": {"brier": {"lo": -1, "hi": 1, "point": 0}}},
            "verdict": {"adopt_gbm": False, "retain_logistic": True},
        },
        "challenger_audit": {
            "dataset_parquet_sha256": PARQ,
            "split_hash": SPLIT,
            "structural_gates_pass": True,
        },
    }


# --------------------------------------------------------------------------
# 1. A self-consistent bundle reproduces.
# --------------------------------------------------------------------------

def test_self_consistent_bundle_reproduces():
    result = report.verify_reproducibility(
        _artifacts(), live_parquet_sha256=PARQ, live_model_sha256=MODELJSON
    )
    assert result["all_results_reproduce"] is True
    assert all(result["checks"].values())


def test_pinned_tuple_is_reported():
    result = report.verify_reproducibility(_artifacts())
    pinned = result["pinned"]
    assert pinned["parquet_sha256"] == PARQ
    assert pinned["split_hash"] == SPLIT
    assert pinned["corpus_hash"] == CORPUS
    assert pinned["model_hash"] == MODELHASH


# --------------------------------------------------------------------------
# 2. Any single broken link flips the gate and pinpoints the check.
# --------------------------------------------------------------------------

def test_tampered_downstream_parquet_hash_breaks_chain():
    arts = _artifacts()
    arts["challenger_audit"]["dataset_parquet_sha256"] = "0000deadbeef0000"
    result = report.verify_reproducibility(arts)
    assert result["checks"]["parquet_hash_chain_consistent"] is False
    assert result["all_results_reproduce"] is False


def test_tampered_split_hash_breaks_chain():
    arts = _artifacts()
    arts["ablation_audit"]["split_hash"] = "1111111111111111"
    result = report.verify_reproducibility(arts)
    assert result["checks"]["split_hash_consistent"] is False
    assert result["all_results_reproduce"] is False


def test_mismatched_model_feature_schema_breaks_chain():
    arts = _artifacts()
    arts["model_audit"]["model_feature_schema_hash"] = "2222222222222222"
    result = report.verify_reproducibility(arts)
    assert result["checks"]["model_feature_schema_consistent"] is False
    assert result["all_results_reproduce"] is False


def test_failing_mart_quality_gate_breaks_reproducibility():
    arts = _artifacts()
    arts["mart_audit"]["gates"]["unique_possession_boundary"] = {
        "failures": 7, "passed": False
    }
    result = report.verify_reproducibility(arts)
    assert result["checks"]["mart_quality_gates_pass"] is False
    assert result["all_results_reproduce"] is False


def test_model_gate_failure_breaks_reproducibility():
    arts = _artifacts()
    arts["model_audit"]["all_gates_pass"] = False
    result = report.verify_reproducibility(arts)
    assert result["checks"]["model_gates_pass"] is False
    assert result["all_results_reproduce"] is False


def test_on_disk_parquet_mismatch_breaks_reproducibility():
    # The strongest check: the mart file on disk must hash to the pinned value.
    result = report.verify_reproducibility(
        _artifacts(), live_parquet_sha256="mismatch", live_model_sha256=MODELJSON
    )
    assert result["checks"]["mart_parquet_on_disk_matches"] is False
    assert result["all_results_reproduce"] is False


def test_on_disk_model_mismatch_breaks_reproducibility():
    result = report.verify_reproducibility(
        _artifacts(), live_parquet_sha256=PARQ, live_model_sha256="mismatch"
    )
    assert result["checks"]["model_json_on_disk_matches"] is False
    assert result["all_results_reproduce"] is False


def test_disk_checks_absent_when_live_hashes_not_supplied():
    # Pure-chain verification without disk access omits the on-disk checks.
    result = report.verify_reproducibility(_artifacts())
    assert "mart_parquet_on_disk_matches" not in result["checks"]
    assert "model_json_on_disk_matches" not in result["checks"]


# --------------------------------------------------------------------------
# 3. Consolidation + boundary validation.
# --------------------------------------------------------------------------

def test_consolidate_pulls_headline_numbers():
    results = report.consolidate_results(_artifacts())
    assert results["held_out_2025"]["model"]["brier"] == pytest.approx(0.16372)
    assert results["rapm_ablation"]["tiers"]["E"]["brier"] == pytest.approx(0.15619)
    assert results["challenger"]["verdict"]["retain_logistic"] is True
    # The best out-of-sample logistic is tier E (adds team strength).
    assert results["best_oos_logistic"]["brier"] == pytest.approx(0.15619)


def test_load_artifacts_missing_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        report.load_artifacts(tmp_path)
    assert "report" in str(exc.value).lower() or "run" in str(exc.value).lower()


def test_build_report_writes_results_json(tmp_path):
    # Write a self-consistent bundle to disk (plus stub parquet/model), build,
    # and confirm results.json carries the reproducibility verdict.
    arts = _artifacts()
    files = {
        "manifest.json": arts["mart_manifest"],
        "audit.json": arts["mart_audit"],
        "winprob_model_manifest.json": arts["model_manifest"],
        "winprob_metrics.json": arts["model_metrics"],
        "winprob_audit.json": arts["model_audit"],
        "ablation_metrics.json": arts["ablation_metrics"],
        "ablation_audit.json": arts["ablation_audit"],
        "challenger_metrics.json": arts["challenger_metrics"],
        "challenger_audit.json": arts["challenger_audit"],
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload))
    # Stub the hashed on-disk files so the live-hash checks have something to read.
    (tmp_path / "fct_game_states.parquet").write_bytes(b"stub")
    (tmp_path / "logistic_model.json").write_text(json.dumps(arts["model_manifest"]))

    reportdoc = report.run(tmp_path)  # run() builds AND writes results.json
    assert "reproducibility" in reportdoc
    assert "results" in reportdoc
    # The stub file hashes won't match the fake pinned hashes -> gate is False,
    # which is the CORRECT behavior (the pinned hashes describe the real mart).
    assert reportdoc["reproducibility"]["all_results_reproduce"] is False
    written = json.loads((tmp_path / "results.json").read_text())
    assert written == reportdoc
