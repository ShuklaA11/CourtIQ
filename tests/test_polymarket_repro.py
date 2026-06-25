"""Reproducibility gate for the published Sprint-6 Polymarket numbers.

Unlike the fixture-based unit tests, this reads the REAL on-disk artifacts
(`data/winprob/polymarket_metrics.json` + `polymarket_audit.json`) and proves the
published numbers reproduce from them: the audit's `metrics_hash` must equal the
canonical hash recomputed from the metrics JSON on disk, every headline figure the
README quotes must trace back to the metrics, and the audit must pin the mart and
the Polymarket snapshot by sha256 plus the split hash.

The artifacts are gitignored data, absent on a fresh clone, so the test SKIPS when
they are missing rather than failing — it is a gate on the published corpus, not a
data-generation step. Run `./polymarket.sh` first to materialize them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from winprob import design, polymarket_compare

DATA_DIR = Path("data/winprob")
METRICS_PATH = DATA_DIR / polymarket_compare.METRICS_JSON_NAME
AUDIT_PATH = DATA_DIR / polymarket_compare.AUDIT_JSON_NAME

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing {path}; run ./polymarket.sh to materialize the artifacts")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def metrics() -> dict:
    return _load(METRICS_PATH)


@pytest.fixture(scope="module")
def audit() -> dict:
    return _load(AUDIT_PATH)


def test_audit_metrics_hash_reproduces_from_on_disk_metrics(metrics, audit):
    # The strongest check: the audit's pinned hash must equal the canonical hash
    # recomputed from the metrics JSON as it sits on disk. If any published number
    # drifted, this hash flips.
    assert audit["metrics_hash"] == design.canonical_hash(metrics)


def test_audit_headline_fields_match_the_metrics(metrics, audit):
    comp = metrics["comparison"]
    gap = metrics["gap_close"]
    assert audit["p3_brier"] == comp["model"]["brier"]
    assert audit["market_brier"] == comp["market"]["brier"]
    assert audit["paired_market_minus_model"] == comp["paired_diff"]["market_minus_model"]
    assert audit["fraction_of_gap_closed"] == gap["fraction_of_gap_closed"]
    assert audit["fraction_of_gap_closed_ci"] == gap["fraction_of_gap_closed_ci"]
    assert audit["n_covered"] == metrics["n_covered"]
    assert audit["chosen_k"] == metrics["chosen_k"]
    assert audit["gates"] == metrics["gates"]
    assert audit["verdict"] == metrics["verdict"]
    assert audit["structural_gates_pass"] == metrics["structural_gates_pass"]


def test_audit_pins_mart_and_snapshot_by_sha256_and_the_split(audit):
    assert _SHA256.match(audit["dataset_parquet_sha256"])
    assert _SHA256.match(audit["snapshot_sha256"])
    assert audit["split_hash"] == design.canonical_hash(design.SPLIT_DEFINITION)
    # When the pinned inputs are present, their live hashes must still match.
    parquet = DATA_DIR / polymarket_compare.PARQUET_NAME
    if parquet.exists():
        assert audit["dataset_parquet_sha256"] == design.file_hash(parquet)
    snapshot = DATA_DIR / polymarket_compare.SNAPSHOT_NAME
    if snapshot.exists():
        assert audit["snapshot_sha256"] == design.file_hash(snapshot)


def test_structural_gates_hold_on_the_published_run(metrics):
    # Structural integrity is a published guarantee: predictions in (0, 1), every
    # snapshot price strictly pre-tip and within the 24h window, test-season only.
    for name in polymarket_compare.STRUCTURAL_GATE_NAMES:
        assert metrics["gates"][name] is True
    assert metrics["structural_gates_pass"] is True


def test_coverage_is_at_least_the_reported_floor(metrics):
    # The Sprint-6 claim is ~96% coverage of the 1,258 test games, well above MGM's
    # 769. Guard the floor so a silently degraded pull is caught.
    assert metrics["n_test_games_total"] == 1258
    assert metrics["coverage_fraction"] >= 0.90
    assert metrics["n_covered"] > 769  # strictly better than the MGM sportsbook slice


# The exact numbers the README's Sprint-6 section publishes. Filled in from the
# metrics JSON once ./polymarket.sh has produced the frozen artifact; if a table
# cell drifts from the metrics, this fails and names the offending figure.
def test_readme_numbers_match_metrics(metrics):
    comp = metrics["comparison"]
    gap = metrics["gap_close"]
    assert comp["model"]["brier"] == pytest.approx(_README["p3_brier"], abs=5e-5)
    assert comp["market"]["brier"] == pytest.approx(_README["market_brier"], abs=5e-5)
    assert gap["baseline_model_brier"] == pytest.approx(_README["tier_e_brier"], abs=5e-5)
    assert gap["fraction_of_gap_closed"] == pytest.approx(_README["frac_closed"], abs=5e-4)
    assert comp["model"]["calibration"]["intercept"] == pytest.approx(
        _README["p3_intercept"], abs=5e-4
    )
    assert comp["model"]["calibration"]["slope"] == pytest.approx(
        _README["p3_slope"], abs=5e-4
    )


# The exact figures the README's Sprint-6 section publishes, pinned from the frozen
# polymarket_metrics.json. If a table cell drifts from the metrics, the test fails
# and names the offending figure.
_README = {
    "p3_brier": 0.20987,
    "market_brier": 0.19692,
    "tier_e_brier": 0.22453,
    "frac_closed": 0.5309,
    "p3_intercept": -0.0083,
    "p3_slope": 0.9563,
}
