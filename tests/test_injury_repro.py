"""Reproducibility gate for the published Sprint-5 injury numbers.

Reads the REAL on-disk artifacts (`data/winprob/injury_metrics.json` +
`injury_audit.json`) and proves the published numbers reproduce from them: the
audit's `metrics_hash` must equal the canonical hash recomputed from the metrics
JSON on disk, every figure the README quotes must trace back to the metrics, and
the audit must pin the mart, the RAPM ratings, and the availability artifact by
sha256 plus the split hash.

The artifacts are gitignored data, absent on a fresh clone, so the test SKIPS when
missing rather than failing — a gate on the published corpus, not a data step. Run
`./injury.sh` first to materialize them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from winprob import design, pregame_injury

DATA_DIR = Path("data/winprob")
METRICS_PATH = DATA_DIR / pregame_injury.METRICS_JSON_NAME
AUDIT_PATH = DATA_DIR / pregame_injury.AUDIT_JSON_NAME

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing {path}; run ./injury.sh to materialize the artifacts")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def metrics() -> dict:
    return _load(METRICS_PATH)


@pytest.fixture(scope="module")
def audit() -> dict:
    return _load(AUDIT_PATH)


def test_audit_metrics_hash_reproduces_from_on_disk_metrics(metrics, audit):
    # The strongest check: the audit's pinned hash equals the canonical hash
    # recomputed from the metrics JSON as it sits on disk. Any drift flips it.
    assert audit["metrics_hash"] == design.canonical_hash(metrics)


def test_audit_tier_brier_matches_the_metrics_tiers(metrics, audit):
    for tier, brier in audit["tier_brier"].items():
        assert brier == metrics["tiers"][tier]["brier"]


def test_audit_headline_fields_match_the_metrics(metrics, audit):
    assert audit["fraction_of_gap_closed"] == metrics["gap_close"]["fraction_of_gap_closed"]
    assert audit["extra_gap_closed"] == metrics["extra_gap_closed"]
    assert audit["chosen_k"] == metrics["chosen_k"]
    assert audit["n_test_games"] == metrics["n_test_games"]
    assert audit["gates"] == metrics["gates"]
    assert audit["verdict"] == metrics["verdict"]
    assert audit["structural_gates_pass"] == metrics["structural_gates_pass"]


def test_audit_pins_mart_ratings_and_availability_by_sha256_and_the_split(audit):
    for key in ("dataset_parquet_sha256", "bayes_ratings_sha256", "game_availability_sha256"):
        assert _SHA256.match(audit[key]), key
    assert audit["split_hash"] == design.canonical_hash(design.SPLIT_DEFINITION)
    # When the pinned inputs are present, their live hashes must still match.
    parquet = DATA_DIR / pregame_injury.PARQUET_NAME
    if parquet.exists():
        assert audit["dataset_parquet_sha256"] == design.file_hash(parquet)
    availability = DATA_DIR / pregame_injury.AVAILABILITY_PARQUET_NAME
    if availability.exists():
        assert audit["game_availability_sha256"] == design.file_hash(availability)
    ratings = DATA_DIR.parent / pregame_injury.RATINGS_SUBPATH
    if ratings.exists():
        assert audit["bayes_ratings_sha256"] == design.file_hash(ratings)


def test_readme_ladder_numbers_match_metrics(metrics):
    # The P3->P4 numbers the README's Sprint-5 section publishes.
    assert metrics["tiers"]["P3"]["brier"] == pytest.approx(0.20981, abs=5e-5)
    assert metrics["tiers"]["P4"]["brier"] == pytest.approx(0.20930, abs=5e-5)
    diff = metrics["paired_diff"]["P4_minus_P3"]["brier"]
    assert diff["point"] == pytest.approx(-0.00051, abs=5e-5)
    assert diff["hi"] > 0.0  # CI straddles zero -> the honest null


def test_readme_covered_game_numbers_match_metrics(metrics):
    p4 = metrics["gap_close"]
    assert p4["p4_brier"] == pytest.approx(0.22218, abs=5e-5)
    assert p4["market_brier"] == pytest.approx(0.21466, abs=5e-5)
    assert p4["fraction_of_gap_closed"] == pytest.approx(0.507, abs=5e-4)
    assert metrics["form_gap_close"]["p3_brier"] == pytest.approx(0.22234, abs=5e-5)
    assert metrics["verdict"]["form_fraction_of_gap_closed"] == pytest.approx(0.496, abs=5e-4)


def test_readme_p4_pregame_calibration_matches_metrics(metrics):
    cal = metrics["tiers"]["P4"]["calibration"]
    assert cal["intercept"] == pytest.approx(0.0056, abs=5e-4)
    assert cal["slope"] == pytest.approx(0.9692, abs=5e-4)


def test_gate_availability_beats_form_is_the_honest_null(metrics):
    # The scientific finding: availability does NOT beat form, so the gate is False
    # while the structural gates pass and calibration holds.
    assert metrics["gates"]["gate_availability_beats_form"] is False
    assert metrics["gates"]["gate_pregame_calibrated"] is True
    assert metrics["structural_gates_pass"] is True
