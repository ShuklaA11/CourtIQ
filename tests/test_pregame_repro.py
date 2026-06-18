"""Reproducibility gate for the published Sprint-4 pre-game numbers.

Unlike the synthetic-data unit tests, this reads the REAL on-disk artifacts
(`data/winprob/pregame_metrics.json` + `pregame_audit.json`) and proves the
published numbers reproduce from them: the audit's `metrics_hash` must equal the
canonical hash recomputed from the metrics JSON on disk (so the audit pins the
exact metrics), every figure the README quotes must trace back to the metrics, and
the audit must pin the mart and the odds file by sha256 plus the split hash.

The artifacts are gitignored data, absent on a fresh clone, so the test SKIPS when
they are missing rather than failing — it is a gate on the published corpus, not a
data-generation step. Run `./pregame.sh` first to materialize them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from winprob import design, pregame

DATA_DIR = Path("data/winprob")
METRICS_PATH = DATA_DIR / pregame.METRICS_JSON_NAME
AUDIT_PATH = DATA_DIR / pregame.AUDIT_JSON_NAME

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing {path}; run ./pregame.sh to materialize the artifacts")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def metrics() -> dict:
    return _load(METRICS_PATH)


@pytest.fixture(scope="module")
def audit() -> dict:
    return _load(AUDIT_PATH)


def test_audit_metrics_hash_reproduces_from_on_disk_metrics(metrics, audit):
    # The single strongest check: the audit's pinned hash must equal the canonical
    # hash recomputed from the metrics JSON as it sits on disk. If any published
    # number drifted, this hash flips.
    assert audit["metrics_hash"] == design.canonical_hash(metrics)


def test_audit_tier_brier_matches_the_metrics_tiers(metrics, audit):
    for tier, brier in audit["tier_brier"].items():
        assert brier == metrics["tiers"][tier]["brier"]


def test_audit_headline_fields_match_the_metrics(metrics, audit):
    gap = metrics["gap_close"]
    assert audit["fraction_of_gap_closed"] == gap["fraction_of_gap_closed"]
    assert audit["fraction_of_gap_closed_ci"] == gap["fraction_of_gap_closed_ci"]
    assert audit["chosen_k"] == metrics["chosen_k"]
    assert audit["n_test_games"] == metrics["n_test_games"]
    assert audit["gates"] == metrics["gates"]
    assert audit["verdict"] == metrics["verdict"]
    assert audit["structural_gates_pass"] == metrics["structural_gates_pass"]


def test_audit_pins_mart_and_odds_by_sha256_and_the_split(audit):
    assert _SHA256.match(audit["dataset_parquet_sha256"])
    assert _SHA256.match(audit["odds_csv_sha256"])
    assert audit["split_hash"] == design.canonical_hash(design.SPLIT_DEFINITION)
    # When the pinned inputs are present, their live hashes must still match.
    parquet = DATA_DIR / pregame.PARQUET_NAME
    if parquet.exists():
        assert audit["dataset_parquet_sha256"] == design.file_hash(parquet)
    odds = DATA_DIR / pregame.ODDS_CSV_NAME
    if odds.exists():
        assert audit["odds_csv_sha256"] == design.file_hash(odds)


# The exact numbers the README's Sprint-4 section publishes. If a table cell drifts
# from the metrics JSON, this fails and names the offending figure.
_README_NUMBERS = {
    "P0_brier": ("tiers", "P0", "brier", 0.24719),
    "P1_brier": ("tiers", "P1", "brier", 0.23296),
    "P2_brier": ("tiers", "P2", "brier", 0.20922),
    "P3_brier": ("tiers", "P3", "brier", 0.20981),
}


def test_readme_ladder_numbers_match_metrics(metrics):
    for _, (section, tier, field, published) in _README_NUMBERS.items():
        assert metrics[section][tier][field] == pytest.approx(published, abs=5e-5)


def test_readme_covered_game_numbers_match_metrics(metrics):
    gap = metrics["gap_close"]
    assert gap["baseline_model_brier"] == pytest.approx(0.22990, abs=5e-5)
    assert gap["p3_brier"] == pytest.approx(0.22234, abs=5e-5)
    assert gap["market_brier"] == pytest.approx(0.21466, abs=5e-5)
    assert gap["fraction_of_gap_closed"] == pytest.approx(0.496, abs=5e-4)
    ci = gap["fraction_of_gap_closed_ci"]
    assert ci["lo"] == pytest.approx(0.0149, abs=5e-4)
    assert ci["hi"] == pytest.approx(0.8619, abs=5e-4)


def test_readme_p3_pregame_calibration_matches_metrics(metrics):
    cal = metrics["tiers"]["P3"]["calibration"]
    assert cal["intercept"] == pytest.approx(-0.0035, abs=5e-4)
    assert cal["slope"] == pytest.approx(0.9568, abs=5e-4)
