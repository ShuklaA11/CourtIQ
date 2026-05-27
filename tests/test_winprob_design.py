import hashlib
import json

from winprob.design import (
    FEATURE_SCHEMA_VERSION,
    canonical_hash,
    coverage_summary,
    split_for_season,
)


def test_forward_chaining_splits_are_explicit():
    assert split_for_season(2021) == "audit_only"
    assert split_for_season(2022) == "train"
    assert split_for_season(2023) == "train"
    assert split_for_season(2024) == "validation"
    assert split_for_season(2025) == "test"


def test_canonical_hash_is_order_independent_for_mapping_keys():
    left = {"version": FEATURE_SCHEMA_VERSION, "splits": {"train": [2022, 2023]}}
    right = {"splits": {"train": [2022, 2023]}, "version": FEATURE_SCHEMA_VERSION}
    assert canonical_hash(left) == canonical_hash(right)
    expected = hashlib.sha256(
        json.dumps(left, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert canonical_hash(left) == expected


def test_coverage_summary_counts_rows_and_replacement_appearances():
    rows = [
        (2022, "full", 8, 0),
        (2022, "partial", 2, 3),
        (2023, "partial", 4, 5),
    ]
    result = coverage_summary(rows)
    assert result["2022"]["rows"] == 10
    assert result["2022"]["full_rows"] == 8
    assert result["2022"]["replacement_player_appearances"] == 3
    assert result["2022"]["full_row_rate"] == 0.8
    assert result["2023"]["full_row_rate"] == 0.0
