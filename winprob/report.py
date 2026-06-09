"""Sprint-3 Phase-5 results consolidation + the final reproducibility gate.

Phase 5 does not recompute the sprint's numbers — Phases 2-4 already emitted them
into pinned metrics/audit JSONs. This module CONSOLIDATES those into one
``results.json`` and enforces the final gate the plan names: *all published
results reproduce from pinned corpus, feature, quality, split, and model
manifests.*

That gate is a HASH CHAIN, not a promise. The mart parquet is the anchor; its
file sha256 is recorded by the mart build and must reappear, byte-identical, in
every downstream audit (model, ablation, challenger). The split definition hash,
the model's feature-schema hash, the model artifact hash, the upstream possession
corpus hash, and the mart's quality gates form the rest of the immutable tuple.
Walking the chain and asserting one consistent tuple turns "trust me, it
reproduces" into a single boolean — and, when the real on-disk mart and model
files are available, re-hashing them proves the artifacts every number was
computed against are the exact ones still on disk.

Everything here is pure with respect to already-loaded JSON (so the chain logic is
tested on synthetic bundles); only ``load_artifacts``/``build_report``/``run``
touch the filesystem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from winprob.design import file_hash

DEFAULT_DATA_DIR = Path("data/winprob")
RESULTS_JSON_NAME = "results.json"
PARQUET_NAME = "fct_game_states.parquet"
MODEL_JSON_NAME = "logistic_model.json"

# Logical artifact name -> filename. Each is produced by an earlier phase's run.
ARTIFACT_FILES: dict[str, str] = {
    "mart_manifest": "manifest.json",
    "mart_audit": "audit.json",
    "model_manifest": "winprob_model_manifest.json",
    "model_metrics": "winprob_metrics.json",
    "model_audit": "winprob_audit.json",
    "ablation_metrics": "ablation_metrics.json",
    "ablation_audit": "ablation_audit.json",
    "challenger_metrics": "challenger_metrics.json",
    "challenger_audit": "challenger_audit.json",
}

# Which phase runner regenerates each missing artifact — surfaced in the error.
_REGEN_HINT = {
    "mart_manifest": "./game_states.sh",
    "mart_audit": "./game_states.sh",
    "model_manifest": "./winprob.sh",
    "model_metrics": "./winprob.sh",
    "model_audit": "./winprob.sh",
    "ablation_metrics": "./ablation.sh",
    "ablation_audit": "./ablation.sh",
    "challenger_metrics": "./challenger.sh",
    "challenger_audit": "./challenger.sh",
}


# --------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------

def load_artifacts(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, dict]:
    """Load every pinned phase artifact; fail fast with an actionable hint."""
    data_dir = Path(data_dir)
    artifacts: dict[str, dict] = {}
    for name, filename in ARTIFACT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            hint = _REGEN_HINT.get(name, "the relevant phase runner")
            raise FileNotFoundError(
                f"missing artifact for the Phase-5 report: {path}. "
                f"Run {hint} first to regenerate it."
            )
        artifacts[name] = json.loads(path.read_text())
    return artifacts


# --------------------------------------------------------------------------
# Provenance extraction + the hash-chain gate.
# --------------------------------------------------------------------------

def collect_provenance(artifacts: dict[str, dict]) -> dict[str, str]:
    """The immutable provenance tuple every published number must trace to."""
    return {
        "corpus_hash": artifacts["mart_manifest"]["source_possession_corpus_hash"],
        "rapm_model_hash": artifacts["mart_manifest"]["rapm_model_hash"],
        "parquet_sha256": artifacts["mart_audit"]["parquet_sha256"],
        "split_hash": artifacts["mart_manifest"]["split_hash"],
        "model_feature_schema_hash": artifacts["model_manifest"]["feature_schema_hash"],
        "model_hash": artifacts["model_manifest"]["model_hash"],
        "model_json_sha256": artifacts["model_audit"]["model_json_sha256"],
    }


def _all_equal(values: list[str]) -> bool:
    """True iff the list is non-empty and every element is identical."""
    return len(values) > 0 and all(v == values[0] for v in values)


def verify_reproducibility(
    artifacts: dict[str, dict],
    live_parquet_sha256: str | None = None,
    live_model_sha256: str | None = None,
) -> dict:
    """Walk the provenance chain; return per-check booleans and the overall gate.

    The parquet-hash and split-hash chains must be internally consistent across
    every phase; the model feature-schema hash must match between the model
    manifest and its audit; the mart quality gates and each phase's exit gates
    must all pass. When the real on-disk hashes are supplied, they must equal the
    pinned values — proving the files still on disk are the ones every result was
    computed against. Pure with respect to ``artifacts``.
    """
    anchor_parquet = artifacts["mart_audit"]["parquet_sha256"]
    parquet_refs = [
        anchor_parquet,
        artifacts["model_manifest"]["dataset_sha256"],
        artifacts["model_audit"]["dataset_parquet_sha256"],
        artifacts["model_audit"]["model_dataset_sha256"],
        artifacts["ablation_audit"]["dataset_parquet_sha256"],
        artifacts["challenger_audit"]["dataset_parquet_sha256"],
    ]
    split_refs = [
        artifacts["mart_manifest"]["split_hash"],
        artifacts["model_manifest"]["split_hash"],
        artifacts["ablation_audit"]["split_hash"],
        artifacts["challenger_audit"]["split_hash"],
    ]
    mart_gates = artifacts["mart_audit"]["gates"]

    checks: dict[str, bool] = {
        "parquet_hash_chain_consistent": _all_equal(parquet_refs),
        "split_hash_consistent": _all_equal(split_refs),
        "model_feature_schema_consistent": (
            artifacts["model_manifest"]["feature_schema_hash"]
            == artifacts["model_audit"]["model_feature_schema_hash"]
        ),
        "corpus_and_rapm_pinned": bool(
            artifacts["mart_manifest"].get("source_possession_corpus_hash")
            and artifacts["mart_manifest"].get("rapm_model_hash")
        ),
        "mart_quality_gates_pass": all(
            bool(g.get("passed")) for g in mart_gates.values()
        ),
        "model_gates_pass": bool(artifacts["model_audit"]["all_gates_pass"]),
        "ablation_structural_gates_pass": bool(
            artifacts["ablation_audit"]["structural_gates_pass"]
        ),
        "challenger_structural_gates_pass": bool(
            artifacts["challenger_audit"]["structural_gates_pass"]
        ),
    }

    if live_parquet_sha256 is not None:
        checks["mart_parquet_on_disk_matches"] = (
            live_parquet_sha256 == anchor_parquet
        )
    if live_model_sha256 is not None:
        checks["model_json_on_disk_matches"] = (
            live_model_sha256 == artifacts["model_audit"]["model_json_sha256"]
        )

    return {
        "pinned": collect_provenance(artifacts),
        "checks": checks,
        "all_results_reproduce": all(checks.values()),
    }


# --------------------------------------------------------------------------
# Consolidation of the headline numbers.
# --------------------------------------------------------------------------

def consolidate_results(artifacts: dict[str, dict]) -> dict:
    """Pull the published headline numbers from the pinned metrics JSONs."""
    mm = artifacts["model_metrics"]
    ab = artifacts["ablation_metrics"]
    ch = artifacts["challenger_metrics"]

    tier_e = ab["tiers"]["E"]
    return {
        "held_out_2025": {
            "n_test_rows": mm["n_test_rows"],
            "n_test_games": mm["n_test_games"],
            "home_win_rate": mm["home_win_rate_test"],
            "model": {
                "brier": mm["model"]["brier"],
                "log_loss": mm["model"]["log_loss"],
                "calibration": mm["calibration"],
            },
            "baselines": {
                name: {"brier": b["brier"], "log_loss": b["log_loss"]}
                for name, b in mm["baselines"].items()
            },
        },
        "rapm_ablation": {
            "tiers": {
                t: {
                    "brier": ab["tiers"][t]["brier"],
                    "log_loss": ab["tiers"][t]["log_loss"],
                    "features": ab["tiers"][t]["features"],
                }
                for t in ab["tiers"]
            },
            "paired_diff": ab["paired_diff"],
            "rolling_folds": ab["rolling"],
            "verdict": ab["verdict"],
        },
        "challenger": {
            "gbm": {
                "brier": ch["gbm"]["brier"],
                "log_loss": ch["gbm"]["log_loss"],
                "config": ch["gbm"]["config"],
            },
            "logistic_tier_e": {
                "brier": ch["logistic"]["brier"],
                "log_loss": ch["logistic"]["log_loss"],
            },
            "paired_diff": ch["paired_diff"],
            "verdict": ch["verdict"],
        },
        # The best out-of-sample logistic is the ablation's tier E (adds prior
        # team strength); the serialized Phase-2 model above is the sparse
        # score+time+possession baseline that never sees ratings.
        "best_oos_logistic": {
            "name": "tier_E",
            "brier": tier_e["brier"],
            "log_loss": tier_e["log_loss"],
            "note": "tier-E logistic: score+time+possession+team strength+lineup RAPM+coverage",
        },
    }


# --------------------------------------------------------------------------
# Assembly + serialization.
# --------------------------------------------------------------------------

def build_report(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load artifacts, verify the chain against on-disk hashes, consolidate."""
    data_dir = Path(data_dir)
    artifacts = load_artifacts(data_dir)

    parquet_path = data_dir / PARQUET_NAME
    model_path = data_dir / MODEL_JSON_NAME
    live_parquet = file_hash(parquet_path) if parquet_path.exists() else None
    live_model = file_hash(model_path) if model_path.exists() else None

    reproducibility = verify_reproducibility(artifacts, live_parquet, live_model)
    results = consolidate_results(artifacts)
    return {
        "reproducibility": reproducibility,
        "results": results,
        "generated_from": dict(ARTIFACT_FILES),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Build the consolidated report and write ``results.json``."""
    data_dir = Path(data_dir)
    reportdoc = build_report(data_dir)
    _write_json(data_dir / RESULTS_JSON_NAME, reportdoc)
    return reportdoc


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

def _print_summary(reportdoc: dict) -> None:
    repro = reportdoc["reproducibility"]
    res = reportdoc["results"]
    pinned = repro["pinned"]
    print("winprob Phase-5 consolidated results\n")
    print("  pinned provenance:")
    for name, value in pinned.items():
        print(f"    {name:28} {str(value)[:16]}")

    ho = res["held_out_2025"]
    print(
        f"\n  held-out 2025: {ho['n_test_rows']:,} rows / {ho['n_test_games']:,} games "
        f"(home win rate {ho['home_win_rate']:.3f})"
    )
    print(
        f"    serialized logistic  Brier {ho['model']['brier']:.5f}  "
        f"log_loss {ho['model']['log_loss']:.5f}  "
        f"calib {ho['model']['calibration']['intercept']:+.3f}/"
        f"{ho['model']['calibration']['slope']:.3f}"
    )
    best = res["best_oos_logistic"]
    print(
        f"    best OOS logistic ({best['name']})  Brier {best['brier']:.5f}  "
        f"log_loss {best['log_loss']:.5f}"
    )
    ch = res["challenger"]
    decision = "ADOPT GBM" if ch["verdict"]["adopt_gbm"] else "RETAIN LOGISTIC"
    print(
        f"    challenger  GBM Brier {ch['gbm']['brier']:.5f} vs "
        f"tier-E {ch['logistic_tier_e']['brier']:.5f}  ->  {decision}"
    )

    print("\n  reproducibility checks:")
    for name, passed in repro["checks"].items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    verdict = "REPRODUCE" if repro["all_results_reproduce"] else "DO NOT REPRODUCE"
    print(f"\n  final gate: all published results {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate Phase 2-4 results and enforce the reproducibility gate"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    reportdoc = run(Path(args.data_dir))
    _print_summary(reportdoc)
    if not reportdoc["reproducibility"]["all_results_reproduce"]:
        raise SystemExit("reproducibility gate failure: published results do not reproduce")


if __name__ == "__main__":
    main()
