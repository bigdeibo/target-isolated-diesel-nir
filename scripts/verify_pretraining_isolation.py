"""Verify target exclusion from spectrum hashes and checkpoint identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED = {
    "xdi": {"target": 784, "corpus": 8555, "weight_sha256": "669410e855e7700e6794cd0c4414aadca59b9b6e4c2d893c2e4ace06013ba278"},
    "xgas": {"target": 60, "corpus": 9279, "weight_sha256": "1f6796283054e035c65642348897578dbd5b60a79c7adb8cf503720fb902fffb"},
}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    hashes = pd.read_csv(ROOT / "data/manifests/pretraining_spectrum_hashes.csv")
    results = {}
    for weight, expected in EXPECTED.items():
        subset = hashes[hashes.weight == weight]
        target = set(subset.loc[subset.role == "excluded_target", "spectrum_sha256"])
        corpus = set(subset.loc[subset.role == "pretraining_corpus", "spectrum_sha256"])
        checkpoint = ROOT / "weights" / f"{weight}_encoder.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checks = {
            "target_count": len(subset[subset.role == "excluded_target"]) == expected["target"],
            "corpus_count": len(subset[subset.role == "pretraining_corpus"]) == expected["corpus"],
            "exact_target_overlap_zero": len(target & corpus) == 0,
            "target_source_excluded_from_corpus_labels": not bool(subset[(subset.role == "pretraining_corpus") & (subset.source == ("swri_diesel" if weight == "xdi" else "kalivas_gasoline"))].shape[0]),
            "checkpoint_sha256_matches": digest == expected["weight_sha256"],
        }
        results[weight] = {
            "passed": all(checks.values()),
            "checks": checks,
            "target_spectra": len(subset[subset.role == "excluded_target"]),
            "retained_corpus_spectra": len(subset[subset.role == "pretraining_corpus"]),
            "exact_target_matches": len(target & corpus),
            "checkpoint_sha256": digest,
        }
    summary = {
        "passed": all(result["passed"] for result in results.values()),
        "hash_representation": "Raw spectrum interpolated by normalized position to 512 float32 values",
        "raw_spectra_required_for_this_check": False,
        "weights": results,
    }
    (output / "pretraining_isolation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/isolation")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
