"""Regenerate nested splits and compare prediction test memberships."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_folds
from src.reproduction import prediction_for_scope, scope_metadata

PROTOCOLS = {
    "swri_diesel": (20261105, 20263000),
    "haffner_diesel": (20261104, 20262026),
    "kalivas_gasoline": (20261116, 20263116),
}


def role_map(frame):
    return dict(zip(frame.sample_index.astype(int), frame.role.replace({"validation": "val"})))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for outer_path in sorted((ROOT / "results/splits").glob("*_outer_splits.csv")):
        scope = outer_path.name.removesuffix("_outer_splits.csv")
        dataset, property_name, group, expected_n = scope_metadata(scope)
        outer_seed, inner_seed = PROTOCOLS[dataset]
        outer = pd.read_csv(outer_path)
        inner = pd.read_csv(ROOT / "results/splits" / f"{scope}_inner_splits.csv")
        labels = outer.drop_duplicates("sample_index").sort_values("sample_index").y_true.to_numpy()
        if len(labels) != expected_n:
            raise AssertionError(f"Unexpected label count in {scope}")
        predictions = prediction_for_scope(scope)
        outer_difference = 0
        inner_difference = 0
        hash_overlap = 0
        prediction_mismatch = 0
        for repetition in sorted(outer.cv_repetition.unique()):
            folds = make_folds(labels, 5, outer_seed + 1009 * (repetition - 1))
            for fold, (train, test) in enumerate(folds):
                expected = {int(index): "train" for index in train} | {int(index): "test" for index in test}
                saved_frame = outer[(outer.cv_repetition == repetition) & (outer.outer_fold == fold + 1)]
                saved = role_map(saved_frame)
                outer_difference += len(set(expected.items()) ^ set(saved.items()))
                hash_overlap += len(set(saved_frame.loc[saved_frame.role == "train", "spectrum_sha256"]) & set(saved_frame.loc[saved_frame.role == "test", "spectrum_sha256"]))
                inner_folds = make_folds(labels[train], 3, inner_seed + 1009 * (repetition - 1) + 17 * fold)
                for inner_fold, (inner_train, inner_validation) in enumerate(inner_folds):
                    expected_inner = {int(train[index]): "train" for index in inner_train} | {int(train[index]): "val" for index in inner_validation}
                    saved_inner_frame = inner[(inner.cv_repetition == repetition) & (inner.outer_fold == fold + 1) & (inner.inner_fold == inner_fold + 1)]
                    inner_difference += len(set(expected_inner.items()) ^ set(role_map(saved_inner_frame).items()))
                expected_test = set(map(int, test))
                prediction_frame = predictions[(predictions.cv_repetition == repetition) & (predictions.outer_fold == fold + 1)]
                for _, group_frame in prediction_frame.groupby(["model", "seed"]):
                    actual = set(group_frame.sample_index.astype(int))
                    prediction_mismatch += len(expected_test ^ actual)
        checks = {
            "outer_split_regeneration": outer_difference == 0,
            "inner_split_regeneration": inner_difference == 0,
            "train_test_hash_disjoint": hash_overlap == 0,
            "prediction_test_membership": prediction_mismatch == 0,
        }
        rows.append({
            "scope": scope,
            "samples": expected_n,
            "outer_symmetric_difference": outer_difference,
            "inner_symmetric_difference": inner_difference,
            "train_test_hash_overlap": hash_overlap,
            "prediction_membership_difference": prediction_mismatch,
            "passed": all(checks.values()),
            "checks": json.dumps(checks, sort_keys=True),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "split_integrity_audit.csv", index=False)
    summary = {
        "passed": bool(audit.passed.all()),
        "scopes": len(audit),
        "outer_fold_regenerations": int(sum(len(pd.read_csv(path).cv_repetition.unique()) * 5 for path in (ROOT / "results/splits").glob("*_outer_splits.csv"))),
        "total_outer_symmetric_difference": int(audit.outer_symmetric_difference.sum()),
        "total_inner_symmetric_difference": int(audit.inner_symmetric_difference.sum()),
        "total_prediction_membership_difference": int(audit.prediction_membership_difference.sum()),
    }
    (output / "split_integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/splits")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
