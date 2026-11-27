"""Reconstruct the three-inner-fold ensemble and first-inner-model predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproduction import prediction_for_scope, scope_metadata


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for member_path in sorted((ROOT / "results/predictions/inner_members").glob("*_inner_member_predictions.csv")):
        scope = member_path.name.removesuffix("_inner_member_predictions.csv")
        _, _, _, expected_n = scope_metadata(scope)
        members = pd.read_csv(member_path)
        predictions = prediction_for_scope(scope)
        keys = ["model", "cv_repetition", "outer_fold", "seed", "sample_index"]
        member_counts = members.groupby(keys + ["inner_fold"]).size()
        if not (member_counts == 1).all():
            raise AssertionError(f"Duplicate member key in {scope}")
        reconstructed = members.groupby(keys, as_index=False).agg(y_pred_recomputed=("y_pred", "mean"), inner_models=("inner_fold", "nunique"))
        saved = predictions[predictions.model.isin(["target_isolated_masked_autoencoder", "matched_random_initialization"])].rename(columns={"y_pred": "y_pred_saved"})
        merged = saved.merge(reconstructed, on=keys, how="outer", indicator=True, validate="one_to_one")
        ensemble_difference = float(np.max(np.abs(merged.y_pred_saved - merged.y_pred_recomputed)))
        single_path = ROOT / "results/predictions/single_models" / f"{scope}_first_inner_model_predictions.csv"
        singles = pd.read_csv(single_path).rename(columns={"y_pred": "y_pred_saved"})
        first = members[members.inner_fold == 1][keys + ["y_pred"]].rename(columns={"y_pred": "y_pred_recomputed"})
        single_merged = singles.merge(first, on=keys, how="outer", indicator=True, validate="one_to_one")
        single_difference = float(np.max(np.abs(single_merged.y_pred_saved - single_merged.y_pred_recomputed)))
        counts = merged.groupby(["model", "cv_repetition", "seed"]).sample_index.agg(["size", "nunique"])
        checks = {
            "all_three_inner_models": bool((reconstructed.inner_models == 3).all()),
            "ensemble_keys_match": bool((merged._merge == "both").all()),
            "single_keys_match": bool((single_merged._merge == "both").all()),
            "complete_oof_counts": bool((counts == expected_n).all().all()),
            "ensemble_tolerance": ensemble_difference < 1e-8,
            "single_model_tolerance": single_difference < 1e-8,
        }
        rows.append({
            "scope": scope,
            "member_rows": len(members),
            "ensemble_rows": len(merged),
            "ensemble_max_absolute_difference": ensemble_difference,
            "single_model_max_absolute_difference": single_difference,
            "passed": all(checks.values()),
            "checks": json.dumps(checks, sort_keys=True),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "ensemble_reconstruction_audit.csv", index=False)
    summary = {
        "passed": bool(audit.passed.all()),
        "scopes": len(audit),
        "maximum_ensemble_difference": float(audit.ensemble_max_absolute_difference.max()),
        "maximum_single_model_difference": float(audit.single_model_max_absolute_difference.max()),
        "tolerance": 1e-8,
    }
    (output / "ensemble_reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/ensemble")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
