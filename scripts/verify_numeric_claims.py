"""Recalculate the 27 numeric claims registered in the evidence package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_manuscript_tables import build_table_three, deepspectra_claims
from src.metrics import relative_improvement
from src.reproduction import manuscript_table, read_prediction, recalculate_runs


def mean_rmsep(runs, model, input_representation=None):
    subset = runs[runs.model == model]
    if input_representation is not None:
        subset = subset[subset.input_representation == input_representation]
    return float(subset.rmsep.mean())


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    swri_predictions = pd.concat([
        read_prediction("swri_seven_property_oof_predictions.csv"),
        read_prediction("swri_cn_initial_oof_predictions.csv"),
    ], ignore_index=True)
    table3 = build_table_three(swri_predictions)
    cn = table3[(table3.property == "CN") & (table3.split_group == "repeat23")].iloc[0]
    claims = {
        "N01": float(cn.target_isolated_masked_autoencoder_rmsep),
        "N02": float(cn.improvement_over_best_classical_pct),
        "N03": float(cn.pretraining_benefit_pct),
    }
    haffner_runs = recalculate_runs(read_prediction("haffner_cn_oof_predictions.csv"))
    target = mean_rmsep(haffner_runs, "target_isolated_masked_autoencoder")
    random = mean_rmsep(haffner_runs, "matched_random_initialization")
    reduced = mean_rmsep(haffner_runs, "pls_reduced_wavelength")
    claims.update({"N04": target, "N05": relative_improvement(random, target), "N06": relative_improvement(reduced, target)})
    confirmation = table3[table3.split_group == "repeat23"]
    for abbreviation in ("CN", "FLASH", "VISC", "D4052", "TOTAL", "BP50", "FREEZE"):
        claims[f"N07_{abbreviation}"] = float(confirmation.loc[confirmation.property == abbreviation, "pretraining_benefit_pct"].iloc[0])
    claims["N08"] = float(confirmation.clear_advantage_pretraining.sum())
    deep = deepspectra_claims(swri_predictions)
    for index, prefix in ((0, "N09"), (1, "N11")):
        claims[prefix] = float(deep[index]["mean_rmsep_advantages"])
        claims[f"N{int(prefix[1:]) + 1}"] = float(deep[index]["clear_advantages"])

    cn_runs = recalculate_runs(read_prediction("cn_representation_oof_predictions.csv"))
    raw = mean_rmsep(cn_runs, "target_isolated_masked_autoencoder", "raw_snv")
    sg1 = mean_rmsep(cn_runs, "target_isolated_masked_autoencoder", "sg1_snv")
    claims["N13"] = relative_improvement(raw, sg1)
    inner = manuscript_table("pretraining_inner_fold_rmse.csv").set_index("arm").mean_inner_fold_rmse
    for seed in (0, 1):
        claims[f"N14_seed{seed}"] = 100 * (inner[f"sg1_s{seed}"] / inner["xdi_existing"] - 1)

    gasoline_runs = recalculate_runs(read_prediction("gasoline_raw_sensitivity_oof_predictions.csv"))
    sg1_candidate = mean_rmsep(gasoline_runs, "target_isolated_masked_autoencoder", "sg1_snv")
    sg1_random = mean_rmsep(gasoline_runs, "matched_random_initialization", "sg1_snv")
    raw_candidate = mean_rmsep(gasoline_runs, "target_isolated_masked_autoencoder", "raw_snv")
    raw_random = mean_rmsep(gasoline_runs, "matched_random_initialization", "raw_snv")
    claims["N15"] = relative_improvement(sg1_random, sg1_candidate)
    claims["N16"] = relative_improvement(raw_random, raw_candidate)
    claims["N17"] = 100 * (raw_candidate - sg1_candidate) / sg1_candidate

    wavelength_runs = recalculate_runs(read_prediction("haffner_wavelength_oof_predictions.csv"))
    full = mean_rmsep(wavelength_runs, "target_isolated_masked_autoencoder", "sg1_snv_full_wavelength")
    common = mean_rmsep(wavelength_runs, "target_isolated_masked_autoencoder", "sg1_snv_common_wavelength")
    common_pls = mean_rmsep(wavelength_runs, "pls_common_wavelength", "sg1_snv_common_wavelength")
    claims["N18"] = full
    claims["N19"] = relative_improvement(common, full)
    claims["N20"] = relative_improvement(common_pls, common)

    registered = manuscript_table("numeric_claims.csv")
    rows = []
    for row in registered.itertuples(index=False):
        recomputed = claims.get(row.claim_id)
        difference = abs(row.value - recomputed) if recomputed is not None else None
        rows.append({
            "claim_id": row.claim_id,
            "registered_value": row.value,
            "recomputed_value": recomputed,
            "absolute_difference": difference,
            "unit": row.unit,
            "passed": recomputed is not None and difference <= 1e-7 * max(1.0, abs(row.value)),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "numeric_claims_audit.csv", index=False)
    missing = sorted(set(claims) - set(registered.claim_id))
    summary = {
        "passed": bool(audit.passed.all()) and not missing,
        "registered_claims": len(registered),
        "recomputed_claims": len(claims),
        "unregistered_recomputed_claims": missing,
        "failed_claims": audit.loc[~audit.passed, "claim_id"].tolist(),
        "maximum_absolute_difference": float(audit.absolute_difference.max()),
    }
    (output / "numeric_claims_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/claims")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
