"""Rebuild manuscript Tables 1–4 from the frozen result data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import relative_improvement
from src.reproduction import manuscript_table, read_prediction, recalculate_runs

PROPERTY_ORDER = ["cetane_number", "flash_point", "kinematic_viscosity", "density_d4052", "total_aromatics", "distillation_50", "freezing_point"]
PROPERTY_LABEL = {
    "cetane_number": "CN",
    "flash_point": "FLASH",
    "kinematic_viscosity": "VISC",
    "density_d4052": "D4052",
    "total_aromatics": "TOTAL",
    "distillation_50": "BP50",
    "freezing_point": "FREEZE",
}
UNITS = {
    "cetane_number": "CN units",
    "flash_point": "degC",
    "kinematic_viscosity": "mm2/s",
    "density_d4052": "g/cm3",
    "total_aromatics": "percent_aromatics",
    "distillation_50": "degC",
    "freezing_point": "degC",
}


def mean_metrics(runs, model):
    subset = runs[runs.model == model]
    if subset.empty:
        raise ValueError(f"No runs for {model}")
    return {
        "rmsep": subset.rmsep.mean(),
        "r_squared": subset.r_squared.mean(),
        "mae": subset.mae.mean(),
        "rpd": subset.rpd.mean(),
        "bias": subset.bias.mean(),
    }


def paired_wins(runs, reference):
    candidate = runs[runs.model == "target_isolated_masked_autoencoder"]
    comparator = runs[runs.model == reference]
    wins = 0
    repeat_better = True
    for row in candidate.itertuples(index=False):
        same_repeat = comparator[comparator.cv_repetition == row.cv_repetition]
        matching = same_repeat[same_repeat.seed == row.seed]
        if matching.empty:
            matching = same_repeat[same_repeat.seed == 0]
        if len(matching) != 1:
            raise AssertionError(f"Cannot pair {reference} with repetition {row.cv_repetition}, seed {row.seed}")
        wins += int(row.rmsep < matching.iloc[0].rmsep)
    for repetition in sorted(candidate.cv_repetition.unique()):
        candidate_mean = candidate[candidate.cv_repetition == repetition].rmsep.mean()
        reference_mean = comparator[comparator.cv_repetition == repetition].rmsep.mean()
        repeat_better = repeat_better and candidate_mean < reference_mean
    return wins, bool(repeat_better)


def repeat_win_count(runs, reference):
    candidate = runs[runs.model == "target_isolated_masked_autoencoder"]
    comparator = runs[runs.model == reference]
    wins = 0
    for repetition in sorted(candidate.cv_repetition.unique()):
        candidate_mean = candidate[candidate.cv_repetition == repetition].rmsep.mean()
        reference_mean = comparator[comparator.cv_repetition == repetition].rmsep.mean()
        wins += int(candidate_mean < reference_mean)
    return wins


def build_table_three(predictions):
    runs = recalculate_runs(predictions[predictions.model != "deepspectra_sg1_snv"])
    rows = []
    for property_name in PROPERTY_ORDER:
        groups = ["initial", "confirmation"]
        for group in groups:
            subset = runs[(runs.property == property_name) & (runs.split_group == group)]
            candidate = mean_metrics(subset, "target_isolated_masked_autoencoder")
            classical_models = ["boss_pls"] if property_name == "cetane_number" else ["pls", "svr"]
            classical = min(classical_models, key=lambda model: mean_metrics(subset, model)["rmsep"])
            classical_rmsep = mean_metrics(subset, classical)["rmsep"]
            random_rmsep = mean_metrics(subset, "matched_random_initialization")["rmsep"]
            deepspectra_rmsep = mean_metrics(subset, "deepspectra_raw")["rmsep"]
            classical_wins, classical_repeats = paired_wins(subset, classical)
            random_wins, random_repeats = paired_wins(subset, "matched_random_initialization")
            rows.append({
                "property": PROPERTY_LABEL[property_name],
                "split_group": "repeat23" if group == "confirmation" else "repeat01",
                "property_name": property_name,
                "unit": UNITS[property_name],
                "evaluation_label": "Confirmation CV (repetitions 3–4)" if group == "confirmation" else "Initial CV (repetitions 1–2)",
                "target_isolated_masked_autoencoder_rmsep": candidate["rmsep"],
                "target_isolated_masked_autoencoder_r_squared": candidate["r_squared"],
                "target_isolated_masked_autoencoder_mae": candidate["mae"],
                "target_isolated_masked_autoencoder_rpd": candidate["rpd"],
                "best_classical_model": classical,
                "best_classical_rmsep": classical_rmsep,
                "matched_random_initialization_rmsep": random_rmsep,
                "deepspectra_raw_rmsep": deepspectra_rmsep,
                "improvement_over_best_classical_pct": relative_improvement(classical_rmsep, candidate["rmsep"]),
                "pretraining_benefit_pct": relative_improvement(random_rmsep, candidate["rmsep"]),
                "improvement_over_deepspectra_raw_pct": relative_improvement(deepspectra_rmsep, candidate["rmsep"]),
                "paired_wins_best_classical": classical_wins,
                "paired_wins_pretraining": random_wins,
                "both_repeats_better_best_classical": classical_repeats,
                "both_repeats_better_pretraining": random_repeats,
                "clear_advantage_best_classical": classical_repeats and classical_wins >= 5,
                "clear_advantage_pretraining": random_repeats and random_wins >= 5,
                "paired_total": len(subset[subset.model == "target_isolated_masked_autoencoder"]),
                "repeat_wins_best_classical": repeat_win_count(subset, classical),
                "repeat_wins_pretraining": repeat_win_count(subset, "matched_random_initialization"),
            })
    return pd.DataFrame(rows)


def build_table_four(swri, haffner):
    rows = []
    specifications = [
        (swri, "swri_diesel", "Confirmation CV (repetitions 3–4)", ["pls", "boss_pls", "deepspectra_raw", "matched_random_initialization", "target_isolated_masked_autoencoder"]),
        (haffner, "haffner_diesel", "Independent-dataset replication with recalibration; Initial CV (repetitions 1–2)", ["pls_full_wavelength", "pls_reduced_wavelength", "deepspectra_full_wavelength", "deepspectra_reduced_wavelength", "ipa_full_wavelength", "ipa_reduced_wavelength", "matched_random_initialization", "target_isolated_masked_autoencoder"]),
    ]
    for predictions, dataset, role, models in specifications:
        runs = recalculate_runs(predictions)
        target_rmsep = mean_metrics(runs, "target_isolated_masked_autoencoder")["rmsep"]
        for model in models:
            subset = runs[runs.model == model]
            rows.append({
                "dataset": dataset,
                "evaluation_role": role,
                "runs": len(subset),
                "rmsep_mean": subset.rmsep.mean(),
                "rmsep_sd": subset.rmsep.std(),
                "r_squared_mean": subset.r_squared.mean(),
                "r_squared_sd": subset.r_squared.std(),
                "mae_mean": subset.mae.mean(),
                "mae_sd": subset.mae.std(),
                "rpd_mean": subset.rpd.mean(),
                "bias_mean": subset.bias.mean(),
                "model": model,
                "improvement_of_target_isolated_masked_autoencoder_over_model_pct": relative_improvement(subset.rmsep.mean(), target_rmsep),
            })
    return pd.DataFrame(rows)


def compare_table(recomputed, expected, label):
    if list(recomputed.columns) != list(expected.columns) or len(recomputed) != len(expected):
        raise AssertionError(f"{label} structure differs from the frozen table")
    for column in recomputed.columns:
        left = recomputed[column].reset_index(drop=True)
        right = expected[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.astype(float), right.astype(float), rtol=1e-9, atol=1e-10, equal_nan=True):
                difference = (left.astype(float) - right.astype(float)).abs()
                raise AssertionError(f"{label} differs in {column}; maximum difference {difference.max()}")
        elif not left.equals(right):
            raise AssertionError(f"{label} differs in {column}")


def deepspectra_claims(predictions):
    runs = recalculate_runs(predictions)
    rows = []
    for input_model in ("deepspectra_raw", "deepspectra_sg1_snv"):
        wins = 0
        clear = 0
        for (property_name, group), subset in runs[runs.property != "cetane_number"].groupby(["property", "split_group"]):
            candidate = mean_metrics(subset, "target_isolated_masked_autoencoder")["rmsep"]
            reference = mean_metrics(subset, input_model)["rmsep"]
            improvement = relative_improvement(reference, candidate)
            paired, repeats = paired_wins(subset, input_model)
            wins += int(improvement > 0)
            clear += int(improvement > 0 and repeats and paired >= 5)
        rows.append({"model": input_model, "mean_rmsep_advantages": wins, "clear_advantages": clear, "comparisons": 12})
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    swri = pd.concat([
        read_prediction("swri_seven_property_oof_predictions.csv"),
        read_prediction("swri_cn_initial_oof_predictions.csv"),
    ], ignore_index=True)
    haffner = read_prediction("haffner_cn_oof_predictions.csv")
    table3 = build_table_three(swri)
    table4 = build_table_four(swri[(swri.property == "cetane_number") & (swri.split_group == "confirmation")], haffner)
    compare_table(table3, manuscript_table("manuscript_table_3_swri_seven_property_performance.csv"), "Table 3")
    compare_table(table4, manuscript_table("manuscript_table_4_cn_haffner_detailed_metrics.csv"), "Table 4")
    table3.to_csv(output / "manuscript_table_3_swri_seven_property_performance.csv", index=False)
    table4.to_csv(output / "manuscript_table_4_cn_haffner_detailed_metrics.csv", index=False)
    for name in ("manuscript_table_1_dataset_roles.csv", "manuscript_table_1_swri_property_summary.csv", "manuscript_table_1_xdi_pretraining_corpus.csv", "manuscript_table_2_model_training_configurations.csv"):
        manuscript_table(name).to_csv(output / name, index=False)

    cn = table3[(table3.property == "CN") & (table3.split_group == "repeat23")].iloc[0]
    haffner_target = table4[(table4.dataset == "haffner_diesel") & (table4.model == "target_isolated_masked_autoencoder")].iloc[0]
    haffner_random = table4[(table4.dataset == "haffner_diesel") & (table4.model == "matched_random_initialization")].iloc[0]
    haffner_pls = table4[(table4.dataset == "haffner_diesel") & (table4.model == "pls_reduced_wavelength")].iloc[0]
    claims = {
        "swri_cn_rmsep": float(cn.target_isolated_masked_autoencoder_rmsep),
        "swri_cn_improvement_over_boss_pct": float(cn.improvement_over_best_classical_pct),
        "swri_cn_pretraining_benefit_pct": float(cn.pretraining_benefit_pct),
        "confirmation_targets_with_clear_pretraining_benefit": int(table3[table3.split_group == "repeat23"].clear_advantage_pretraining.sum()),
        "haffner_rmsep": float(haffner_target.rmsep_mean),
        "haffner_improvement_over_random_pct": float(relative_improvement(haffner_random.rmsep_mean, haffner_target.rmsep_mean)),
        "haffner_improvement_over_reduced_pls_pct": float(relative_improvement(haffner_pls.rmsep_mean, haffner_target.rmsep_mean)),
        "deepspectra_comparisons": deepspectra_claims(swri),
        "table_3_matches_frozen_source": True,
        "table_4_matches_frozen_source": True,
    }
    summary = {"passed": True, "tables_written": 6, "claims": claims}
    (output / "table_reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/tables")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
