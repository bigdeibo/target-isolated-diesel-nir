"""Recalculate every stored OOF metric from per-sample predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproduction import compare_with_stored

PAIRS = [
    ("swri_seven_property_oof_predictions.csv", "swri_stored_metrics_by_run.csv"),
    ("haffner_cn_oof_predictions.csv", "haffner_stored_metrics_by_run.csv"),
    ("gasoline_octane_oof_predictions.csv", "gasoline_stored_metrics_by_run.csv"),
    ("gasoline_raw_sensitivity_oof_predictions.csv", "gasoline_raw_sensitivity_stored_metrics_by_run.csv"),
    ("cn_representation_oof_predictions.csv", "cn_representation_stored_metrics_by_run.csv"),
    ("haffner_wavelength_oof_predictions.csv", "haffner_wavelength_stored_metrics_by_run.csv"),
]


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    audits = []
    run_counts = {}
    for prediction_name, stored_name in PAIRS:
        predictions = ROOT.joinpath("results/predictions", prediction_name)
        stored = ROOT.joinpath("results/metrics", stored_name)
        import pandas as pd
        audit, computed = compare_with_stored(pd.read_csv(predictions), pd.read_csv(stored))
        audit.insert(0, "prediction_file", prediction_name)
        audits.append(audit)
        run_counts[prediction_name] = len(computed)
    import pandas as pd
    combined = pd.concat(audits, ignore_index=True)
    combined.to_csv(output / "metric_recalculation_audit.csv", index=False)
    summary = {
        "passed": bool(combined.passed.all()),
        "oof_runs_by_file": run_counts,
        "oof_runs_total": sum(run_counts.values()),
        "metric_comparisons": len(combined),
        "failed_comparisons": int((~combined.passed).sum()),
        "maximum_absolute_difference": float(combined.absolute_difference.max()),
        "tolerance": "1e-8 * max(1, abs(stored value))",
    }
    (output / "metric_recalculation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/metrics")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
