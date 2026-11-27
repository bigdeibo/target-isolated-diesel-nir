"""Run the complete result-level reproduction workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import (
    generate_manuscript_tables,
    verify_ensemble_predictions,
    verify_numeric_claims,
    verify_prediction_metrics,
    verify_pretraining_isolation,
    verify_split_integrity,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {
        "metrics": verify_prediction_metrics.run(args.output / "metrics"),
        "splits": verify_split_integrity.run(args.output / "splits"),
        "ensembles": verify_ensemble_predictions.run(args.output / "ensemble"),
        "pretraining_isolation": verify_pretraining_isolation.run(args.output / "isolation"),
        "tables": generate_manuscript_tables.run(args.output / "tables"),
        "numeric_claims": verify_numeric_claims.run(args.output / "claims"),
    }
    summary = {
        "passed": all(item["passed"] for item in summaries.values()),
        "workflow": "Frozen OOF predictions, splits, ensembles, target isolation, manuscript tables, and numeric claims",
        "model_training_performed": False,
        "summaries": summaries,
    }
    (args.output / "reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": summary["passed"],
        "oof_runs": summaries["metrics"]["oof_runs_total"],
        "metric_comparisons": summaries["metrics"]["metric_comparisons"],
        "split_scopes": summaries["splits"]["scopes"],
        "ensemble_scopes": summaries["ensembles"]["scopes"],
        "numeric_claims": summaries["numeric_claims"]["recomputed_claims"],
        "output": str(args.output),
    }, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
