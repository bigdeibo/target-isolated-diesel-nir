"""Shared readers and checks for the frozen manuscript evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
RUN_KEYS = ["dataset", "property", "split_group", "model", "input_representation", "cv_repetition", "seed"]


def read_prediction(name):
    return pd.read_csv(ROOT / "results/predictions" / name)


def read_stored_metric(name):
    return pd.read_csv(ROOT / "results/metrics" / name)


def recalculate_runs(predictions):
    rows = []
    for key, group in predictions.groupby(RUN_KEYS, sort=False):
        if group.sample_index.nunique() != len(group):
            raise AssertionError(f"Duplicate OOF sample in {key}")
        row = dict(zip(RUN_KEYS, key if isinstance(key, tuple) else (key,)))
        row.update(regression_metrics(group.y_true, group.y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_stored(predictions, stored):
    computed = recalculate_runs(predictions)
    merged = computed.merge(stored, on=RUN_KEYS, how="outer", suffixes=("_recomputed", "_stored"), indicator=True, validate="one_to_one")
    merged = merged.rename(columns={"_merge": "merge_status"})
    rows = []
    for row in merged.itertuples(index=False):
        for metric in ("n", "rmsep", "r_squared", "mae", "rpd", "bias"):
            recomputed = getattr(row, f"{metric}_recomputed")
            saved = getattr(row, f"{metric}_stored")
            difference = abs(recomputed - saved) if pd.notna(recomputed) and pd.notna(saved) else np.nan
            rows.append({
                **{key: getattr(row, key) for key in RUN_KEYS},
                "merge_status": row.merge_status,
                "metric": metric,
                "stored": saved,
                "recomputed": recomputed,
                "absolute_difference": difference,
                "passed": row.merge_status == "both" and pd.notna(difference) and difference <= 1e-8 * max(1.0, abs(saved)),
            })
    return pd.DataFrame(rows), computed


def manuscript_table(name):
    return pd.read_csv(ROOT / "results/manuscript_source_data" / name)


def scope_metadata(scope):
    if scope == "swri_cn_confirmation":
        return "swri_diesel", "cetane_number", "confirmation", 381
    if scope == "haffner_cn":
        return "haffner_diesel", "cetane_number", "initial", 179
    if scope == "gasoline_octane":
        return "kalivas_gasoline", "octane_number", "initial", 60
    parts = scope.split("_")
    group = parts[-1]
    property_name = "_".join(parts[1:-1])
    expected = 381 if property_name == "cetane_number" else 395
    return "swri_diesel", property_name, group, expected


def prediction_for_scope(scope):
    dataset, property_name, group, _ = scope_metadata(scope)
    filename = {
        "swri_diesel": "swri_seven_property_oof_predictions.csv",
        "haffner_diesel": "haffner_cn_oof_predictions.csv",
        "kalivas_gasoline": "gasoline_octane_oof_predictions.csv",
    }[dataset]
    frame = read_prediction(filename)
    return frame[(frame.property == property_name) & (frame.split_group == group)].copy()
