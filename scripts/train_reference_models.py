"""Train classical and deep reference models on the manuscript splits."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classical import PREPROCESSING, boss_style_select, fit_pls, fit_svr, pls_predict
from src.data import PROPERTY_CODES, load_gasoline, load_haffner, load_haffner_native, load_swri, make_folds, scale_xy, snv_401
from src.metrics import rmsep
from src.training import predict_network, train_reference_network


def protocol(dataset):
    values = {
        "swri_diesel": (20261105, 20263000, 20264000),
        "haffner_diesel": (20261104, 20262026, 20263030),
        "kalivas_gasoline": (20261116, 20263116, 20264116),
    }
    return values[dataset]


def load_input(dataset, property_name, wavelength):
    if dataset == "swri_diesel":
        X, y, ids, hashes = load_swri(property_name)
    elif dataset == "kalivas_gasoline":
        X, y, ids, hashes = load_gasoline()
    elif wavelength == "common":
        X, y, ids, hashes, _ = load_haffner(common_wavelength=True)
    else:
        full, reduced, y, ids, hashes, _ = load_haffner_native()
        X = full if wavelength == "full" else reduced
    return X, y, ids, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["swri_diesel", "haffner_diesel", "kalivas_gasoline"], required=True)
    parser.add_argument("--property", choices=list(PROPERTY_CODES))
    parser.add_argument("--model", choices=["pls", "svr", "boss_pls", "deepspectra_raw", "deepspectra_sg1_snv", "ipa"], required=True)
    parser.add_argument("--evaluation-group", choices=["initial", "confirmation"], default="initial")
    parser.add_argument("--wavelength", choices=["common", "full", "reduced"], default="common")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "swri_diesel" and not args.property:
        parser.error("--property is required for SWRI")
    if args.model == "boss_pls" and (args.dataset != "swri_diesel" or args.property != "cetane_number"):
        parser.error("BOSS-based PLS is used for SWRI cetane number")
    if args.model == "ipa" and args.dataset != "haffner_diesel":
        parser.error("IPA is used for the independent diesel comparison")
    if args.dataset != "swri_diesel" and args.evaluation_group == "confirmation":
        parser.error("The independent diesel and gasoline protocols use the initial evaluation group")
    if args.dataset != "haffner_diesel" and args.wavelength != "common":
        parser.error("Full and reduced wavelength identifiers apply to the independent diesel data")
    if args.model == "deepspectra_sg1_snv" and args.dataset == "haffner_diesel":
        parser.error("The SG1-SNV DeepSpectra arm is used for SWRI properties")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    property_name = args.property if args.dataset == "swri_diesel" else ("octane_number" if args.dataset == "kalivas_gasoline" else "cetane_number")
    X, y, ids, hashes = load_input(args.dataset, property_name, args.wavelength)
    outer_seed, deep_inner_seed, classical_inner_seed = protocol(args.dataset)
    repetitions = ([0, 1] if args.evaluation_group == "initial" else [2, 3]) if args.dataset == "swri_diesel" else ([0, 1] if args.dataset == "haffner_diesel" else [0, 1, 2, 3, 4])
    if args.model == "deepspectra_sg1_snv":
        X = snv_401(X)
    if args.model == "boss_pls":
        X = savgol_filter(X, 11, 2, deriv=1, axis=1)

    model_name = args.model
    if args.dataset == "haffner_diesel" and args.wavelength in ("full", "reduced"):
        model_name = f"{args.model}_{args.wavelength}_wavelength"
    prediction_rows = []
    selection_rows = []
    started = time.time()
    for repeat in repetitions:
        outer = make_folds(y, 5, outer_seed + 1009 * repeat)
        for fold, (train, test) in enumerate(outer):
            if set(hashes[train]) & set(hashes[test]):
                raise AssertionError("A spectral hash occurs in both outer roles")
            deep_inner = make_folds(y[train], 3, deep_inner_seed + 1009 * repeat + 17 * fold)
            classical_inner = make_folds(y[train], 3, classical_inner_seed + 1009 * repeat + 17 * fold)
            if args.model in ("pls", "svr"):
                candidates = []
                preprocessing_options = ("center", "snv") if args.dataset == "haffner_diesel" else PREPROCESSING
                for preprocessing in preprocessing_options:
                    if args.model == "pls":
                        settings = [(preprocessing, components) for components in range(1, 16)]
                    elif args.dataset == "haffner_diesel":
                        settings = [("center", C, epsilon) for C in (1, 10, 100) for epsilon in (0.05, 0.1, 0.2)]
                        if preprocessing == "snv":
                            continue
                    else:
                        settings = [(preprocessing, C, epsilon) for C in (1, 10, 100) for epsilon in (0.05, 0.1, 0.2)]
                    for setting in settings:
                        errors = []
                        for inner_train, inner_validation in classical_inner:
                            if args.model == "pls":
                                prediction = fit_pls(X[train][inner_train], y[train][inner_train], X[train][inner_validation], *setting)
                            else:
                                prediction = fit_svr(X[train][inner_train], y[train][inner_train], X[train][inner_validation], *setting)
                            errors.append(rmsep(y[train][inner_validation], prediction))
                        candidates.append((float(np.mean(errors)), setting))
                candidates.sort(key=lambda item: item[0])
                inner_rmsep, setting = candidates[0]
                prediction = fit_pls(X[train], y[train], X[test], *setting) if args.model == "pls" else fit_svr(X[train], y[train], X[test], *setting)
                selection_rows.append({"cv_repetition": repeat + 1, "outer_fold": fold + 1, "seed": 0, "setting": json.dumps(setting), "inner_cv_rmsep": inner_rmsep})
                seeds = [0]
                predictions = [(0, prediction)]
            elif args.model == "boss_pls":
                predictions = []
                for seed in args.seeds:
                    selected, _ = boss_style_select(
                        X[train], y[train], 20267000 + 100000 * seed + 1009 * repeat + 17 * fold,
                        classical_inner, n_iter=20,
                    )
                    prediction = pls_predict(X[train], y[train], X[test], selected["variables"], selected["components"])
                    selection_rows.append({"cv_repetition": repeat + 1, "outer_fold": fold + 1, "seed": seed, "n_variables": len(selected["variables"]), "components": selected["components"], "inner_cv_rmsep": selected["cv_rmsep"]})
                    predictions.append((seed, prediction))
            else:
                selected_epochs = []
                for inner_fold, (inner_train, inner_validation) in enumerate(deep_inner):
                    train_indices = train[inner_train]
                    validation_indices = train[inner_validation]
                    A, a, B, y_mean, y_scale = scale_xy(X[train_indices], y[train_indices], X[validation_indices])
                    b = ((y[validation_indices] - y_mean) / y_scale).astype(np.float32).reshape(-1, 1)
                    network_name = "ipa" if args.model == "ipa" else "deepspectra_raw"
                    network, info = train_reference_network(network_name, A, a, B, b, 1, args.max_epochs, args.patience, args.device)
                    validation_prediction = predict_network(network, B, args.device) * y_scale + y_mean
                    selected_epochs.append(info["best_epoch"] + 1)
                    selection_rows.append({"cv_repetition": repeat + 1, "outer_fold": fold + 1, "inner_fold": inner_fold + 1, "seed": 1, "selected_epochs": info["best_epoch"] + 1, "inner_validation_rmsep": rmsep(y[validation_indices], validation_prediction)})
                epochs = int(max(1, round(float(np.median(selected_epochs)))))
                predictions = []
                for seed in args.seeds:
                    A, a, B, y_mean, y_scale = scale_xy(X[train], y[train], X[test])
                    network_name = "ipa" if args.model == "ipa" else "deepspectra_raw"
                    network, _ = train_reference_network(network_name, A, a, None, None, seed, epochs, args.patience, args.device)
                    prediction = predict_network(network, B, args.device) * y_scale + y_mean
                    predictions.append((seed, prediction))
            for seed, prediction in predictions:
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite reference prediction")
                for index, value in zip(test, prediction):
                    prediction_rows.append({
                        "dataset": args.dataset,
                        "property": property_name,
                        "split_group": args.evaluation_group,
                        "model": model_name,
                        "initialization": "not_applicable",
                        "input_representation": args.model,
                        "wavelength_protocol": args.wavelength,
                        "cv_repetition": repeat + 1,
                        "outer_fold": fold + 1,
                        "seed": seed,
                        "sample_index": int(index),
                        "sample_id": str(ids[index]),
                        "y_true": y[index],
                        "y_pred": float(value),
                    })
                pd.DataFrame(prediction_rows).to_csv(args.output / "reference_oof_predictions.csv", index=False)
                pd.DataFrame(selection_rows).to_csv(args.output / "inner_selection_records.csv", index=False)
            print(f"model={model_name} repeat={repeat + 1} fold={fold + 1} complete", flush=True)

    frame = pd.DataFrame(prediction_rows)
    counts = frame.groupby(["cv_repetition", "seed"]).sample_index.nunique()
    expected = len(repetitions) * (1 if args.model in ("pls", "svr") else len(args.seeds))
    summary = {
        "dataset": args.dataset,
        "property": property_name,
        "model": model_name,
        "evaluation_group": args.evaluation_group,
        "wavelength_protocol": args.wavelength,
        "samples": len(y),
        "oof_runs": len(counts),
        "expected_oof_runs": expected,
        "complete": len(counts) == expected and bool((counts == len(y)).all()),
        "outer_test_used_for_selection": False,
        "seconds": time.time() - started,
    }
    (args.output / "reference_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
