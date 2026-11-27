"""Run repeated nested evaluation for the target-isolated ResNet estimator."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROPERTY_CODES, load_gasoline, load_haffner, load_swri, make_folds, sg1_snv
from src.metrics import rmsep
from src.training import train_resnet

PROTOCOLS = {
    "swri_diesel": {"outer_seed": 20261105, "inner_seed": 20263000, "weight": "xdi"},
    "haffner_diesel": {"outer_seed": 20261104, "inner_seed": 20262026, "weight": "xdi"},
    "kalivas_gasoline": {"outer_seed": 20261116, "inner_seed": 20263116, "weight": "xgas"},
}
WEIGHT_HASH = {
    "xdi": "669410e855e7700e6794cd0c4414aadca59b9b6e4c2d893c2e4ace06013ba278",
    "xgas": "1f6796283054e035c65642348897578dbd5b60a79c7adb8cf503720fb902fffb",
}


def load_encoder(weight):
    path = ROOT / "weights" / f"{weight}_encoder.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHT_HASH[weight]:
        raise ValueError(f"Unexpected {weight} checkpoint checksum")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 60 or checkpoint["arch"] != "resnet1" or checkpoint["emb_dim"] != 128 or checkpoint["in_len"] != 512:
        raise ValueError("Checkpoint metadata do not match the frozen encoder")
    return checkpoint["enc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(PROTOCOLS), required=True)
    parser.add_argument("--property", choices=list(PROPERTY_CODES))
    parser.add_argument("--evaluation-group", choices=["initial", "confirmation"], default="initial")
    parser.add_argument("--initializations", nargs="+", choices=["target_isolated", "random"], default=["target_isolated", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "swri_diesel" and not args.property:
        parser.error("--property is required for SWRI")
    if args.dataset != "swri_diesel" and args.evaluation_group == "confirmation":
        parser.error("The independent diesel and gasoline protocols use the initial evaluation group")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == "swri_diesel":
        X_raw, y, ids, hashes = load_swri(args.property)
        property_name = args.property
    elif args.dataset == "haffner_diesel":
        X_raw, y, ids, hashes, _ = load_haffner(common_wavelength=True)
        property_name = "cetane_number"
    else:
        X_raw, y, ids, hashes = load_gasoline()
        property_name = "octane_number"
    X = sg1_snv(X_raw)
    protocol = PROTOCOLS[args.dataset]
    if args.dataset == "swri_diesel":
        repetitions = [0, 1] if args.evaluation_group == "initial" else [2, 3]
    elif args.dataset == "haffner_diesel":
        repetitions = [0, 1]
    else:
        repetitions = [0, 1, 2, 3, 4]
    model_names = {
        "target_isolated": "target_isolated_masked_autoencoder",
        "random": "matched_random_initialization",
    }
    encoder_states = {
        "target_isolated": load_encoder(protocol["weight"]),
        "random": None,
    }
    outer_by_repeat = {
        repeat: make_folds(y, 5, protocol["outer_seed"] + 1009 * repeat)
        for repeat in repetitions
    }
    split_rows = []
    inner_split_rows = []
    for repeat, folds in outer_by_repeat.items():
        for fold, (train, test) in enumerate(folds):
            if set(hashes[train]) & set(hashes[test]):
                raise AssertionError("A spectral hash occurs in both outer roles")
            for role, indices in (("train", train), ("test", test)):
                for index in indices:
                    split_rows.append({
                        "property": property_name,
                        "split_group": args.evaluation_group,
                        "cv_repetition": repeat + 1,
                        "outer_fold": fold + 1,
                        "sample_index": int(index),
                        "sample_id": str(ids[index]),
                        "role": role,
                        "spectrum_sha256": hashes[index],
                        "y_true": y[index],
                    })
            inner = make_folds(y[train], 3, protocol["inner_seed"] + 1009 * repeat + 17 * fold)
            for inner_fold, (inner_train, inner_validation) in enumerate(inner):
                for role, local in (("train", inner_train), ("validation", inner_validation)):
                    for index in train[local]:
                        inner_split_rows.append({
                            "property": property_name,
                            "split_group": args.evaluation_group,
                            "cv_repetition": repeat + 1,
                            "outer_fold": fold + 1,
                            "inner_fold": inner_fold + 1,
                            "sample_index": int(index),
                            "role": role,
                        })
    pd.DataFrame(split_rows).to_csv(args.output / "outer_splits.csv", index=False)
    pd.DataFrame(inner_split_rows).to_csv(args.output / "inner_splits.csv", index=False)

    member_rows = []
    single_rows = []
    ensemble_rows = []
    model_rows = []
    for repeat in repetitions:
        for fold, (train, test) in enumerate(outer_by_repeat[repeat]):
            inner = make_folds(y[train], 3, protocol["inner_seed"] + 1009 * repeat + 17 * fold)
            for initialization in args.initializations:
                model_name = model_names[initialization]
                for seed in args.seeds:
                    seed_members = []
                    for inner_fold, (inner_train, inner_validation) in enumerate(inner):
                        train_indices = train[inner_train]
                        validation_indices = train[inner_validation]
                        y_mean = float(y[train_indices].mean())
                        y_scale = float(y[train_indices].std())
                        train_target = ((y[train_indices] - y_mean) / y_scale).astype(np.float32).reshape(-1, 1)
                        validation_target = ((y[validation_indices] - y_mean) / y_scale).astype(np.float32).reshape(-1, 1)
                        validation_prediction, test_prediction, info = train_resnet(
                            X[train_indices], train_target,
                            X[validation_indices], validation_target,
                            X[test], seed, args.device,
                            encoder_states[initialization], args.max_epochs, args.patience,
                        )
                        validation_prediction = validation_prediction * y_scale + y_mean
                        test_prediction = test_prediction * y_scale + y_mean
                        model_rows.append({
                            "model": model_name,
                            "cv_repetition": repeat + 1,
                            "outer_fold": fold + 1,
                            "inner_fold": inner_fold + 1,
                            "seed": seed,
                            "selected_epochs": info["best_epoch"] + 1,
                            "inner_validation_rmsep": rmsep(y[validation_indices], validation_prediction),
                        })
                        for index, prediction in zip(test, test_prediction):
                            row = {
                                "dataset": args.dataset,
                                "property": property_name,
                                "split_group": args.evaluation_group,
                                "model": model_name,
                                "initialization": "target_isolated_masked_autoencoder" if initialization == "target_isolated" else "random",
                                "input_representation": "sg1_snv",
                                "cv_repetition": repeat + 1,
                                "outer_fold": fold + 1,
                                "inner_fold": inner_fold + 1,
                                "seed": seed,
                                "sample_index": int(index),
                                "sample_id": str(ids[index]),
                                "y_true": y[index],
                                "y_pred": float(prediction),
                            }
                            member_rows.append(row)
                            seed_members.append(row)
                    members = pd.DataFrame(seed_members)
                    single_rows.extend(members[members.inner_fold == 1].to_dict("records"))
                    ensemble = members.groupby(["sample_index", "sample_id", "y_true"], as_index=False).y_pred.mean()
                    for row in ensemble.itertuples(index=False):
                        ensemble_rows.append({
                            "dataset": args.dataset,
                            "property": property_name,
                            "split_group": args.evaluation_group,
                            "model": model_name,
                            "initialization": initialization,
                            "input_representation": "sg1_snv",
                            "cv_repetition": repeat + 1,
                            "outer_fold": fold + 1,
                            "seed": seed,
                            "sample_index": int(row.sample_index),
                            "sample_id": row.sample_id,
                            "y_true": row.y_true,
                            "y_pred": row.y_pred,
                        })
                    pd.DataFrame(member_rows).to_csv(args.output / "inner_member_predictions.csv", index=False)
                    pd.DataFrame(single_rows).to_csv(args.output / "first_inner_model_predictions.csv", index=False)
                    pd.DataFrame(ensemble_rows).to_csv(args.output / "ensemble_oof_predictions.csv", index=False)
                    pd.DataFrame(model_rows).to_csv(args.output / "inner_model_metrics.csv", index=False)
                    print(f"repeat={repeat + 1} fold={fold + 1} model={model_name} seed={seed} complete", flush=True)

    expected_runs = len(repetitions) * len(args.initializations) * len(args.seeds)
    ensembles = pd.DataFrame(ensemble_rows)
    counts = ensembles.groupby(["model", "cv_repetition", "seed"]).sample_index.nunique()
    summary = {
        "dataset": args.dataset,
        "property": property_name,
        "evaluation_group": args.evaluation_group,
        "samples": len(y),
        "oof_runs": len(counts),
        "expected_oof_runs": expected_runs,
        "complete": len(counts) == expected_runs and bool((counts == len(y)).all()),
        "outer_test_used_for_selection": False,
        "full_outer_pool_refit": False,
        "batch_normalization_recalibration": False,
        "seed_or_inner_member_selection": False,
        "final_seeds": args.seeds,
        "outer_seed_rule": f"{protocol['outer_seed']}+1009*zero_based_repeat",
        "inner_seed_rule": f"{protocol['inner_seed']}+1009*zero_based_repeat+17*zero_based_outer_fold",
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
