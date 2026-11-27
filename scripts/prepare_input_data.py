"""Convert the released SWRI, Haffner, and gasoline files to model inputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OFFSET = 44.15632183908046
TEST_OFFSET = 41.221333333333334


def clean(value):
    return value.strip().strip('"').strip()


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.reader(handle))


def prepare_swri(data_dir):
    directory = data_dir / "swri_diesel"
    spectra_rows = read_rows(directory / "diesel_spec.csv")
    property_rows = read_rows(directory / "diesel_prop.csv")
    axis = None
    data_start = None
    for index, row in enumerate(spectra_rows):
        if row and clean(row[0]) == "Axisscale":
            axis = np.array([float(value) for value in row[2:] if value.strip()], dtype=float)
            data_start = index + 1
            break
    if axis is None:
        raise ValueError("SWRI wavelength axis was not found")
    ids = []
    spectra = []
    for row in spectra_rows[data_start:]:
        if len(row) < 2 + len(axis) or not clean(row[1]):
            continue
        try:
            values = [float(value) if value.strip() else np.nan for value in row[2:2 + len(axis)]]
        except ValueError:
            continue
        ids.append(clean(row[1]))
        spectra.append(values)
    header_index = next(index for index, row in enumerate(property_rows) if clean(row[0]) == "Label" and any(clean(value) == "CN" for value in row))
    header = [clean(value) for value in property_rows[header_index]]
    names = [value for value in header[2:] if value]
    property_ids = []
    properties = []
    for row in property_rows[header_index + 1:]:
        if len(row) < 3 or not clean(row[1]):
            continue
        values = []
        for value in row[2:2 + len(names)]:
            try:
                values.append(float(value) if value.strip() else np.nan)
            except ValueError:
                values.append(np.nan)
        property_ids.append(clean(row[1]))
        properties.append(values)
    values = np.asarray(properties, dtype=np.float64)
    positions = {sample_id: index for index, sample_id in enumerate(property_ids)}
    Y = np.full((len(ids), len(names)), np.nan)
    for index, sample_id in enumerate(ids):
        if sample_id in positions:
            Y[index] = values[positions[sample_id]]
    X = np.asarray(spectra, dtype=np.float64)
    if X.shape != (784, 401) or Y.shape != (784, 7):
        raise ValueError(f"Unexpected SWRI shapes: {X.shape}, {Y.shape}")
    np.savez_compressed(directory / "swri_cn.npz", X=X, Y=Y, wavelength_nm=axis, property_names=np.asarray(names), ids=np.asarray(ids))


def prepare_haffner(data_dir):
    directory = data_dir / "haffner_ipa"
    full = loadmat(directory / "raw-kennard-full_range.mat")
    reduced = loadmat(directory / "raw-kennard-reduced_range.mat")
    y_train_scaled = np.loadtxt(directory / "y_train_scaled.csv", delimiter=",")
    y_test_scaled = np.loadtxt(directory / "y_test_scaled.csv", delimiter=",")
    np.savez_compressed(
        directory / "haffner_cn.npz",
        X_train_full=np.asarray(full["X_train"], dtype=np.float64),
        X_test_full=np.asarray(full["X_test"], dtype=np.float64),
        X_train_reduced=np.asarray(reduced["X_train"], dtype=np.float64),
        X_test_reduced=np.asarray(reduced["X_test"], dtype=np.float64),
        y_train=y_train_scaled,
        y_test=y_test_scaled,
    )
    y_train_centered = full["y_train"].ravel()
    y_test_centered = full["y_test"].ravel()
    recovered = np.concatenate([y_train_centered + TRAIN_OFFSET, y_test_centered + TEST_OFFSET])
    pd.DataFrame({
        "split": ["train"] * len(y_train_centered) + ["test"] * len(y_test_centered),
        "row_within_split": np.r_[np.arange(len(y_train_centered)), np.arange(len(y_test_centered))],
        "y_centered_mat": np.r_[y_train_centered, y_test_centered],
        "y_scaled_csv": np.r_[y_train_scaled, y_test_scaled],
        "y_cn_recovered": recovered,
    }).to_csv(directory / "recovered_cn_labels.csv", index=False)
    np.savetxt(directory / "wavenumber_full_cm-1.csv", full["W"].ravel(), delimiter=",")
    np.savetxt(directory / "wavenumber_reduced_cm-1.csv", reduced["W"].ravel(), delimiter=",")


def prepare_gasoline(data_dir):
    directory = data_dir / "gasoline_kalivas"
    archive = np.load(directory / "gasoline_arrays.npz")
    y = archive["arr_0"].astype(np.float64)
    X = archive["arr_1"].astype(np.float64)
    if X.shape != (60, 401) or y.shape != (60,):
        raise ValueError("Unexpected gasoline array shapes")
    np.savez_compressed(
        directory / "gasoline_octane.npz",
        X=X,
        y_octane=y,
        wavelength_nm=np.arange(900, 1701, 2),
        ids=np.asarray([f"gas{index}" for index in range(1, 61)]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["swri_diesel", "haffner_diesel", "kalivas_gasoline", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.dataset in ("swri_diesel", "all"):
        prepare_swri(args.data_dir)
    if args.dataset in ("haffner_diesel", "all"):
        prepare_haffner(args.data_dir)
    if args.dataset in ("kalivas_gasoline", "all"):
        prepare_gasoline(args.data_dir)
    print("Input arrays prepared. Compressed NPZ byte hashes can differ across runs; array values are the reproducibility target.")


if __name__ == "__main__":
    main()
