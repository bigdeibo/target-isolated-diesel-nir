"""Data loading, deterministic splits, and spectral preprocessing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROPERTY_CODES = {
    "cetane_number": "CN",
    "distillation_50": "BP50",
    "density_d4052": "D4052",
    "flash_point": "FLASH",
    "freezing_point": "FREEZE",
    "total_aromatics": "TOTAL",
    "kinematic_viscosity": "VISC",
}


def row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def make_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = pd.qcut(pd.Series(y), q=5, labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), strata))


def resample_snv(X: np.ndarray, length: int = 512) -> np.ndarray:
    old = np.linspace(0, 1, X.shape[1])
    new = np.linspace(0, 1, length)
    values = np.stack([np.interp(new, old, row) for row in X])
    scale = values.std(axis=1, keepdims=True) + 1e-8
    return ((values - values.mean(axis=1, keepdims=True)) / scale).astype(np.float32)[:, None, :]


def sg1_snv(X: np.ndarray) -> np.ndarray:
    return resample_snv(savgol_filter(X, 11, 2, deriv=1, axis=1))


def snv_401(X: np.ndarray) -> np.ndarray:
    values = savgol_filter(X, 11, 2, deriv=1, axis=1)
    return ((values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def scale_xy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    scale = X_train.std(0)
    scale[scale == 0] = 1.0
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale == 0:
        y_scale = 1.0
    A = ((X_train - mean) / scale).astype(np.float32)[:, None, :]
    B = ((X_test - mean) / scale).astype(np.float32)[:, None, :]
    a = ((y_train - y_mean) / y_scale).astype(np.float32).reshape(-1, 1)
    return A, a, B, y_mean, y_scale


def load_swri(property_name: str, data_dir: Path | None = None):
    if property_name not in PROPERTY_CODES:
        raise ValueError(f"Unsupported SWRI property: {property_name}")
    path = (data_dir or ROOT / "data") / "swri_diesel/swri_cn.npz"
    archive = np.load(path)
    names = [str(value) for value in archive["property_names"]]
    column = names.index(PROPERTY_CODES[property_name])
    mask = np.isfinite(archive["Y"][:, column])
    X = archive["X"][mask].astype(np.float64)
    y = archive["Y"][mask, column].astype(np.float64)
    ids = archive["ids"][mask]
    hashes = np.array([row_hash(row) for row in X])
    expected = 381 if property_name == "cetane_number" else 395
    if X.shape != (expected, 401) or len(set(hashes)) != expected:
        raise ValueError(f"Unexpected SWRI content for {property_name}: {X.shape}")
    return X, y, ids, hashes


def load_haffner_native(data_dir: Path | None = None):
    base = data_dir or ROOT / "data"
    archive = np.load(base / "haffner_ipa/haffner_cn.npz")
    labels = pd.read_csv(base / "haffner_ipa/recovered_cn_labels.csv")
    X_full = np.vstack([archive["X_train_full"], archive["X_test_full"]]).astype(np.float64)
    X_reduced = np.vstack([archive["X_train_reduced"], archive["X_test_reduced"]]).astype(np.float64)
    y_all = labels.y_cn_recovered.to_numpy(dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(X_full):
        groups.setdefault(row_hash(row), []).append(index)
    keep = np.array(sorted(indices[0] for indices in groups.values()), dtype=int)
    if len(keep) != 179:
        raise ValueError(f"Expected 179 unique independent-diesel spectra, found {len(keep)}")
    for indices in groups.values():
        if np.ptp(y_all[indices]) != 0:
            raise ValueError("Exact spectral copies have inconsistent reference values")
        for index in indices[1:]:
            if not np.array_equal(X_full[indices[0]], X_full[index]) or not np.array_equal(X_reduced[indices[0]], X_reduced[index]):
                raise ValueError("Full and reduced spectral deduplication disagree")
    ids = keep.astype(str)
    hashes = np.array([row_hash(row) for row in X_full[keep]])
    return X_full[keep], X_reduced[keep], y_all[keep], ids, hashes, keep


def load_haffner(data_dir: Path | None = None, common_wavelength: bool = True):
    base = data_dir or ROOT / "data"
    X, _, y, ids, hashes, _ = load_haffner_native(base)
    wavenumber = pd.read_csv(base / "haffner_ipa/wavenumber_full_cm-1.csv", header=None).iloc[:, 0].to_numpy(dtype=float)
    wavelength_desc = 1e7 / wavenumber
    low = max(750.0, float(wavelength_desc.min())) if common_wavelength else float(wavelength_desc.min())
    high = min(1550.0, float(wavelength_desc.max())) if common_wavelength else float(wavelength_desc.max())
    mask = (wavelength_desc >= low) & (wavelength_desc <= high)
    wavelength = wavelength_desc[mask][::-1]
    cropped = X[:, mask][:, ::-1]
    grid = np.linspace(low, high, 401)
    X_grid = np.stack([np.interp(grid, wavelength, row) for row in cropped])
    axis = {"range_nm": [low, high], "points": 401, "common_wavelength": common_wavelength}
    return X_grid, y, ids, hashes, axis


def load_gasoline(data_dir: Path | None = None):
    path = (data_dir or ROOT / "data") / "gasoline_kalivas/gasoline_octane.npz"
    archive = np.load(path)
    X = archive["X"].astype(np.float64)
    y = archive["y_octane"].astype(np.float64)
    ids = archive["ids"]
    hashes = np.array([row_hash(row) for row in X])
    if X.shape != (60, 401) or len(set(hashes)) != 60:
        raise ValueError("Expected 60 unique gasoline spectra with 401 points")
    return X, y, ids, hashes


def load_pretraining_corpus(weight: str, data_dir: Path | None = None) -> np.ndarray:
    base = data_dir or ROOT / "data"
    directory = base / f"pretrain_corpus_{weight}"
    path = directory / ("spectra_nondiesel.npy" if weight == "xdi" else "spectra.npy")
    X = np.load(path).astype(np.float32)
    if X.shape[1] != 512:
        raise ValueError(f"Expected 512-point pretraining spectra, found {X.shape}")
    return ((X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
