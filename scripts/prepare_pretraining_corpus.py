"""Remove target spectra from the full pretraining corpus and record hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", choices=["xdi", "xgas"], required=True)
    parser.add_argument("--source-corpus", type=Path, required=True, help="Directory containing spectra.npy and sources.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    X = np.load(args.source_corpus / "spectra.npy")
    sources = pd.read_csv(args.source_corpus / "sources.csv")
    if X.shape != (9339, 512) or len(sources) != 9339:
        raise ValueError("The source corpus must contain 9,339 spectra with 512 positions")
    target = "diesel" if args.weight == "xdi" else "gasoline"
    keep = ~sources.source.isin([target]).to_numpy()
    expected_excluded = 784 if args.weight == "xdi" else 60
    if int((~keep).sum()) != expected_excluded:
        raise ValueError(f"Expected {expected_excluded} excluded {target} spectra")
    retained = np.ascontiguousarray(X[keep]).astype(np.float32)
    retained_sources = sources.loc[keep].reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    spectra_name = "spectra_nondiesel.npy" if args.weight == "xdi" else "spectra.npy"
    sources_name = "sources_nondiesel.csv" if args.weight == "xdi" else "sources.csv"
    np.save(args.output / spectra_name, retained)
    retained_sources.to_csv(args.output / sources_name, index=False)
    manifest = {
        "weight": args.weight,
        "excluded_source": target,
        "excluded_rows": expected_excluded,
        "retained_rows": len(retained),
        "array_shape": list(retained.shape),
        "dtype": str(retained.dtype),
        "source_counts": retained_sources.source.value_counts().to_dict(),
        "source_spectra_sha256": sha256(args.source_corpus / "spectra.npy"),
        "source_labels_sha256": sha256(args.source_corpus / "sources.csv"),
        "output_spectra_sha256": sha256(args.output / spectra_name),
        "output_labels_sha256": sha256(args.output / sources_name),
        "normalization": "Per-spectrum SNV is applied when the corpus is loaded for pretraining",
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
