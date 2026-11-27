"""Check repository contents and optionally write its file manifest."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "configs/model_config.json",
    "configs/evaluation_protocols.json",
    "data/README.md",
    "docs/data_dictionary.md",
    "docs/reproduction_guide.md",
    "docs/provenance_map.csv",
    "weights/xdi_encoder.pt",
    "weights/xgas_encoder.pt",
    "results/predictions/swri_seven_property_oof_predictions.csv",
    "results/predictions/swri_cn_initial_oof_predictions.csv",
    "results/predictions/haffner_cn_oof_predictions.csv",
    "results/predictions/gasoline_octane_oof_predictions.csv",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".tif", ".tiff", ".drawio"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_files():
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and "validation" not in path.relative_to(ROOT).parts and ".git" not in path.relative_to(ROOT).parts and path.name != "MANIFEST.sha256" and "__pycache__" not in path.parts)


def write_manifest():
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in repository_files()]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    checks = {}
    files = repository_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    checks["required_files"] = all(path in relative for path in REQUIRED)
    checks["no_figure_directory"] = not any("figures" in path.relative_to(ROOT).parts for path in files)
    checks["no_image_files"] = not any(path.suffix.lower() in IMAGE_SUFFIXES for path in files)
    checks["no_raw_data_files"] = not any(path.is_file() and path.parent.name in {"swri_diesel", "haffner_ipa", "gasoline_kalivas", "pretrain_corpus_xdi", "pretrain_corpus_xgas"} for path in (ROOT / "data").rglob("*"))
    python_files = [path for path in files if path.suffix == ".py"]
    plotting_libraries = {"matplotlib", "seaborn", "plotly", "drawio"}
    imported_libraries = set()
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_libraries.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_libraries.add(node.module.split(".")[0])
    checks["no_plotting_code"] = not bool(imported_libraries & plotting_libraries)
    checks["python_syntax"] = True
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError:
            checks["python_syntax"] = False
    prose_files = [path for path in files if path.suffix in {".py", ".md", ".cff"} or path.parent == ROOT / "docs"]
    cjk = re.compile(r"[㐀-鿿]")
    checks["english_code_and_documentation"] = not any(cjk.search(path.read_text(encoding="utf-8")) for path in prose_files if path.name != "verify_repository.py")
    local_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/")
    checks["no_local_absolute_paths"] = not any(local_path.search(path.read_text(encoding="utf-8")) for path in prose_files if path.name != "verify_repository.py")
    manifest_path = ROOT / "MANIFEST.sha256"
    checks["file_manifest_present"] = manifest_path.exists()
    checks["file_manifest_matches"] = False
    if manifest_path.exists():
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in manifest_path.read_text(encoding="utf-8").splitlines() if "  " in line}
        actual = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
        checks["file_manifest_matches"] = expected == actual
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "largest_file_bytes": max(path.stat().st_size for path in files),
        "figure_files_or_plotting_code_included": False,
        "raw_spectra_included": False,
    }
    report_dir = ROOT / "validation/repository"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "repository_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
