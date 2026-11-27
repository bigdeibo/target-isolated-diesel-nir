# Target-isolated masked autoencoding for multi-property NIR analysis of diesel fuels

This repository contains the model code, target-excluded encoder weights, frozen out-of-fold (OOF) predictions, and verification scripts for the diesel and gasoline experiments reported in the manuscript.

The main diesel study covers seven independently calibrated properties. Target-isolated pretraining improves confirmation-CV RMSEP for six of them. For cetane number, the confirmation-CV RMSEP is 2.132765. Independent-dataset recalibration gives an RMSEP of 1.835093. The complete framework has lower mean RMSEP than DeepSpectra in all 12 comparisons with raw input and all 12 comparisons with SG1–SNV input.

## What is included

- The 1D ResNet masked autoencoder and supervised readout.
- PLS, SVR, BOSS-based PLS, DeepSpectra, and IPA reference implementations.
- The `xdi` and `xgas` encoder checkpoints, with SHA-256 values and pretraining histories.
- Per-sample OOF predictions, individual inner-member predictions, and nested split assignments.
- Source data for Tables 1–4 and the other quantitative analyses in the manuscript.
- Scripts that recalculate metrics, rebuild tables, reconstruct ensembles, regenerate splits, and verify target exclusion.

Figures, graphical abstracts, and plotting scripts are outside the scope of this repository. Raw spectra are also excluded because redistribution terms differ across sources. Their access routes and processing steps are documented in `data/README.md`.

## Verify the reported results

Python 3.10 or later is recommended. From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
python -I scripts/reproduce_manuscript_results.py
python -I -m unittest discover -s tests -v
```

The reproduction workflow uses the supplied predictions and does not train a model. It writes recalculated tables and audit reports to `validation/`.

A successful run checks:

- 590 OOF runs and 3,540 stored metric values, including sensitivity analyses;
- 15 nested-split scopes and 15 ensemble-reconstruction scopes;
- target exclusion for both pretraining checkpoints;
- manuscript Tables 3 and 4 against their frozen source data;
- all 27 registered numeric claims.

The 590 runs include overlapping primary and sensitivity evaluations. They are verification records, not independent experimental samples.

## Repository layout

```text
configs/       Model and evaluation settings
data/          Data access notes and hash manifests
docs/          Field definitions, reproduction instructions, and provenance
results/       Frozen predictions, metrics, splits, audits, and table data
scripts/       Data preparation, training, and verification entry points
src/           Model, preprocessing, training, and metric implementations
tests/         Code-level tests that do not train models
weights/       Target-excluded encoders and pretraining histories
```

## Train with locally prepared data

After preparing the data as described in `docs/reproduction_guide.md`, a supervised SWRI run can be started with:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset swri_diesel \
  --property cetane_number \
  --evaluation-group confirmation \
  --output training_output/swri_cn_confirmation
```

The independent diesel analysis uses recalibration on its own OOF splits:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset haffner_diesel \
  --evaluation-group initial \
  --output training_output/haffner_cn
```

Training is computationally intensive and is separate from result-level verification. Small numerical differences can occur across PyTorch versions and hardware. The supplied OOF predictions define the manuscript results.

## Provenance and citation

`docs/provenance_map.csv` records the source and SHA-256 of every exported evidence file. `MANIFEST.sha256` covers the repository files. `CITATION.cff` identifies the associated manuscript; formal publication metadata can be added when available.

The code is released under the MIT License. Raw datasets, third-party software, and referenced publications remain subject to their own terms.
