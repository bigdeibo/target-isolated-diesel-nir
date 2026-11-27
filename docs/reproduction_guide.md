# Reproduction guide

## 1. Result-level verification

Install the dependencies and run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -I scripts/reproduce_manuscript_results.py
python -I -m unittest discover -s tests -v
python -I scripts/verify_repository.py
```

The main workflow calls six checks in sequence:

1. `verify_prediction_metrics.py` recalculates each run from per-sample OOF predictions and compares it with the stored metrics.
2. `verify_split_integrity.py` regenerates outer and inner splits and checks prediction test membership.
3. `verify_ensemble_predictions.py` averages the three inner members and reconstructs the saved ensemble.
4. `verify_pretraining_isolation.py` checks target hashes and checkpoint identities.
5. `generate_manuscript_tables.py` rebuilds Tables 3 and 4 and compares them with the frozen table data. Table 3 includes Initial and Confirmation CV for all seven properties. The separate `swri_cn_initial_oof_predictions.csv` file supplies the CN Initial candidate, matched-random, and classical-reference predictions; Table 4 continues to use CN Confirmation only.
6. `verify_numeric_claims.py` recalculates all 27 registered numeric claims.

Reports are written under `validation/`. The tolerance for stored-metric comparisons is `1e-8 * max(1, abs(stored value))`. The largest observed differences are at floating-point precision.

Individual checks can also be run directly, for example:

```bash
python -I scripts/verify_prediction_metrics.py --output validation/metrics
python -I scripts/generate_manuscript_tables.py --output validation/tables
```

## 2. Data preparation

Follow `data/README.md` to obtain the released datasets and place them in the expected directories. Then run:

```bash
python -I scripts/prepare_input_data.py --dataset all
```

The data preparation step is not required for result-level verification because predictions, reference values, split assignments, and spectrum hashes are already supplied.

## 3. Target-isolated pretraining

Prepare the target-excluded corpus as described in `data/README.md`. Pretraining is optional when using the supplied encoder weights.

```bash
python -I scripts/pretrain_target_isolated_mae.py \
  --weight xdi \
  --output training_output/xdi_pretraining
```

The script uses raw-spectrum interpolation and per-spectrum SNV, three to six masked bands, 60 epochs, and seed 0 by default. Only the encoder is transferred to supervised training. The gasoline extension requires `--weight xgas` and its separately prepared corpus.

## 4. Supervised fine-tuning

SWRI cetane number, confirmation CV:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset swri_diesel \
  --property cetane_number \
  --evaluation-group confirmation \
  --output training_output/swri_cn_confirmation
```

Another SWRI property, initial CV:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset swri_diesel \
  --property flash_point \
  --evaluation-group initial \
  --output training_output/swri_flash_initial
```

Independent-dataset recalibration:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset haffner_diesel \
  --evaluation-group initial \
  --output training_output/haffner_cn
```

Gasoline extension:

```bash
python -I scripts/train_supervised_resnet.py \
  --dataset kalivas_gasoline \
  --evaluation-group initial \
  --output training_output/gasoline_octane
```

Each run trains both the target-isolated and matched-random initializations unless `--initializations` is supplied. Every estimator contains three complete inner-fold models. Their best-validation states are averaged with equal weights. There is no full outer-pool refit, batch-normalization recalibration, or selection of a seed or inner member.

## 5. Reference methods

PLS and SVR share the outer splits and use inner-validation model selection:

```bash
python -I scripts/train_reference_models.py \
  --dataset swri_diesel \
  --property total_aromatics \
  --model pls \
  --evaluation-group confirmation \
  --output training_output/swri_total_pls
```

BOSS-based PLS is available for SWRI cetane number. DeepSpectra supports `deepspectra_raw` and `deepspectra_sg1_snv`. IPA is available for the independent-diesel comparison:

```bash
python -I scripts/train_reference_models.py \
  --dataset haffner_diesel \
  --model ipa \
  --wavelength reduced \
  --output training_output/haffner_ipa_reduced
```

DeepSpectra and IPA use their established inner-epoch selection and fixed-epoch outer-pool refit. This differs deliberately from the three-inner-fold prediction ensemble used by the target-isolated estimator.

## 6. Environment and numerical agreement

The local verification environment used Python 3.14.6, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, and PyTorch 2.12.1+cu126. The requirements file gives compatible minimum versions rather than forcing a platform-specific CUDA build.

Frozen OOF predictions are the reference for manuscript numbers. Retraining can produce small differences with other library versions, operating systems, BLAS implementations, or GPU kernels. No model training is part of the default verification workflow.
