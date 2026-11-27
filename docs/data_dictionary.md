# Data dictionary

Repository identifiers follow the manuscript terminology. CV repetitions, outer folds, and inner folds are one-based in exported files. The split-seed formulas use the corresponding zero-based indices.

## Prediction fields

| Field | Meaning |
|---|---|
| `dataset` | `swri_diesel`, `haffner_diesel`, or `kalivas_gasoline` |
| `property` | Single prediction target, such as `cetane_number` or `density_d4052` |
| `split_group` | `initial` for repetitions 1–2 or `confirmation` for repetitions 3–4 |
| `model` | Canonical model or reference-method identifier |
| `initialization` | `target_isolated_masked_autoencoder`, `random`, or `not_applicable` |
| `input_representation` | Spectral transformation used by the model |
| `cv_repetition` | One-based CV repetition; one complete outer-five-fold partition |
| `outer_fold` | One-based outer fold from 1 to 5 |
| `inner_fold` | One-based inner fold from 1 to 3 in member-level files |
| `seed` | Final training seed; classical deterministic fits use 0 |
| `sample_index` | Zero-based numeric row index within the deduplicated source dataset |
| `sample_id` | Source sample identifier |
| `original_index` | Row in the released independent-diesel data before deduplication |
| `y_true` | Reference property value in the property unit |
| `y_pred` | Out-of-fold prediction in the same unit |

The main prediction files store the three-inner-fold equal-weight ensemble. Files under `inner_members/` contain all three complete inner models. Files under `single_models/` contain the first inner model for the estimator comparison; this model is not selected by performance.

## Split fields

| Field | Meaning |
|---|---|
| `role` | `train`, `test`, or `val` (`validation` in newly generated files) |
| `spectrum_sha256` | SHA-256 of the source spectral row used for overlap checks |
| `y_true` | Reference value used to regenerate quantile-stratified splits |

Inner splits are subsets of the corresponding outer-training pool. Outer test samples do not appear in an inner role.

## Metric fields

| Field | Definition |
|---|---|
| `n` | Number of unique OOF samples |
| `rmsep` | Root mean squared error of prediction |
| `r_squared` | Coefficient of determination calculated from the same OOF predictions |
| `mae` | Mean absolute error |
| `rpd` | Sample standard deviation of reference values divided by RMSEP |
| `bias` | Mean of predicted minus reference values |

Run-level metrics are calculated after all five outer folds are assembled. Means and sample standard deviations are then calculated across the stated CV–seed runs. Runs are paired analysis units and do not increase the independent experimental sample size.

Relative RMSEP improvement is `100 * (reference_rmsep - candidate_rmsep) / reference_rmsep`. Positive values favor the candidate. Cross-property average R² is not used.

## Property and model identifiers

| Property | Identifier |
|---|---|
| Cetane number | `cetane_number` |
| 50% distillation temperature | `distillation_50` |
| Density by ASTM D4052 | `density_d4052` |
| Flash point | `flash_point` |
| Freezing point | `freezing_point` |
| Total aromatic content | `total_aromatics` |
| Kinematic viscosity | `kinematic_viscosity` |
| Gasoline octane number | `octane_number` |

The two target-isolated weights are `xdi` and `xgas`. `xdi` excludes all SWRI diesel spectra and is used for diesel calibration. `xgas` excludes the gasoline spectra and is used only for the gasoline extension.

`deepspectra_raw` retains the 401-point raw input. `deepspectra_sg1_snv` applies SG1 and SNV while retaining 401 points. Independent-diesel references use explicit `full_wavelength` or `reduced_wavelength` suffixes where applicable. The primary candidate protocol uses the common wavelength range.

## Source-field mapping

Historical result files used zero-based `repeat` and `fold` fields. Export adds one to these values and renames `fold` to `outer_fold`. Historical `family` values are mapped to `model`. Historical `r2` and `bias_pred_minus_true` are exported as `r_squared` and `bias`. The source path and transformation of each exported file are recorded in `docs/provenance_map.csv`.
