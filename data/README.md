# Data access and preparation

Raw spectral data are not bundled. The files in `manifests/` allow the target-isolation audit and file-identity checks to run without redistributing the spectra.

## Main datasets

| Dataset | Source | Files expected under `data/` |
|---|---|---|
| SWRI diesel | [Eigenvector Research, SWRI data](https://www.eigenvector.com/data/SWRI/); Wang et al., DOI [10.1016/j.infrared.2019.103111](https://doi.org/10.1016/j.infrared.2019.103111) | `swri_diesel/diesel_spec.csv`, `swri_diesel/diesel_prop.csv` |
| Independent diesel | Haffner et al., DOI [10.1016/j.fuel.2024.133016](https://doi.org/10.1016/j.fuel.2024.133016), accompanying public data | `haffner_ipa/raw-kennard-full_range.mat`, `raw-kennard-reduced_range.mat`, `y_train_scaled.csv`, `y_test_scaled.csv` |
| Gasoline | Kalivas, DOI [10.1016/S0169-7439(97)00038-5](https://doi.org/10.1016/S0169-7439(97)00038-5); [R `pls` package](https://cran.r-project.org/package=pls) | `gasoline_kalivas/gasoline_arrays.npz` |

The SWRI dataset contains 784 spectra. Cetane number has 381 valid reference values; each of the other six properties has 395. The independent diesel dataset is deduplicated from 249 released rows to 179 unique spectra. Its public split is not used as an external test because exact spectral copies occur across that split. The gasoline dataset contains 60 unique spectra.

## Pretraining sources

The diesel-excluded `xdi` corpus contains 8,555 spectra after all 784 SWRI spectra are removed. It comprises 6,000 simulated spectra, 1,995 olive-oil spectra, 162 mayonnaise spectra, 98 NIST reference spectra, 240 corn spectra measured on three instruments, and 60 gasoline spectra. The independent diesel dataset is absent.

The gasoline-excluded `xgas` corpus contains 9,279 spectra after the 60 gasoline spectra are removed. Diesel spectra are allowed in this corpus because gasoline is the target material.

Source publications and access points:

- Olive oil: DOI [10.1016/j.crfs.2024.100913](https://doi.org/10.1016/j.crfs.2024.100913).
- Mayonnaise: DOI [10.1016/S0169-7439(99)00023-4](https://doi.org/10.1016/S0169-7439(99)00023-4).
- Corn and other reference datasets: [Eigenvector Research data sets](https://eigenvector.com/resources/data-sets/).
- NIST reference spectra: [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).

The simulated spectra use mixtures of 98 NIST spectra and 13 class-mean spectra from the olive-oil and mayonnaise datasets. Target-dataset spectra and class means are not used. `pretraining_corpus_manifest.json` records source counts and checkpoint identities.

## Prepare model inputs

Place the released files in the directories listed above, then run:

```bash
python -I scripts/prepare_input_data.py --dataset all
```

This creates the NPZ files expected by the training code. NPZ containers can have different byte hashes across creation dates and NumPy versions. Reproducibility applies to their array values; the original released-file hashes are listed separately.

The independent-diesel reference values in physical CN units use offsets inferred from the centered labels and the published one-decimal CN grid. They are not author-supplied scaling parameters. The preparation script preserves the source labels alongside the recovered values.

## Prepare a target-excluded pretraining corpus

A source corpus directory must contain `spectra.npy` with shape `(9339, 512)` and a row-aligned `sources.csv`. Build either target exclusion with:

```bash
python -I scripts/prepare_pretraining_corpus.py \
  --weight xdi \
  --source-corpus /path/to/full_corpus \
  --output data/pretrain_corpus_xdi
```

Use `--weight xgas` and `data/pretrain_corpus_xgas` for the gasoline extension. The script checks source counts and writes SHA-256 values. It does not modify the source corpus.

## Hash-based isolation audit

`pretraining_spectrum_hashes.csv` contains hashes rather than spectral intensities. Each raw spectrum is interpolated by normalized position to 512 float32 values before hashing. `scripts/verify_pretraining_isolation.py` confirms that target hashes are absent from the corresponding retained corpus and that the supplied checkpoints match their registered SHA-256 values.

`raw_data_checksums.csv` distinguishes released files, derived arrays, and files included in this repository. A value of `False` in `included_in_repository` means that the file must be obtained or generated locally.
