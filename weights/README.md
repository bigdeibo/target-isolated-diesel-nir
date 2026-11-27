# Encoder weights

`xdi_encoder.pt` contains the encoder pretrained after excluding all 784 SWRI diesel spectra. Its corpus has 8,555 retained spectra, and the independent diesel dataset is absent.

SHA-256:

`669410e855e7700e6794cd0c4414aadca59b9b6e4c2d893c2e4ace06013ba278`

`xgas_encoder.pt` contains the encoder pretrained after excluding the 60 gasoline spectra. Its corpus has 9,279 retained spectra.

SHA-256:

`1f6796283054e035c65642348897578dbd5b60a79c7adb8cf503720fb902fffb`

Both checkpoints use the 1D ResNet encoder with 512 input positions and a 128-dimensional embedding. Pretraining lasted 60 epochs with raw-spectrum input and per-spectrum SNV. The reconstruction decoder is not used in supervised fine-tuning.

The CSV files contain the actual pretraining histories. The training scripts verify checkpoint hashes before loading them. These weights are research artifacts for reproducing the reported evaluations; they are not deployment models trained on the full target datasets.
