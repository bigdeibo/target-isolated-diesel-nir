"""Pretrain the raw-spectrum masked autoencoder on a target-excluded corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_pretraining_corpus
from src.models import MaskedSpectralAutoencoder


def band_mask(rng, batch, length=512, total=(0.25, 0.40), segments=(3, 6)):
    mask = np.zeros((batch, length), dtype=bool)
    for row in range(batch):
        coverage = rng.uniform(*total)
        count = int(rng.integers(segments[0], segments[1] + 1))
        width = max(1, int(length * coverage / count))
        for _ in range(count):
            start = int(rng.integers(0, max(1, length - width)))
            mask[row, start:start + width] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", choices=["xdi", "xgas"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X = load_pretraining_corpus(args.weight)
    expected = 8555 if args.weight == "xdi" else 9279
    if len(X) != expected:
        raise ValueError(f"{args.weight} requires {expected} retained spectra, found {len(X)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MaskedSpectralAutoencoder().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    spectra = torch.from_numpy(X).to(args.device)
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(spectra), generator=generator, device=args.device)
        loss_sum = 0.0
        mask_count = 0
        for begin in range(0, len(permutation), args.batch_size):
            indices = permutation[begin:begin + args.batch_size]
            batch = spectra[indices]
            mask = torch.from_numpy(band_mask(rng, len(indices))).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.masked_fill(mask, 0.0)[:, None])
            loss = (((prediction - batch) ** 2) * mask.float()).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            count = int(mask.sum())
            loss_sum += float(loss.detach()) * count
            mask_count += count
        history.append({"epoch": epoch + 1, "masked_mse": loss_sum / mask_count})
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} masked_mse={history[-1]['masked_mse']:.8f}", flush=True)

    checkpoint = {
        "enc": {key: value.detach().cpu() for key, value in model.encoder.state_dict().items()},
        "arch": "resnet1",
        "epoch": args.epochs,
        "emb_dim": 128,
        "in_len": 512,
        "weight": args.weight,
        "seed": args.seed,
    }
    path = args.output / f"{args.weight}_encoder.pt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with (args.output / f"{args.weight}_pretraining_history.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,masked_mse\n")
        handle.writelines(f"{row['epoch']},{row['masked_mse']:.8f}\n" for row in history)
    summary = {
        "weight": args.weight,
        "spectra": len(X),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "mask_coverage": [0.25, 0.40],
        "mask_segments": [3, 6],
        "decoder_retained_for_fine_tuning": False,
        "checkpoint_sha256": digest,
        "seconds": time.time() - started,
        "final_masked_mse": history[-1]["masked_mse"],
    }
    (args.output / "pretraining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
