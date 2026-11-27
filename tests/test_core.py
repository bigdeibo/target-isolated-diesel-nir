"""Small code-level checks that do not train or evaluate models."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_folds, resample_snv
from src.metrics import regression_metrics, relative_improvement
from src.models import DeepSpectra, IPA, ResNet1Encoder, SpectralResNetRegressor


class CoreTests(unittest.TestCase):
    def test_checkpoint_loads_strictly(self):
        checkpoint = torch.load(ROOT / "weights/xdi_encoder.pt", map_location="cpu", weights_only=True)
        encoder = ResNet1Encoder(in_len=512, emb_dim=128)
        encoder.load_state_dict(checkpoint["enc"], strict=True)
        self.assertEqual(checkpoint["epoch"], 60)

    def test_model_output_shapes(self):
        model = SpectralResNetRegressor()
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.zeros(2, 1, 512)).shape), (2, 1))
            self.assertEqual(tuple(DeepSpectra()(torch.zeros(2, 1, 401)).shape), (2, 1))
            self.assertEqual(tuple(IPA()(torch.zeros(2, 1, 401)).shape), (2, 1))

    def test_preprocessing_shape_and_normalization(self):
        values = np.arange(20, dtype=float).reshape(2, 10)
        transformed = resample_snv(values)
        self.assertEqual(transformed.shape, (2, 1, 512))
        np.testing.assert_allclose(transformed[:, 0].mean(axis=1), 0, atol=1e-7)
        np.testing.assert_allclose(transformed[:, 0].std(axis=1), 1, atol=1e-6)

    def test_deterministic_disjoint_folds(self):
        y = np.linspace(0, 1, 50)
        first = make_folds(y, 5, 123)
        second = make_folds(y, 5, 123)
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
            self.assertFalse(set(train_a) & set(test_a))

    def test_metrics(self):
        metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(metrics["rmsep"], np.sqrt(1 / 3))
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(relative_improvement(2.0, 1.5), 25.0)


if __name__ == "__main__":
    unittest.main()
