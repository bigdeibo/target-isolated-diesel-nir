"""One-dimensional networks used in the manuscript."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, 9, stride=stride, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.conv2 = nn.Conv1d(channels_out, channels_out, 9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.short = None
        if stride != 1 or channels_in != channels_out:
            self.short = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(channels_out),
            )

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + identity)


class ResNet1Encoder(nn.Module):
    def __init__(self, in_len: int = 512, emb_dim: int = 128, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, 25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.stage1 = ResBlock1D(width, width)
        self.stage2 = ResBlock1D(width, 2 * width, stride=2)
        self.stage3 = ResBlock1D(2 * width, 4 * width, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Linear(4 * width, emb_dim)
        self.out_dim = emb_dim

    def forward(self, x):
        h = self.stem(x)
        h = self.stage3(self.stage2(self.stage1(h)))
        return self.proj(self.head(h))


class SpectralResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet1Encoder(in_len=512, emb_dim=128)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        return self.regressor(self.encoder(x))


class MaskedSpectralAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet1Encoder(in_len=512, emb_dim=128)
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class BasicConv1D(nn.Module):
    def __init__(self, channels_in, filters, kernel_size, strides=1, slope=0.3):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, filters, kernel_size, stride=strides)
        self.act = nn.LeakyReLU(negative_slope=slope)

    def forward(self, x):
        return self.act(self.conv(x))


class DeepSpectraModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(nn.MaxPool1d(2), BasicConv1D(8, 4, 1, 3))
        self.branch2 = nn.Sequential(BasicConv1D(8, 4, 1, 3), BasicConv1D(4, 4, 7, 3))
        self.branch3 = nn.Sequential(BasicConv1D(8, 4, 1, 3), BasicConv1D(4, 4, 7, 3))
        self.branch4 = BasicConv1D(8, 4, 1, 3)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=2)


class DeepSpectra(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = BasicConv1D(1, 8, 9, 5)
        self.module = DeepSpectraModule()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.regressor_1 = nn.LazyLinear(16)
        self.regressor_2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.module(x)
        x = self.regressor_1(self.dropout(self.flatten(x)))
        return self.regressor_2(x)


class IPAModule(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.branch1 = nn.Sequential(nn.MaxPool1d(2), BasicConv1D(16, in_channels * 2, 1, 2))
        self.branch2 = nn.Sequential(BasicConv1D(16, in_channels, 1, 2), BasicConv1D(in_channels, in_channels * 2, 3, 1))
        self.branch3 = nn.Sequential(
            BasicConv1D(16, in_channels, 1, 2),
            BasicConv1D(in_channels, in_channels * 2, 3, 2),
            BasicConv1D(in_channels * 2, in_channels * 2, 3, 2),
        )
        self.branch4 = BasicConv1D(16, in_channels * 2, 1, 2)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=2)


class IPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv1D(1, 16, 3, 2),
            BasicConv1D(16, 16, 3, 1),
            BasicConv1D(16, 16, 3, 1),
        )
        self.module = IPAModule(32)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.LazyLinear(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.module(x)
        return self.regressor(self.dropout(self.flatten(x)))


def build_reference_model(name: str, sample: torch.Tensor, seed: int):
    if name == "deepspectra_raw":
        model = DeepSpectra()
    elif name == "ipa":
        model = IPA()
    else:
        raise ValueError(f"Unsupported reference network: {name}")
    model(sample[:2])
    torch.manual_seed(seed)
    np.random.seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model
