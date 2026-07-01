"""Short-chunk CNN (Won et al., 2020) — 7 conv blocks + global pool + FC.
Architecture reused from the MIT sota-music-tagging-models repo; weights trained
fresh on permissive FMA. Input is a log-mel tensor (feature extraction is done
outside this module / outside the ONNX graph)."""
from __future__ import annotations
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ShortChunkCNN(nn.Module):
    def __init__(self, num_classes: int, n_channels: int = 128):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(1)
        c = n_channels
        self.layers = nn.Sequential(
            _ConvBlock(1, c), _ConvBlock(c, c), _ConvBlock(c, c * 2),
            _ConvBlock(c * 2, c * 2), _ConvBlock(c * 2, c * 2),
            _ConvBlock(c * 2, c * 2), _ConvBlock(c * 2, c * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c * 4, c * 4), nn.BatchNorm1d(c * 4), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(c * 4, num_classes),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.bn_in(mel)))
