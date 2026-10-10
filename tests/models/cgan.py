"""Conditional GAN: both networks receive the class label as a second input."""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 16, classes: int = 10) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(classes, classes)
        self.fc = nn.Sequential(nn.Linear(z_dim + classes, 64 * 4 * 4), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.deconv(self.fc(x).view(-1, 64, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(classes, 16 * 16)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = self.label_emb(labels).view(-1, 1, 16, 16)
        return self.head(self.conv(torch.cat([image, condition], dim=1)))
