from __future__ import annotations

import torch
from torch import nn


class VAE(nn.Module):
    """MLP VAE with the reparameterisation trick; returns (reconstruction, μ, log σ²)."""

    def __init__(self, features: int = 784, hidden: int = 64, latent: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, features), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar
