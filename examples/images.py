"""A picture to run a vision model on, drawn in a few lines of numpy: no download, no licence.

Sky, a sun, hills, a striped and a checked shape: edges, colours and textures that a
convolution's feature maps pick up even with untrained weights. Normalised as torchvision's
ImageNet models expect.

    from images import sample_image
    tg.draw(model, sample_image(), activations=True).animate()
"""

from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_image(size: int = 224) -> torch.Tensor:
    """A ``1 × 3 × size × size`` picture, normalised (ImageNet mean and std)."""
    y, x = np.mgrid[0:size, 0:size] / size
    r, g, b = 0.35 + 0.4 * (1 - y), 0.55 + 0.35 * (1 - y), 0.85 + 0.1 * (1 - y)  # sky
    sun = (x - 0.75) ** 2 + (y - 0.25) ** 2 < 0.012
    r[sun], g[sun], b[sun] = 1.0, 0.85, 0.3
    hill = y > 0.62 + 0.08 * np.sin(x * 7)
    r[hill], g[hill], b[hill] = 0.25, 0.55 + 0.1 * np.sin(x[hill] * 40), 0.2
    flag = (abs(x - 0.25) < 0.12) & (abs(y - 0.4) < 0.15)
    stripe = (np.floor((x + y) * 30) % 2) == 0
    r[flag & stripe], g[flag & stripe], b[flag & stripe] = 0.9, 0.2, 0.2
    r[flag & ~stripe], g[flag & ~stripe], b[flag & ~stripe] = 1.0, 1.0, 1.0
    board = (abs(x - 0.55) < 0.08) & (abs(y - 0.78) < 0.08)
    white = ((np.floor(x * 24) + np.floor(y * 24)) % 2) == 0
    for channel in (r, g, b):
        channel[board] = white[board]
    picture = np.stack([r, g, b])
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    return torch.tensor((picture - mean) / std, dtype=torch.float32)[None]
