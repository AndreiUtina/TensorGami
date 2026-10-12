"""AlexNet (Krizhevsky et al. 2012) drawn by Tensorgami, end to end.

The model is defined exactly as in ``torchvision.models.alexnet`` (so torchvision is not
needed) and drawn with one call, ``tg.draw(model, example_input)``. The notebook
``alexnet.ipynb`` shows the figures inline, in every style and view::

    uv run python examples/alexnet.py           # open the figure in a window
    uv run python examples/alexnet.py figures/  # save PNG + SVG variants into figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

import tensorgami as tg


class AlexNet(nn.Module):
    """Identical to ``torchvision.models.AlexNet``."""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def draw(**options: object) -> tg.Figure:
    """AlexNet on one 224×224 image; ``options`` go to ``tg.draw`` (level, style, direction)."""
    return tg.draw(AlexNet(), torch.randn(1, 3, 224, 224), **options)


def main(argv: list[str]) -> None:
    """Without arguments: open AlexNet in a window. With a folder: save a few variants there
    as PNG (300 dpi) and SVG."""
    if not argv:
        draw().show()
        return
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "alexnet": {},
        "alexnet-stages": {"level": 1},
        "alexnet-volume": {"view": "volume"},
        "alexnet-origami": {"style": "origami"},
        "alexnet-origami-extended": {"style": "origami-extended"},
    }
    for name, options in variants.items():
        fig = draw(**options)
        fig.save(out / f"{name}.png", dpi=300)
        fig.save(out / f"{name}.svg")
        print(f"wrote {out / name}.png and .svg — {fig!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
