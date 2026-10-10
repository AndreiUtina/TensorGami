"""AlexNet (Krizhevsky et al. 2012) drawn by Tensorgami, end to end.

The model is defined exactly as in ``torchvision.models.alexnet`` (so torchvision is not
needed), traced on one 224×224 image, simplified and laid out automatically::

    uv run python examples/alexnet.py

Writes ``alexnet.svg`` (layer level, left → right), ``alexnet-vertical.svg`` (top → bottom)
and ``alexnet-stages.svg`` (stage level) next to this script.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tensorgami.importers.torch import trace
from tensorgami.layout import Direction, LayoutOptions, layout
from tensorgami.render import render_svg
from tensorgami.style import load_style
from tensorgami.transforms import simplify

HERE = Path(__file__).parent


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


def main() -> None:
    model = AlexNet()
    graph = trace(model, torch.randn(1, 3, 224, 224))  # runs once, leaves the model untouched
    print(simplify(graph, level=2).summary())

    theme = load_style("default")  # the standard style (ADR-0008)
    figures = {
        "alexnet.svg": (2, LayoutOptions()),
        "alexnet-vertical.svg": (2, LayoutOptions(direction=Direction.TOP_BOTTOM)),
        "alexnet-stages.svg": (1, LayoutOptions()),
    }
    for name, (level, options) in figures.items():
        scene = layout(simplify(graph, level), theme, options)
        (HERE / name).write_text(render_svg(scene, theme), encoding="utf-8")
        print(f"wrote examples/{name} ({scene.width:.0f}×{scene.height:.0f} px)")


if __name__ == "__main__":
    main()
