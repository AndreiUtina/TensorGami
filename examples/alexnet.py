"""AlexNet (Krizhevsky et al. 2012) drawn by Tensorgami, end to end.

The model is defined exactly as in ``torchvision.models.alexnet`` (so torchvision is not
needed), traced on one 224×224 image, simplified and laid out automatically::

    uv run python examples/alexnet.py

Writes ``alexnet`` (layer level, left → right), ``alexnet-vertical`` (top → bottom) and
``alexnet-stages`` (stage level), each as ``.svg`` and as a 300 dpi ``.png``, next to this
script. In a notebook, ``draw()`` shows the figure inline (see ``alexnet.ipynb``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import tensorgami as tg
from tensorgami.importers.torch import trace
from tensorgami.layout import LayoutOptions, layout
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


def draw(level: int = 2, style: str = "default", **layout_options: object) -> tg.Figure:
    """AlexNet as a Tensorgami figure (step 10 will shorten this to ``tg.draw(...)``).

    ``layout_options`` override the style's layout preferences, e.g. ``direction="TB"``.
    """
    graph = simplify(trace(AlexNet(), torch.randn(1, 3, 224, 224)), level)
    theme = load_style(style)  # "default" or "origami" (ADR-0008)
    return tg.Figure(
        layout(graph, theme, LayoutOptions.for_style(theme, **layout_options)), theme, graph
    )


def main() -> None:
    figures = {
        "alexnet": dict(level=2),
        "alexnet-vertical": dict(level=2, direction="TB"),
        "alexnet-stages": dict(level=1),
        "alexnet-origami": dict(level=2, style="origami"),
        "alexnet-origami-vertical": dict(level=2, style="origami", direction="TB"),
        "alexnet-origami-stages": dict(level=1, style="origami"),
    }
    for name, settings in figures.items():
        fig = draw(**settings)
        fig.save(HERE / f"{name}.svg")  # vector, for the docs
        fig.save(HERE / f"{name}.png", dpi=300)  # for reports and slides
        print(f"wrote examples/{name}.svg and .png — {fig!r}")


if __name__ == "__main__":
    main()
