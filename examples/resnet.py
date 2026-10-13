"""ResNet (He et al. 2016) drawn by Tensorgami, end to end.

The models are defined as in ``torchvision.models.resnet`` (so torchvision is not needed;
BSD-3-Clause, see THIRD_PARTY_NOTICES.md), with the same module names, so torchvision's
pretrained weights load with ``load_state_dict``. Only the options for grouped and dilated
convolutions (ResNeXt, segmentation backbones) are left out. The notebook ``resnet.ipynb``
shows the figures inline::

    uv run python examples/resnet.py           # open ResNet-18 in a window
    uv run python examples/resnet.py figures/  # save PNG + SVG variants into figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

import tensorgami as tg


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two 3×3 convolutions and the identity (or a 1×1 projection) added back: ResNet-18/34."""

    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """1×1 down, 3×3, 1×1 up (×4 channels), plus the identity or a projection: ResNet-50+."""

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """A 7×7 stem, four stages of residual blocks (64 → 512 channels, each after the first
    halving the resolution), global average pooling and a linear classifier."""

    def __init__(
        self, block: type[BasicBlock | Bottleneck], layers: list[int], num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock | Bottleneck], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def draw(model: nn.Module | None = None, **options: object) -> tg.Figure:
    """A ResNet (default: ResNet-18) on one 224×224 image; ``options`` go to ``tg.draw``."""
    return tg.draw(model or resnet18(), torch.randn(1, 3, 224, 224), **options)


def main(argv: list[str]) -> None:
    """Without arguments: open ResNet-18 in a window. With a folder: save a few variants
    there as PNG (300 dpi) and SVG."""
    if not argv:
        draw().show()
        return
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "resnet18": {},
        "resnet18-stages": {"level": 1},
        "resnet18-origami": {"style": "origami"},
        "resnet50": {"model": resnet50()},
    }
    for name, options in variants.items():
        fig = draw(**options)
        fig.save(out / f"{name}.png", dpi=300)
        fig.save(out / f"{name}.svg")
        print(f"wrote {out / name}.png and .svg — {fig!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
