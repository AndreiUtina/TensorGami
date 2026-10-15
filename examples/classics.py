"""The ten most-cited deep-learning papers with an architecture figure, as PyTorch models.

Each model follows its paper: the layer sizes, the inputs the paper uses, and the structure its
architecture figure shows. ``classics.ipynb`` draws each one with Tensorgami next to a link to
the paper's own figure. Where a paper leaves something out or a detail cannot be traced as
layers (proposal selection, Gaussian output units), the docstring says what stands in for it.

ResNet comes from ``resnet.py`` and is not repeated here. Citation counts are in the notebook.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- AlexNet (2012)


class AlexNet2012(nn.Module):
    """AlexNet as in the paper (Krizhevsky, Sutskever & Hinton, NeurIPS 2012): 96-256-384-
    384-256 kernels, local response normalisation, overlapping max pooling, two dense layers of
    4096. The paper splits the network over two GPUs; ``groups=2`` reproduces that split
    (conv 2, 4 and 5 only see the kernels on their own GPU). torchvision's AlexNet is the
    later single-GPU variant (64-192-384-256-256)."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # across both GPUs
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


# --------------------------------------------------------------------------- Transformer (2017)


def _sinusoids(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div)
    pe[0, :, 1::2] = torch.cos(position * div)
    return pe


class Transformer(nn.Module):
    """The Transformer (Vaswani et al., NeurIPS 2017), base model: N = 6 encoder and 6 decoder
    layers, d_model 512, 8 heads, d_ff 2048, post-norm ("Add & Norm"), sinusoidal positional
    encodings, embeddings scaled by √d_model, and a linear layer with softmax on top. The
    decoder's self-attention is masked (a causal mask, kept as a buffer)."""

    def __init__(
        self,
        vocab: int = 37000,
        d_model: int = 512,
        heads: int = 8,
        layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab, d_model)
        self.tgt_embed = nn.Embedding(vocab, d_model)
        self.register_buffer("positions", _sinusoids(max_len, d_model))
        causal = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal", causal)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, d_ff, batch_first=True),
            layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, heads, d_ff, batch_first=True), layers
        )
        self.generator = nn.Linear(d_model, vocab)
        # one shared matrix for both embeddings and the pre-softmax projection (section 3.4)
        self.tgt_embed.weight = self.src_embed.weight
        self.generator.weight = self.src_embed.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(self.d_model)
        memory = self.encoder(self.src_embed(src) * scale + self.positions[:, : src.size(1)])
        n = tgt.size(1)
        out = self.decoder(
            self.tgt_embed(tgt) * scale + self.positions[:, :n],
            memory,
            tgt_mask=self.causal[:n, :n],
            tgt_is_causal=True,  # else PyTorch compares the mask with a causal one, as ops
        )
        return F.softmax(self.generator(out), dim=-1)


# --------------------------------------------------------------------------- BERT (2019)


class BertEmbeddings(nn.Module):
    """Token + segment + position embeddings, summed, then LayerNorm and dropout (Fig. 2)."""

    def __init__(self, vocab: int, hidden: int, max_len: int = 512) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab, hidden)
        self.segment = nn.Embedding(2, hidden)
        self.position = nn.Embedding(max_len, hidden)
        self.register_buffer("position_ids", torch.arange(max_len).unsqueeze(0))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        positions = self.position_ids[:, : tokens.size(1)]
        x = self.token(tokens) + self.segment(segments) + self.position(positions)
        return self.dropout(self.norm(x))


class Bert(nn.Module):
    """BERT-base (Devlin et al., NAACL 2019) with its two pre-training heads, as in Fig. 1:
    12 Transformer encoder layers (hidden 768, 12 heads, GELU), the [CLS] output C pooled for
    next-sentence prediction (NSP), every token's output T_i into the masked-LM head."""

    def __init__(self, vocab: int = 30522, hidden: int = 768, layers: int = 12) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(vocab, hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 12, 4 * hidden, activation="gelu", batch_first=True),
            layers,
            enable_nested_tensor=False,
        )
        self.pooler = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.nsp = nn.Linear(hidden, 2)
        self.mlm = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, vocab)
        )
        self.mlm[-1].weight = self.embeddings.token.weight  # tied, as in the released model

    def forward(
        self, tokens: torch.Tensor, segments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.embeddings(tokens, segments))
        nsp = self.nsp(self.pooler(hidden[:, 0]))  # C, the [CLS] output
        return nsp, self.mlm(hidden)


# --------------------------------------------------------------------------- LSTM (1997)


class LSTMNet(nn.Module):
    """A recurrent net with LSTM memory cells and a linear output layer, the size of the
    paper's Fig. 2 example (8 inputs, 4 outputs). PyTorch's LSTM is the modern one, with the
    forget gate Gers et al. added in 2000; the 1997 cell has input and output gates only."""

    def __init__(self, inputs: int = 8, cells: int = 4, outputs: int = 4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(inputs, cells, batch_first=True)
        self.out = nn.Linear(cells, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.out(sequence[:, -1])


# --------------------------------------------------------------------------- U-Net (2015)


def _double_conv(inputs: int, outputs: int) -> nn.Sequential:
    """Two unpadded 3×3 convolutions with ReLU: each one trims a pixel on every side."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3),
        nn.ReLU(inplace=True),
    )


def _crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """The centre ``size`` × ``size`` of a feature map ("copy and crop")."""
    top = (x.size(-1) - size) // 2
    return x[..., top : top + size, top : top + size]


class UNet(nn.Module):
    """U-Net (Ronneberger, Fischer & Brox, MICCAI 2015) as in Fig. 1: unpadded convolutions,
    so a 572 × 572 tile gives a 388 × 388 map of 2 classes; each skip is cropped to the size of
    the up-convolution it joins. The crops are taken on the encoder side, right after each
    level (their sizes follow from the tile size), which is the same computation."""

    def __init__(self, channels: int = 1, classes: int = 2) -> None:
        super().__init__()
        widths = [64, 128, 256, 512]
        self.down = nn.ModuleList()
        previous = channels
        for width in widths:
            self.down.append(_double_conv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottom = _double_conv(512, 1024)
        self.up = nn.ModuleList(nn.ConvTranspose2d(2 * w, w, 2, stride=2) for w in widths[::-1])
        self.dec = nn.ModuleList(_double_conv(2 * w, w) for w in widths[::-1])
        self.head = nn.Conv2d(64, classes, 1)

    @staticmethod
    def _skip_sizes(size: int, levels: int = 4) -> list[int]:
        """The size each skip is cropped to, for an input tile of ``size``."""
        encoder = []
        for _ in range(levels):
            size -= 4
            encoder.append(size)
            size //= 2
        size -= 4  # the bottom
        crops = []
        for _ in range(levels):
            size *= 2
            crops.append(size)
            size -= 4
        return crops[::-1]  # outermost level first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = self._skip_sizes(x.size(-1))
        skips = []
        for level, block in enumerate(self.down):
            x = block(x if level == 0 else self.pool(x))
            skips.append(_crop(x, sizes[level]))
        x = self.bottom(self.pool(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.head(x)


# --------------------------------------------------------------------------- Faster R-CNN (2015)


def _vgg16_convs() -> nn.Sequential:
    """VGG-16's 13 convolutional layers and 4 poolings (stride 16), the paper's backbone."""
    layers: list[nn.Module] = []
    previous = 3
    for stage, (width, repeat) in enumerate([(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]):
        for _ in range(repeat):
            layers += [nn.Conv2d(previous, width, 3, padding=1), nn.ReLU(inplace=True)]
            previous = width
        if stage < 4:  # no pool5: the RPN works at stride 16
            layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class RegionProposalNetwork(nn.Module):
    """A 3×3 sliding window over the feature map, then two sibling 1×1 layers: objectness
    scores (2k) and box regressions (4k) for k = 9 anchors per position (Fig. 3)."""

    def __init__(self, channels: int = 512, anchors: int = 9) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 512, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_logits = nn.Conv2d(512, 2 * anchors, 1)
        self.bbox_pred = nn.Conv2d(512, 4 * anchors, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv(features))
        return self.cls_logits(x), self.bbox_pred(x)


class ProposalLayer(nn.Module):
    """Anchors moved by the predicted regressions, the ``top`` most object-like kept as
    proposals (the paper also applies non-maximum suppression). Drawn as one box."""

    def __init__(self, top: int = 8, stride: int = 16, sizes=(128, 256, 512)) -> None:
        super().__init__()
        self.top, self.stride = top, stride
        ratios = (0.5, 1.0, 2.0)
        shapes = [(s * math.sqrt(1 / r), s * math.sqrt(r)) for s in sizes for r in ratios]
        self.register_buffer("shapes", torch.tensor(shapes))  # (k, 2): width, height

    def forward(self, scores: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        _, _, h, w = scores.shape
        k = self.shapes.size(0)
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        centres = torch.stack([xs, ys], -1).reshape(-1, 1, 2) * self.stride + self.stride / 2
        sizes = self.shapes.expand(h * w, k, 2)
        d = deltas.view(k, 4, h, w).permute(2, 3, 0, 1).reshape(-1, k, 4)
        centre = centres + d[..., :2] * sizes
        size = sizes * torch.exp(d[..., 2:])
        boxes = torch.cat([centre - size / 2, centre + size / 2], -1).reshape(-1, 4)
        objectness = scores.view(2, k, h, w).softmax(0)[1].permute(1, 2, 0).reshape(-1)
        return boxes[objectness.topk(self.top).indices]


class RoIPool(nn.Module):
    """Max-pools the feature map inside each proposal to a fixed 7 × 7 grid. Drawn as one box."""

    def __init__(self, output: int = 7, stride: int = 16) -> None:
        super().__init__()
        self.output, self.stride = output, stride

    def forward(self, features: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        h, w = features.shape[-2:]
        pooled = []
        for x0, y0, x1, y1 in (proposals / self.stride).round().int().tolist():
            x0, y0 = min(max(x0, 0), w - 1), min(max(y0, 0), h - 1)
            x1, y1 = max(min(x1, w), x0 + 1), max(min(y1, h), y0 + 1)
            region = features[:, :, y0:y1, x0:x1]
            pooled.append(F.adaptive_max_pool2d(region, self.output))
        return torch.cat(pooled)


class FasterRCNN(nn.Module):
    """Faster R-CNN (Ren, He, Girshick & Sun, NeurIPS 2015) with VGG-16, as in Fig. 2: shared
    conv layers, the region proposal network on their feature maps, RoI pooling of the
    proposals from the same feature maps, and the Fast R-CNN classifier (21 PASCAL VOC classes
    with per-class box regression)."""

    def __init__(self, classes: int = 21, proposals: int = 8) -> None:
        super().__init__()
        self.backbone = _vgg16_convs()
        self.rpn = RegionProposalNetwork()
        self.proposals = ProposalLayer(top=proposals)
        self.roi_pool = RoIPool()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, classes)
        self.bbox_pred = nn.Linear(4096, 4 * classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        scores, deltas = self.rpn(features)
        rois = self.roi_pool(features, self.proposals(scores, deltas))
        x = self.head(rois)
        return self.cls_score(x), self.bbox_pred(x)


#: draw the two non-layer steps as one box each
FASTER_RCNN_OVERRIDES = {
    "ProposalLayer": {"leaf": True, "label": "Proposals", "category": "other"},
    "RoIPool": {"leaf": True, "label": "RoI pooling", "category": "resample"},
}


# --------------------------------------------------------------------------- ViT (2021)


class VisionTransformer(nn.Module):
    """ViT-B/16 (Dosovitskiy et al., ICLR 2021), as in Fig. 1: 16 × 16 patches linearly
    projected (a strided convolution), a learnable [class] token prepended, learnable position
    embeddings added, 12 pre-norm Transformer encoder layers, and an MLP head on the class
    token's output."""

    def __init__(
        self,
        image: int = 224,
        patch: int = 16,
        hidden: int = 768,
        layers: int = 12,
        heads: int = 12,
        classes: int = 1000,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image // patch) ** 2 + 1, hidden))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden, heads, 4 * hidden, activation="gelu", batch_first=True, norm_first=True
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 196, 768)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.encoder(x))
        return self.head(x[:, 0])


# --------------------------------------------------------------------------- LeNet-5 (1998)


class Subsampling(nn.Module):
    """LeNet's S layers: 2 × 2 averages, times a trainable coefficient plus a trainable bias
    per map, through a sigmoidal squashing. Drawn as one box."""

    def __init__(self, maps: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.scale = nn.Conv2d(maps, maps, 1, groups=maps)  # coefficient and bias per map
        self.squash = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squash(self.scale(self.pool(x)))


class GaussianConnections(nn.Module):
    """The output layer: one Euclidean radial basis function unit per class, the squared
    distance between F6's output and the class's (fixed) parameter vector. Drawn as one box."""

    def __init__(self, inputs: int = 84, classes: int = 10) -> None:
        super().__init__()
        self.centres = nn.Parameter(torch.randn(classes, inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.unsqueeze(1) - self.centres) ** 2).sum(-1)


class LeNet5(nn.Module):
    """LeNet-5 (LeCun, Bottou, Bengio & Haffner, Proc. IEEE 1998) as in Fig. 2: C1, S2, C3,
    S4, C5, F6 and the Gaussian output, on a 32 × 32 image. C3 is fully connected to S2 here
    (the paper connects each C3 map to a subset of S2's maps, Table I); the paper's squashing
    is a scaled tanh."""

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh())
        self.s2 = Subsampling(6)
        self.c3 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.Tanh())
        self.s4 = Subsampling(16)
        self.c5 = nn.Sequential(nn.Conv2d(16, 120, 5), nn.Tanh())
        self.f6 = nn.Sequential(nn.Flatten(), nn.Linear(120, 84), nn.Tanh())
        self.output = GaussianConnections(84, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s4(self.c3(self.s2(self.c1(x))))
        return self.output(self.f6(self.c5(x)))


LENET_OVERRIDES = {
    "Subsampling": {"leaf": True, "label": "Subsampling", "category": "resample"},
    "GaussianConnections": {"leaf": True, "label": "Gaussian connections", "category": "linear"},
}


# --------------------------------------------------------------------------- YOLO (2016)


def _conv(inputs: int, outputs: int, kernel: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2), nn.LeakyReLU(0.1)]


class Reduce(nn.Sequential):
    """YOLO's 1 × 1 reduction followed by a 3 × 3 convolution (Fig. 3: "}×4", "}×2")."""

    def __init__(self, channels: int, reduced: int) -> None:
        super().__init__(*_conv(channels, reduced, 1), *_conv(reduced, channels, 3))


class YOLO(nn.Module):
    """YOLO v1 (Redmon, Divvala, Girshick & Farhadi, CVPR 2016) as in Fig. 3: 24 convolutional
    layers in six blocks (the figure's columns) and 2 fully connected layers, leaky ReLU, on a
    448 × 448 image; the output is the 7 × 7 × 30 grid of box and class predictions."""

    def __init__(self, grid: int = 7, boxes: int = 2, classes: int = 20) -> None:
        super().__init__()
        self.grid, self.depth = grid, boxes * 5 + classes
        self.block1 = nn.Sequential(*_conv(3, 64, 7, 2), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(*_conv(64, 192, 3), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(
            *_conv(192, 128, 1),
            *_conv(128, 256, 3),
            *_conv(256, 256, 1),
            *_conv(256, 512, 3),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            *(Reduce(512, 256) for _ in range(4)),
            *_conv(512, 512, 1),
            *_conv(512, 1024, 3),
            nn.MaxPool2d(2, 2),
        )
        self.block5 = nn.Sequential(
            *(Reduce(1024, 512) for _ in range(2)),
            *_conv(1024, 1024, 3),
            *_conv(1024, 1024, 3, 2),
        )
        self.block6 = nn.Sequential(*_conv(1024, 1024, 3), *_conv(1024, 1024, 3))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid * grid, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid * grid * self.depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in (self.block1, self.block2, self.block3, self.block4, self.block5):
            x = block(x)
        x = self.head(self.block6(x))
        return x.view(-1, self.grid, self.grid, self.depth)
