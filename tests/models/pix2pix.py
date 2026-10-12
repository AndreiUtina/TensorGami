"""pix2pix (Isola et al. 2017): the U-Net generator and the PatchGAN discriminator, as in the
authors' official implementation (junyanz/pytorch-CycleGAN-and-pix2pix, models/networks.py,
BSD licence); the same code as ``examples/pix2pix.ipynb``. The generator's skips nest inside
each other (a U-Net), and each level opens with a LeakyReLU applied in place.
"""

from __future__ import annotations

import torch
from torch import nn


class UnetSkipConnectionBlock(nn.Module):
    """One U-Net level: down-sample, the inner levels (submodule), up-sample, skip connection.

    |-- down --|-- submodule --|-- up --|
    x -------------------------------- concat --> output
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)  # skip connection


class UnetGenerator(nn.Module):
    """U-Net generator, built from the innermost level outwards; ``num_downs=8`` maps a
    256×256 image to a 1×1 bottleneck and back."""

    def __init__(
        self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False
    ):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True
        )
        for _ in range(num_downs - 5):  # intermediate levels with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator; ``n_layers=3`` gives a 70×70 receptive field per output value."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw, padw = 4, 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),  # 1-channel map
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConditionalDiscriminator(nn.Module):
    """pix2pix judges an image B together with its input image A: the training code calls
    ``netD(torch.cat((real_A, fake_B), 1))``. This wrapper makes the two images two inputs, so a
    figure can show where each comes from."""

    def __init__(self, netD):
        super().__init__()
        self.netD = netD

    def forward(self, A, B):
        return self.netD(torch.cat([A, B], 1))
