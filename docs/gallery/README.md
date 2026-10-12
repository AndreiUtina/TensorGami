# Gallery (preview)

Every figure on this page is drawn **automatically** from a PyTorch model: the model is traced
(step 6), simplified to the layer level (step 7) and laid out (step 8). Nothing is positioned
by hand. The models are the small reference implementations in
[`tests/models/`](../../tests/models/). Rebuild with
`uv run python docs/gallery/build_gallery.py`.

## CNN with residual blocks: layer level

The two identical residual blocks fold into one frame marked ×2. The skip connection enters
through a port and runs above the block's main path. Activations are folded into the layer
before them, and shapes appear only where they change.

![CNN](cnn.svg)

## The same CNN: stage level (L1)

![CNN stages](cnn-stages.svg)

## LSTM classifier

The fused `nn.LSTM` (2 layers) is one box; `view="unrolled"` draws it as a grid: a column
per time step, a row per layer, the per-step inputs x_t below and state passed along each row.
The 5-step sequence shows steps 1–3, "⋯" and step 5.

![LSTM](lstm.svg)

![LSTM, unrolled](lstm-unrolled.svg)

## Conditional GAN: generator and discriminator

![cGAN generator](cgan-generator.svg)

![cGAN discriminator](cgan-discriminator.svg)

## VAE

The noise ε is a dashed source with a dotted arrow, kept separate from μ (the
reparameterisation trick).

![VAE](vae.svg)

## ConvLSTM encoder

The cells are called in a Python loop over time, so `view="auto"` unrolls them: the frames
x₁…x₃ enter the first layer, h and c pass from step to step (green), and the last state of the
top layer feeds the 1×1 convolution.

![ConvLSTM, unrolled](convlstm.svg)

With `view="block"` each layer is one cell, drawn once with its recurrent h/c state loops
(■ = one-step delay) and marked ×3 for the three time steps.

![ConvLSTM, folded](convlstm-folded.svg)

With `view="cell"`, the cell itself, as its gates: the LSTM is recognised from the traced
operations, and its gates are coloured as the convolution that computes them.

![ConvLSTM cell](convlstm-cell.svg)

The fused `nn.LSTM` of the LSTM classifier, the same way:

![LSTM cell](lstm-cell.svg)

## pix2pix: U-Net generator and PatchGAN discriminator

The generator's skip connections nest inside each other, so it is drawn as a U: one row per
level, encoder on the left, decoder on the right, each skip a short arrow into its concat (‖),
the bottleneck below. Each level's in-place LeakyReLU is folded into the layer before it, and
the decoder's ReLU into the concat. (Small sizes here: 5 levels, 32×32 images.)

![pix2pix generator](pix2pix-generator.svg)

![pix2pix discriminator](pix2pix-discriminator.svg)

## Transformer encoder: top to bottom

![Transformer](transformer.svg)

## AlexNet

From [`examples/alexnet.py`](../../examples/alexnet.py): torchvision's AlexNet traced on one
224×224 image, at the layer level, top to bottom, and at the stage level. The collapsed
blocks are named after their attributes and coloured by their contents.

![AlexNet](alexnet.svg)

![AlexNet, stage level](alexnet-stages.svg)

<img src="alexnet-vertical.svg" alt="AlexNet, top to bottom" width="320">

With `view="volume"`, every layer is drawn as the tensor it produces, C × H × W, as in the
original paper; the dense layers are bars.

![AlexNet, volume view](alexnet-volume.svg)

## Origami style

The same models drawn with `style="origami"`: upright cards, left → right (see [the style
page](../style/origami.md)).

![CNN, origami](origami/cnn.svg)

![ConvLSTM, origami](origami/convlstm.svg)

![VAE, origami](origami/vae.svg)

![cGAN generator, origami](origami/cgan-generator.svg)

<img src="origami/transformer.svg" alt="Transformer, origami" width="360">

![AlexNet, origami](origami/alexnet.svg)

## Origami-extended style

With `style="origami-extended"`: flat single-line cards, top → bottom.

<img src="origami-extended/cnn.svg" alt="CNN, origami-extended" width="420">
<img src="origami-extended/vae.svg" alt="VAE, origami-extended" width="420">

<img src="origami-extended/convlstm.svg" alt="ConvLSTM, origami-extended" width="360">
<img src="origami-extended/transformer.svg" alt="Transformer, origami-extended" width="360">
