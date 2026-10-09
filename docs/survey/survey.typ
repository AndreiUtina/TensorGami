// Tensorgami — Step 1 literature survey.
// Build: uv run python docs/survey/build.py   (figures + PDF)

#let accent = rgb("#2b6cb0")
#let muted = rgb("#4a5568")
#let sans = ("Helvetica Neue", "Helvetica", "Arial", "Libertinus Serif")

#set document(
  title: "Drawing Deep-Learning Architectures: A Survey of Diagram Styles, Conventions and Tools",
  author: "Tensorgami project",
)
#set page(
  paper: "a4",
  margin: (x: 1.85cm, top: 1.9cm, bottom: 1.9cm),
  numbering: "1",
  number-align: center,
  footer: context {
    set text(size: 7.5pt, fill: muted, font: sans)
    grid(
      columns: (1fr, auto),
      [Tensorgami · Step 1 · Literature survey],
      counter(page).display("1 / 1", both: true),
    )
  },
)
#set text(font: "Libertinus Serif", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.85em)
#set heading(numbering: "1.1")
#show heading: set text(font: sans, fill: accent)
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 1): it => block(above: 1.15em, below: 0.6em, it)
#show heading.where(level: 2): set text(size: 10.5pt)
#show heading.where(level: 2): it => block(above: 0.95em, below: 0.5em, it)
#show figure.caption: set text(size: 8.5pt)
#show figure: set block(spacing: 1.0em)
#set figure(gap: 0.55em, placement: auto)
#show table: set text(size: 7.8pt)
#set table(
  stroke: (x, y) => (
    top: if y == 0 { 0.8pt } else if y == 1 { 0.5pt } else { 0.25pt + luma(200) },
    bottom: 0.8pt,
  ),
  inset: (x: 4pt, y: 3.2pt),
  align: left + horizon,
)
#show table.cell.where(y: 0): set text(weight: "bold", font: sans, size: 7.6pt)
#show link: set text(fill: accent)

#let req(id, body) = block(spacing: 0.45em)[*#id* #h(0.3em) #body]

// ------------------------------------------------------------------ title block
#block(width: 100%, inset: (bottom: 4pt))[
  #set par(justify: false)
  #text(font: sans, size: 8.5pt, fill: muted, tracking: 0.06em)[TENSORGAMI · DESIGN STEP 1 · LITERATURE SURVEY]
  #v(2pt)
  #text(font: sans, size: 17pt, weight: "bold")[Drawing Deep-Learning Architectures]
  #v(-6pt)
  #text(font: sans, size: 12pt, fill: muted)[A survey of diagram styles, conventions and tools — and what they imply for Tensorgami]
  #v(2pt)
  #text(size: 9pt, fill: muted)[October 2026]
]

#block(
  fill: rgb("#f4f7fb"),
  inset: 9pt,
  radius: 3pt,
  width: 100%,
)[
  #set text(size: 9.2pt)
  *Abstract.* Architecture diagrams are the most common figure in deep-learning papers, yet
  no notation standard exists: most figures are drawn by hand shortly before publication,
  and automatic tools mostly target debugging rather than communication. This report surveys
  how deep-learning models are drawn. It covers (i) a taxonomy of nine static diagram styles
  plus interactive explainers, (ii) the canonical depictions of the main architecture families
  (MLPs, CNNs, residual and U-shaped networks, RNN/LSTM/GRU, ConvLSTM, autoencoders and VAEs,
  GANs, Transformers), (iii) the tool landscape, from operator-graph viewers (TensorBoard,
  Netron, Model Explorer) to publication-oriented generators (NN-SVG, PlotNeuralNet, Net2Vis),
  and (iv) the available empirical evidence on what makes such diagrams effective. It closes
  with ten design requirements for Tensorgami, a locally installed Python tool for drawing the
  architecture and information flow of ML models.
]

= Introduction

Diagrams appear in almost every paper that introduces a new neural system
@marshall2020. Readers use them for a holistic overview of the system, and some use them to
decide whether a paper is relevant at all, before reading the text @marshall2025. Outside of
publishing, architecture drawings help developers *understand and debug* a model (is the skip
connection where I think it is? which tensor shape enters the decoder?) and help *teach*
the mechanics of a cell or block @hohman2019. These three purposes (communication,
comprehension, teaching) call for different levels of detail and fidelity. That difference
is a recurring theme in this survey.

Despite this ubiquity, Marshall et al. note that there are currently no established,
evidence-based conventions for presenting neural-network diagrams @marshall2025. The
Net2Vis authors observed that most figures are hand-crafted shortly before publication, which
produces inconsistent conventions, wasted effort and occasional errors. They also observed
that the automatic tools of the time were designed for debugging networks rather than for
producing publication figures @bauerle2021.

*Scope and method.* We reviewed (i) the architecture papers whose figures established today's
visual conventions, (ii) the drawing tools for these models and the visual grammars they
implement, and (iii) the empirical and HCI work on diagram effectiveness. We focus on
*static* depictions of architecture and information flow. Visualisations of learned content
(saliency maps, feature visualisation, activation atlases) are out of scope, as is most of
the training-dynamics tooling. Figures 1–3 are schematic re-drawings made for this report,
not reproductions of the original artwork.

= A taxonomy of diagram styles

Across papers and tools we identify nine recurring static styles (@fig-styles,
@tab-styles). They differ along three axes: _what a glyph stands for_ (a neuron, a tensor,
an operator, an axis), _how much of the computation is shown_ (from system level down to
individual gates), and _which dimension the layout encodes_ (data flow, time, resolution).

#figure(
  image("figures/fig1-styles.svg", width: 100%),
  caption: [Nine static styles for drawing neural networks (schematic re-drawings). (a) neuron
    graph; (b) LeNet-style feature-map planes; (c) AlexNet/PlotNeuralNet tensor volumes;
    (d) layer-block flowchart with an "N×" repeat; (e) operator dataflow graph as in Netron;
    (f) gate-level LSTM cell after Olah; (g) folded vs. unrolled recurrence with a delay marker;
    (h) encoder–latent–decoder hourglass; (i) tensor-axis circuit diagram after Abbott.],
) <fig-styles>

*S1 Neuron graph* (@fig-styles a). Units are circles and weights are lines. This is the
textbook picture of an MLP and the basis of the Asimov Institute's _Neural Network Zoo_,
which compares dozens of architectures by colour-coding cell types @vanveen2016. NN-SVG
generates it parametrically @lenail2019. It shows connectivity patterns intuitively (dense,
sparse, recurrent, stochastic units), but it stops working beyond a few dozen units. Its
explicit unit counts also suggest a precision that is rarely intended, which Marshall et
al. warn against @marshall2025.

*S2 Feature-map planes* (@fig-styles b). Starting with LeNet-5 @lecun1998, convolutional
layers are drawn as stacks of 2-D maps whose size shows spatial resolution and whose count
shows channels. The style suits small CNNs and teaching, but becomes unreadable once a
layer has hundreds of channels.

*S3 Tensor volumes* (@fig-styles c). The AlexNet figure @krizhevsky2012 made the 2.5-D
cuboid the emblem of CNNs. Height and depth stand for spatial size and width for channel
count, and small inset cuboids mark kernels. PlotNeuralNet (LaTeX/TikZ) @iqbal2018, NN-SVG's
"AlexNet" mode @lenail2019 and visualkeras' layered view @gavrikov2020 produce this style.
Net2Vis flattens it to 2-D glyphs whose height encodes spatial resolution and whose width
encodes the number of features @bauerle2021. Marshall et al. cite the 3-D format as *the*
available convention for CNNs @marshall2025. Its weaknesses are perspective, dimensions that are rarely to scale (they are
usually log-scaled or clipped), and poor support for branching topologies.

*S4 Layer-block flowchart* (@fig-styles d). Each layer or block is a (rounded) rectangle and
arrows carry tensors. This is the dominant style in modern papers: residual and Inception
blocks @he2016 @szegedy2015 and the Transformer @vaswani2017 are drawn this way. The
Transformer figure shows the style's typical devices: fill colour by sub-layer type, a frame
with an "N×" label instead of drawing repeats, and residual arrows that bypass blocks.
Keras' `plot_model` @kerasplot and torchview @kurttutan2022 generate it via Graphviz
@gansner2000. It scales through hierarchy and handles branches and merges, but it loses
tensor sizes unless edges are annotated.

*S5 Operator dataflow graph* (@fig-styles e). Every primitive operation becomes a node and
every tensor an edge, often with attribute rows and shape labels. Examples are the
TensorBoard graph @wongsuphasawat2018, Netron @roeder2017, PyTorchViz's autograd graph
@zagoruyko2018, TorchLens @taylor2023 and Google's Model Explorer @google2024. These graphs
are exact and fully automatic, but real models produce thousands of nodes, so every tool in
this class must collapse hierarchy and hide clutter to stay legible (§5).

*S6 Cell internals* (@fig-styles f). Gated recurrent cells are explained at gate level.
Olah's LSTM diagrams @olah2015 established the vocabulary: the cell state is a horizontal
"conveyor belt", learned layers ($sigma$, $tanh$) are boxes and point-wise operations
($times$, $+$) are circles. The same grammar is reused for GRUs @cho2014 and for many
attention variants. This style explains the mathematics of a cell, but it applies only at
the lowest level of detail.

*S7 Folded vs. unrolled recurrence* (@fig-styles g). A recurrent network is drawn either as
a circuit with a self-loop (a black square marks the one-step delay) or unfolded into a
chain over time @goodfellow2016. Unrolling makes time an explicit layout axis, usually
horizontal, with network depth stacked vertically. Sequence-to-sequence models
@sutskever2014 and attention-based decoders @bahdanau2015 are almost always drawn unrolled.

*S8 Encoder–latent–decoder* (@fig-styles h). Trapezoids that narrow and then widen show
compression into, and expansion out of, a latent code: the hourglass of autoencoders and VAEs
@kingma2014. The U-Net @ronneberger2015 bends the hourglass into a "U", in which _vertical
position encodes resolution_ and horizontal arrows carry the skip connections.

*S9 Formal tensor diagrams* (@fig-styles i). Neural circuit diagrams @abbott2024 adapt string
or tensor-network diagrams: wires are tensor axes, boxes are operations, and dashed lines
separate tuple elements. Broadcasting, contraction and copying become visible, and the
drawings are precise enough to derive an implementation. The cost is a learning curve and
density that general audiences find hard to read.

*Interactive explainers* complete the picture: CNN Explainer @wang2021, Bycroft's 3-D LLM
visualisation @bycroft2023 and Treescope's foldable model trees @johnson2024. They use hover,
animation and expand/collapse, which print cannot provide, but they show that _progressive
disclosure_ of detail is the key to scale.

#figure(
  table(
    columns: (auto, 1.2fr, 1.5fr, auto, 1.5fr),
    table.header([Style], [A glyph is…], [Encodes], [Scales to], [Typical use]),
    [S1 Neuron graph], [a unit; edges = weights], [connectivity pattern], [≈ 50 units], [teaching, small MLPs, taxonomies],
    [S2 Planes], [a feature map], [resolution, \#channels], [≈ 6 layers], [small CNNs, teaching],
    [S3 Volumes], [a tensor (cuboid)], [C × H × W, kernel size], [≈ 20 layers], [CNN papers],
    [S4 Block flowchart], [a layer / block], [op type (colour), topology], [100s via nesting], [most papers; generic DL],
    [S5 Operator graph], [a primitive op], [exact topology, shapes, params], [10⁴+ with collapse], [debugging, deployment],
    [S6 Cell internals], [a gate / point-wise op], [equations of one cell], [one cell], [LSTM/GRU/attention explanation],
    [S7 Unrolled time], [a cell at step _t_], [time × depth], [≈ 5 steps shown], [RNN, seq2seq, ConvLSTM],
    [S8 Hourglass / U], [an encoder / decoder stage], [compression, resolution levels], [≈ 5 levels], [AE, VAE, U-Net, diffusion],
    [S9 Tensor circuit], [a tensor axis (wire)], [contractions, broadcasting], [one block], [precise specification],
  ),
  caption: [Comparison of the static diagram styles. "Scales to" gives the approximate size at
    which the style stays legible on one page without aggregation.],
) <tab-styles>

A cross-cutting observation is the *duality of node semantics*. In S4, S5 and S6 nodes are
_operators_ and edges carry tensors. In S2, S3 and the U-Net figure, nodes are _tensors_
and the operations sit on the arrows; the U-Net legend, for example, assigns one arrow colour
per operation type @ronneberger2015. Both conventions are established, so a drawing tool
should be able to render either view from the same model description.

= Architecture families and their canonical depictions

Each family has settled on a dominant style and a set of elements that readers expect to see
(@fig-families, @tab-families). Marshall et al. recommend meeting those expectations, and
saying explicitly when an expected element such as pooling is absent @marshall2025.

#figure(
  image("figures/fig2-families.svg", width: 100%),
  caption: [Canonical depictions of architecture families (schematic re-drawings). (a) VAE
    with explicit reparameterisation $z = mu + sigma dot.o epsilon$ and loss terms drawn
    dashed; (b) GAN with the adversarial training loop; (c) U-Net, where vertical position
    encodes resolution; (d) ConvLSTM encoder–forecaster with 3-D states unrolled in time;
    (e) residual block with identity shortcut; (f) unrolled sequence-to-sequence model with
    attention.],
) <fig-families>

*General DL and MLPs.* Small MLPs are drawn as neuron graphs (S1). Anything larger is drawn
as a chain of blocks labelled with widths (e.g. "Linear 784→256, ReLU"). Activations are
usually merged into the label of the preceding layer rather than drawn as separate nodes.

*CNNs.* LeNet and AlexNet established planes and volumes (S2, S3). VGG-era papers moved to
tables and flat block chains, and ResNet and Inception made the block flowchart (S4)
standard because their key ideas are topological: the identity shortcut merging at "⊕"
(@fig-families e) and parallel branches joined by concatenation @he2016 @szegedy2015.
Readers expect to see kernel size, channel count, stride or pooling, and the type of each
merge (sum vs. concatenation). An unlabelled merge node is a common source of ambiguity.

*Encoder–decoder and U-Net.* U-Net's figure @ronneberger2015 combines tensor-centric
glyphs (feature-map bars labelled with channels and resolution) with operation-coded arrows
(convolution, copy-and-crop, max-pool, up-convolution). It is the archetype for
segmentation and for the denoising networks of diffusion models @ho2020.

*RNNs, seq2seq and attention.* Recurrent models are drawn unrolled (S7), with time on the
horizontal axis and stacked layers on the vertical axis. Bidirectional layers are drawn as
two rows with opposite arrows. Seq2seq figures @sutskever2014 separate the encoder and the
decoder chains. Attention @bahdanau2015 adds fan-in edges from all encoder states into a
context vector, and the edges are sometimes weighted by attention strength
(@fig-families f).

*LSTM and GRU.* At the macro level an LSTM is a single recurrent block annotated with its
hidden size. Explaining the cell requires the gate-level view (S6) @olah2015
@hochreiter1997 @cho2014, preferably next to its equations. A stable symbol set has emerged:
$sigma$ gates as boxes, point-wise $times$ and $+$ as circles, and the cell-state line on top.

*ConvLSTM.* Shi et al. @shi2015 replace the matrix products of the LSTM with convolutions,
so inputs, hidden states and cell states become 3-D tensors. Their figures therefore combine
_tensor volumes_ (S3) with _unrolled time_ (S7). In the encoding–forecasting network,
stacked ConvLSTM layers encode the input sequence, the forecaster's initial states are copied
from the encoder's last states, and the forecaster's states are concatenated and passed
through a 1×1 convolution to produce the predictions (@fig-families d). A faithful diagram
must therefore show three axes: time, layer depth and the shape of each state.

*Autoencoders and VAEs.* The hourglass (S8) is universal. VAE figures add the stochastic
layer: the encoder outputs $mu$ and $sigma$, a noise source $epsilon tilde cal(N)(0, I)$ enters
from outside, and $z = mu + sigma dot.o epsilon$ is formed by the reparameterisation trick
@kingma2014. The loss terms (reconstruction and KL) are drawn as training-only annotations
(@fig-families a). The sampling node is the element most often left implicit, and drawing it
explicitly removes the most common misunderstanding of the model.

*GANs.* Two networks are drawn side by side. A real-data source and the generator both feed
the discriminator, and a dashed loop carries the adversarial gradient back @goodfellow2014.
The main design problem is to separate _training-time_ flow from _inference-time_ flow, a
distinction the VAE and diffusion models share.

*Transformers.* The original figure @vaswani2017 is a model of multi-level depiction. The
macro view shows encoder and decoder stacks, each a frame marked "N×" containing coloured
sub-layers (attention, feed-forward, Add & Norm) with residual arrows. Separate "zoom" figures
then show multi-head and scaled dot-product attention. Graph-level tools need hierarchical
collapsing to show Transformers at all: Model Explorer was built because large
Transformer-based models overwhelm conventional graph visualisers @google2024.

#figure(
  table(
    columns: (auto, 1.1fr, 1.9fr, 1.5fr),
    table.header([Family], [Dominant style], [Elements readers expect], [Frequent pitfalls]),
    [MLP], [S1 / S4], [layer widths, activation], [misleading unit counts],
    [CNN], [S3 / S4 / S2], [kernel, channels, resolution, pooling/stride], [volumes not to scale; missing pooling],
    [ResNet, Inception], [S4], [shortcut, merge type (⊕ vs. concat), repeats], [unlabelled merges],
    [U-Net, AE], [S8], [resolution levels, skip copies], [one arrow style for many ops],
    [RNN, seq2seq], [S7], [time axis, state passing, direction], [folded loop ambiguity],
    [LSTM, GRU], [S6 (+ S4 macro)], [gates, cell state, point-wise ops], [ad-hoc symbols],
    [ConvLSTM], [S7 × S3], [3-D states, layer stack, state copy], [visual overload],
    [VAE], [S8 + stochastic node], [$mu$, $sigma$, $z$, $epsilon$, loss terms], [sampling left implicit],
    [GAN], [S4 loop], [real/fake sources, adversarial loss], [training vs. inference mixed],
    [Transformer], [S4 + zooms], [N× stacks, residual + norm, attention detail], [no tensor shapes],
  ),
  caption: [Architecture families, their dominant styles (cf. @tab-styles) and the
    elements a reader expects to find.],
) <tab-families>

= Tool landscape

@tab-tools summarises the software that produces such diagrams. Two clusters stand out.
*Automatic, engineering-oriented* tools trace a real model and draw an exact operator or
module graph. The TensorBoard graph, Netron, Model Explorer, PyTorchViz, torchview,
HiddenLayer and TorchLens belong here. Their output is correct and complete, but it is
styled for inspection, not for papers. *Publication-oriented* tools produce attractive
figures but are manual or parametric (NN-SVG, PlotNeuralNet) or restricted to mostly
sequential CNNs (visualkeras, Net2Vis). In practice many paper figures are still drawn by
hand in TikZ, draw.io, PowerPoint or Figma.

#figure(
  table(
    columns: (auto, 1.2fr, auto, 1.5fr, 1.6fr),
    table.header([Tool], [Input], [Style], [Layout / output], [Notable features / limits]),
    [TensorBoard Graph @wongsuphasawat2018], [TF graphs], [S5], [layered, interactive], [namespace clustering, auxiliary-node extraction, repeat detection],
    [Netron @roeder2017], [ONNX, TFLite, Keras, PyTorch, …], [S5], [interactive app], [broad format support; operator level only],
    [Model Explorer @google2024], [JAX, PyTorch, TF, TFLite], [S5], [hierarchical, GPU-rendered], [10⁴+ nodes, expand/collapse, overlays],
    [TorchLens @taylor2023], [PyTorch], [S5], [Graphviz], [all ops incl. non-module calls; activations],
    [torchview @kurttutan2022], [PyTorch], [S4/S5], [Graphviz], [module depth control, shapes, `meta` device],
    [PyTorchViz @zagoruyko2018], [PyTorch autograd], [S5], [Graphviz], [backward graph; hard to read],
    [HiddenLayer @abdulla2018], [PyTorch, TF, Keras], [S4], [Graphviz], [rule-based folding of common op patterns],
    [Keras `plot_model` @kerasplot], [Keras], [S4], [Graphviz], [shapes on nodes; few styling options],
    [visualkeras @gavrikov2020], [Keras], [S3/S4], [raster image], [layered 3-D view; mostly sequential models],
    [Net2Vis @bauerle2021], [Keras-style code], [S3 (2-D)], [SVG/PDF, web], [visual grammar, aggregation, legend],
    [NN-SVG @lenail2019], [web form], [S1/S2/S3], [SVG], [parametric, not from a model],
    [PlotNeuralNet @iqbal2018], [Python → TikZ], [S3], [LaTeX/PDF], [high quality; manual, needs LaTeX],
    [Treescope @johnson2024], [JAX/PyTorch objects], [tree], [HTML (notebook)], [foldable model structure, arrays],
  ),
  caption: [Representative tools for drawing neural-network architectures.],
) <tab-tools>

This leaves a gap for a tool that extracts structure automatically and renders it with
publication-grade conventions, and that also handles the non-CNN families: recurrent
unrolling, ConvLSTM state tensors, stochastic latent nodes and training-only flows. Most
generators also depend on Graphviz or LaTeX, which complicates local installation.

= What makes an architecture diagram effective

*Corpus evidence (Net2Vis).* Bäuerle et al. collected 751 architecture figures from
ICCV/CVPR 2013–2019 and interviewed authors of highly cited papers @bauerle2021. They
derived a visual grammar with four parts: _visual encoding_ (glyph shape encodes tensor
dimensions), _layout_ (data flows left to right), _layer aggregation_ (repeated sequences
replaced by one aggregate glyph) and _automatic legend generation_. Their user study found
improvements in both creating and reading the figures.

*Making dataflow graphs legible (TensorBoard).* Wongsuphasawat et al. showed that raw
operator graphs become usable only after a series of graph transformations @wongsuphasawat2018.
These transformations move non-critical, high-degree nodes out of the main layout, build an
expandable cluster hierarchy from the code's namespaces, and detect and highlight repeated
structures. The transformed graph can then be laid out with standard layered drawing
@sugiyama1981. Model Explorer follows the same principle, letting users descend layer by
layer through an expandable hierarchy @google2024.

*Reader studies (Marshall et al.).* The only evidence-based guidance specific to
neural-network diagrams comes from interviews, a mixed-methods experiment and a corpus
analysis @marshall2025. All interviewees said consistency matters, and 10 of 12 considered
hyperparameters unnecessary in a system diagram. Papers whose diagrams followed the resulting
guidelines received more citations. The twelve guidelines are: (A) use conventional
graphical objects; (B) use one arrow type for information flow and reserve other arrow types
for fundamentally different uses; (C) use precision with care, since drawing four circles
implies "four"; (D) show the input and output of the whole system; (E) consider a single
consistent example; (F) use visual encodings (proximity, alignment) meaningfully; (G) make
navigation easy with labels and linear alignment; (H) do not use colour for aesthetics,
only for grouping; (I) use available conventions such as the 3-D CNN format with numeric
filter sizes; (J) show what readers expect, or say when it is absent; (K) be specific
("BERT", not "embedding"); (L) make the diagram usable without the text.

*Who is looking, and why.* Hohman et al.'s interrogative survey @hohman2019 separates model
developers, model users and non-experts, and separates purposes such as interpretability,
debugging and comparison, and teaching. Each audience needs a different level of detail.
This supports generating _several_ views of one model rather than a single
"correct" diagram.

*Colour and accessibility.* Colour should encode categories redundantly with labels or
shapes, and palettes should stay distinguishable under colour-vision deficiency @okabe2008.
Guideline H reaches the same conclusion from the communication side.

= Implications: design requirements for Tensorgami

The survey translates into ten requirements. They define the drawing style baseline (step 2)
and the architecture of the later phases.

#req[R1][*Multiple levels of detail from one model*: L0 system, L1 stages, L2 layers, L3
operators and L4 cell internals (@fig-levels), with deterministic collapse and expand.]
#req[R2][*Both node semantics*: an operator-centric block view (S4/S5) and a tensor-centric
view (S2/S3/S8), rendered from the same intermediate representation.]
#req[R3][*Semantic, consistent encoding*: fill colour only for the operation category; shape
for the kind of element (data, operation, merge, stochastic, group); one arrow style for
data flow, with distinct styles reserved for recurrent state, sampling and training-only
signals (guidelines B, H).]
#req[R4][*Tensor shapes as first-class annotations* (C×H×W on edges, batch omitted) and
optionally to-scale (log) glyph sizes in the volume view.]
#req[R5][*Aggregation of repeated structures* into "N×" groups, plus folding of trivial
operations (activation into the previous layer, reshape, dropout).]
#req[R6][*Automatic legend* listing only the categories that appear.]
#req[R7][*Time-aware layouts*: folded and unrolled recurrence and a time × depth grid for
RNN, seq2seq and ConvLSTM.]
#req[R8][*Explicit stochastic and training-time elements*: sampling nodes, external noise
inputs, and dashed loss and gradient paths (VAE, GAN, diffusion).]
#req[R9][*Two input routes into one IR*: automatic extraction (PyTorch first, then ONNX
and Keras) and a hand-written spec for models that cannot be imported.]
#req[R10][*Deterministic vector output* (SVG first, then PDF/PNG) with themeable styles and
no heavyweight system dependencies (no Graphviz or LaTeX), so the tool installs locally
with `pip`/`uv` alone.]

#figure(
  image("figures/fig3-levels.svg", width: 100%),
  caption: [The same residual network at four levels of detail. L4 (cell internals,
    @fig-styles f) completes the hierarchy for recurrent cells. Tensorgami should produce
    any of these views from one model description (R1).],
) <fig-levels>

= Conclusion

Drawing deep-learning models is a mature practice without a standard. Nine static styles
coexist, each tied to the architecture families and audiences it serves best. The evidence
that does exist (the Net2Vis corpus, the TensorBoard design study and Marshall et al.'s
reader studies) agrees on a short list of principles: left-to-right flow, semantic rather
than decorative colour, a single arrow type for data, aggregation of repeated structure,
explicit tensor shapes, automatic legends and several levels of detail. No current tool
combines automatic extraction, publication-quality styling and coverage of recurrent,
convolutional-recurrent and generative families. Tensorgami's baseline style (step 2)
encodes these principles directly.

#v(0.4em)
#set text(size: 8.2pt)
#bibliography("refs.bib", title: "References", style: "ieee")
