"""The volume view: CNN layers as tensor volumes (step 11c; no torch needed)."""

from __future__ import annotations

from ir_fixtures import residual_cnn, unet
from layout_checks import check_scene

from tensorgami.ir import VOLUME
from tensorgami.layout import LayoutOptions
from tensorgami.layout.volume import volume_geometry, volume_label
from tensorgami.style import Shape
from tensorgami.transforms import simplify
from tensorgami.transforms.volume import MarkVolumes
from tensorgami.views import get_view
from tensorgami.vocabulary import OpCategory as C


def test_image_tensors_become_volumes_and_vectors_bars():
    g = MarkVolumes().apply(simplify(residual_cnn(), 2))
    marked = {n.op: n.params.get(VOLUME) for n in g.nodes}
    assert marked["input"] == (3, 32, 32)  # the input image is a volume too
    assert marked["Conv2d"] == (16, 32, 32)
    assert marked["Linear"] == (10,)  # a flat vector
    assert marked["add"] is None and marked["output"] is None  # merges and outputs stay


def test_volumes_grow_with_their_dimensions():
    small, large = volume_geometry((64, 13, 13)), volume_geometry((64, 55, 55))
    assert large[1] > small[1] and large[2] > small[2]  # taller, deeper
    assert volume_geometry((256, 13, 13))[0] > small[0]  # more channels: thicker
    assert volume_geometry((4096,))[0] < small[0]  # a vector: a thin bar
    assert volume_label((64, 55, 55)) == "64×55×55"


def draw(graph, theme, **options):
    return get_view("volume").draw(
        graph, theme, level=2, options=LayoutOptions(wrap_width=None, **options)
    )


def test_volume_view_keeps_the_invariants(theme):
    shown, scene = draw(residual_cnn(), theme)
    check_scene(scene, shown, theme)


def test_volumes_are_cuboids_named_with_their_shape(theme):
    _, scene = draw(residual_cnn(), theme)
    volumes = [n for n in scene.nodes if n.shape is Shape.CUBOID]
    assert {n.label for n in volumes} >= {"image", "Conv 3×3", "MaxPool", "Linear"}
    conv = next(n for n in volumes if n.label == "Conv 3×3")
    assert conv.detail == "16×32×32" and conv.depth > 0
    assert not [e for e in scene.edges if e.label]  # the shapes are on the volumes
    assert not [n for n in scene.nodes if n.label == "BatchNorm"]  # folded into the conv


def test_wires_between_volumes_stay_level(theme):
    _, scene = draw(residual_cnn(), theme)
    faces = {n.id: n.box for n in scene.nodes}
    for e in scene.edges:
        if e.source in faces and e.target in faces and not e.via:
            assert faces[e.source].cy == faces[e.target].cy


def test_volume_view_runs_left_to_right(theme):
    _, scene = draw(residual_cnn(), theme, direction="TB")
    assert scene.width > scene.height
    _, merged = draw(unet(3), theme)
    assert [n for n in merged.nodes if n.category is C.MERGE and n.shape is None]
