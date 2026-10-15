"""Several models in one figure, links, inputs, losses and notes (step 11e; no torch)."""

from __future__ import annotations

import pytest
from layout_checks import check_scene

from tensorgami.ir import CAPTION, MODELS, TRAINS, GraphBuilder
from tensorgami.layout import LayoutOptions, layout
from tensorgami.transforms import simplify
from tensorgami.transforms.annotate import (
    UnknownReference,
    add_input,
    add_loss,
    add_note,
    compose,
    link,
    resolve,
)
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C


def generator():
    b = GraphBuilder("G", root_type="Generator")
    z = b.input((1, 16), label="z")
    fc = b.group("fc", "Sequential", module="fc")
    lin = b.node("Linear", category=C.LINEAR, module="fc.0", group=fc, outputs=[(1, 64)])
    b.edge(z, lin)
    out = b.node("Linear", category=C.LINEAR, module="out", outputs=[(1, 64)])
    b.edge(lin, out)
    b.output(out)
    return b.build()


def discriminator():
    b = GraphBuilder("D", root_type="Discriminator")
    image = b.input((1, 64), label="image")
    lin = b.node("Linear", category=C.LINEAR, module="score", outputs=[(1, 1)])
    b.edge(image, lin)
    b.output(lin)
    return b.build()


def gan():
    g = compose({"generator": generator(), "discriminator": discriminator()})
    g = add_input(g, "real image", to="discriminator.image")
    g = link(g, "generator", "discriminator.image", "fake image")
    return add_loss(g, "adversarial", ["discriminator"], trains=["generator", "discriminator"])


def test_compose_frames_each_model_and_prefixes_its_names():
    g = compose({"generator": generator(), "discriminator": discriminator()})
    assert g.metadata[MODELS] == "generator,discriminator"
    tops = {x.id: x.label for x in g.groups if x.parent == ""}
    assert tops == {"generator": "generator", "discriminator": "discriminator"}
    assert {n.module for n in g.nodes if n.module} == {
        "generator.fc.0",
        "generator.out",
        "discriminator.score",
    }
    assert all(n.group == "" for n in g.nodes if n.is_input or n.is_output)  # outside the frames


def test_references_name_inputs_outputs_modules_and_models():
    g = compose({"generator": generator(), "discriminator": discriminator()})
    assert g.node(resolve(g, "generator.z")).label == "z"
    assert g.node(resolve(g, "generator")).is_output  # a model's name: its output
    assert g.node(resolve(g, "generator.fc")).module == "generator.fc.0"  # a module: its last node
    assert g.node(resolve(g, "discriminator.score")).module == "discriminator.score"
    with pytest.raises(UnknownReference, match="nothing called 'generator.nope'; try one of: z"):
        resolve(g, "generator.nope")


def test_a_link_feeds_one_model_from_another():
    g = gan()
    labels = {n.label for n in g.nodes}
    assert "image" not in labels and {"fake image", "real image"} <= labels
    fake = next(n for n in g.nodes if n.label == "fake image")
    assert fake.category is C.TENSOR and not fake.is_output  # now inside the figure
    score = next(n for n in g.nodes if n.module == "discriminator.score")
    feeding = {g.node(e.source).label for e in g.in_edges(score.id)}
    assert feeding == {"fake image", "real image"}
    with pytest.raises(UnknownReference, match="ends at a model input"):
        link(g, "generator.z", "discriminator.score")


def test_a_loss_compares_tensors_and_trains_models():
    g = gan()
    loss = next(n for n in g.nodes if n.category is C.LOSS)
    assert loss.label == "adversarial" and loss.params[TRAINS] == ("generator", "discriminator")
    (into,) = g.in_edges(loss.id)
    assert g.node(into.source).module == "discriminator.score"  # behind the output pill
    assert any(n.is_output for n in g.nodes)  # which stays an output
    single = add_loss(generator(), "l2", ["output"], trains=["out"])  # a layer, not a frame
    edge = next(e for e in single.edges if e.kind is EdgeKind.TRAINING)
    assert single.node(edge.target).module == "out"


def test_notes_set_a_second_line():
    g = add_note(generator(), "a projection", at="out")
    assert next(n for n in g.nodes if n.module == "out").params[CAPTION] == "a projection"


def test_levels_count_from_the_models():
    boxes = simplify(gan(), 0)  # each model one box, not the whole figure one box
    for model in ("generator", "discriminator"):
        assert len(boxes.nodes_under(model)) == 1
    assert {n.label for n in boxes.nodes if n.category is C.LOSS} == {"adversarial"}
    assert {n.label for n in boxes.nodes if n.is_input} == {"z", "real image"}


@pytest.mark.parametrize("direction", ["LR", "TB"])
def test_training_arrows_rise_into_the_frames(theme, direction):
    g = simplify(gan(), 2)
    scene = layout(g, theme, LayoutOptions(direction=direction, wrap_width=None, wrap_height=None))
    check_scene(scene, g, theme)
    frames = {f.id: f.box for f in scene.groups}  # a one-layer model has no frame: its box
    frames.setdefault(
        "discriminator", next(n.box for n in scene.nodes if n.id == "discriminator/n1")
    )
    arrows = [e for e in scene.edges if e.kind is EdgeKind.TRAINING and isinstance(e.source, str)]
    assert len(arrows) == 2
    for arrow, model in zip(arrows, ("generator", "discriminator"), strict=True):
        end = arrow.target
        box = frames[model]
        from_below = box.x <= end.x <= box.right and abs(end.y - box.bottom) < 0.5
        from_above = box.x <= end.x <= box.right and abs(end.y - box.y) < 0.5  # over the top
        from_left = abs(end.x - box.x) < 0.5 and box.y <= end.y <= box.bottom  # all blocked
        assert from_below or from_above or from_left


def test_arrows_from_several_losses_are_named(theme):
    g = add_loss(gan(), "l1", ["generator", "real image"], trains=["generator"])
    scene = layout(simplify(g, 2), theme, LayoutOptions(wrap_width=None, wrap_height=None))
    into = [e for e in scene.edges if e.kind is EdgeKind.TRAINING and isinstance(e.source, str)]
    named = {e.label for e in into if e.label}
    assert named == {"adversarial", "l1"}  # the generator gets one from each loss
    spots = [e.label_at for e in into if e.label]
    assert len({(round(p.x), round(p.y)) for p in spots}) == len(spots)  # apart


def test_the_loss_loops_back_over_the_top_into_the_model_beside_it(theme):
    """GAN loss → discriminator: out of the loss's right side, up past the output, left
    above the discriminator and down into it (counterclockwise); the generator, beyond the
    discriminator, is reached from underneath."""
    g = simplify(gan(), 2)
    scene = layout(g, theme, LayoutOptions(wrap_width=None, wrap_height=None))
    loss = next(n.box for n in scene.nodes if n.id.startswith("loss:"))
    arrows = {
        e.via[-1].x: e
        for e in scene.edges
        if e.kind is EdgeKind.TRAINING and e.source == "loss:adversarial"
    }
    into = {}
    for e in arrows.values():
        frame = next((f for f in scene.groups if f.box.x <= e.target.x <= f.box.right), None)
        into[frame.id if frame else "discriminator"] = e
    over = into["discriminator"]
    assert over.exit.value == "right" and over.via[0].x > loss.right  # out of its side
    assert all(p.y < loss.y for p in over.via[1:])  # up and over
    assert over.target.y > over.via[-1].y  # then down into the discriminator's top
    under = into["generator"]
    assert under.exit.value == "bottom" and under.via[0].y > loss.bottom
