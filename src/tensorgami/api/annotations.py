"""Annotations: what a figure shows beyond the traced modules (ADR-0007 ``annotations=``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class Loss:
    """A loss term: a box fed by the tensors it compares, e.g.
    ``tg.Loss("reconstruction", "output 0", "input")``. ``trains`` names the models or modules
    its gradient updates; dashed arrows go back to them (the adversarial loop of a GAN)."""

    name: str
    inputs: tuple[str, ...]
    trains: tuple[str, ...]

    def __init__(self, name: str, *inputs: str, trains: tuple[str, ...] | list[str] = ()) -> None:
        if not inputs:
            raise ValueError(f"Loss({name!r}) needs the tensors it compares, e.g. 'output'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "trains", tuple(trains))


@dataclass(frozen=True)
class Input:
    """Another input for a tensor of the figure: it feeds what that tensor feeds, e.g. the real
    images a discriminator also judges, ``tg.Input("real image", to="discriminator.image")``."""

    name: str
    to: str


@dataclass(frozen=True)
class Note:
    """A line of text under a node, e.g. ``tg.Note("ε ~ N(0, I)", at="...")``."""

    text: str
    at: str


Annotation = Loss | Input | Note
