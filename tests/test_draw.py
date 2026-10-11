"""``tg.draw``: from a ``torch.nn.Module`` to a figure in one call (step 10, ADR-0007)."""

from __future__ import annotations

import subprocess
import sys
from importlib import resources

import pytest

torch = pytest.importorskip("torch")

import matplotlib.pyplot as plt  # noqa: E402
from models import REFERENCE  # noqa: E402
from models.cnn import SmallCNN  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

import tensorgami as tg  # noqa: E402
from tensorgami.importers import ImporterError, UnclassifiedOpWarning  # noqa: E402
from tensorgami.importers.torch import trace  # noqa: E402
from tensorgami.layout import LayoutOptions, layout  # noqa: E402
from tensorgami.render import render_svg  # noqa: E402
from tensorgami.style import ThemeError, load_style  # noqa: E402
from tensorgami.transforms import simplify  # noqa: E402
from tensorgami.views import VIEWS, available_views, get_view  # noqa: E402


@pytest.fixture(scope="module")
def cnn():
    return SmallCNN()


@pytest.fixture(scope="module")
def x():
    return torch.randn(1, 3, 32, 32)


# --------------------------------------------------------------------------- end to end


@pytest.mark.parametrize("style", ["default", "origami", "origami-extended"])
@pytest.mark.parametrize("name", REFERENCE)
def test_every_reference_model_is_drawn(name, style):
    make, example = REFERENCE[name]
    fig = tg.draw(make(), example(), style=style)
    assert isinstance(fig, tg.Figure)
    assert fig.style.name == style
    assert fig.scene.nodes and fig.graph is not None
    assert repr(fig).startswith(f"<tensorgami.Figure '{fig.graph.name}'")


def test_from_a_module_to_a_png(cnn, x, tmp_path):
    fig = tg.draw(cnn, x)
    path = fig.save(tmp_path / "cnn.png", dpi=300)
    w, h = fig.size
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert abs(image.width - w * 300 / 96) <= 1 and abs(image.height - h * 300 / 96) <= 1
    png, shown = fig._repr_png_()  # what Jupyter displays
    assert png.startswith(b"\x89PNG") and shown == {"width": round(w), "height": round(h)}


def test_draw_is_the_pipeline_in_one_call(cnn, x):
    theme = load_style("origami")
    graph = simplify(trace(cnn, x), 1)
    manual = render_svg(layout(graph, theme, LayoutOptions.for_style(theme, direction="TB")), theme)
    fig = tg.draw(cnn, x, level=1, style="origami", direction="TB")
    assert fig.to_svg() == manual
    assert fig.graph == graph  # the graph that is drawn, simplified to the level


# --------------------------------------------------------------------------- options


def test_options_after_the_input_are_keyword_only(cnn, x):
    with pytest.raises(TypeError):
        tg.draw(cnn, x, 2)


def test_level_by_number_or_name(cnn, x):
    assert tg.draw(cnn, x, level="stages").to_svg() == tg.draw(cnn, x, level=1).to_svg()
    assert len(tg.draw(cnn, x, level="system").graph.nodes) == 3  # input → model → output


def test_style_by_name_path_or_theme(cnn, x, tmp_path):
    expected = tg.draw(cnn, x, style="origami").to_svg()
    path = tmp_path / "mine.toml"
    path.write_text((resources.files("tensorgami.style.themes") / "origami.toml").read_text())
    assert tg.draw(cnn, x, style=path).to_svg() == expected
    assert tg.draw(cnn, x, style=str(path)).to_svg() == expected
    assert tg.draw(cnn, x, style=load_style("origami")).to_svg() == expected


def test_layout_options_pass_through(cnn, x):
    assert tg.draw(cnn, x, direction="tb").to_svg() == tg.draw(cnn, x, direction="TB").to_svg()
    tall = tg.draw(cnn, x, direction="TB").size
    assert tall[1] > tall[0]
    plain = tg.draw(cnn, x, legend=False)
    assert not [n for n in plain.scene.nodes if n.id.startswith("legend:")]
    assert [n for n in tg.draw(cnn, x).scene.nodes if n.id.startswith("legend:")]


def test_overrides_reach_the_importer(cnn, x):
    fig = tg.draw(cnn, x, overrides={"classifier.2": {"label": "Classifier head"}})
    assert "Classifier head" in {node.label for node in fig.scene.nodes}


def test_views(cnn, x):
    assert available_views() == ["auto", "block", "unrolled"]
    assert VIEWS["auto"].choose(trace(cnn, x)) is get_view("block")
    assert tg.draw(cnn, x, view="block").to_svg() == tg.draw(cnn, x).to_svg()


# --------------------------------------------------------------------------- no side effects


def test_draw_leaves_the_model_the_caller_and_the_disk_alone(x, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = SmallCNN().train()  # dropout and batch-norm statistics would change in train mode
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    figures = plt.get_fignums()

    tg.draw(model, x)._repr_png_()

    assert all(m.training for m in model.modules())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert plt.get_fignums() == figures  # no pyplot figures left open
    assert list(tmp_path.iterdir()) == []  # nothing saved until fig.save(...)


def test_import_stays_light():
    code = (
        "import sys, tensorgami as tg; tg.draw; "
        "print('torch' in sys.modules, 'matplotlib' in sys.modules, 'draw' in dir(tg))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # matplotlib comes with tg.Figure; torch only when a model is actually drawn
    assert result.stdout.split()[0] == "False" and result.stdout.split()[2] == "True"


# --------------------------------------------------------------------------- errors


class MustNotRun(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover - the arguments are checked first
        raise AssertionError("the model ran before the arguments were checked")


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"directon": "TB"}, TypeError, "did you mean 'direction'"),
        ({"theme": "origami"}, TypeError, "did you mean 'style'"),
        ({"colour": "red"}, TypeError, "layout options are: direction"),
        ({"style": "origamy"}, ThemeError, "built-in styles: default, origami"),
        ({"level": 9}, ValueError, "level must be one of 0 system"),
        ({"direction": "XY"}, ValueError, "direction must be 'LR'"),
        ({"view": "volume"}, NotImplementedError, "planned for roadmap step 11"),
        ({"view": "fancy"}, ValueError, "unknown view 'fancy'"),
        ({"wrap_width": "wide"}, TypeError, None),
    ],
)
def test_bad_arguments_fail_before_the_model_runs(options, error, message):
    with pytest.raises(error, match=message):
        tg.draw(MustNotRun(), torch.randn(1, 4), **options)


def test_model_errors_say_what_to_do(cnn):
    with pytest.raises(ImporterError, match=r"x\.unsqueeze\(0\)"):
        tg.draw(cnn, torch.randn(3, 32, 32))  # no batch dimension
    with pytest.raises(ImporterError, match=r"create it first, e\.g\. SmallCNN\(\)"):
        tg.draw(SmallCNN, torch.randn(1, 3, 32, 32))  # the class, not a model


class Spectrum(nn.Module):
    def forward(self, x):
        return torch.fft.rfft(x).abs()


def test_warnings_point_at_the_callers_line():
    with pytest.warns(UnclassifiedOpWarning, match="fft_rfft") as record:
        tg.draw(Spectrum(), torch.randn(1, 8))
    assert record[0].filename == __file__


def test_recurrent_models_get_the_unrolled_view():
    make, example = REFERENCE["convlstm"]
    fig = tg.draw(make(), example())  # view="auto": a cell called step after step
    assert any(g.type_name == "unrolled" for g in fig.graph.groups)
    folded = tg.draw(make(), example(), view="block")
    assert not any(g.type_name == "unrolled" for g in folded.graph.groups)
    make, example = REFERENCE["lstm"]
    fig = tg.draw(make(), example(), view="unrolled")  # a fused nn.LSTM, on request
    cells = [n for n in fig.graph.nodes if n.op == "LSTMCell"]
    assert len(cells) == 2 * 4  # 2 layers × 4 of the 5 steps (1, 2, 3, ⋯, 5)
