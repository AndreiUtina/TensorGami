"""The animation player's controls, run in a real browser (roadmap step 14c): headless Chrome,
when one is installed (GitHub's Ubuntu runners have it)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from ir_fixtures import transformer

from tensorgami.api.figure import Figure
from tensorgami.layout import layout
from tensorgami.transforms import simplify

CHROMES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def chrome() -> str | None:
    for name in CHROMES:
        found = shutil.which(name) or (name if Path(name).is_file() else None)
        if found:
            return found
    return None


pytestmark = pytest.mark.skipif(chrome() is None, reason="needs Chrome")

#: clicks the controls once the page has loaded, and notes what the player shows after each
PROBE = """<script>
window.addEventListener("load", function () {
  var root = document.querySelector(".tg-player"), svg = root.querySelector("svg");
  function part(name) { return root.querySelector('[data-tg="' + name + '"]'); }
  function key(name) {
    root.dispatchEvent(new KeyboardEvent("keydown", {key: name, bubbles: true}));
  }
  var log = [];
  function note() {
    log.push([part("step").textContent, part("time").value,
              part("play").getAttribute("aria-label"), svg.animationsPaused()].join("|"));
  }
  note();
  part("next").click(); part("next").click(); part("next").click(); note();
  part("previous").click(); note();
  part("time").value = SEEK; part("time").dispatchEvent(new Event("input")); note();
  part("previous").click(); note();
  key("ArrowRight"); note();
  key(" "); note();
  for (var i = 0; i < 40; i++) part("next").click(); note();
  var out = document.createElement("pre");
  out.id = "probe";
  out.textContent = log.join("\\n");
  document.body.appendChild(out);
});
</script>"""


def test_the_controls_step_seek_and_play(theme, tmp_path):
    graph = simplify(transformer(layers=2), 2)
    anim = Figure(layout(graph, theme), theme, graph).animate()
    _, _, starts = anim._timed()
    n = len(starts)
    seek = starts[5] + 0.5  # well into step 6
    page = anim.save(tmp_path / "player.html").read_text(encoding="utf-8")
    probe = tmp_path / "probe.html"
    probe.write_text(page.replace("</body>", PROBE.replace("SEEK", str(seek)) + "</body>"))
    run = subprocess.run(
        [
            chrome(),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--enable-logging=stderr",
            "--v=0",
            "--virtual-time-budget=2000",
            "--dump-dom",
            probe.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Uncaught" not in run.stderr, run.stderr
    found = re.search(r'<pre id="probe">(.*?)</pre>', run.stdout, re.S)
    assert found, run.stdout[-2000:]
    rows = [row.split("|") for row in found.group(1).splitlines()]

    def at(k: int) -> str:  # the slider's value at the start of step k (from 0)
        return f"{round(starts[k], 3):g}"

    assert rows == [
        [f"step 1 / {n}", "0", "Pause", "true"],  # playing, on the player's own clock
        [f"step 4 / {n}", at(3), "Play", "true"],  # → → →: paused, at the step's start
        [f"step 3 / {n}", at(2), "Play", "true"],  # ←: the step before
        [f"step 6 / {n}", f"{seek:g}", "Play", "true"],  # the slider
        [f"step 6 / {n}", at(5), "Play", "true"],  # ← well into a step: back to its start
        [f"step 7 / {n}", at(6), "Play", "true"],  # the → key
        [f"step 7 / {n}", at(6), "Pause", "true"],  # space: playing again
        [f"step {n} / {n}", at(n - 1), "Play", "true"],  # → stops at the last step
    ]
