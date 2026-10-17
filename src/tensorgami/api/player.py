"""The player of an animation (roadmap step 14c): the animated SVG with play/pause, step back
and forward, a time slider, the step shown and the speed.

The controls drive the SVG's own clock (``pauseAnimations``, ``setCurrentTime``), so the player
needs no widget library and no kernel: it works in JupyterLab, the classic notebook, VS Code
and Colab, and in a saved HTML page. Where scripts do not run (a notebook not trusted), the
SVG still plays on its own, looping.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from xml.sax.saxutils import escape

_PLAYERS = itertools.count(1)  # two players of one animation on a page stay apart


def _icon(path: str) -> str:
    return (
        '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">'
        f'<path d="{path}" fill="currentColor"/></svg>'
    )


PLAY = _icon("M3 1.5v9l7.5-4.5z")
PAUSE = _icon("M2.5 1.5h2.6v9H2.5zM6.9 1.5h2.6v9H6.9z")
PREVIOUS = _icon("M1.5 1.5h1.7v9H1.5zM10.5 1.5v9L4 6z")
NEXT = _icon("M8.8 1.5h1.7v9H8.8zM1.5 1.5v9L8 6z")

_BUTTON = (
    "display:inline-flex;align-items:center;justify-content:center;width:28px;height:24px;"
    "padding:0;border:1px solid rgba(128,128,128,.45);border-radius:4px;background:transparent;"
    "color:inherit;cursor:pointer"
)

_SCRIPT = """(function () {
  var root = document.getElementById(__ID__);
  if (!root || root.getAttribute("data-ready")) return;
  root.setAttribute("data-ready", "1");
  var svg = root.querySelector("svg.tg-animation");
  if (!svg || !svg.pauseAnimations) return;  // no SMIL clock: the SVG plays on its own
  var total = __TOTAL__, starts = __STARTS__, icons = __ICONS__;
  function part(name) { return root.querySelector('[data-tg="' + name + '"]'); }
  var play = part("play"), time = part("time"), label = part("step"), speed = part("speed");
  var t = 0, playing = true, last = null;
  svg.pauseAnimations();
  function stepAt(x) {
    var i = 0;
    while (i + 1 < starts.length && starts[i + 1] <= x + 1e-6) i++;
    return i;
  }
  function show() {
    svg.setCurrentTime(t);
    time.value = t;
    label.textContent = "step " + (stepAt(t) + 1) + " / " + starts.length;
    play.innerHTML = playing ? icons.pause : icons.play;
    play.setAttribute("aria-label", playing ? "Pause" : "Play");
    play.title = playing ? "Pause (space)" : "Play (space)";
  }
  function tick(now) {
    if (!root.isConnected) return;
    if (playing) {
      if (last !== null) t = (t + (now - last) / 1000 * parseFloat(speed.value)) % total;
      last = now;
      show();
    }
    requestAnimationFrame(tick);
  }
  function seek(x) {
    playing = false;
    last = null;
    t = Math.max(0, Math.min(total - 0.001, x));
    show();
  }
  function toggle() {
    playing = !playing;
    last = null;
    show();
  }
  function back() {
    var i = stepAt(t);
    seek(starts[t - starts[i] > 0.25 ? i : Math.max(0, i - 1)]);
  }
  function forward() { seek(starts[Math.min(starts.length - 1, stepAt(t) + 1)]); }
  play.addEventListener("click", toggle);
  part("previous").addEventListener("click", back);
  part("next").addEventListener("click", forward);
  time.addEventListener("input", function () { seek(parseFloat(time.value)); });
  root.addEventListener("keydown", function (e) {
    if (e.target === speed) return;
    if (e.key === " ") toggle();
    else if (e.key === "ArrowRight") forward();
    else if (e.key === "ArrowLeft") back();
    else return;
    e.preventDefault();
  });
  show();
  requestAnimationFrame(tick);
})();"""


def player_html(svg: str, *, total: float, starts: list[float], title: str = "") -> str:
    """The player around ``svg`` (an animated SVG document): ``total``, the loop's length;
    ``starts``, when each step lights up (seconds)."""
    digest = hashlib.sha1(svg.encode("utf-8")).hexdigest()[:8]
    ident = f"tg-player-{digest}-{next(_PLAYERS)}"
    inline = svg.split("\n", 1)[1] if svg.startswith("<?xml") else svg
    inline = inline.replace(
        "<svg ", '<svg class="tg-animation" style="display:block;max-width:100%;height:auto" ', 1
    )
    script = (
        _SCRIPT.replace("__ID__", json.dumps(ident))
        .replace("__TOTAL__", json.dumps(round(total, 3)))
        .replace("__STARTS__", json.dumps([round(s, 3) for s in starts]))
        .replace("__ICONS__", json.dumps({"play": PLAY, "pause": PAUSE}))
    )
    label = escape(title or "animation")
    speeds = "".join(
        f'<option value="{v}"{" selected" if v == "1" else ""}>{v}×</option>'
        for v in ("0.5", "1", "2")
    )
    return (
        f'<div id="{ident}" class="tg-player" tabindex="0" role="group" '
        f'aria-label="{label}" style="display:inline-block;max-width:100%;outline:none">\n'
        f"{inline}"
        '<div style="display:flex;align-items:center;gap:6px;margin-top:6px;'
        'font:12px/1.2 system-ui,-apple-system,sans-serif;opacity:.85">'
        f'<button type="button" data-tg="previous" title="Previous step (←)" '
        f'aria-label="Previous step" style="{_BUTTON}">{PREVIOUS}</button>'
        f'<button type="button" data-tg="play" title="Pause (space)" aria-label="Pause" '
        f'style="{_BUTTON}">{PAUSE}</button>'
        f'<button type="button" data-tg="next" title="Next step (→)" aria-label="Next step" '
        f'style="{_BUTTON}">{NEXT}</button>'
        f'<input type="range" data-tg="time" min="0" max="{round(total, 3)}" step="0.01" '
        'value="0" aria-label="Time" style="flex:1;min-width:60px;margin:0 4px">'
        '<span data-tg="step" style="min-width:6.5em;text-align:right;'
        f'font-variant-numeric:tabular-nums">step 1 / {len(starts)}</span>'
        f'<select data-tg="speed" title="Speed" aria-label="Speed" style="color:inherit;'
        f'background:transparent;border:1px solid rgba(128,128,128,.45);border-radius:4px">'
        f"{speeds}</select>"
        "</div>\n"
        f"<script>\n{script}\n</script>\n"
        "</div>"
    )


def page_html(player: str, title: str) -> str:
    """A standalone page holding one player."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n</head>\n"
        '<body style="margin:24px;font-family:system-ui,-apple-system,sans-serif">\n'
        f"{player}\n</body>\n</html>\n"
    )
