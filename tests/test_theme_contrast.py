"""Every theme keeps text readable (WCAG AA, 4.5:1) and defines the same variables."""
import os
import re

CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "assets", "app.css")


def _lum(h):
    h = h.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4     # noqa: E731
    return 0.2126 * f(ch[0]) + 0.7152 * f(ch[1]) + 0.0722 * f(ch[2])


def _cr(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def themes():
    css = open(CSS, encoding="utf-8").read()
    out = {}
    for name, body in re.findall(r'(\[data-theme="[^"]+"\])\s*\{([^}]*)\}', css):
        out[name] = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{3,6})", body))
    return out


def test_eight_themes_with_the_same_variables():
    t = themes()
    assert len(t) == 8
    keys = set.intersection(*[set(v) for v in t.values()])
    for need in ("bg", "ink", "muted", "faint", "accent", "accent-ink", "low", "med", "high", "crit", "on-sem"):
        assert need in keys, need


def test_text_and_badges_meet_wcag_aa_in_every_theme():
    for name, v in themes().items():
        for k in ("low", "med", "high", "crit"):
            assert _cr(v[k], v["on-sem"]) >= 4.5, (name, k, _cr(v[k], v["on-sem"]))
        assert _cr(v["accent"], v["accent-ink"]) >= 4.5, (name, "accent button")
        assert _cr(v["ink"], v["bg"]) >= 7, (name, "body text")
        assert _cr(v["muted"], v["bg"]) >= 4.5, (name, "muted text")
        assert _cr(v["faint"], v["bg"]) >= 4.5, (name, "faint text")
