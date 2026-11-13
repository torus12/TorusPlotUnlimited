"""Colour palettes.

All palettes are plain colour definitions that are free to use:

* **Okabe-Ito**: Okabe M, Ito K (2002/2008) "Color Universal Design: how to make
  figures and presentations that are friendly to colorblind people"
  (https://jfly.uni-koeln.de/color/). Eight colours distinguishable under the common
  forms of colour-vision deficiency.
* **viridis family** (viridis, magma, plasma, inferno, cividis): perceptually uniform
  colormaps shipped with matplotlib (released under CC0 by their authors).
* **Torus** and the neutral palettes below were designed for TorusPlot.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]

PALETTES: dict[str, list[str]] = {
    "Okabe-Ito": OKABE_ITO,
    "Torus": ["#27318B", "#E4572E", "#17A398", "#F2A541", "#7FC8F8", "#7A1F2B", "#A8E0C0", "#16181D"],
    "Torus Soft": ["#5B6BD6", "#F28C6A", "#4FC3B5", "#F6C177", "#7A1F2B", "#6A2C8F", "#6B7A1E", "#16181D"],
    "Classic Lab": ["#1A1A1A", "#8C8C8C", "#D0D0D0", "#FFFFFF", "#4D4D4D", "#B3B3B3"],
    "Grayscale": ["#222222", "#555555", "#888888", "#AAAAAA", "#CCCCCC", "#E5E5E5"],
    "Blue-Orange": ["#1F5FA8", "#F28E2B", "#7FB3E6", "#FFBE7D", "#0B3C6F", "#B85C00"],
}

#: Palettes whose colours all stay distinguishable (CIE76 Delta-E >= 10 between every pair) for
#: normal vision and simulated deuteranopia, protanopia and tritanopia, at any number of groups
#: up to the palette length. Verified by tests/plot/test_styles_palettes.py.
CVD_SAFE_PALETTES = ("Okabe-Ito", "Torus", "Torus Soft")

#: Sequential colormaps that can be sampled into discrete palettes or used for heat maps.
SEQUENTIAL = ("viridis", "magma", "plasma", "inferno", "cividis")
DIVERGING = ("RdBu_r", "coolwarm", "PuOr_r", "BrBG")

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


class PaletteError(ValueError):
    pass


def palette_names() -> list[str]:
    return list(PALETTES) + list(SEQUENTIAL)


def is_hex(color: str) -> bool:
    return bool(isinstance(color, str) and _HEX.match(color.strip()))


def normalize_color(color: str) -> str:
    """Any matplotlib colour (name or hex) -> '#RRGGBB'. Raises PaletteError if invalid."""
    try:
        return mcolors.to_hex(color.strip() if isinstance(color, str) else color, keep_alpha=False).upper()
    except (ValueError, AttributeError) as exc:
        raise PaletteError(f"'{color}' is not a valid colour. Use hex like #1F77B4.") from exc


def sample_colormap(name: str, n: int, lo: float = 0.08, hi: float = 0.92) -> list[str]:
    """``n`` evenly spaced colours from a colormap (avoiding the extreme ends)."""
    try:
        cmap = mpl.colormaps[name]
    except KeyError as exc:
        raise PaletteError(f"Unknown colormap '{name}'.") from exc
    if n <= 0:
        return []
    points = [0.5 * (lo + hi)] if n == 1 else np.linspace(lo, hi, n)
    return [mcolors.to_hex(cmap(float(p))).upper() for p in points]


def get_palette(palette: str | Sequence[str] | None, n: int) -> list[str]:
    """Resolve a palette name or custom hex list to exactly ``n`` colours (cycled if short)."""
    if n <= 0:
        return []
    if palette is None:
        base = PALETTES["Torus"]
    elif isinstance(palette, str):
        if palette in PALETTES:
            base = PALETTES[palette]
        elif palette in mpl.colormaps:
            return sample_colormap(palette, n)
        else:
            matches = [k for k in PALETTES if k.lower() == palette.lower()]
            if not matches:
                raise PaletteError(f"Unknown palette '{palette}'. Available: {', '.join(palette_names())}")
            base = PALETTES[matches[0]]
    else:
        base = [normalize_color(c) for c in palette]
        if not base:
            raise PaletteError("A custom palette needs at least one colour.")
    return [base[i % len(base)] for i in range(n)]


def resolve_colors(
    names: Sequence[str],
    palette: str | Sequence[str] | None = None,
    overrides: dict[str, str] | None = None,
    gradient: str | None = None,
) -> dict[str, str]:
    """Map each group name to a colour: palette (or gradient) first, then per-group overrides."""
    names = [str(n) for n in names]
    base = sample_colormap(gradient, len(names)) if gradient else get_palette(palette, len(names))
    out = dict(zip(names, base))
    for name, color in (overrides or {}).items():
        if name in out:
            out[name] = normalize_color(color)
    return out


def darken(color: str, amount: float = 0.35) -> str:
    """Darker shade of a colour (mix with black)."""
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb * (1.0 - amount), 0, 1)).upper()


def lighten(color: str, amount: float = 0.5) -> str:
    """Lighter tint of a colour (mix with white)."""
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb + (1.0 - rgb) * amount, 0, 1)).upper()


def luminance(color: str) -> float:
    """Relative luminance (WCAG definition) in [0, 1]."""
    rgb = np.array(mcolors.to_rgb(color))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def contrast_text_color(background: str) -> str:
    """Black or white, whichever reads better on ``background``."""
    return "#000000" if luminance(background) > 0.35 else "#FFFFFF"
