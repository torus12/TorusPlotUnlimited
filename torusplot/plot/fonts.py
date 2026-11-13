"""Bundled fonts (SIL OFL 1.1 only) and font-family resolution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent / "fonts"

#: Families shipped with TorusPlot (see fonts/README.md for licenses).
BUNDLED_FAMILIES = ("Inter", "Source Sans 3", "Liberation Sans")
DEFAULT_FAMILY = "Inter"

_registered = False


def register_bundled_fonts() -> list[str]:
    """Make the bundled fonts available to matplotlib (idempotent). Returns file names.

    The bundled files always win over a system-installed font of the same family name
    (system copies of those families are hidden from matplotlib for this process), so a
    figure renders identically on every computer.
    """
    global _registered
    files = sorted(FONT_DIR.glob("*/*.ttf"))
    if not _registered:
        manager = font_manager.fontManager
        for path in files:
            manager.addfont(str(path))
        root = str(FONT_DIR.resolve())
        bundled, others = [], []
        for entry in manager.ttflist:
            try:
                inside = str(Path(entry.fname).resolve()).startswith(root)
            except OSError:
                inside = False
            if inside:
                bundled.append(entry)
            elif entry.name not in BUNDLED_FAMILIES:
                others.append(entry)
        manager.ttflist = bundled + others
        cache = getattr(manager, "_findfont_cached", None)
        if cache is not None and hasattr(cache, "cache_clear"):
            cache.cache_clear()
        _registered = True
        _system_families.cache_clear()
    return [p.name for p in files]


@lru_cache(maxsize=1)
def _system_families() -> tuple[str, ...]:
    return tuple(sorted({f.name for f in font_manager.fontManager.ttflist}))


def available_families() -> list[str]:
    """Bundled families first, then every other font installed on this computer."""
    register_bundled_fonts()
    others = [f for f in _system_families() if f not in BUNDLED_FAMILIES]
    return list(BUNDLED_FAMILIES) + others


def resolve_family(family: str | None) -> tuple[str, str | None]:
    """Return (usable family, warning or None). Unknown fonts fall back to the bundled default."""
    register_bundled_fonts()
    if not family:
        return DEFAULT_FAMILY, None
    known = {f.lower(): f for f in _system_families()}
    if family.lower() in known:
        return known[family.lower()], None
    return (
        DEFAULT_FAMILY,
        f"Font '{family}' is not installed on this computer; using the bundled '{DEFAULT_FAMILY}' instead.",
    )
