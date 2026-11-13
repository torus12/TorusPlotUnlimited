"""Figure export: SVG (editable text), PDF, TIFF (LZW), PNG.

* SVG text stays text (``svg.fonttype = 'none'``) and every element sits in a named
  group, so files open cleanly in vector editors (Inkscape, Graphite, Illustrator...).
* PDF embeds TrueType fonts (type 42), so text stays selectable/editable.
* TIFF is flattened RGB with LZW compression at 300 or 600 dpi, as journals ask.
* Size is exact: the figure's millimetre size is preserved (no "tight" cropping).
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from matplotlib.figure import Figure

from .. import __version__
from ..plot.styles import style_context

FORMATS = ("svg", "pdf", "tiff", "png")
_CREATOR = f"TorusPlot {__version__}"


class ExportError(ValueError):
    pass


def _rc(fig: Figure) -> dict[str, Any]:
    rc = dict(getattr(fig, "_torusplot_rc", {}))
    rc.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "torusplot"})
    return rc


def figure_bytes(fig: Figure, fmt: str = "png", dpi: int = 300, transparent: bool = False) -> bytes:
    """Render a figure to bytes in the given format."""
    fmt = fmt.lower().lstrip(".")
    fmt = {"tif": "tiff", "jpg": "png", "jpeg": "png"}.get(fmt, fmt)
    if fmt not in FORMATS:
        raise ExportError(f"Unsupported export format '{fmt}'. Choose from {FORMATS}.")
    buf = io.BytesIO()
    with style_context(_rc(fig)):
        if fmt == "svg":
            fig.savefig(buf, format="svg", transparent=transparent, metadata={"Date": None, "Creator": _CREATOR})
        elif fmt == "pdf":
            fig.savefig(buf, format="pdf", transparent=transparent,
                        metadata={"Creator": _CREATOR, "Producer": _CREATOR, "CreationDate": None})
        elif fmt == "png":
            fig.savefig(buf, format="png", dpi=dpi, transparent=transparent, metadata={"Software": _CREATOR})
        else:  # tiff
            from PIL import Image

            raw = io.BytesIO()
            fig.savefig(raw, format="png", dpi=dpi, transparent=transparent)
            raw.seek(0)
            image = Image.open(raw)
            if transparent:
                image = image.convert("RGBA")
            else:
                rgba = image.convert("RGBA")
                image = Image.new("RGB", rgba.size, "#FFFFFF")
                image.paste(rgba, mask=rgba.split()[3])
            image.save(buf, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi), software=_CREATOR)
    return buf.getvalue()


def export_figure(fig: Figure, path: str | Path, fmt: str | None = None, dpi: int = 300,
                  transparent: bool = False) -> Path:
    """Write a figure to ``path``; the format comes from ``fmt`` or the file extension."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "png").lower()
    fmt = {"tif": "tiff"}.get(fmt, fmt)
    if not path.suffix:
        path = path.with_suffix("." + ("tif" if fmt == "tiff" else fmt))
    data = figure_bytes(fig, fmt, dpi=dpi, transparent=transparent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def export_all(fig: Figure, base: str | Path, formats: list[str] | tuple[str, ...] = ("svg", "pdf", "png"),
               dpi: int = 300, transparent: bool = False) -> dict[str, Path]:
    """Export one figure in several formats: ``base`` + .svg/.pdf/.tif/.png."""
    base = Path(base)
    if base.suffix.lower().lstrip(".") in {"svg", "pdf", "png", "tif", "tiff"}:
        base = base.with_suffix("")
    out = {}
    for fmt in formats:
        ext = "tif" if fmt in {"tiff", "tif"} else fmt
        out[fmt] = export_figure(fig, base.parent / f"{base.name}.{ext}", fmt=fmt, dpi=dpi, transparent=transparent)
    return out


def figure_size_mm(fig: Figure) -> tuple[float, float]:
    w, h = fig.get_size_inches()
    return float(w * 25.4), float(h * 25.4)


def validate_svg(svg: str | bytes | Path) -> dict[str, Any]:
    """Check that an SVG is well-formed XML with real text and named groups.

    Returns ``{"ok", "text_elements", "group_ids", "duplicate_ids", "has_paths_for_text",
    "width", "height", "problems"}``. This is what vector editors need to open the file
    with editable text.
    """
    if isinstance(svg, Path):
        data = svg.read_bytes()
    elif isinstance(svg, str) and not svg.lstrip().startswith("<"):
        data = Path(svg).read_bytes()
    else:
        data = svg.encode("utf-8") if isinstance(svg, str) else svg
    problems: list[str] = []
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        return {"ok": False, "problems": [f"Not well-formed XML: {exc}"], "text_elements": 0,
                "group_ids": [], "duplicate_ids": [], "width": None, "height": None}
    ns = "{http://www.w3.org/2000/svg}"
    if root.tag != f"{ns}svg":
        problems.append("Root element is not <svg> in the SVG namespace.")
    texts = [el for el in root.iter(f"{ns}text")]
    ids = [el.get("id") for el in root.iter() if el.get("id")]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    groups = [el.get("id") for el in root.iter(f"{ns}g") if el.get("id")]
    if not texts:
        problems.append("No <text> elements: text was converted to outlines and is not editable.")
    if dup:
        problems.append(f"Duplicate element ids: {dup[:5]}")
    hrefs = set()
    for el in root.iter():
        for key, value in el.attrib.items():
            if key.endswith("href") and value.startswith("#"):
                hrefs.add(value[1:])
            for m in re.finditer(r"url\(#([^)]+)\)", value):
                hrefs.add(m.group(1))
    dangling = sorted(h for h in hrefs if h not in set(ids))
    if dangling:
        problems.append(f"References to missing ids: {dangling[:5]}")
    return {
        "ok": not problems,
        "problems": problems,
        "text_elements": len(texts),
        "group_ids": groups,
        "duplicate_ids": dup,
        "width": root.get("width"),
        "height": root.get("height"),
    }
