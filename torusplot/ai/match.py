""""Match this figure": copy the STYLE of a reference image onto the user's own data.

Only style is copied (colours, spines, grid, background, proportions). The reference
figure's data, artwork, annotations and statistics are never read into the result: the
figure is always drawn from the user's table by TorusPlot's own renderer.

Everything here works without AI. The measurements (:func:`dominant_colors`,
:func:`analyze_image`) are deterministic image processing with fixed parameters. A vision
model, when one is enabled, only adds guesses that pixels cannot give (plot type, font
category ...) and its colours are cross-checked against the measured palette.

Heuristics and their limits (stated honestly):

* Colours are found by Pillow median-cut quantisation + k-means refinement, then merged in
  CIE Lab. Each colour is reported as the most frequent exact pixel value of its cluster,
  so flat fills are recovered exactly. Small anti-aliasing blends between two larger
  colours are folded into their neighbours; a genuinely distinct colour that covers under
  2 % of the image *and* lies on the line between two larger colours can be lost that way.
* "background" is the commonest colour on the image border; "ink" is a low-chroma colour
  covering little area (axes, text); everything else is "data". Rare colours that are
  nearly the background colour (JPEG halos) are ignored. The style palette keeps only
  data colours covering at least 8 % of the area of the largest one, so small marker or
  outline colours do not end up in the palette.
* The plot area is found from long thin runs of ink (the axes). Figures without visible
  axis lines give ``plot_area = None``.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ..data.tables import DataTable
from ..spec.figure import PLOT_TYPES, FigureSpec, plot_types_for
from .jsonutil import JSONExtractError, extract_json
from .prompts import (
    default_binding,
    normalize_results,
    normalize_tables,
    resolve_table,
    style_system_prompt,
    style_user_prompt,
    table_summary,
)
from .providers.base import Provider, ProviderError
from .settings import MAX_FEEDBACK_LOOPS_CAP

# ---- fixed parameters (changing any of these changes results: keep them constant)
_PALETTE_MAX_SIDE = 256  # image is reduced (nearest neighbour) to this size for colour statistics
_QUANT_COLORS = 32  # median-cut boxes before merging
_KMEANS_ITERATIONS = 4  # k-means refinement passes of the quantiser
_MERGE_DELTA_E = 7.0  # clusters closer than this (CIE76) are one colour
_BLEND_MAX_FRACTION = 0.02  # only colours rarer than this can be treated as anti-aliasing blends
_BLEND_RESIDUAL = 14.0  # max RGB distance from the line between two larger colours
_HALO_DELTA_E = 18.0  # rare colours this close to the background are compression halo, not data
_PALETTE_MIN_RELATIVE = 0.08  # a palette colour covers at least this share of the largest data colour
_MIN_FRACTION = 0.0004  # colours rarer than this are noise
_ACHROMATIC_CHROMA = 10.0  # Lab chroma below this = gray/black/white
_FILL_FRACTION = 0.05  # an achromatic colour covering more than this is a fill, not ink
_INK_CONTRAST = 60  # gray-level difference from the background that counts as "ink"
_AXIS_MIN_RUN = 0.35  # an axis line spans at least this fraction of the image
_ANALYSIS_MAX_SIDE = 2000

#: A model colour further than this (CIE76) from every measured colour is replaced.
VISION_DELTA_E = 15.0

DEFAULT_PRESET = "Torus Signature"
MIN_HEIGHT_MM, MAX_HEIGHT_MM = 40.0, 200.0

SSIM_NOTE = (
    "The similarity score (SSIM) is only a rough signal: your data differ from the reference "
    "figure's, so the images can never be identical."
)

_LEGEND_POSITIONS = (
    "best", "upper right", "upper left", "lower left", "lower right", "right", "outside right", "top", "bottom",
)
_ERROR_BARS = ("sem", "sd", "ci95", "range", "iqr", "none")
#: Plot types that need an analysis result (or run one): never chosen from a picture.
_NEEDS_RESULT = {"residuals", "estimation", "dose_response", "survival", "qq"}
_SIMPLE_CATEGORICAL = {"bar", "scatter", "box", "violin", "box_violin", "beeswarm"}


# ====================================================================== image helpers
def load_rgb(image: bytes | Image.Image | np.ndarray) -> np.ndarray:
    """PNG/JPEG bytes, a PIL image or an array -> ``(H, W, 3) uint8`` (transparency over white)."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            return load_rgb(Image.fromarray(arr))
        return np.ascontiguousarray(arr[..., :3])
    if isinstance(image, bytes | bytearray):
        try:
            pil = Image.open(io.BytesIO(bytes(image)))
            pil.load()
        except Exception as exc:  # PIL raises several unrelated types
            raise ValueError("This file is not an image TorusPlot can read (use PNG or JPEG).") from exc
    else:
        pil = image
    if pil.mode in {"RGBA", "LA", "P"} or "transparency" in pil.info:
        rgba = pil.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
        pil = flat
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def to_png_bytes(image: bytes | Image.Image | np.ndarray) -> bytes:
    """Any supported image as PNG bytes (RGB, transparency flattened on white)."""
    buf = io.BytesIO()
    Image.fromarray(load_rgb(image)).save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_hex(rgb: Any) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    """``#RGB`` / ``#RRGGBB`` -> (r, g, b); None if it is not a hex colour."""
    if not isinstance(color, str):
        return None
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def rgb_to_lab(rgb: Any) -> np.ndarray:
    """sRGB (0-255) -> CIE Lab (D65), any array shape ending in 3."""
    from skimage.color import rgb2lab

    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    return rgb2lab(arr.reshape(-1, 1, 3)).reshape(arr.shape)


def delta_e(color_a: str | Any, color_b: str | Any) -> float:
    """CIE76 colour difference between two colours (hex strings or RGB triples)."""
    a = hex_to_rgb(color_a) if isinstance(color_a, str) else color_a
    b = hex_to_rgb(color_b) if isinstance(color_b, str) else color_b
    if a is None or b is None:
        raise ValueError("delta_e needs hex colours such as #1F77B4.")
    lab = rgb_to_lab(np.array([a, b], dtype=float))
    return float(np.linalg.norm(lab[0] - lab[1]))


# ====================================================================== palette
@dataclass
class _Cluster:
    rgb: tuple[int, int, int]
    count: int
    lab: np.ndarray
    members: list[int] = field(default_factory=list)  # quantiser indices


def _small(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(h, w) / _PALETTE_MAX_SIDE
    if scale <= 1:
        return rgb
    size = (max(1, round(w / scale)), max(1, round(h / scale)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def _clusters(rgb: np.ndarray) -> tuple[list[_Cluster], np.ndarray]:
    """Merged colour clusters (largest first) and the cluster index of every pixel."""
    image = Image.fromarray(rgb)
    quant = image.quantize(
        colors=_QUANT_COLORS, method=Image.Quantize.MEDIANCUT, kmeans=_KMEANS_ITERATIONS, dither=Image.Dither.NONE
    )
    idx = np.asarray(quant, dtype=np.uint8)
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    raw: list[tuple[int, int, int]] = []  # (count, modal colour, quantiser index)
    for i in np.unique(idx):
        members = packed[idx == i]
        values, counts = np.unique(members, return_counts=True)
        raw.append((int(members.size), int(values[int(counts.argmax())]), int(i)))
    raw.sort(key=lambda t: (-t[0], t[1]))

    clusters: list[_Cluster] = []
    for count, mode, qi in raw:
        color = ((mode >> 16) & 255, (mode >> 8) & 255, mode & 255)
        lab = rgb_to_lab(np.array(color, dtype=float))
        target = None
        for cluster in clusters:
            if float(np.linalg.norm(cluster.lab - lab)) < _MERGE_DELTA_E:
                target = cluster
                break
        if target is None:
            clusters.append(_Cluster(color, count, lab, [qi]))
        else:
            target.count += count
            target.members.append(qi)

    # fold anti-aliasing blends (rare colours on the line between two larger colours) into a neighbour
    total = float(idx.size)
    kept: list[_Cluster] = []
    for cluster in clusters:
        host = None
        if cluster.count / total < _BLEND_MAX_FRACTION and len(kept) >= 2:
            c = np.array(cluster.rgb, dtype=float)
            for i, a in enumerate(kept):
                for b in kept[i + 1:]:
                    if min(a.count, b.count) < 2 * cluster.count:
                        continue
                    pa, pb = np.array(a.rgb, dtype=float), np.array(b.rgb, dtype=float)
                    ab = pb - pa
                    t = float(np.dot(c - pa, ab) / max(float(np.dot(ab, ab)), 1e-9))
                    if 0.06 < t < 0.94 and float(np.linalg.norm(pa + t * ab - c)) <= _BLEND_RESIDUAL:
                        host = a if t < 0.5 else b
                        break
                if host is not None:
                    break
        if host is None:
            kept.append(cluster)
        else:
            host.count += cluster.count
            host.members.extend(cluster.members)
    kept.sort(key=lambda cl: (-cl.count, cl.rgb))
    lut = np.zeros(256, dtype=np.int32)
    for ci, cluster in enumerate(kept):
        for qi in cluster.members:
            lut[qi] = ci
    return kept, lut[idx]


def _palette(rgb: np.ndarray) -> list[dict[str, Any]]:
    """Every colour of an image with its area fraction and role, largest first."""
    small = _small(rgb)
    clusters, labels = _clusters(small)
    total = float(labels.size)
    h, w = labels.shape
    m = max(1, round(0.02 * min(h, w)))
    border = np.concatenate([labels[:m].ravel(), labels[-m:].ravel(), labels[:, :m].ravel(), labels[:, -m:].ravel()])
    background = int(np.bincount(border, minlength=len(clusters)).argmax())
    bg_l = float(clusters[background].lab[0])
    out = []
    for i, cluster in enumerate(clusters):
        fraction = cluster.count / total
        chroma = float(math.hypot(cluster.lab[1], cluster.lab[2]))
        halo = float(np.linalg.norm(cluster.lab - clusters[background].lab)) < _HALO_DELTA_E
        if i == background:
            role = "background"
        elif fraction < _MIN_FRACTION or (halo and fraction < _BLEND_MAX_FRACTION):
            continue
        elif chroma < _ACHROMATIC_CHROMA and fraction < _FILL_FRACTION:
            role = "ink"
        else:
            role = "data"
        out.append(
            {
                "hex": rgb_to_hex(cluster.rgb),
                "fraction": round(fraction, 6),
                "role": role,
                "chroma": round(chroma, 2),
                "contrast": round(abs(float(cluster.lab[0]) - bg_l), 2),
            }
        )
    return out


def dominant_colors(image_bytes: bytes | Image.Image | np.ndarray, k: int = 6) -> list[dict[str, Any]]:
    """The ``k`` largest colours of an image, ordered by area.

    Each item is ``{"hex": "#RRGGBB", "fraction": 0..1, "role": "background" | "ink" | "data"}``.
    Deterministic: the same image always gives the same list.
    """
    colors = _palette(load_rgb(image_bytes))
    return [{"hex": c["hex"], "fraction": c["fraction"], "role": c["role"]} for c in colors[: max(1, int(k))]]


def data_colors(image_bytes: bytes | Image.Image | np.ndarray, k: int = 8) -> list[str]:
    """Hex colours of the data series (background and ink removed), ordered by area."""
    return [c["hex"] for c in _palette(load_rgb(image_bytes)) if c["role"] == "data"][: max(0, int(k))]


# ====================================================================== plot-area geometry
def _longest_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of a boolean array: (length of the longest run of True, its start index)."""
    c = np.cumsum(mask, axis=1, dtype=np.int32)
    reset = np.maximum.accumulate(np.where(mask, 0, c), axis=1)
    run = c - reset
    length = run.max(axis=1)
    start = run.argmax(axis=1) - length + 1
    return length, start


def _groups(flags: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive True values as (first index, last index)."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.int16)


def find_axes(rgb: np.ndarray) -> dict[str, Any] | None:
    """Locate the axis lines of a plot image. None when no left + bottom axis is found.

    Returns pixel indices ``left, top, right, bottom`` of the plot area (the axis lines
    themselves), ``has_top`` / ``has_right`` (is the area boxed on those sides) and
    ``line_width`` (thickness of the axis lines in pixels).
    """
    h0, w0 = rgb.shape[:2]
    factor = max(1, math.ceil(max(h0, w0) / _ANALYSIS_MAX_SIDE))
    if factor > 1:
        size = (max(1, w0 // factor), max(1, h0 // factor))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX), dtype=np.uint8)
    gray = _gray(rgb)
    h, w = gray.shape
    if h < 8 or w < 8:
        return None
    m = max(1, round(0.02 * min(h, w)))
    border = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(), gray[:, :m].ravel(), gray[:, -m:].ravel()])
    background = int(np.median(border))
    ink = np.abs(gray - background) > _INK_CONTRAST
    row_len, row_start = _longest_runs(ink)
    col_len, col_start = _longest_runs(ink.T)
    max_thick = max(3, round(0.012 * min(h, w)))
    h_lines = [g for g in _groups(row_len >= _AXIS_MIN_RUN * w) if g[1] - g[0] + 1 <= max_thick]
    v_lines = [g for g in _groups(col_len >= _AXIS_MIN_RUN * h) if g[1] - g[0] + 1 <= max_thick]
    if not h_lines or not v_lines:
        return None
    bottom_group, left_group = h_lines[-1], v_lines[0]
    bottom = (bottom_group[0] + bottom_group[1]) // 2
    left = (left_group[0] + left_group[1]) // 2
    b_start, b_len = int(row_start[bottom]), int(row_len[bottom])
    l_start, l_len = int(col_start[left]), int(col_len[left])
    tol = max(3, round(0.03 * max(h, w)))

    top_group = None
    for g in h_lines[:-1]:
        row = (g[0] + g[1]) // 2
        same_extent = abs(int(row_start[row]) - b_start) <= tol and abs(int(row_len[row]) - b_len) <= 2 * tol
        if bottom - row > 0.3 * h and same_extent:
            top_group = g
            break
    right_group = None
    for g in reversed(v_lines[1:]):
        col = (g[0] + g[1]) // 2
        same_extent = abs(int(col_start[col]) - l_start) <= tol and abs(int(col_len[col]) - l_len) <= 2 * tol
        if col - left > 0.3 * w and same_extent:
            right_group = g
            break
    top = (top_group[0] + top_group[1]) // 2 if top_group else l_start
    right = (right_group[0] + right_group[1]) // 2 if right_group else b_start + b_len - 1
    if right - left < 0.2 * w or bottom - top < 0.2 * h:
        return None
    thickness = max(bottom_group[1] - bottom_group[0] + 1, left_group[1] - left_group[0] + 1)
    return {
        "left": left * factor,
        "top": top * factor,
        "right": right * factor,
        "bottom": bottom * factor,
        "has_top": top_group is not None,
        "has_right": right_group is not None,
        "line_width": thickness * factor,
        "background_gray": background,
    }


def _has_grid(rgb: np.ndarray, axes: dict[str, Any]) -> bool:
    """True when at least two thin horizontal lines cross (almost) the whole plot area."""
    pad = int(axes["line_width"]) + 2
    left, right = axes["left"] + pad, axes["right"] - pad
    top, bottom = axes["top"] + pad, axes["bottom"] - pad
    if right - left < 10 or bottom - top < 10:
        return False
    inner = np.abs(_gray(rgb[top:bottom, left:right]) - int(axes["background_gray"])) > 8
    covered = inner.mean(axis=1) >= 0.9
    max_thick = max(3, round(0.012 * min(rgb.shape[:2])))
    thin = [g for g in _groups(covered) if g[1] - g[0] + 1 <= max_thick]
    return len(thin) >= 2


def analyze_image(image_bytes: bytes | Image.Image | np.ndarray) -> dict[str, Any]:
    """Deterministic measurements of a figure image (no AI).

    Keys: ``width``, ``height`` (pixels), ``aspect_ratio`` (width / height), ``background``
    and ``text_color`` (hex or None), ``colors`` (see :func:`dominant_colors`, all of them),
    ``plot_area`` (``(left, top, right, bottom)`` pixel indices or None),
    ``has_top_spine``, ``has_right_spine``, ``has_top_right_spines`` and ``grid``
    (bool, or None when the plot area was not found).
    """
    rgb = load_rgb(image_bytes)
    h, w = rgb.shape[:2]
    colors = _palette(rgb)
    background = next((c["hex"] for c in colors if c["role"] == "background"), None)
    achromatic = [c for c in colors if c["role"] != "background" and c["chroma"] < _ACHROMATIC_CHROMA]
    text = max(achromatic, key=lambda c: (c["contrast"], c["fraction"]))["hex"] if achromatic else None
    try:
        axes = find_axes(rgb)
    except Exception:  # pragma: no cover - measurements must never break the feature
        axes = None
    out: dict[str, Any] = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / h, 4),
        "background": background,
        "text_color": text,
        "colors": [{"hex": c["hex"], "fraction": c["fraction"], "role": c["role"]} for c in colors],
        "plot_area": None,
        "has_top_spine": None,
        "has_right_spine": None,
        "has_top_right_spines": None,
        "grid": None,
    }
    if axes is not None:
        out.update(
            plot_area=(axes["left"], axes["top"], axes["right"], axes["bottom"]),
            has_top_spine=bool(axes["has_top"]),
            has_right_spine=bool(axes["has_right"]),
            has_top_right_spines=bool(axes["has_top"] and axes["has_right"]),
            grid=_has_grid(rgb, axes),
        )
    return out


# ====================================================================== style extraction
class StyleExtraction(BaseModel):
    """Style properties read from a reference image. Contains no data and no statistics."""

    model_config = ConfigDict(extra="forbid")

    plot_type: str | None = Field(default=None, description="Guess of a vision model; None without one")
    palette: list[str] = Field(default_factory=list, description="Hex colours of the data series, by area")
    background: str | None = None
    text_color: str | None = None
    has_top_right_spines: bool | None = None
    grid: bool | None = None
    font_category: Literal["sans", "serif"] | None = None
    font_family: str | None = Field(default=None, description="Bundled font family to use (None = preset's)")
    bar_edge: str | None = Field(default=None, description="'none', 'black', 'group' or a hex colour")
    error_bars: str | None = Field(default=None, description="Guess only; never applied automatically")
    points_shown: bool | None = None
    legend_position: str | None = None
    aspect_ratio: float | None = Field(default=None, description="Image width / height")
    suggested_width_mm: float | None = None
    suggested_height_mm: float | None = None
    notes: list[str] = Field(default_factory=list)
    source: Literal["deterministic", "deterministic+vision"] = "deterministic"

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


def _preset_width(preset: str) -> float:
    from ..plot.styles import StyleError, get_preset

    try:
        return float(get_preset(preset)["width_mm"])
    except StyleError:
        return float(get_preset(DEFAULT_PRESET)["width_mm"])


def _height_for(width_mm: float, aspect: float | None) -> float | None:
    if not aspect or aspect <= 0:
        return None
    return round(min(MAX_HEIGHT_MM, max(MIN_HEIGHT_MM, width_mm / aspect)), 1)


def cross_check_colors(
    model_colors: list[Any], measured: list[str], *, threshold: float = VISION_DELTA_E
) -> tuple[list[str], list[str]]:
    """Check a vision model's colours against the measured palette.

    A model colour is kept only if it is within ``threshold`` (CIE76 Delta-E) of a measured
    colour; otherwise it is replaced by the nearest measured colour. Returns
    ``(colours, notes)``; with nothing measured the model's colours are not trusted at all.
    """
    notes: list[str] = []
    out: list[str] = []
    measured_rgb = [hex_to_rgb(c) for c in measured]
    for raw in model_colors:
        rgb = hex_to_rgb(raw) if isinstance(raw, str) else None
        if rgb is None:
            notes.append(f"The vision model gave an invalid colour ({raw!r}); it was ignored.")
            continue
        color = rgb_to_hex(rgb)
        if not measured:
            notes.append(f"The vision model's colour {color} could not be confirmed in the image; it was ignored.")
            continue
        distances = [delta_e(rgb, m) for m in measured_rgb]
        nearest = int(np.argmin(distances))
        if distances[nearest] > threshold:
            notes.append(
                f"The vision model's colour {color} is not in the image; the nearest measured colour "
                f"{measured[nearest]} was used instead."
            )
            color = measured[nearest]
        if color not in out:
            out.append(color)
    return out, notes


def _apply_vision(extraction: StyleExtraction, answer: dict[str, Any], measured: dict[str, Any]) -> None:
    """Merge a vision model's style guesses into the deterministic extraction (validated field by field)."""
    plot_type = answer.get("plot_type")
    if isinstance(plot_type, str):
        key = plot_type.strip().lower().replace(" ", "_").replace("-", "_")
        if key in PLOT_TYPES:
            extraction.plot_type = key
        else:
            extraction.note(f"The vision model's plot type '{plot_type}' is not one TorusPlot knows; ignored.")

    model_colors = answer.get("colors")
    if isinstance(model_colors, list) and model_colors:
        pool = [c["hex"] for c in measured["colors"] if c["role"] == "data"]
        pool = pool or [c["hex"] for c in measured["colors"] if c["role"] != "background"]
        checked, notes = cross_check_colors(model_colors, pool)
        for text in notes:
            extraction.note(text)
        rest = [c for c in extraction.palette if all(delta_e(c, k) > VISION_DELTA_E for k in checked)]
        extraction.palette = checked + rest

    for key in ("has_top_right_spines", "grid"):
        value = answer.get(key)
        if isinstance(value, bool) and getattr(extraction, key) is None:
            setattr(extraction, key, value)  # measurements win when they exist
    if isinstance(answer.get("points_shown"), bool):
        extraction.points_shown = answer["points_shown"]

    font = answer.get("font_category")
    if isinstance(font, str) and font.strip().lower() in {"sans", "serif"}:
        extraction.font_category = font.strip().lower()  # type: ignore[assignment]

    edge = answer.get("bar_edge")
    if isinstance(edge, str):
        value = edge.strip().lower()
        if value in {"none", "black", "group"}:
            extraction.bar_edge = value
        elif hex_to_rgb(edge) is not None:
            extraction.bar_edge = rgb_to_hex(hex_to_rgb(edge))

    bars = answer.get("error_bars")
    if isinstance(bars, str) and bars.strip().lower() in _ERROR_BARS:
        extraction.error_bars = bars.strip().lower()

    legend = answer.get("legend_position")
    if isinstance(legend, str) and legend.strip().lower() in (*_LEGEND_POSITIONS, "none"):
        extraction.legend_position = legend.strip().lower()

    for key in ("background", "text_color"):
        value = answer.get(key)
        if getattr(extraction, key) is None and isinstance(value, str) and hex_to_rgb(value) is not None:
            checked, _ = cross_check_colors([value], [c["hex"] for c in measured["colors"]])
            if checked:
                setattr(extraction, key, checked[0])


def extract_style(image_bytes: bytes | Image.Image | np.ndarray, provider: Provider | None = None) -> StyleExtraction:
    """Read the style of a reference figure.

    Without a provider (or with one that has no vision) the result is purely measured
    (``source == "deterministic"``). A vision provider is asked **only** for style
    properties, as JSON, and is told not to read data values; if it fails or answers
    nonsense, the measured result is returned with a note.
    """
    measured = analyze_image(image_bytes)
    series = [c for c in measured["colors"] if c["role"] == "data"]
    floor = _PALETTE_MIN_RELATIVE * max((c["fraction"] for c in series), default=0.0)
    extraction = StyleExtraction(
        palette=[c["hex"] for c in series if c["fraction"] >= floor][:8],
        background=measured["background"],
        text_color=measured["text_color"],
        has_top_right_spines=measured["has_top_right_spines"],
        grid=measured["grid"],
        aspect_ratio=measured["aspect_ratio"],
    )
    width = _preset_width(DEFAULT_PRESET)
    extraction.suggested_width_mm = width
    extraction.suggested_height_mm = _height_for(width, extraction.aspect_ratio)
    if measured["plot_area"] is None:
        extraction.note("No axis lines were found in the image, so spines and grid could not be measured.")
    if not extraction.palette:
        extraction.note("No distinct data colours were found; the style preset's palette is kept.")

    if provider is not None and provider.supports_vision:
        try:
            raw = provider.complete(
                style_system_prompt(), style_user_prompt(), images=[to_png_bytes(image_bytes)], json_mode=True
            )
            answer = extract_json(raw)
        except (ProviderError, JSONExtractError) as exc:
            extraction.note(f"The vision model could not be used ({exc}); only measured properties were applied.")
        else:
            _apply_vision(extraction, answer, measured)
            extraction.source = "deterministic+vision"
    elif provider is not None:
        extraction.note(f"The '{provider.name}' provider cannot look at images; only measured properties were applied.")

    if extraction.font_category == "serif":
        extraction.font_family = "Source Sans 3"
        extraction.note(
            "The reference uses a serif font. TorusPlot bundles only open-licensed sans-serif fonts, "
            "so 'Source Sans 3' is used; choose an installed serif font in Style if you prefer."
        )
    elif extraction.font_category == "sans":
        extraction.font_family = "Inter"
    return extraction


# ====================================================================== style -> FigureSpec
def _default_plot(table: DataTable, extraction: StyleExtraction) -> dict[str, Any]:
    """``{"plot_type", "data"}`` for the user's table, honouring the extraction's guess when it is safe."""
    binding = default_binding(table_summary(table))
    guess = extraction.plot_type
    if not guess or guess == binding["plot_type"]:
        return binding
    valid = plot_types_for(table.kind)
    usable = guess in valid and guess not in _NEEDS_RESULT
    if usable and table.kind == "multiple_variables":
        usable = guess in _SIMPLE_CATEGORICAL and binding["plot_type"] == "bar"
    if usable and table.kind == "xy":
        usable = guess in {"xy", "histogram"}
    if usable:
        return {"plot_type": guess, "data": binding["data"]}
    extraction.note(
        f"The reference looks like a '{guess}' plot, which cannot be drawn directly from your "
        f"{table.kind_label} table; a '{binding['plot_type']}' plot is used instead."
    )
    return binding


def style_to_spec(
    extraction: StyleExtraction,
    table: DataTable | str,
    tables: Any,
    *,
    base_spec: FigureSpec | None = None,
) -> FigureSpec:
    """Build a FigureSpec for the USER'S table that carries the extracted style.

    * Without ``base_spec``: plot type = the extraction's guess when it is valid for the
      table kind and needs no analysis result, else a sensible default for the kind
      (bar for column/grouped, xy for xy ...), with a note added to ``extraction.notes``.
      The new figure has no annotations and no statistics.
    * With ``base_spec`` (the user's own figure of the same table): its plot type, data,
      layers and annotations are kept; only style, layout and legend are restyled.

    Style overrides: palette (hex list), spines, grid, background, text colour, bar edge,
    font. Layout: width stays the preset's width, height follows the reference's aspect
    ratio, clamped to 40-200 mm. An error-bar guess is **not** applied (which error bar a
    figure shows is a scientific choice, not a style).
    """
    known = normalize_tables(tables)
    target = resolve_table(table, known)
    if target is None:
        raise ValueError(f"Table '{table}' was not found. Available tables: {list(known)}")

    if base_spec is not None and base_spec.data.table == target.name:
        data = base_spec.model_dump(mode="python")
        if extraction.plot_type and extraction.plot_type != base_spec.plot_type:
            extraction.note(
                f"The reference looks like a '{extraction.plot_type}' plot; your figure stays a "
                f"'{base_spec.plot_type}' plot (only the style was copied)."
            )
    else:
        data = {**_default_plot(target, extraction), "style": {"preset": DEFAULT_PRESET}}
        if base_spec is not None:
            data["style"] = base_spec.style.model_dump(mode="python")

    style: dict[str, Any] = dict(data.get("style") or {})
    style.setdefault("preset", DEFAULT_PRESET)
    if extraction.palette:
        style["palette"] = list(extraction.palette[:8])
        style["gradient"] = None
    if extraction.has_top_right_spines is True:
        style["spines"] = ["left", "bottom", "top", "right"]
    elif extraction.has_top_right_spines is False:
        style["spines"] = ["left", "bottom"]
    if extraction.grid is not None:
        style["grid"] = extraction.grid
    if extraction.background:
        style["background"] = extraction.background
    if extraction.text_color:
        style["text_color"] = extraction.text_color
        style["axis_color"] = extraction.text_color
    if extraction.bar_edge == "none":
        style["bar_edge_width"] = 0.0
    elif extraction.bar_edge == "black":
        style["bar_edge_color"] = "#000000"
    elif extraction.bar_edge:
        style["bar_edge_color"] = extraction.bar_edge
    if extraction.font_family:
        style["font_family"] = extraction.font_family
    data["style"] = style

    layout: dict[str, Any] = dict(data.get("layout") or {})
    width = _preset_width(style["preset"])
    height = _height_for(width, extraction.aspect_ratio)
    if height is not None:
        layout["width_mm"] = width
        layout["height_mm"] = height
    data["layout"] = layout

    legend: dict[str, Any] = dict(data.get("legend") or {})
    if extraction.legend_position == "none":
        legend["show"] = False
    elif extraction.legend_position in _LEGEND_POSITIONS:
        legend["position"] = extraction.legend_position
    data["legend"] = legend

    if extraction.error_bars and extraction.error_bars != "none":
        extraction.note(
            f"The reference may show {extraction.error_bars.upper()} error bars. TorusPlot did not change "
            "your error bars: choose them yourself in the figure options."
        )
    if extraction.points_shown is False:
        extraction.note("The reference hides individual data points; TorusPlot keeps showing yours.")
    return FigureSpec.model_validate(data)


# ====================================================================== the loop
@dataclass
class MatchOutcome:
    """Result of :func:`match_figure`."""

    spec: FigureSpec
    extraction: StyleExtraction
    history: list[dict[str, Any]]
    reference_png: bytes
    render_png: bytes
    best_ssim: float
    notes: list[str] = field(default_factory=list)


def match_figure(
    image_bytes: bytes,
    table: DataTable | str,
    tables: Any,
    results: Any = None,
    *,
    provider: Provider | None = None,
    max_loops: int = 3,
    base_spec: FigureSpec | None = None,
) -> MatchOutcome:
    """Restyle the user's data to look like a reference figure.

    1. :func:`extract_style` reads the style (measured; plus a vision model if enabled).
    2. :func:`style_to_spec` builds a figure of the user's table with that style.
    3. Feedback loop, at most ``max_loops`` passes (default 3, hard cap 3): render -> SSIM
       against the reference -> (vision provider only) ask for a style patch -> apply the
       whitelisted part of the patch -> render again. The spec with the best SSIM is kept.
       The loop stops early when the model says it is done, proposes nothing usable, or
       SSIM stops improving.

    Without a vision provider there is exactly one deterministic pass (one history entry).
    ``history`` items: ``{"loop", "ssim", "applied", "rejected", "critique"}``.
    """
    from .feedback import apply_style_patch, critique_patch, render_png, ssim_score

    known = normalize_tables(tables)
    computed = normalize_results(results)
    loops = max(1, min(int(max_loops), MAX_FEEDBACK_LOOPS_CAP))
    vision = provider if (provider is not None and provider.supports_vision) else None

    reference = to_png_bytes(image_bytes)
    extraction = extract_style(image_bytes, provider=provider)
    spec = style_to_spec(extraction, table, known, base_spec=base_spec)
    notes = [SSIM_NOTE, "Only the style of the reference was copied; its data and artwork were not used."]

    history: list[dict[str, Any]] = []
    best: tuple[float, FigureSpec, bytes] | None = None
    for loop in range(1, loops + 1):
        try:
            png = render_png(spec, known, computed)
        except Exception as exc:
            if best is None:
                raise
            notes.append(f"A suggested style change could not be drawn ({exc}); the best earlier version is kept.")
            break
        score = ssim_score(reference, png)
        entry: dict[str, Any] = {"loop": loop, "ssim": score, "applied": [], "rejected": [], "critique": None}
        history.append(entry)
        improved = best is None or score > best[0]
        if improved:
            best = (score, spec, png)
        if vision is None or loop == loops or not improved:
            break
        critique = critique_patch(vision, reference, png, spec)
        entry["critique"] = critique
        new_spec, applied, rejected = apply_style_patch(spec, critique.get("patch") or {})
        entry["applied"], entry["rejected"] = applied, rejected
        if critique.get("done") or not applied:
            break
        spec = new_spec

    if best is None:  # pragma: no cover - the first pass either renders or raises
        raise RuntimeError("The figure could not be rendered.")
    notes.extend(n for n in extraction.notes if n not in notes)
    return MatchOutcome(
        spec=best[1],
        extraction=extraction,
        history=history,
        reference_png=reference,
        render_png=best[2],
        best_ssim=best[0],
        notes=notes,
    )
