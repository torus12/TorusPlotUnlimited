"""Visual feedback loop of "Match this figure": similarity score, critique, whitelisted patches.

THE HARD RULE applies here in its strictest form. A patch proposed by a vision model may
only touch **cosmetics**:

* ``style.*`` (every StyleSpec field), ``layout.*``, ``legend.*``;
* axis cosmetics: ``axes.x.*`` / ``axes.y.*`` (scale, limits, label, ticks ...) and
  ``axes.zero_line``;
* cosmetic fields of *existing* layers: ``layers.<i>.<color | edge_color | alpha | size |
  width | linewidth | linestyle | marker | fill | cap_size | jitter>``.

Everything else is rejected by :func:`apply_style_patch`: ``data``, ``annotations``,
``texts``, ``title``, ``plot_type``, layer ``type`` / ``stat_result_id`` / ``series``, adding
or removing layers. Text values that read like a statistic are rejected too. The patched
spec is fully re-validated.
"""

from __future__ import annotations

import copy
import io
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError

from ..spec.figure import AxisSpec, FigureSpec, LayoutSpec, LegendSpec, StyleSpec, default_layers
from .describe import format_validation_error, looks_like_statistic
from .jsonutil import JSONExtractError, extract_json
from .prompts import STATS_RULE, critique_system_prompt, critique_user_prompt
from .providers.base import Provider, ProviderError

SSIM_SIZE = 256

STYLE_FIELDS = tuple(StyleSpec.model_fields)
LAYOUT_FIELDS = tuple(LayoutSpec.model_fields)
LEGEND_FIELDS = tuple(LegendSpec.model_fields)
AXIS_FIELDS = tuple(AxisSpec.model_fields)
LAYER_COSMETIC_FIELDS = (
    "color", "edge_color", "alpha", "size", "width", "linewidth", "linestyle", "marker", "fill", "cap_size",
    "jitter",
)

_COLOR_PATHS = {
    "style.background", "style.text_color", "style.axis_color", "style.grid_color", "style.point_edge_color",
    "style.bar_edge_color",
}
_TEXT_PATHS = {"axes.x.label", "axes.y.label", "legend.title", "axes.x.tick_labels", "axes.y.tick_labels"}


# ---------------------------------------------------------------------- similarity
def _gray_square(image: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(image))
    if pil.mode in {"RGBA", "LA", "P"}:
        rgba = pil.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
        pil = flat
    gray = pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float64)


def ssim_score(reference_png: bytes, render_png: bytes) -> float:
    """Structural similarity (SSIM) of two images, in [-1, 1]; 1.0 means identical.

    Both images are converted to grayscale and resized to 256 x 256 first, so the score is
    deterministic and independent of the original sizes. It is only a rough signal when
    the two figures show different data.
    """
    from skimage.metrics import structural_similarity

    a, b = _gray_square(reference_png), _gray_square(render_png)
    if np.array_equal(a, b):
        return 1.0
    return float(structural_similarity(a, b, data_range=255.0))


def render_png(spec: FigureSpec, tables: Any, results: Any = None, dpi: int = 100) -> bytes:
    """Render a spec with TorusPlot's own renderer and return PNG bytes."""
    from ..export.figure import figure_bytes
    from ..plot.render import render_figure

    return figure_bytes(render_figure(spec, tables, results), "png", dpi=dpi)


# ---------------------------------------------------------------------- whitelist
def allowed_patch_paths() -> list[str]:
    """Every dot-path a style patch may set (``<i>`` stands for a layer index)."""
    paths = [f"style.{f}" for f in STYLE_FIELDS]
    paths += [f"layout.{f}" for f in LAYOUT_FIELDS]
    paths += [f"legend.{f}" for f in LEGEND_FIELDS]
    paths += [f"axes.{axis}.{f}" for axis in ("x", "y") for f in AXIS_FIELDS]
    paths += ["axes.zero_line"]
    paths += [f"layers.<i>.{f}" for f in LAYER_COSMETIC_FIELDS]
    return paths


def is_allowed_path(path: str) -> bool:
    """True if ``path`` is on the cosmetic whitelist."""
    parts = path.split(".")
    if len(parts) == 2:
        section, name = parts
        if section == "style":
            return name in STYLE_FIELDS
        if section == "layout":
            return name in LAYOUT_FIELDS
        if section == "legend":
            return name in LEGEND_FIELDS
        return section == "axes" and name == "zero_line"
    if len(parts) == 3:
        section, middle, name = parts
        if section == "axes":
            return middle in {"x", "y"} and name in AXIS_FIELDS
        if section == "layers":
            return middle.isdigit() and name in LAYER_COSMETIC_FIELDS
        return section == "style" and middle == "colors" and bool(name)
    return False


def _flatten(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested patch dicts -> dot-paths (``style.colors`` stays a dict; layer lists are indexed)."""
    out: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and path != "style.colors":
            out.update(_flatten(value, f"{path}."))
        elif path == "layers" and isinstance(value, list) and all(isinstance(v, Mapping) for v in value):
            for i, item in enumerate(value):
                index = item.get("index", i)
                rest = {k: v for k, v in item.items() if k != "index"}
                out.update(_flatten(rest, f"layers.{index}."))
        else:
            out[path] = value
    return out


def _value_problem(path: str, value: Any) -> str | None:
    """Checks beyond the schema: statistics-like text, unknown presets/palettes, invalid colours."""
    if path in _TEXT_PATHS:
        texts = value if isinstance(value, list) else [value]
        if any(looks_like_statistic(t) for t in texts):
            return f"text that reads like a statistical result is not allowed. {STATS_RULE}"
    if value is None:
        return None
    from ..plot.palettes import PaletteError, get_palette, normalize_color, sample_colormap
    from ..plot.styles import preset_names

    try:
        if path == "style.preset" and value not in preset_names():
            return f"unknown style preset '{value}'"
        if path == "style.palette":
            get_palette(value, 1)
        elif path in {"style.colormap", "style.gradient"}:
            sample_colormap(str(value), 1)
        elif path in _COLOR_PATHS or path.startswith("style.colors.") or path.endswith((".color", ".edge_color")):
            if not (path == "style.bar_edge_color" and value == "group"):
                normalize_color(value)
        elif path == "style.colors":
            if not isinstance(value, Mapping):
                return "must be an object of {group name: hex colour}"
            for color in value.values():
                normalize_color(color)
    except (PaletteError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def _set(data: dict[str, Any], path: str, value: Any) -> str | None:
    """Set ``path`` in a dumped spec. Returns a rejection reason or None."""
    parts = path.split(".")
    if parts[0] == "layers":
        if not data["layers"]:
            # materialise the default layers (drawing is unchanged) so one of them can be restyled
            data["layers"] = [lay.model_dump(mode="python") for lay in default_layers(data["plot_type"])]
        index = int(parts[1])
        if index >= len(data["layers"]):
            return f"there is no layer {index} (layers cannot be added)"
        data["layers"][index][parts[2]] = value
        return None
    if path == "style.colors":
        data["style"]["colors"] = {**data["style"]["colors"], **{str(k): v for k, v in value.items()}}
        return None
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return None


def apply_style_patch(spec: FigureSpec, patch: Mapping[str, Any]) -> tuple[FigureSpec, list[str], list[str]]:
    """Apply the whitelisted part of a style patch.

    ``patch`` maps dot-paths to values (nested dicts are accepted and flattened). Returns
    ``(new spec, applied paths, rejected)`` where each rejected item is ``"path: reason"``.
    The input spec is never modified; ``data`` and ``annotations`` of the result are always
    identical to the input's.
    """
    applied: list[str] = []
    rejected: list[str] = []
    if not isinstance(patch, Mapping):
        return spec, applied, ["(patch): must be a JSON object of dot-paths"]
    data = spec.model_dump(mode="python")
    for path, value in _flatten(patch).items():
        if not is_allowed_path(path):
            rejected.append(f"{path}: not a style setting (only style, layout, legend, axis and layer cosmetics may change)")
            continue
        problem = _value_problem(path, value)
        if problem:
            rejected.append(f"{path}: {problem}")
            continue
        trial = copy.deepcopy(data)
        problem = _set(trial, path, value)
        if problem:
            rejected.append(f"{path}: {problem}")
            continue
        try:
            FigureSpec.model_validate(trial)
        except ValidationError as exc:
            rejected.append(f"{path}: {format_validation_error(exc)}")
            continue
        data = trial
        applied.append(path)
    if not applied:
        return spec, applied, rejected
    new_spec = FigureSpec.model_validate(data)
    # belt and braces: the whitelist must never let these change
    if new_spec.data != spec.data or new_spec.annotations != spec.annotations or new_spec.texts != spec.texts:
        raise AssertionError("A style patch changed data, annotations or texts.")  # pragma: no cover
    return new_spec, applied, rejected


# ---------------------------------------------------------------------- critique
def critique_patch(provider: Provider, reference: bytes, render: bytes, spec: FigureSpec) -> dict[str, Any]:
    """Ask a vision model how to make ``render`` look more like ``reference`` (style only).

    Returns ``{"patch": {dot-path: value}, "done": bool, "comment": str}`` (plus ``"error"``
    when the model could not be used or answered with something that is not JSON; the loop
    then stops). The patch is NOT trusted: pass it through :func:`apply_style_patch`.
    """
    try:
        raw = provider.complete(
            critique_system_prompt(),
            critique_user_prompt(spec, allowed_patch_paths()),
            images=[reference, render],
            json_mode=True,
        )
        answer = extract_json(raw)
    except (ProviderError, JSONExtractError) as exc:
        return {"patch": {}, "done": True, "comment": "", "error": str(exc)}
    patch = answer.get("patch")
    done = answer.get("done")
    comment = answer.get("comment")
    comment = " ".join(str(comment).split())[:300] if isinstance(comment, str) else ""
    if looks_like_statistic(comment):
        comment = ""
    return {
        "patch": dict(patch) if isinstance(patch, Mapping) else {},
        "done": done is True or (isinstance(done, str) and done.strip().lower() == "true"),
        "comment": comment,
    }
