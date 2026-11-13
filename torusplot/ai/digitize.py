""""Digitize": read APPROXIMATE data points off an image of a plot. Deterministic, no AI.

Workflow (the usual one for this kind of tool):

1. The user calibrates each axis with two reference pixels whose values they know
   (:class:`AxisCalibration`; log axes interpolate in log10 space).
   :func:`detect_plot_area` can suggest where the axes are; the *values* always come from
   the user.
2. The user picks the colour of one series (:func:`candidate_colors` lists likely ones).
3. :func:`digitize` segments that colour in CIE Lab inside the plot area and reduces the
   pixels to points (``points`` mode: one per marker), a sampled line (``curve`` mode) or
   bar heights (``bars`` mode).
4. :func:`to_table` wraps the result in an XY table flagged ``approximate=True``.
   TorusPlot's statistics refuse approximate tables unless the user explicitly overrides.

Pixel convention: ``(px, py)`` are array indices (column, row), i.e. the centre of the
top-left pixel is ``(0, 0)`` and ``py`` grows downwards. Use the same convention for the
calibration pixels.

Limits: overlapping markers are split only when a blob is clearly several markers big
(noted in the result); a legend entry or text in the series colour inside the plot area
is read as data; touching bars of one colour are read as one bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image

from ..data.tables import XYTable
from .match import _palette, find_axes, hex_to_rgb, load_rgb, rgb_to_hex, rgb_to_lab

APPROXIMATE_SUFFIX = " (APPROXIMATE)"
APPROXIMATE_NOTE = "APPROXIMATE: for reference only. Digitized from an image; values are estimates."

DEFAULT_TOLERANCE = 20.0  # CIE76 Delta-E around the chosen colour
_DEFAULT_MAX_POINTS = {"points": 5000, "curve": 200, "bars": 500}
_MAX_SPLIT = 6

Mode = Literal["points", "curve", "bars"]


class DigitizeError(ValueError):
    """Digitizing could not be done (message is user-facing)."""


# ====================================================================== calibration
@dataclass(frozen=True)
class AxisCalibration:
    """Two reference pixels of one axis and the data values they stand for.

    For the X axis the pixels are column indices, for the Y axis row indices. With
    ``log=True`` the axis is logarithmic: interpolation is linear in log10(value).
    """

    pixel_a: float
    value_a: float
    pixel_b: float
    value_b: float
    log: bool = False

    def __post_init__(self) -> None:
        try:
            numbers = [float(v) for v in (self.pixel_a, self.value_a, self.pixel_b, self.value_b)]
        except (TypeError, ValueError):
            numbers = [float("nan")]
        if not all(math.isfinite(v) for v in numbers):
            raise DigitizeError("Calibration pixels and values must be finite numbers.")
        if self.pixel_a == self.pixel_b:
            raise DigitizeError("The two calibration points of an axis must be at different pixels.")
        if self.value_a == self.value_b:
            raise DigitizeError("The two calibration points of an axis must have different values.")
        if self.log and (self.value_a <= 0 or self.value_b <= 0):
            raise DigitizeError("Values on a logarithmic axis must be greater than zero.")

    def to_data(self, pixel: Any) -> Any:
        """Pixel position(s) -> data value(s)."""
        t = (np.asarray(pixel, dtype=float) - self.pixel_a) / (self.pixel_b - self.pixel_a)
        if self.log:
            la, lb = math.log10(self.value_a), math.log10(self.value_b)
            out = 10.0 ** (la + t * (lb - la))
        else:
            out = self.value_a + t * (self.value_b - self.value_a)
        return float(out) if np.ndim(out) == 0 else out

    def to_pixel(self, value: Any) -> Any:
        """Data value(s) -> pixel position(s)."""
        v = np.asarray(value, dtype=float)
        if self.log:
            la, lb = math.log10(self.value_a), math.log10(self.value_b)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (np.log10(v) - la) / (lb - la)
        else:
            t = (v - self.value_a) / (self.value_b - self.value_a)
        out = self.pixel_a + t * (self.pixel_b - self.pixel_a)
        return float(out) if np.ndim(out) == 0 else out


def pixel_to_data(px: Any, py: Any, x_axis: AxisCalibration, y_axis: AxisCalibration) -> tuple[Any, Any]:
    """Pixel coordinates ``(column, row)`` -> data coordinates ``(x, y)``."""
    return x_axis.to_data(px), y_axis.to_data(py)


def calibration_from_plot_area(
    plot_area: tuple[float, float, float, float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    x_log: bool = False,
    y_log: bool = False,
) -> tuple[AxisCalibration, AxisCalibration]:
    """Calibrations for the common case "the axes run from min to max across the plot area".

    ``plot_area`` is ``(left, top, right, bottom)`` in pixels: ``x_min`` is at ``left``,
    ``x_max`` at ``right``, ``y_min`` at ``bottom`` and ``y_max`` at ``top``. The values are
    the user's; only the pixels can come from :func:`detect_plot_area`.
    """
    left, top, right, bottom = plot_area
    return (
        AxisCalibration(left, x_min, right, x_max, log=x_log),
        AxisCalibration(bottom, y_min, top, y_max, log=y_log),
    )


# ====================================================================== helpers for the UI
def detect_plot_area(
    image: bytes | Image.Image | np.ndarray, *, inner: bool = False
) -> tuple[int, int, int, int] | None:
    """Find the axes of a plot image: ``(left, top, right, bottom)`` pixel indices, or None.

    The axes are recognised as long thin dark lines. With ``inner=True`` the box is shrunk
    to lie just inside the axis lines (what :func:`digitize` searches by default).
    """
    axes = find_axes(load_rgb(image))
    if axes is None:
        return None
    if not inner:
        return axes["left"], axes["top"], axes["right"], axes["bottom"]
    return _inner_box(axes)


def _inner_box(axes: dict[str, Any]) -> tuple[int, int, int, int]:
    pad = int(axes["line_width"]) // 2 + 2
    left = axes["left"] + pad
    bottom = axes["bottom"] - pad
    top = axes["top"] + pad if axes["has_top"] else axes["top"]
    right = axes["right"] - pad if axes["has_right"] else axes["right"]
    return left, top, right, bottom


def candidate_colors(image: bytes | Image.Image | np.ndarray, k: int = 6) -> list[dict[str, Any]]:
    """Colours a series might have: ``[{"hex", "fraction", "role"}]``, data colours first.

    The background is left out. "ink" colours (black/gray) come last: they are usually
    axes and text, but can be picked to digitize a black series.
    """
    colors = [c for c in _palette(load_rgb(image)) if c["role"] != "background"]
    colors.sort(key=lambda c: (c["role"] != "data", -c["fraction"]))
    return [{"hex": c["hex"], "fraction": c["fraction"], "role": c["role"]} for c in colors[: max(1, int(k))]]


# ====================================================================== result
@dataclass
class DigitizeResult:
    """Approximate points read off an image.

    ``points`` has columns ``x``, ``y`` (data units, sorted by x); ``pixel_points`` the
    matching ``px``, ``py`` (column, row). ``notes`` always starts with the approximate-data
    warning and lists anything doubtful (overlaps that were split, missing plot area ...).
    """

    points: pd.DataFrame
    n: int
    mode: str
    notes: list[str]
    pixel_points: pd.DataFrame
    series_name: str = "Digitized"
    color: str = ""
    plot_area: tuple[int, int, int, int] | None = None
    approximate: bool = field(default=True, init=False)


# ====================================================================== segmentation
def _mask(rgb: np.ndarray, color: str, tolerance: float) -> np.ndarray:
    target = hex_to_rgb(color)
    if target is None:
        raise DigitizeError(f"'{color}' is not a hex colour such as #1F77B4.")
    lab = rgb_to_lab(rgb)
    ref = rgb_to_lab(np.array(target, dtype=float))
    return np.sqrt(((lab - ref) ** 2).sum(axis=-1)) <= float(tolerance)


def _split_blob(coords: np.ndarray, n: int) -> np.ndarray:
    """Split one blob's pixel coordinates into ``n`` centres (deterministic k-means)."""
    pts = coords.astype(float)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    order = np.argsort(centred @ vt[0], kind="stable")
    centres = np.array([pts[chunk].mean(axis=0) for chunk in np.array_split(order, n)])
    for _ in range(12):
        dist = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        nearest = dist.argmin(axis=1)
        updated = np.array(
            [pts[nearest == j].mean(axis=0) if np.any(nearest == j) else centres[j] for j in range(n)]
        )
        if np.allclose(updated, centres):
            break
        centres = updated
    return centres


def _points(mask: np.ndarray, max_points: int, notes: list[str]) -> np.ndarray:
    from skimage.measure import label, regionprops

    regions = regionprops(label(mask, connectivity=2))
    if not regions:
        return np.empty((0, 2))
    areas = np.array([r.area for r in regions], dtype=float)
    keep = [r for r in regions if r.area >= max(2.0, 0.25 * float(np.median(areas)))]
    dropped = len(regions) - len(keep)
    if dropped:
        notes.append(f"{dropped} tiny speck(s) in the series colour were ignored.")
    typical = float(np.median([r.area for r in keep]))
    centres: list[tuple[float, float]] = []
    split_blobs = 0
    extra = 0
    for region in keep:
        count = round(region.area / typical) if typical > 0 else 1
        if len(keep) >= 3 and region.area > 1.6 * typical and count >= 2:
            count = min(int(count), _MAX_SPLIT)
            for row, col in _split_blob(region.coords, count):
                centres.append((float(col), float(row)))
            split_blobs += 1
            extra += count - 1
        else:
            row, col = region.centroid
            centres.append((float(col), float(row)))
    if split_blobs:
        notes.append(
            f"{split_blobs} blob(s) looked like overlapping markers and were split into "
            f"{split_blobs + extra} points by size. Check these points: overlaps are a guess."
        )
    if len(keep) == 1 and keep[0].area > 400:
        notes.append("The series is one connected shape: if it is a line, use curve mode instead.")
    if len(centres) > max_points:
        notes.append(f"More than {max_points} markers were found; only the first {max_points} (by x) are kept.")
        centres = sorted(centres)[:max_points]
    return np.array(centres, dtype=float).reshape(-1, 2)


def _curve(mask: np.ndarray, max_points: int, notes: list[str]) -> np.ndarray:
    columns = np.flatnonzero(mask.any(axis=0))
    if columns.size == 0:
        return np.empty((0, 2))
    n = max(1, min(int(max_points), columns.size))
    chosen = np.unique(columns[np.round(np.linspace(0, columns.size - 1, n)).astype(int)])
    out = []
    thick = 0
    for col in chosen:
        rows = np.flatnonzero(mask[:, col])
        if rows[-1] - rows[0] + 1 > 3 * rows.size:
            thick += 1
        out.append((float(col), float(np.median(rows))))
    if thick > 0.2 * len(out):
        notes.append(
            "Many columns contain several separate pieces of the series colour (crossing lines, "
            "markers or text): the curve is the median of them and may be off there."
        )
    return np.array(out, dtype=float)


def _bars(
    mask: np.ndarray, max_points: int, baseline_row: float, notes: list[str]
) -> np.ndarray:
    from skimage.measure import label, regionprops

    regions = regionprops(label(mask, connectivity=1))
    if not regions:
        return np.empty((0, 2))
    largest = max(r.area for r in regions)
    keep = [r for r in regions if r.area >= max(4.0, 0.03 * largest)]
    if len(regions) - len(keep):
        notes.append(f"{len(regions) - len(keep)} small speck(s) in the bar colour were ignored.")
    out = []
    for region in keep[:max_points]:
        min_row, min_col, max_row, max_col = region.bbox
        image = region.image
        filled = image.any(axis=0)
        tops = min_row + image.argmax(axis=0)[filled]
        bottoms = max_row - 1 - image[::-1].argmax(axis=0)[filled]
        top_edge = float(np.median(tops)) - 0.5
        bottom_edge = float(np.median(bottoms)) + 0.5
        edge = top_edge if abs(top_edge - baseline_row) >= abs(bottom_edge - baseline_row) else bottom_edge
        out.append(((min_col + max_col - 1) / 2.0, edge))
    if len(keep) > max_points:
        notes.append(f"More than {max_points} bars were found; only the first {max_points} are kept.")
    return np.array(out, dtype=float).reshape(-1, 2)


# ====================================================================== public entry points
def digitize(
    image_bytes: bytes | Image.Image | np.ndarray,
    x_axis: AxisCalibration,
    y_axis: AxisCalibration,
    *,
    color: str,
    tolerance: float = DEFAULT_TOLERANCE,
    mode: Mode = "points",
    max_points: int | None = None,
    plot_area: tuple[int, int, int, int] | None = None,
    series_name: str = "Digitized",
) -> DigitizeResult:
    """Extract approximate data points of one colour from a plot image.

    Parameters
    ----------
    x_axis, y_axis:
        Calibrations (two known pixels each).
    color:
        Hex colour of the series to read. ``tolerance`` is the allowed CIE76 Delta-E
        (colour distance in CIE Lab) around it.
    mode:
        ``"points"``: one point per marker (centroid of each connected blob).
        ``"curve"``: up to ``max_points`` evenly spaced columns; y = median of the
        matching pixels in each column.
        ``"bars"``: one point per bar: x = centre of the bar, y = its end away from zero.
    max_points:
        Upper limit of points (default: 5000 markers, 200 curve samples, 500 bars).
    plot_area:
        ``(left, top, right, bottom)`` pixel box to search (inclusive). Default: just inside
        the automatically detected axes; the whole image if no axes are found.

    The result is always flagged approximate; see :func:`to_table`.
    """
    if mode not in _DEFAULT_MAX_POINTS:
        raise DigitizeError(f"Unknown mode '{mode}'. Use 'points', 'curve' or 'bars'.")
    limit = int(max_points) if max_points is not None else _DEFAULT_MAX_POINTS[mode]
    if limit < 1:
        raise DigitizeError("max_points must be at least 1.")
    rgb = load_rgb(image_bytes)
    height, width = rgb.shape[:2]
    notes = [APPROXIMATE_NOTE]

    if plot_area is None:
        axes = find_axes(rgb)
        if axes is None:
            notes.append(
                "The plot area could not be detected, so the whole image was searched: text, legend "
                "entries or axis lines in the series colour may have been read as data."
            )
            area = (0, 0, width - 1, height - 1)
        else:
            area = _inner_box(axes)
    else:
        if len(plot_area) != 4:
            raise DigitizeError("plot_area must be (left, top, right, bottom).")
        area = tuple(int(round(v)) for v in plot_area)  # type: ignore[assignment]
    left, top, right, bottom = area
    left, top = max(0, left), max(0, top)
    right, bottom = min(width - 1, right), min(height - 1, bottom)
    if right <= left or bottom <= top:
        raise DigitizeError("The plot area is empty: check (left, top, right, bottom).")

    full = _mask(rgb, color, tolerance)
    mask = np.zeros_like(full)
    mask[top : bottom + 1, left : right + 1] = full[top : bottom + 1, left : right + 1]
    normalized = rgb_to_hex(hex_to_rgb(color))  # type: ignore[arg-type]
    if not mask.any():
        raise DigitizeError(
            f"No pixels close to {normalized} were found in the plot area. Pick the colour with "
            "candidate_colors() or increase the tolerance."
        )

    if mode == "points":
        pixels = _points(mask, limit, notes)
    elif mode == "curve":
        pixels = _curve(mask, limit, notes)
    else:
        zero_row = float(y_axis.to_pixel(0.0)) if not y_axis.log else float("nan")
        baseline = zero_row if (math.isfinite(zero_row) and top - 2 <= zero_row <= bottom + 2) else float(bottom)
        pixels = _bars(mask, limit, baseline, notes)
    if pixels.shape[0] == 0:
        raise DigitizeError(f"No {mode} could be read for colour {normalized}.")

    xs, ys = pixel_to_data(pixels[:, 0], pixels[:, 1], x_axis, y_axis)
    frame = pd.DataFrame({"x": np.asarray(xs, dtype=float), "y": np.asarray(ys, dtype=float),
                          "px": pixels[:, 0], "py": pixels[:, 1]})
    frame = frame.sort_values(["x", "y"], kind="stable").reset_index(drop=True)
    return DigitizeResult(
        points=frame[["x", "y"]].copy(),
        n=int(len(frame)),
        mode=mode,
        notes=notes,
        pixel_points=frame[["px", "py"]].copy(),
        series_name=str(series_name),
        color=normalized,
        plot_area=(left, top, right, bottom),
    )


def to_table(result: DigitizeResult, name: str | None = None) -> XYTable:
    """Wrap digitized points in an XY table flagged ``approximate=True``.

    The table name always ends in " (APPROXIMATE)" and its notes say the values are
    estimates. TorusPlot's statistics refuse such a table unless ``allow_approximate=True``
    is passed explicitly.
    """
    if result.n == 0:
        raise DigitizeError("There are no digitized points to put in a table.")
    base = (name or result.series_name or "Digitized").strip() or "Digitized"
    if not base.endswith(APPROXIMATE_SUFFIX.strip()):
        base = base + APPROXIMATE_SUFFIX
    series = (result.series_name or "Digitized").strip() or "Digitized"
    if series == "X":
        series = "Y"
    return XYTable.from_arrays(
        base,
        result.points["x"].to_numpy(dtype=float),
        {series: result.points["y"].to_numpy(dtype=float)},
        x_name="X",
        approximate=True,
        notes=APPROXIMATE_NOTE,
    )
