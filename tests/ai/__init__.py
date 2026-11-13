"""Shared helpers for the AI-layer tests. Everything is offline: mock provider or fakes only."""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from torusplot.data.samples import load_sample
from torusplot.stats.result import Comparison, StatResult

T_SIZE = "Nanoparticle diameter (nm)"
T_DOSE = "Viability (%)"
T_UPTAKE = "Uptake (a.u.)"
T_CELLS = "Per-cell uptake"
T_SURV = "Survival study"
T_PAIRED = "Tumour volume (mm3)"
T_CONT = "Response by treatment"

SAMPLE_NAMES = (
    "nanoparticle_size", "dose_response", "cell_uptake", "cell_uptake_cells", "survival", "paired", "contingency",
)

RESULT_ID = "sr_test0001"

#: Distinctive numbers stored in the fake result. None of them may ever reach a prompt.
SECRET_NUMBERS = (
    "0.0123456", "0.0234567", "0.0456789", "0.000098765", "9.8765e-05", "0.00019753", "31.4159", "21.2345",
    "56.7891", "0.0345678", "0.0678901", "0.8123456", "1.2345678",
)


@lru_cache(maxsize=1)
def _samples() -> dict:
    out = {}
    for name in SAMPLE_NAMES:
        table = load_sample(name)
        out[table.name] = table
    return out


def sample_tables() -> dict:
    """All sample tables by name (the tables are shared: do not modify them)."""
    return dict(_samples())


def make_result(result_id: str = RESULT_ID, table: str = T_SIZE) -> StatResult:
    """A hand-built one-way ANOVA result with distinctive p values."""
    result = StatResult(
        id=result_id,
        analysis="anova_oneway",
        test_name="One-way ANOVA",
        table=table,
        statistic_name="F",
        statistic=31.4159,
        df=3,
        df2=28,
        p_value=0.0123456,
        comparisons=[
            Comparison(id="", group_a="Bulk mixing", group_b="Sonication", estimate=21.2345,
                       p_value=0.0234567, p_adjusted=0.0456789),
            Comparison(id="", group_a="Bulk mixing", group_b="Microfluidic", estimate=56.7891,
                       p_value=0.000098765, p_adjusted=0.00019753),
            Comparison(id="", group_a="Sonication", group_b="Extrusion", estimate=1.2345678,
                       p_value=0.8123456, p_adjusted=0.8123456),
            Comparison(id="", group_a="Extrusion", group_b="Microfluidic", estimate=18.0,
                       p_value=0.0345678, p_adjusted=0.0678901),
        ],
    )
    return result.finalize()


def results_map() -> dict[str, StatResult]:
    result = make_result()
    return {result.id: result}


def all_prompt_text(provider: Any) -> str:
    """Everything a mock provider was sent, as one string."""
    return "\n".join(call["system"] + "\n" + call["prompt"] for call in provider.calls)


# ---------------------------------------------------------------------- synthetic images
def png_of(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def jpeg_of(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def flat_figure(
    colors: tuple[str, ...] = ("#1F77B4", "#FF7F0E", "#2CA02C"),
    *,
    size: tuple[int, int] = (400, 300),
    background: str = "#FFFFFF",
    ink: str = "#000000",
    box: bool = False,
    grid: bool = False,
) -> Image.Image:
    """A crisp (no anti-aliasing) bar-chart-like picture with exactly known colours.

    Axes: left line at x=50, bottom line at y=250 (2 px thick), spanning to x=380 / y=20.
    Bars get narrower from left to right, so areas are ordered like ``colors``.
    """
    width, height = size
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    if grid:
        for y in (80, 130, 180):
            draw.line([(52, y), (380, y)], fill="#DDDDDD", width=1)
    x = 70
    for i, color in enumerate(colors):
        bar_width = 70 - 12 * i
        draw.rectangle([x, 60 + 30 * i, x + bar_width, 248], fill=color)
        x += bar_width + 25
    draw.rectangle([50, 20, 51, 251], fill=ink)  # left axis
    draw.rectangle([50, 250, 380, 251], fill=ink)  # bottom axis
    if box:
        draw.rectangle([50, 20, 380, 21], fill=ink)
        draw.rectangle([379, 20, 380, 251], fill=ink)
    return image


class PlotImage:
    """A matplotlib-rendered plot with exact pixel <-> data calibration for digitizer tests."""

    def __init__(self, *, xlog: bool = False, ylog: bool = False, xlim=(0.0, 10.0), ylim=(0.0, 100.0),
                 size=(6.0, 4.0), dpi: int = 100) -> None:
        self.fig = Figure(figsize=size, dpi=dpi)
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot()
        if xlog:
            self.ax.set_xscale("log")
        if ylog:
            self.ax.set_yscale("log")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.spines[["top", "right"]].set_visible(False)
        self.xlog, self.ylog, self.xlim, self.ylim, self.dpi = xlog, ylog, xlim, ylim, dpi

    def png(self) -> bytes:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=self.dpi)
        return buf.getvalue()

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        """Data point -> (column, row) pixel-centre indices of the saved image."""
        self.fig.canvas.draw()
        dx, dy = self.ax.transData.transform((x, y))
        return float(dx - 0.5), float(self.fig.bbox.height - dy - 0.5)

    def calibrations(self):
        from torusplot.ai.digitize import AxisCalibration

        (x0, x1), (y0, y1) = self.xlim, self.ylim
        px0, py0 = self.pixel(x0, y0)
        px1, py1 = self.pixel(x1, y1)
        return (
            AxisCalibration(px0, x0, px1, x1, log=self.xlog),
            AxisCalibration(py0, y0, py1, y1, log=self.ylog),
        )

    def tolerance(self, axis: str, fraction: float = 0.015) -> float:
        """Allowed error: ``fraction`` of the axis range (of the log10 range on log axes)."""
        lo, hi = self.xlim if axis == "x" else self.ylim
        is_log = self.xlog if axis == "x" else self.ylog
        return fraction * (np.log10(hi) - np.log10(lo) if is_log else hi - lo)

    def error(self, axis: str, measured: Any, truth: Any) -> np.ndarray:
        """Absolute error in axis units (log10 units on log axes)."""
        is_log = self.xlog if axis == "x" else self.ylog
        m, t = np.asarray(measured, dtype=float), np.asarray(truth, dtype=float)
        return np.abs(np.log10(m) - np.log10(t)) if is_log else np.abs(m - t)
