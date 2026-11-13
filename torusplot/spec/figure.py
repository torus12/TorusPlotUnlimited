"""FigureSpec: the declarative description of a TorusPlot figure.

A figure is *data + spec*: the spec says which table to draw, which plot type and
layers to use, how it is styled and sized, and which statistics results annotate it.
Rendering the same spec with the same data always gives the same figure, which is what
makes ``.tplot`` projects reproducible and lets AI features work safely.

Design notes
------------
* Layers follow a small verb vocabulary (``mean_bar``, ``sem_errorbar``,
  ``data_points_beeswarm``, ``boxplot`` ...): add a layer to add an element, remove it
  to remove the element.
* **Statistics never live in a spec.** A :class:`StatAnnotation` points to a
  ``StatResult`` by ID (and optionally to specific comparison IDs / group pairs). The
  renderer looks the numbers up; free text that looks like a p value or significance
  stars is rejected by validation. AI features can therefore choose *which* computed
  comparisons to show, but can never invent or alter a number.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = 1

PlotType = Literal[
    "bar",
    "scatter",
    "box",
    "violin",
    "box_violin",
    "beeswarm",
    "superplot",
    "estimation",
    "paired",
    "xy",
    "dose_response",
    "survival",
    "grouped_bar",
    "stacked_bar",
    "histogram",
    "heatmap",
    "qq",
    "residuals",
]

LayerType = Literal[
    # central tendency
    "mean_bar",
    "median_bar",
    "mean_dash",
    "median_dash",
    "mean_dot",
    "mean_line",
    "stack_bar",
    # spread / uncertainty
    "sem_errorbar",
    "sd_errorbar",
    "ci95_errorbar",
    "range_errorbar",
    "iqr_errorbar",
    # raw data
    "data_points",
    "data_points_jitter",
    "data_points_beeswarm",
    "paired_lines",
    "replicate_means",
    # distributions
    "boxplot",
    "violin",
    "histogram",
    "density",
    # model / statistics-driven layers (drawn from a StatResult)
    "curve_fit",
    "confidence_band",
    "prediction_band",
    "km_curve",
    "km_ci",
    "censor_marks",
    "effect_axis",
    "qq_points",
    "qq_line",
    "residual_points",
    # other
    "heatmap",
    "reference_line",
    "value_labels",
    "n_labels",
]

#: Human-readable catalogue used by the UI, docs and the AI prompt template.
PLOT_TYPES: dict[str, dict[str, Any]] = {
    "bar": {
        "label": "Bar + individual points",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Bars at the mean with error bars; every data point is shown by default.",
    },
    "scatter": {
        "label": "Scatter dot plot (mean ± error)",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Every value as a dot with a line at the mean and an error bar.",
    },
    "box": {
        "label": "Box plot + points",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Median, quartiles and whiskers, with the raw points on top.",
    },
    "violin": {
        "label": "Violin plot + points",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Smoothed distribution shape for each group with the raw points.",
    },
    "box_violin": {
        "label": "Violin + box + points",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Violin outline, a slim box plot inside, and the raw points.",
    },
    "beeswarm": {
        "label": "Beeswarm",
        "tables": ("column", "grouped", "multiple_variables"),
        "description": "Points arranged so none overlap, with a line at the mean.",
    },
    "superplot": {
        "label": "SuperPlot (replicate-coloured)",
        "tables": ("multiple_variables",),
        "description": "Small dots = technical replicates, large dots = mean of each biological "
        "replicate, coloured by replicate; statistics use the replicate means.",
    },
    "estimation": {
        "label": "Estimation plot",
        "tables": ("column", "multiple_variables"),
        "description": "Raw data on the left, the mean difference and its 95% CI on the right.",
    },
    "paired": {
        "label": "Before-after (paired lines)",
        "tables": ("column", "multiple_variables"),
        "description": "One line per subject connecting its values across conditions.",
    },
    "xy": {
        "label": "XY (points, error bars, lines)",
        "tables": ("xy", "multiple_variables"),
        "description": "Y versus X with error bars from replicates; optional fitted line or curve.",
    },
    "dose_response": {
        "label": "Dose-response curve",
        "tables": ("xy",),
        "description": "Response versus dose on a log axis with the fitted sigmoidal curve.",
    },
    "survival": {
        "label": "Survival curve (Kaplan-Meier)",
        "tables": ("multiple_variables",),
        "description": "Staircase survival curves with tick marks for censored subjects.",
    },
    "grouped_bar": {
        "label": "Grouped bars",
        "tables": ("grouped", "multiple_variables"),
        "description": "Side-by-side bars for two grouping variables, with points and error bars.",
    },
    "stacked_bar": {
        "label": "Stacked bars",
        "tables": ("grouped", "multiple_variables", "contingency"),
        "description": "Bars stacked on top of each other (means or counts).",
    },
    "histogram": {
        "label": "Histogram",
        "tables": ("column", "multiple_variables", "xy"),
        "description": "Frequency distribution of values.",
    },
    "heatmap": {
        "label": "Heat map",
        "tables": ("grouped", "multiple_variables", "contingency", "column"),
        "description": "A colour-coded grid of values (means, counts or correlations).",
    },
    "qq": {
        "label": "Q-Q plot (normality)",
        "tables": ("column", "multiple_variables"),
        "description": "Data quantiles against normal quantiles; points on the line = normal.",
    },
    "residuals": {
        "label": "Residual plot",
        "tables": ("xy", "multiple_variables", "column", "grouped"),
        "description": "Residuals of a fitted model versus X or fitted value (needs a fit result).",
    },
}

_ERROR_LAYERS = ("sem_errorbar", "sd_errorbar", "ci95_errorbar", "range_errorbar", "iqr_errorbar")
_POINT_LAYERS = ("data_points", "data_points_jitter", "data_points_beeswarm")

_TO_MEDIAN = {"mean_bar": "median_bar", "mean_dash": "median_dash"}
_TO_MEAN = {v: k for k, v in _TO_MEDIAN.items()}

_P_TEXT = re.compile(
    r"(\bp\s*(-?\s*value)?\s*[<=>≤≥]\s*0?\.\d)|(^\s*(\*{1,4}|n\.?s\.?)\s*$)|(\bp\s*adj)", re.IGNORECASE
)


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataBinding(_Spec):
    """Which table to draw and the role of its columns."""

    table: str = Field(description="Name of the data table in the project")
    groups: list[str] | None = Field(
        default=None,
        description="Subset and order of groups (Column table columns, XY series, or the "
        "column-factor levels of a Grouped table). None = all, in table order.",
    )
    rows: list[str] | None = Field(
        default=None, description="Subset and order of row levels of a Grouped table. None = all."
    )
    x: str | None = Field(default=None, description="Multiple-variables table: column for the X axis / categories")
    y: str | None = Field(default=None, description="Multiple-variables table: numeric column to plot")
    hue: str | None = Field(default=None, description="Multiple-variables table: column for the colour grouping")
    replicate: str | None = Field(
        default=None, description="Multiple-variables table: biological-replicate column (SuperPlots)"
    )
    subject: str | None = Field(default=None, description="Multiple-variables table: subject ID (paired plots)")
    time: str | None = Field(default=None, description="Survival: time column")
    event: str | None = Field(default=None, description="Survival: event column (1 = event, 0 = censored)")
    columns: list[str] | None = Field(
        default=None, description="Multiple-variables table: columns for heat maps / histograms / Q-Q plots"
    )
    stat_result_id: str | None = Field(
        default=None,
        description="ID of the StatResult a statistics-driven plot is drawn from (survival curves, "
        "residuals, correlation heat map).",
    )
    transpose: bool = Field(default=False, description="Grouped table: swap the row and column factors")


class Layer(_Spec):
    """One drawn element. Unset options fall back to the style preset."""

    type: LayerType
    color: str | None = Field(default=None, description="Hex colour overriding the palette for this layer")
    edge_color: str | None = None
    alpha: float | None = Field(default=None, ge=0, le=1)
    size: float | None = Field(default=None, ge=0, description="Marker size in points")
    width: float | None = Field(default=None, gt=0, le=1, description="Bar/box/violin width (fraction of slot)")
    linewidth: float | None = Field(default=None, ge=0)
    linestyle: Literal["solid", "dashed", "dotted", "dashdot"] | None = None
    marker: str | None = Field(default=None, description="Matplotlib marker: o s ^ v D ...")
    fill: bool | None = None
    cap_size: float | None = Field(default=None, ge=0, description="Error-bar cap size in points")
    jitter: float | None = Field(default=None, ge=0, le=0.5, description="Horizontal spread of points")
    direction: Literal["both", "up", "down"] | None = Field(default=None, description="Error-bar direction")
    bins: int | None = Field(default=None, ge=1, le=500, description="Histogram bins (None = automatic)")
    value: float | None = Field(default=None, description="Position of a reference line")
    orientation: Literal["horizontal", "vertical"] | None = None
    label: str | None = Field(default=None, description="Legend label / text for reference lines")
    series: str | None = Field(default=None, description="Restrict this layer to one series/group")
    stat_result_id: str | None = Field(
        default=None, description="StatResult this layer is drawn from (curve fits, survival, effect axis)"
    )

    @field_validator("label")
    @classmethod
    def _no_stats_in_text(cls, v: str | None) -> str | None:
        return _reject_stat_text(v, "Layer label")


class StatAnnotation(_Spec):
    """Significance brackets / result text driven by a computed StatResult.

    There is deliberately no field for a p value or star string: the renderer reads
    them from the referenced result.
    """

    stat_result_id: str = Field(description="ID of the StatResult (e.g. 'sr_3f9a1c2b')")
    kind: Literal["brackets", "summary_text"] = Field(
        default="brackets",
        description="'brackets' draws pairwise comparison brackets; 'summary_text' writes the overall "
        "test and its P value in a corner.",
    )
    comparisons: Literal["all", "significant"] | list[str] = Field(
        default="all",
        description="Which comparisons of the result to draw: 'all', 'significant', or a list of "
        "comparison IDs ('sr_xxx:c1').",
    )
    pairs: list[tuple[str, str]] | None = Field(
        default=None, description="Alternative selection by group names, e.g. [['Control', 'Drug A']]"
    )
    format: Literal["stars", "p", "p_short", "stars_p"] = Field(
        default="stars",
        description="stars: ns/*/**/***/****; p: 'P = 0.0123'; p_short: '0.012'; stars_p: both",
    )
    show_ns: bool = Field(default=True, description="Draw brackets for non-significant comparisons")
    star_style: Literal["gp", "apa"] = Field(default="gp", description="gp: up to 4 stars; apa: up to 3")
    position: Literal["top left", "top right", "bottom left", "bottom right"] = Field(
        default="top left", description="Corner for summary_text"
    )
    engine: Literal["torus", "statannotations"] = Field(
        default="torus",
        description="Bracket drawing engine. 'torus' works for every plot type; 'statannotations' "
        "uses that library for simple categorical plots.",
    )
    line_width: float | None = Field(default=None, ge=0)
    text_size: float | None = Field(default=None, gt=0)
    tip_length: float | None = Field(default=None, ge=0, le=0.2, description="Bracket tip, fraction of axis height")
    step: float | None = Field(default=None, gt=0, le=0.5, description="Vertical spacing between stacked brackets")


class TextAnnotation(_Spec):
    """Free text placed on the figure (labels, notes). Not for statistics."""

    text: str
    x: float = Field(description="X position (axes fraction 0-1 by default)")
    y: float = Field(description="Y position (axes fraction 0-1 by default)")
    coords: Literal["axes", "data"] = "axes"
    size: float | None = Field(default=None, gt=0)
    color: str | None = None
    ha: Literal["left", "center", "right"] = "left"
    va: Literal["top", "center", "bottom"] = "top"
    weight: Literal["normal", "bold"] = "normal"
    style: Literal["normal", "italic"] = "normal"

    @field_validator("text")
    @classmethod
    def _no_stats_in_text(cls, v: str) -> str:
        return _reject_stat_text(v, "Text annotation") or ""


def _reject_stat_text(value: str | None, where: str) -> str | None:
    if value is not None and _P_TEXT.search(value):
        raise ValueError(
            f"{where} looks like a statistical result ('{value}'). Statistics cannot be typed into a "
            "figure: add a StatAnnotation that references a computed StatResult ID instead."
        )
    return value


class AxisSpec(_Spec):
    label: str | None = Field(default=None, description="Axis title; None = automatic from the data")
    scale: Literal["auto", "linear", "log10", "log2", "symlog"] = Field(
        default="auto",
        description="'auto' = linear, except the X axis of dose-response plots, which becomes log10 "
        "when the doses are on a linear scale",
    )
    min: float | None = None
    max: float | None = None
    ticks: list[float] | None = Field(default=None, description="Explicit tick positions")
    tick_labels: list[str] | None = Field(default=None, description="Labels for the explicit ticks")
    tick_rotation: float | None = Field(default=None, ge=-90, le=90)
    tick_format: str | None = Field(
        default=None, description="'%.1f'-style format, or 'sci', 'percent', 'log_exp' (10^n labels)"
    )
    grid: bool | None = None
    invert: bool = False
    show: bool = Field(default=True, description="Draw this axis (ticks, labels and spine)")

    @model_validator(mode="after")
    def _check(self) -> AxisSpec:
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("Axis min must be smaller than max.")
        if self.tick_labels is not None and (self.ticks is None or len(self.ticks) != len(self.tick_labels)):
            raise ValueError("tick_labels needs ticks of the same length.")
        return self


class AxesSpec(_Spec):
    x: AxisSpec = Field(default_factory=AxisSpec)
    y: AxisSpec = Field(default_factory=AxisSpec)
    zero_line: bool = Field(default=False, description="Draw a thin line at y = 0")


class StyleSpec(_Spec):
    """Style = a named preset plus optional overrides (None = use the preset's value)."""

    preset: str = Field(default="Torus Signature", description="Style preset name")
    palette: str | list[str] | None = Field(
        default=None, description="Palette name (e.g. 'Okabe-Ito', 'viridis') or a list of hex colours"
    )
    colors: dict[str, str] = Field(
        default_factory=dict, description="Per-group colour overrides: {group name: hex colour}"
    )
    font_family: str | None = None
    font_size: float | None = Field(default=None, gt=0, description="Base font size in points")
    title_size: float | None = Field(default=None, gt=0)
    label_size: float | None = Field(default=None, gt=0)
    tick_size: float | None = Field(default=None, gt=0)
    legend_size: float | None = Field(default=None, gt=0)
    axis_linewidth: float | None = Field(default=None, ge=0)
    line_width: float | None = Field(default=None, ge=0, description="Width of data lines and curves")
    errorbar_linewidth: float | None = Field(default=None, ge=0)
    cap_size: float | None = Field(default=None, ge=0)
    point_size: float | None = Field(default=None, ge=0)
    point_alpha: float | None = Field(default=None, ge=0, le=1)
    point_edge_color: str | None = None
    point_edge_width: float | None = Field(default=None, ge=0)
    point_color_mode: Literal["group", "dark", "black", "white"] | None = Field(
        default=None, description="Colour of raw points drawn over bars/boxes"
    )
    bar_width: float | None = Field(default=None, gt=0, le=1)
    bar_alpha: float | None = Field(default=None, ge=0, le=1)
    bar_edge_color: str | None = Field(default=None, description="Hex colour, or 'group' to match the fill")
    bar_edge_width: float | None = Field(default=None, ge=0)
    bar_fill: bool | None = Field(default=None, description="False = outlined (hollow) bars")
    spines: list[Literal["left", "bottom", "top", "right"]] | None = None
    tick_direction: Literal["out", "in", "inout"] | None = None
    tick_length: float | None = Field(default=None, ge=0)
    grid: bool | None = None
    grid_color: str | None = None
    background: str | None = None
    text_color: str | None = None
    axis_color: str | None = None
    colormap: str | None = Field(default=None, description="Colormap for heat maps / gradients")
    gradient: str | None = Field(
        default=None, description="Colour groups along a gradient: a colormap name (e.g. 'viridis')"
    )


class LayoutSpec(_Spec):
    width_mm: float | None = Field(default=None, gt=10, le=1000, description="Figure width in mm")
    height_mm: float | None = Field(default=None, gt=10, le=1000, description="Figure height in mm")
    orientation: Literal["vertical", "horizontal"] = Field(
        default="vertical", description="horizontal draws categories on the Y axis"
    )


class LegendSpec(_Spec):
    show: bool | None = Field(default=None, description="None = automatic (shown when there is >1 series)")
    position: Literal[
        "best", "upper right", "upper left", "lower left", "lower right", "right", "outside right", "top", "bottom"
    ] = "best"
    title: str | None = None
    frame: bool = False
    ncol: int = Field(default=1, ge=1, le=10)


class ExportSpec(_Spec):
    formats: list[Literal["svg", "pdf", "tiff", "png"]] = Field(default_factory=lambda: ["svg", "pdf", "png"])
    dpi: int = Field(default=300, ge=72, le=1200)
    transparent: bool = False


class FigureSpec(_Spec):
    """Complete, re-renderable description of one figure."""

    version: int = Field(default=SPEC_VERSION, description="Spec format version")
    name: str = Field(default="Figure 1", description="Figure name inside the project")
    plot_type: PlotType = Field(description="Kind of graph")
    data: DataBinding
    layers: list[Layer] = Field(
        default_factory=list,
        description="Elements to draw, bottom to top. Empty = the default layers of the plot type.",
    )
    annotations: list[StatAnnotation] = Field(default_factory=list)
    texts: list[TextAnnotation] = Field(default_factory=list)
    title: str | None = None
    axes: AxesSpec = Field(default_factory=AxesSpec)
    style: StyleSpec = Field(default_factory=StyleSpec)
    layout: LayoutSpec = Field(default_factory=LayoutSpec)
    legend: LegendSpec = Field(default_factory=LegendSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)

    @field_validator("title")
    @classmethod
    def _no_stats_in_title(cls, v: str | None) -> str | None:
        return _reject_stat_text(v, "Figure title")

    @model_validator(mode="after")
    def _check_version(self) -> FigureSpec:
        if self.version > SPEC_VERSION:
            raise ValueError(
                f"This figure was made with a newer spec version ({self.version}); "
                f"this TorusPlot understands up to version {SPEC_VERSION}."
            )
        return self

    # ------------------------------------------------------------------ helpers
    def effective_layers(self) -> list[Layer]:
        """The layers that will be drawn (defaults filled in when ``layers`` is empty)."""
        return list(self.layers) if self.layers else default_layers(self.plot_type)

    def stat_result_ids(self) -> list[str]:
        """Every StatResult ID this figure depends on."""
        ids = [a.stat_result_id for a in self.annotations]
        ids += [lay.stat_result_id for lay in self.layers if lay.stat_result_id]
        if self.data.stat_result_id:
            ids.append(self.data.stat_result_id)
        return list(dict.fromkeys(ids))

    def with_error_bars(self, kind: str | None) -> FigureSpec:
        """Return a copy using ``sem`` / ``sd`` / ``ci95`` / ``range`` / ``iqr`` error bars (None removes them)."""
        layers = [lay for lay in self.effective_layers() if lay.type not in _ERROR_LAYERS]
        if kind:
            # keep the centre consistent with the spread: IQR goes with the median, the others
            # (SEM, SD, CI, range) with the mean, so the graph and the methods sentence agree
            swap = _TO_MEDIAN if kind == "iqr" else _TO_MEAN
            layers = [lay.model_copy(update={"type": swap[lay.type]}) if lay.type in swap else lay for lay in layers]
            new = Layer(type=f"{kind}_errorbar")  # type: ignore[arg-type]
            idx = next((i for i, lay in enumerate(layers) if lay.type in _POINT_LAYERS), len(layers))
            layers.insert(idx, new)
        return self.model_copy(update={"layers": layers})

    def with_points(self, kind: str | None) -> FigureSpec:
        """Return a copy with ``jitter`` / ``beeswarm`` / ``plain`` points (None hides them)."""
        layers = [lay for lay in self.effective_layers() if lay.type not in _POINT_LAYERS]
        if kind:
            name = {"jitter": "data_points_jitter", "beeswarm": "data_points_beeswarm", "plain": "data_points"}[kind]
            layers.append(Layer(type=name))  # type: ignore[arg-type]
        return self.model_copy(update={"layers": layers})


class PanelSpec(_Spec):
    """One panel of a multi-panel figure."""

    figure: str = Field(description="Name of a figure in the project")
    row: int = Field(ge=0, description="Top row of the panel (0-based)")
    col: int = Field(ge=0, description="Left column of the panel (0-based)")
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    label: str | None = Field(default=None, description="Panel letter override (None = automatic)")


class CompositionSpec(_Spec):
    """Multi-panel layout on a grid with mm-exact outer size."""

    version: int = SPEC_VERSION
    name: str = "Composition 1"
    rows: int = Field(default=1, ge=1, le=12)
    cols: int = Field(default=2, ge=1, le=12)
    panels: list[PanelSpec] = Field(default_factory=list)
    width_mm: float = Field(default=183.0, gt=10, le=2000)
    height_mm: float = Field(default=90.0, gt=10, le=2000)
    panel_labels: Literal["A", "a", "(A)", "(a)", "1", "none"] = Field(default="A", description="Letter style")
    panel_label_size: float | None = Field(default=None, gt=0)
    panel_label_weight: Literal["normal", "bold"] = "bold"
    share_x: Literal["none", "row", "col", "all"] = "none"
    share_y: Literal["none", "row", "col", "all"] = "none"
    shared_legend: bool = Field(default=False, description="One legend for the whole figure")
    legend_position: Literal["bottom", "top", "right"] = "bottom"
    style_preset: str | None = Field(default=None, description="Apply one preset to every panel")
    wspace_mm: float = Field(default=6.0, ge=0, le=100)
    hspace_mm: float = Field(default=6.0, ge=0, le=100)
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="after")
    def _check_grid(self) -> CompositionSpec:
        occupied: set[tuple[int, int]] = set()
        for p in self.panels:
            if p.row + p.rowspan > self.rows or p.col + p.colspan > self.cols:
                raise ValueError(f"Panel '{p.figure}' does not fit in the {self.rows} x {self.cols} grid.")
            cells = {(r, c) for r in range(p.row, p.row + p.rowspan) for c in range(p.col, p.col + p.colspan)}
            if occupied & cells:
                raise ValueError(f"Panel '{p.figure}' overlaps another panel.")
            occupied |= cells
        return self


def default_layers(plot_type: str) -> list[Layer]:
    """Default layer stack for each plot type.

    Group-comparison plots (bar, scatter, box, violin, beeswarm, SuperPlot, estimation, paired)
    include the raw data points by default; the user can hide them explicitly.
    """

    def L(*names: str) -> list[Layer]:
        return [Layer(type=n) for n in names]  # type: ignore[arg-type]

    table = {
        "bar": L("mean_bar", "sem_errorbar", "data_points_jitter"),
        "grouped_bar": L("mean_bar", "sem_errorbar", "data_points_jitter"),
        "stacked_bar": L("stack_bar"),
        "scatter": L("data_points_jitter", "mean_dash", "sd_errorbar"),
        "box": L("boxplot", "data_points_jitter"),
        "violin": L("violin", "data_points_jitter", "median_dash"),
        "box_violin": L("violin", "boxplot", "data_points_jitter"),
        "beeswarm": L("data_points_beeswarm", "mean_dash", "sd_errorbar"),
        "superplot": L("data_points_beeswarm", "replicate_means", "mean_dash", "sem_errorbar"),
        "estimation": L("data_points_beeswarm", "mean_dash", "ci95_errorbar", "effect_axis"),
        "paired": L("paired_lines", "data_points"),
        "xy": L("mean_line", "sem_errorbar", "mean_dot"),
        "dose_response": L("sem_errorbar", "mean_dot", "curve_fit"),
        "survival": L("km_curve", "censor_marks"),
        "histogram": L("histogram"),
        "heatmap": L("heatmap"),
        "qq": L("qq_line", "qq_points"),
        "residuals": L("reference_line", "residual_points"),
    }
    if plot_type not in table:
        raise ValueError(f"Unknown plot type '{plot_type}'.")
    return table[plot_type]


def plot_types_for(table_kind: str) -> list[str]:
    """Plot types that make sense for a table kind (drives the UI's graph picker)."""
    return [k for k, info in PLOT_TYPES.items() if table_kind in info["tables"]]
