"""MCP server for TorusPlot: a thin layer over :class:`~torusplot.mcp_server.session.TorusSession`.

Built on the high-level ``MCPServer`` class of the MCP Python SDK 2.x
(``mcp.server.mcpserver``; it was called ``FastMCP`` in SDK 1.x). Every tool here only
forwards its typed arguments to the session method of the same name and turns a
:class:`SessionError` into the SDK's ``ToolError`` (reported to the client as a tool
result with ``is_error=True`` and the user-facing message; no traceback).

The tool descriptions are what the calling model reads, so they state valid values and
what to call next. Annotations are evaluated eagerly in this module on purpose (no
``from __future__ import annotations``): the SDK builds each tool's input schema from them.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .. import __version__
from ..data.samples import SAMPLES
from ..plot import palette_names, preset_names
from ..spec import PLOT_TYPES
from .session import (
    ANNOTATION_FORMATS,
    AXIS_SCALES,
    LEGEND_POSITIONS,
    PANEL_LABEL_STYLES,
    SHARE_MODES,
    TABLE_FILE_SUFFIXES,
    SessionError,
    TorusSession,
)

SERVER_NAME = "torusplot"

#: The eight tools every TorusPlot MCP server must offer (checked by the tests).
REQUIRED_TOOLS = (
    "load_table",
    "recommend_test",
    "run_analysis",
    "make_figure",
    "restyle_figure",
    "compose_panels",
    "export_figure",
    "get_stats_report",
)

INSTRUCTIONS = """\
TorusPlot is a local scientific graphing and statistics engine. Use these tools to load \
data, run statistics and build publication-quality figures. Nothing leaves this computer.

Typical workflow:
1. load_table: create a table from pasted text, a file or a bundled synthetic sample.
2. recommend_test (or list_analyses): find the right analysis for the design.
3. run_analysis: compute it. The result has an ID such as 'sr_3f9a1c2b'.
4. make_figure: draw the table; attach statistics with annotate_result_id / fit_result_ids.
5. restyle_figure / compose_panels: adjust the look, build multi-panel figures.
6. export_figure / get_stats_report: write SVG/PDF/PNG/TIFF files and the statistics report.

THE HARD RULE: you choose what to analyse and how the figure looks, but every statistic \
comes from run_analysis. Never type, estimate, round differently or invent a p value, \
significance stars, a confidence interval or a test statistic: there is no tool input for \
them, and text that looks like one is rejected. Figures show statistics only by \
referencing a result ID. When you tell the user about results, quote the numbers exactly \
as the tools returned them (test_name, statistic, p_value / p_adjusted, effect sizes, \
interpretation, methods_text).
"""

TableKind = Literal["column", "xy", "grouped", "multiple_variables", "contingency"]
ErrorBars = Literal["sem", "sd", "ci95", "range", "iqr", "none"]
Points = Literal["jitter", "beeswarm", "plain", "none"]
ExportFormat = Literal["svg", "pdf", "png", "tiff"]
ReportFormat = Literal["markdown", "csv"]
AnnotationKind = Literal["auto", "brackets", "summary_text"]

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
_ADDITIVE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
_REPLACING = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)


def _quoted(values: Any) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _call(method: Any, /, **kwargs: Any) -> dict[str, Any]:
    """Run a session method; report anticipated failures the way the SDK expects (ToolError)."""
    try:
        return method(**kwargs)
    except SessionError as exc:
        raise ToolError(str(exc)) from exc


def _with_preview(session: TorusSession, payload: dict[str, Any]) -> CallToolResult:
    """Structured payload as JSON text + the PNG preview as an MCP image content item."""
    content: list[Any] = [TextContent(type="text", text=json.dumps(payload, indent=1, ensure_ascii=False))]
    png = session.preview_png(payload["name"])
    if png:
        content.append(Image(data=png, format="png").to_image_content())
    return CallToolResult(content=content, structured_content=payload)


def build_server(
    session: TorusSession | None = None,
    output_dir: str | Path | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
) -> MCPServer:
    """Create the MCP server with every TorusPlot tool registered (does not start it).

    One :class:`TorusSession` (one in-memory project) backs the server; it is available
    afterwards as ``server.torus_session``. The output directory comes from
    ``output_dir``, else the ``TORUSPLOT_OUTPUT_DIR`` environment variable, else
    ``./torusplot_output``.
    """
    if session is None:
        session = TorusSession(output_dir)
    server = MCPServer(
        SERVER_NAME,
        title="TorusPlot",
        description="Scientific graphing and statistics through TorusPlot's own engine (local, no network).",
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level=log_level,
    )
    server.torus_session = session  # type: ignore[attr-defined]

    samples = "; ".join(f"'{key}' ({info['kind']})" for key, info in SAMPLES.items())
    plot_types = "; ".join(
        f"'{key}' ({info['label']}; tables: {', '.join(info['tables'])})" for key, info in PLOT_TYPES.items()
    )
    presets = _quoted(preset_names())
    palettes = _quoted(palette_names())
    suffixes = ", ".join(TABLE_FILE_SUFFIXES)

    # ------------------------------------------------------------------ data
    @server.tool(
        name="load_table",
        title="Load a data table",
        annotations=_REPLACING,
        description=(
            "Create a data table from exactly ONE source: pasted delimited text (`text`), a file on this "
            f"computer (`path`; only {suffixes}), or a bundled SYNTHETIC sample (`sample`). The table type "
            "decides which analyses and graphs are possible:\n"
            "- 'column': each column is one group, rows are replicates (t tests, one-way ANOVA, bar/box plots).\n"
            "- 'xy': first column is X, the others are Y series, optionally with side-by-side replicates "
            "(regression, dose-response, time courses).\n"
            "- 'grouped': first column holds row labels, the others are groups with replicate sub-columns "
            "(two-way ANOVA, grouped bars).\n"
            "- 'multiple_variables': tidy data, one row per observation and one column per variable "
            "(survival, nested/SuperPlot data, multiple regression, correlation matrix).\n"
            "- 'contingency': first column holds row labels, the others are counts (chi-square, Fisher).\n"
            "Returns the table summary (shape, group/series/column names, row levels, n per group, first rows), "
            "the analyses valid for it (key, label, category) and the valid plot types. A table with the same "
            "name is replaced. Next: recommend_test or list_analyses, then run_analysis; or make_figure."
        ),
    )
    def load_table(
        name: Annotated[str, Field(description="Name for the table; later tools refer to the table by this name.")],
        kind: Annotated[
            TableKind | None,
            Field(description="Table type. Required with `text` or `path`; optional with `sample` (each sample has a fixed type)."),
        ] = None,
        text: Annotated[
            str | None,
            Field(description="Delimited text including the header row (tab, comma or semicolon separated; as copied from a spreadsheet)."),
        ] = None,
        path: Annotated[str | None, Field(description=f"Path of a data file on this computer ({suffixes}).")] = None,
        sample: Annotated[str | None, Field(description=f"Key of a bundled synthetic sample: {samples}.")] = None,
        replicates: Annotated[
            int | None,
            Field(description="xy / grouped tables: number of side-by-side replicate columns per series or group. Omit when the columns are titled 'Name:1', 'Name:2', ... or there are no replicates."),
        ] = None,
        row_label: Annotated[str | None, Field(description="grouped / contingency: name of the row factor (e.g. 'Time').")] = None,
        column_label: Annotated[str | None, Field(description="grouped / contingency: name of the column factor (e.g. 'Treatment').")] = None,
        sheet: Annotated[str | None, Field(description="Worksheet name for .xlsx files (default: the first sheet).")] = None,
    ) -> dict[str, Any]:
        return _call(session.load_table, name=name, kind=kind, text=text, path=path, sample=sample,
                     replicates=replicates, row_label=row_label, column_label=column_label, sheet=sheet)

    @server.tool(
        name="list_state",
        title="List tables, results and figures",
        annotations=_READ_ONLY,
        description=(
            "List everything in the current session with one-line summaries: tables, statistics results "
            "(with their IDs), figures, multi-panel compositions, the output directory and the bundled samples. "
            "Call it to recover names and result IDs."
        ),
    )
    def list_state() -> dict[str, Any]:
        return _call(session.list_state)

    # ------------------------------------------------------------------ statistics
    @server.tool(
        name="list_analyses",
        title="List analyses and their options",
        annotations=_READ_ONLY,
        description=(
            "List every analysis that is valid for a table, each with its parameter specs (name, kind, default, "
            "choices, help) and the table's group/series/column names to use as parameter values. Pass "
            "`analysis` to get the options of just one analysis (a much shorter answer). "
            "Next: run_analysis(table, analysis=<key>, params={...})."
        ),
    )
    def list_analyses(
        table: Annotated[str, Field(description="Name of a loaded table.")],
        analysis: Annotated[str | None, Field(description="Optional analysis key: return only this analysis.")] = None,
    ) -> dict[str, Any]:
        return _call(session.list_analyses, table=table, analysis=analysis)

    @server.tool(
        name="recommend_test",
        title="Recommend a statistical test",
        annotations=_READ_ONLY,
        description=(
            "Inspect a table's design and data (number of groups, pairing, normality, equal variances) and "
            "recommend an analysis in plain English. Returns the analysis key, ready-to-use params, the reason, "
            "the decision steps, the assumption checks, alternatives and cautions. It does NOT run the test: "
            "call run_analysis with the returned `analysis` and `params` next."
        ),
    )
    def recommend_test(
        table: Annotated[str, Field(description="Name of a loaded table.")],
        paired: Annotated[
            bool | None,
            Field(description="True if the same subjects were measured repeatedly (rows of a column table, replicate positions of a grouped table). Omit for independent groups."),
        ] = None,
        control: Annotated[str | None, Field(description="column tables: name of the control group (3+ groups then compare against it, Dunnett style).")] = None,
        groups: Annotated[list[str] | None, Field(description="column tables: subset of groups to compare (default: all).")] = None,
        goal: Annotated[
            str | None,
            Field(description="Optional hint. xy tables: 'dose_response' or 'relationship'. multiple_variables tables: 'survival', 'nested', 'compare', 'regression' or 'ancova'."),
        ] = None,
    ) -> dict[str, Any]:
        return _call(session.recommend_test, table=table, paired=paired, control=control, groups=groups, goal=goal)

    @server.tool(
        name="run_analysis",
        title="Run a statistical analysis",
        annotations=_ADDITIVE,
        description=(
            "Run a registered analysis on a table with TorusPlot's statistics engine and store the result. "
            "`analysis` is a key from load_table / list_analyses / recommend_test (e.g. 'anova_oneway', "
            "'ttest_welch', 'nonlinear_regression'); `params` are that analysis's options (see list_analyses; "
            "omitted options use their defaults). Returns the computed result verbatim: id, test_name, why_chosen, "
            "statistic (+ statistic_name, df, df2), p_value, p_adjusted, significant, effect_sizes, parameters, "
            "anova_table, comparisons (each with its own id), assumptions (status + detail), warnings, "
            "interpretation and methods_text. Quote these numbers exactly; never restate them from memory. "
            "Next: make_figure(..., annotate_result_id=<id>) for brackets or a test summary, "
            "make_figure(..., fit_result_ids=[<id>]) for fitted curves, or get_stats_report."
        ),
    )
    def run_analysis(
        table: Annotated[str, Field(description="Name of a loaded table.")],
        analysis: Annotated[str, Field(description="Analysis key, e.g. 'anova_oneway'.")],
        params: Annotated[
            dict[str, Any] | None,
            Field(description="Options of the analysis as {name: value}, e.g. {\"posthoc\": \"dunnett\", \"control\": \"Vehicle\"}."),
        ] = None,
        allow_approximate: Annotated[
            bool,
            Field(description="Only for tables flagged as approximate (digitized from an image): run anyway and flag the result."),
        ] = False,
    ) -> dict[str, Any]:
        return _call(session.run_analysis, table=table, analysis=analysis, params=params,
                     allow_approximate=allow_approximate)

    @server.tool(
        name="get_result",
        title="Get a stored statistics result",
        annotations=_READ_ONLY,
        description="Return a stored statistics result by ID (same payload as run_analysis). Use it to re-read exact numbers.",
    )
    def get_result(
        result_id: Annotated[str, Field(description="Result ID returned by run_analysis, e.g. 'sr_3f9a1c2b'.")],
    ) -> dict[str, Any]:
        return _call(session.get_result, result_id=result_id)

    # ------------------------------------------------------------------ figures
    @server.tool(
        name="make_figure",
        title="Make a figure",
        annotations=_REPLACING,
        description=(
            "Build a figure from a table, store it under `name` (an existing figure with that name is replaced), "
            "render it and return its spec, rendering warnings, the drawn brackets and a PNG preview (also saved "
            f"as <output_dir>/<name>_preview.png). Plot types: {plot_types}.\n"
            "Statistics are attached by result ID only: `annotate_result_id` draws significance brackets "
            "(categorical plots, results with pairwise comparisons) or a one-line test summary, with labels "
            "computed from that result; `fit_result_ids` draws fitted curves of linear/nonlinear regression "
            "results on 'xy' / 'dose_response' plots; `data_result_id` draws 'survival', 'residuals', 'qq' or a "
            "correlation 'heatmap' from a result. Titles, labels and texts that look like a p value or stars "
            "are rejected.\n"
            "multiple_variables tables need column roles: `y` (numeric values) plus `x` (categories or numeric "
            "X), optional `hue`, `replicate` (SuperPlot), `subject` (paired). Use `spec` only for options the "
            "simple arguments do not cover (see get_figure_schema); simple arguments override it. "
            "Next: restyle_figure, compose_panels or export_figure."
        ),
    )
    def make_figure(
        name: Annotated[str, Field(description="Figure name (also the export file name).")],
        table: Annotated[str | None, Field(description="Name of the table to plot (required unless spec.data.table is given).")] = None,
        plot_type: Annotated[str | None, Field(description="Plot type key, e.g. 'bar', 'box', 'xy', 'dose_response' (required unless spec.plot_type is given). Must suit the table type.")] = None,
        spec: Annotated[dict[str, Any] | None, Field(description="Optional full or partial FigureSpec (JSON object) for advanced control: layers, axes, legend, texts...")] = None,
        annotate_result_id: Annotated[str | None, Field(description="ID of a run_analysis result whose comparisons / test summary annotate the figure.")] = None,
        comparisons: Annotated[
            str | list[str],
            Field(description="Which comparisons get a bracket: 'all', 'significant', or a list of comparison IDs such as ['sr_3f9a1c2b:c1']."),
        ] = "all",
        annotation_format: Annotated[str, Field(description=f"Bracket label format, one of {_quoted(ANNOTATION_FORMATS)}: stars (ns, *, **...), p ('P = 0.0123'), p_short ('0.0123'), stars_p (both).")] = "stars",
        fit_result_ids: Annotated[list[str] | None, Field(description="IDs of regression results whose fitted curves are drawn (xy / dose_response plots).")] = None,
        title: Annotated[str | None, Field(description="Figure title.")] = None,
        x_label: Annotated[str | None, Field(description="X axis title (default: from the data).")] = None,
        y_label: Annotated[str | None, Field(description="Y axis title (default: from the data).")] = None,
        style_preset: Annotated[str | None, Field(description=f"Style preset, one of {presets}.")] = None,
        palette: Annotated[str | list[str] | None, Field(description=f"Palette name ({palettes}) or a list of hex colours.")] = None,
        width_mm: Annotated[float | None, Field(description="Figure width in millimetres (default: from the preset, e.g. 89).")] = None,
        height_mm: Annotated[float | None, Field(description="Figure height in millimetres.")] = None,
        x: Annotated[str | None, Field(description="multiple_variables tables: column for the X axis / categories.")] = None,
        y: Annotated[str | None, Field(description="multiple_variables tables: numeric column to plot.")] = None,
        hue: Annotated[str | None, Field(description="multiple_variables tables: column for the colour grouping (also the group column of survival plots).")] = None,
        replicate: Annotated[str | None, Field(description="multiple_variables tables: biological-replicate column (SuperPlots).")] = None,
        subject: Annotated[str | None, Field(description="multiple_variables tables: subject ID column (paired plots).")] = None,
        groups: Annotated[list[str] | None, Field(description="Subset and order of groups / series to draw (default: all, in table order).")] = None,
        data_result_id: Annotated[str | None, Field(description="ID of the result a statistics-driven plot is drawn from (survival, residuals, qq, correlation heatmap).")] = None,
        annotation_kind: Annotated[AnnotationKind, Field(description="'auto' (brackets when the result has pairwise comparisons and the plot is categorical, else a summary line), 'brackets' or 'summary_text'.")] = "auto",
    ) -> CallToolResult:
        payload = _call(
            session.make_figure, name=name, table=table, plot_type=plot_type, spec=spec,
            annotate_result_id=annotate_result_id, comparisons=comparisons, annotation_format=annotation_format,
            fit_result_ids=fit_result_ids, title=title, x_label=x_label, y_label=y_label,
            style_preset=style_preset, palette=palette, width_mm=width_mm, height_mm=height_mm, x=x, y=y, hue=hue,
            replicate=replicate, subject=subject, groups=groups, data_result_id=data_result_id,
            annotation_kind=annotation_kind,
        )
        return _with_preview(session, payload)

    @server.tool(
        name="restyle_figure",
        title="Restyle a figure",
        annotations=_REPLACING,
        description=(
            "Change ONLY the styling and layout of an existing figure (never its data or statistics), re-render "
            "it and return the same payload as make_figure, with a new PNG preview. Give only what should "
            "change; everything else is kept. An empty string clears a title or label. Call list_styles for "
            "presets, palettes and every style option. Next: export_figure."
        ),
    )
    def restyle_figure(
        name: Annotated[str, Field(description="Name of an existing figure.")],
        style_preset: Annotated[str | None, Field(description=f"Style preset, one of {presets}.")] = None,
        palette: Annotated[str | list[str] | None, Field(description=f"Palette name ({palettes}) or a list of hex colours.")] = None,
        colors: Annotated[dict[str, str] | None, Field(description="Per-group colour overrides, {group name: hex colour}.")] = None,
        font_family: Annotated[str | None, Field(description="Font family, e.g. 'Inter', 'Source Sans 3', 'Liberation Sans' (bundled) or an installed font.")] = None,
        font_size: Annotated[float | None, Field(description="Base font size in points.")] = None,
        width_mm: Annotated[float | None, Field(description="Figure width in millimetres.")] = None,
        height_mm: Annotated[float | None, Field(description="Figure height in millimetres.")] = None,
        title: Annotated[str | None, Field(description="Figure title ('' removes it).")] = None,
        x_label: Annotated[str | None, Field(description="X axis title ('' restores the automatic one).")] = None,
        y_label: Annotated[str | None, Field(description="Y axis title ('' restores the automatic one).")] = None,
        y_min: Annotated[float | None, Field(description="Lower limit of the Y axis.")] = None,
        y_max: Annotated[float | None, Field(description="Upper limit of the Y axis.")] = None,
        y_scale: Annotated[str | None, Field(description=f"Y axis scale, one of {_quoted(AXIS_SCALES)}.")] = None,
        x_scale: Annotated[str | None, Field(description=f"X axis scale, one of {_quoted(AXIS_SCALES)}.")] = None,
        error_bars: Annotated[ErrorBars | None, Field(description="Error bar type: SEM, SD, 95% CI, range, interquartile range, or none.")] = None,
        points: Annotated[Points | None, Field(description="How individual data points are drawn: jitter, beeswarm, plain (no spread) or none.")] = None,
        legend_position: Annotated[str | None, Field(description=f"Legend position, one of {_quoted(LEGEND_POSITIONS)}.")] = None,
        style_overrides: Annotated[
            dict[str, Any] | None,
            Field(description="Other style options merged into the figure's style, e.g. {\"bar_width\": 0.6, \"point_size\": 4, \"spines\": [\"left\", \"bottom\"], \"grid\": true}. See list_styles -> style_options."),
        ] = None,
    ) -> CallToolResult:
        payload = _call(
            session.restyle_figure, name=name, style_preset=style_preset, palette=palette, colors=colors,
            font_family=font_family, font_size=font_size, width_mm=width_mm, height_mm=height_mm, title=title,
            x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max, y_scale=y_scale, x_scale=x_scale,
            error_bars=error_bars, points=points, legend_position=legend_position, style_overrides=style_overrides,
        )
        return _with_preview(session, payload)

    @server.tool(
        name="compose_panels",
        title="Compose a multi-panel figure",
        annotations=_REPLACING,
        description=(
            "Arrange existing figures as lettered panels of one multi-panel figure with an exact size in "
            "millimetres, store it under `name`, render it and return its spec and a PNG preview. `figures` is a "
            "list of figure names placed row by row on an automatic grid (set `rows` / `cols` to control it); or "
            "give an explicit `layout` = [{\"figure\": name, \"row\": 0, \"col\": 0, \"rowspan\": 1, \"colspan\": 1}, ...] "
            "(0-based). 183 mm is a typical double-column journal width, 89 mm single-column. "
            "Next: export_figure(name=<composition name>)."
        ),
    )
    def compose_panels(
        name: Annotated[str, Field(description="Name for the composition (must differ from figure names; also the export file name).")],
        figures: Annotated[list[str] | None, Field(description="Names of existing figures, in reading order.")] = None,
        rows: Annotated[int | None, Field(description="Grid rows (default: automatic).")] = None,
        cols: Annotated[int | None, Field(description="Grid columns (default: automatic).")] = None,
        width_mm: Annotated[float, Field(description="Total width in millimetres.")] = 183.0,
        height_mm: Annotated[float | None, Field(description="Total height in millimetres (default: from the grid shape).")] = None,
        panel_labels: Annotated[str, Field(description=f"Panel letter style, one of {_quoted(PANEL_LABEL_STYLES)}.")] = "A",
        share_x: Annotated[str, Field(description=f"Give panels the same X range: one of {_quoted(SHARE_MODES)}.")] = "none",
        share_y: Annotated[str, Field(description=f"Give panels the same Y range: one of {_quoted(SHARE_MODES)}.")] = "none",
        shared_legend: Annotated[bool, Field(description="One legend for the whole figure instead of one per panel.")] = False,
        style_preset: Annotated[str | None, Field(description=f"Apply one style preset to every panel: {presets}.")] = None,
        layout: Annotated[list[dict[str, Any]] | None, Field(description="Explicit panel placement (replaces `figures`).")] = None,
    ) -> CallToolResult:
        payload = _call(
            session.compose_panels, name=name, figures=figures, rows=rows, cols=cols, width_mm=width_mm,
            height_mm=height_mm, panel_labels=panel_labels, share_x=share_x, share_y=share_y,
            shared_legend=shared_legend, style_preset=style_preset, layout=layout,
        )
        return _with_preview(session, payload)

    # ------------------------------------------------------------------ export
    @server.tool(
        name="export_figure",
        title="Export a figure",
        annotations=_ADDITIVE,
        description=(
            "Export a figure or a composition (looked up by name) into the output directory. SVG keeps text as "
            "editable text with named groups (for vector editors); PDF embeds fonts; TIFF is LZW-compressed; "
            "PNG/TIFF use `dpi`. With `with_stats_report` it also writes <name>_stats.md and <name>_stats.csv "
            "for the statistics results the figure(s) reference. Files are always written inside the output "
            "directory; paths outside it are refused. Returns the file paths and sizes."
        ),
    )
    def export_figure(
        name: Annotated[str, Field(description="Name of a figure or composition.")],
        formats: Annotated[list[ExportFormat], Field(description="File formats to write.")] = ["svg", "pdf", "png"],  # noqa: B006
        dpi: Annotated[int, Field(description="Resolution of PNG / TIFF files (72-1200; journals usually ask for 300 or 600).")] = 300,
        transparent: Annotated[bool, Field(description="Transparent background.")] = False,
        with_stats_report: Annotated[bool, Field(description="Also write the statistics report (Markdown + CSV).")] = True,
        filename: Annotated[str | None, Field(description="Base file name inside the output directory (default: the figure name).")] = None,
    ) -> dict[str, Any]:
        return _call(session.export_figure, name=name, formats=list(formats), dpi=dpi, transparent=transparent,
                     with_stats_report=with_stats_report, filename=filename)

    @server.tool(
        name="get_stats_report",
        title="Get the statistics report",
        annotations=_READ_ONLY,
        description=(
            "Return the statistics report (Markdown, or tidy CSV) and a ready-to-paste methods paragraph for the "
            "results a figure or composition references (`figure`), for explicit `result_ids`, or for every "
            "result of the session when neither is given. All numbers are taken verbatim from the stored results."
        ),
    )
    def get_stats_report(
        figure: Annotated[str | None, Field(description="Name of a figure or composition.")] = None,
        result_ids: Annotated[list[str] | None, Field(description="Result IDs returned by run_analysis.")] = None,
        format: Annotated[ReportFormat, Field(description="'markdown' (readable report) or 'csv' (one row per number).")] = "markdown",
    ) -> dict[str, Any]:
        return _call(session.get_stats_report, figure=figure, result_ids=result_ids, format=format)

    # ------------------------------------------------------------------ project
    @server.tool(
        name="save_project",
        title="Save the project",
        annotations=_ADDITIVE,
        description=(
            "Save the whole session (tables, statistics results, figures, compositions) as a .tplot project "
            "file inside the output directory. The file also opens in the TorusPlot app. Returns its path."
        ),
    )
    def save_project(
        filename: Annotated[str, Field(description="File name inside the output directory, e.g. 'study1.tplot'.")] = "project.tplot",
    ) -> dict[str, Any]:
        return _call(session.save_project, filename=filename)

    @server.tool(
        name="load_project",
        title="Load a project",
        annotations=_REPLACING,
        description=(
            "Load a .tplot project file and REPLACE the current session with it (unsaved tables, results and "
            "figures are lost; call save_project first to keep them). Returns the new state, as list_state does."
        ),
    )
    def load_project(
        path: Annotated[str, Field(description="Path of a .tplot file (a bare file name is looked up in the output directory).")],
    ) -> dict[str, Any]:
        return _call(session.load_project, path=path)

    # ------------------------------------------------------------------ catalogues
    @server.tool(
        name="list_styles",
        title="List styles, palettes and plot types",
        annotations=_READ_ONLY,
        description=(
            "List the style presets, colour palettes and plot types with descriptions, plus the valid values of "
            "error_bars, points, annotation formats, axis scales, legend positions and style_overrides keys."
        ),
    )
    def list_styles() -> dict[str, Any]:
        return _call(session.list_styles)

    @server.tool(
        name="get_figure_schema",
        title="Get the figure spec schema",
        annotations=_READ_ONLY,
        description=(
            "Return the JSON Schema of the FigureSpec and a minimal example, for sending a full or partial "
            "`spec` to make_figure (custom layers, axis ticks, legend, text labels, reference lines). "
            "The schema has no field for a p value or star string: statistics are referenced by result ID."
        ),
    )
    def get_figure_schema() -> dict[str, Any]:
        return _call(session.get_figure_schema)

    return server
