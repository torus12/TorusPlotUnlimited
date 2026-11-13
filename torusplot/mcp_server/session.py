"""Session state and tool logic of the TorusPlot MCP server.

Everything an AI agent can do through the MCP server is a method of
:class:`TorusSession`. The class is plain Python (it does not import the ``mcp``
package), takes and returns JSON-serialisable values, and wraps one in-memory
:class:`torusplot.data.Project` plus an output directory. ``server.py`` is only a thin
layer that registers these methods as MCP tools.

The hard rule
-------------
An agent chooses *what* to analyse and *how the figure looks*; every statistic comes
from :mod:`torusplot.stats`:

* no tool accepts a p value, significance stars or free statistical text;
* figure annotations are attached by ``StatResult`` ID only (the FigureSpec schema has
  no field that could hold a number or a star string);
* text that looks like a typed statistic is rejected in titles, axis labels, legend
  titles, tick labels, panel labels and table headers;
* a result ID that does not exist is an error, never a drawn placeholder;
* tool outputs are dumped verbatim from the ``StatResult`` objects.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import os
import re
import threading
import warnings as _warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ValidationError

from ..data.io import import_table
from ..data.parse import build_table, table_from_text
from ..data.project import Project, ProjectError
from ..data.samples import SAMPLES, sample_frame
from ..data.tables import (
    KIND_DESCRIPTIONS,
    TABLE_KINDS,
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    TableError,
    XYTable,
)
from ..export import (
    ExportError,
    export_all,
    figure_bytes,
    figure_size_mm,
    methods_paragraph,
    significance_legend,
    stats_report_csv,
    stats_report_markdown,
    validate_svg,
    write_stats_report,
)
from ..plot import (
    PALETTES,
    PaletteError,
    PlotError,
    StyleError,
    figure_warnings,
    get_palette,
    get_preset,
    palette_names,
    preset_names,
    render_figure,
    summary_text,
)
from ..plot.compose import render_composition
from ..plot.palettes import normalize_color
from ..plot.render import CATEGORICAL
from ..spec import (
    PLOT_TYPES,
    AxisSpec,
    CompositionSpec,
    FigureSpec,
    LegendSpec,
    StatAnnotation,
    StyleSpec,
    default_layers,
    figure_spec_schema,
    minimal_example,
    plot_types_for,
)
from ..stats import StatResult, StatsError
from ..stats import registry as _registry

#: File types ``load_table(path=...)`` is allowed to read.
TABLE_FILE_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx")
#: Largest data file ``load_table(path=...)`` will open.
MAX_TABLE_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_OUTPUT_DIR = "torusplot_output"
OUTPUT_DIR_ENV = "TORUSPLOT_OUTPUT_DIR"



def _literal_choices(model: type[BaseModel], field: str) -> tuple[str, ...]:
    """The allowed text values of a spec field, read from the schema so they never drift."""

    def walk(annotation: Any) -> list[str]:
        if get_origin(annotation) is Literal:
            return [a for a in get_args(annotation) if isinstance(a, str)]
        return [value for arg in get_args(annotation) for value in walk(arg)]

    return tuple(dict.fromkeys(walk(model.model_fields[field].annotation)))


EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
ERROR_BAR_KINDS = ("sem", "sd", "ci95", "range", "iqr", "none")
POINT_KINDS = ("jitter", "beeswarm", "plain", "none")
ANNOTATION_KINDS = ("auto", *_literal_choices(StatAnnotation, "kind"))
ANNOTATION_FORMATS = _literal_choices(StatAnnotation, "format")
AXIS_SCALES = _literal_choices(AxisSpec, "scale")
LEGEND_POSITIONS = _literal_choices(LegendSpec, "position")
PANEL_LABEL_STYLES = _literal_choices(CompositionSpec, "panel_labels")
SHARE_MODES = _literal_choices(CompositionSpec, "share_x")

_XY_TYPES = frozenset({"xy", "dose_response"})
_PREVIEW_DPI = 150
_PREVIEW_MAX_PX = 1600
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)

# Text that reads like a typed statistical result. Drawn text (titles, axis labels, legend
# titles, tick labels, panel labels) is checked with _STAT_TEXT; names that come from data
# (column headers, group levels) with the slightly more tolerant _STAT_NAME.
_P_NUMBER = r"(0?[.,]\d|\d([.,]\d+)?\s*([eE]|[x×]\s*10\s*\^?)\s*[-−]\s*\d)"
_P_VALUE = (
    r"(\b[pq]\s*(-?\s*val(ue)?s?)?\s*(\(?\s*adj\w*\.?\s*\)?)?\s*[<=>≤≥]\s*" + _P_NUMBER + r")"
    r"|(\bp\s*-?\s*adj)|(\bfdr\s*[<=>≤≥])"
)
_STAT_TEXT = re.compile(
    _P_VALUE + r"|(^\s*(\*+|n\.?\s?s\.?)\s*$)|(\*{2,})|(\s\*\s*$)|(^\s*\*\s)", re.IGNORECASE
)
_STAT_NAME = re.compile(_P_VALUE + r"|(^\s*\*+\s*$)|(\*{2,})", re.IGNORECASE)

_F = TypeVar("_F", bound=Callable[..., Any])


class SessionError(ValueError):
    """A tool failure with a message written for the calling model / end user.

    Every anticipated problem (unknown table, invalid spec, statistics that cannot be
    computed, unsafe path...) is raised as this type, so the MCP layer can report it as
    a clean tool error without a traceback.
    """


_DOMAIN_ERRORS = (StatsError, PlotError, TableError, ProjectError, ExportError, PaletteError, StyleError)


# ============================================================================ helpers
def jsonable(obj: Any) -> Any:
    """Recursively convert a value to strict-JSON-safe plain Python.

    numpy scalars/arrays become Python numbers/lists, pydantic models become dicts,
    paths become strings, ``NaN`` becomes ``None`` and infinities become the strings
    ``"Infinity"`` / ``"-Infinity"`` (bare NaN/Infinity tokens are not valid JSON and
    break strict parsers in MCP clients).
    """
    if obj is None or isinstance(obj, str | bool):
        return obj
    if isinstance(obj, BaseModel):
        return jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(obj, Path):
        return str(obj)
    try:  # pandas.NA / NaT and friends
        if obj != obj:
            return None
    except (TypeError, ValueError):
        return None
    return str(obj)


def _drop_none(obj: Any) -> Any:
    """Remove ``None``-valued keys from nested dicts (keeps lists and their order)."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def validation_message(exc: ValidationError, what: str = "input") -> str:
    """Compact, traceback-free rendering of a pydantic ValidationError."""
    lines = []
    for err in exc.errors(include_url=False)[:12]:
        loc = ".".join(str(part) for part in err.get("loc", ())) or "(root)"
        msg = str(err.get("msg", "invalid value"))
        msg = msg.removeprefix("Value error, ")
        lines.append(f"- {loc}: {msg}")
    more = exc.error_count() - len(lines)
    if more > 0:
        lines.append(f"- ... and {more} more problem(s)")
    return f"Invalid {what} ({exc.error_count()} problem(s)):\n" + "\n".join(lines)


def _tool(func: _F) -> _F:
    """Decorator for tool methods: serialise access, clean the output, tidy the errors."""

    @functools.wraps(func)
    def wrapper(self: TorusSession, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            try:
                return jsonable(func(self, *args, **kwargs))
            except SessionError:
                raise
            except ValidationError as exc:
                raise SessionError(validation_message(exc)) from exc
            except _DOMAIN_ERRORS as exc:
                raise SessionError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


def looks_like_statistic(text: Any, *, data_name: bool = False) -> bool:
    """True if ``text`` reads like a typed statistical result (p value, stars, 'ns')."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool((_STAT_NAME if data_name else _STAT_TEXT).search(text))


def _reject_stat_text(value: Any, where: str) -> None:
    if looks_like_statistic(value):
        raise SessionError(
            f"{where} looks like a statistical result ('{value}'). Statistics cannot be typed into a "
            "figure: run run_analysis and attach its result with annotate_result_id, so the numbers "
            "come from the TorusPlot statistics engine."
        )


def _clean_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionError(f"{what} must be a non-empty text value.")
    name = value.strip()
    if len(name) > 120:
        raise SessionError(f"{what} is too long ({len(name)} characters; the limit is 120).")
    if any(ord(ch) < 32 for ch in name):
        raise SessionError(f"{what} must not contain control characters.")
    return name


def safe_stem(name: str, fallback: str = "figure") -> str:
    """A file-name stem that is safe on Windows, macOS and Linux (letters, digits, '_' and '-' only)."""
    stem = re.sub(r"[^\w\-]+", "_", str(name), flags=re.UNICODE)
    stem = re.sub(r"_{2,}", "_", stem).strip("_ ")
    stem = stem[:100].rstrip("_ ") or fallback
    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    return stem


@contextmanager
def _captured_warnings(into: list[str]) -> Iterator[None]:
    """Collect UserWarnings raised while drawing (e.g. a layout that does not fit) as tool warnings.

    Other warning categories are passed on unchanged once the block ends.
    """
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always", UserWarning)
        yield
    for item in caught:
        if not issubclass(item.category, UserWarning):
            _warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
            continue
        text = " ".join(str(item.message).split())
        if "constrained_layout not applied" in text or "collapsed to zero" in text:
            text = (
                "The content does not fit at this size, so the automatic layout could not be applied. "
                "Increase width_mm / height_mm, use fewer panels per row, or shorten the labels."
            )
        if text not in into:
            into.append(text)


def _as_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        if not all(isinstance(v, str | int | float) and not isinstance(v, bool) for v in value):
            raise SessionError(f"{what} must be a list of text values.")
        return [str(v) for v in value]
    raise SessionError(f"{what} must be a list of text values.")


def _as_dict(value: Any, what: str) -> dict[str, Any]:
    """Accept a dict (or a JSON object typed as text, which some clients send)."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError) as exc:
            raise SessionError(f"{what} must be a JSON object, not free text.") from exc
    if not isinstance(value, dict):
        raise SessionError(f"{what} must be a JSON object (dictionary).")
    return copy.deepcopy(value)


def _sub(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    child = parent.setdefault(key, {})
    if child is None:
        child = parent[key] = {}
    if not isinstance(child, dict):
        raise SessionError(f"{where}.{key} must be a JSON object.")
    return child


def _choice(value: Any, choices: tuple[str, ...], what: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise SessionError(f"{what} must be one of {list(choices)} (got {value!r}).")
    return value


def _number(value: Any, what: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SessionError(f"{what} must be a number (got {value!r}).")
    if (low is not None and value < low) or (high is not None and value > high):
        raise SessionError(f"{what} must be between {low:g} and {high:g} (got {value:g}).")
    return float(value)


# ============================================================================ session
class TorusSession:
    """One MCP session: an in-memory project plus an output directory.

    Every public method decorated as a tool returns plain JSON-safe data and raises
    :class:`SessionError` (and nothing else) for anticipated problems.
    """

    def __init__(self, output_dir: str | Path | None = None, project: Project | None = None) -> None:
        raw = output_dir or os.environ.get(OUTPUT_DIR_ENV) or DEFAULT_OUTPUT_DIR
        self.output_dir: Path = Path(raw).expanduser().resolve()
        self.project: Project = project if project is not None else Project("TorusPlot MCP session")
        self._lock = threading.RLock()
        self._previews: dict[str, bytes] = {}
        #: Default axis title of bundled samples, by table name (used when no y_label is given).
        self._default_labels: dict[str, str] = {}
        #: Results computed on a table that was replaced afterwards.
        self._stale_results: set[str] = set()

    # ------------------------------------------------------------------ lookups
    def preview_png(self, name: str) -> bytes | None:
        """PNG bytes of the most recent preview of a figure/composition (None if never rendered)."""
        return self._previews.get(name)

    def _table(self, name: Any) -> DataTable:
        if isinstance(name, str) and name in self.project.tables:
            return self.project.tables[name]
        if not self.project.tables:
            raise SessionError(f"Table '{name}' not found: no tables are loaded yet. Call load_table first.")
        raise SessionError(f"Table '{name}' not found. Loaded tables: {list(self.project.tables)}")

    def _result(self, result_id: Any) -> StatResult:
        res = self.project.results.get(result_id) if isinstance(result_id, str) else None
        if res is None:
            known = [f"{rid} ({self._result_obj(r).test_name})" for rid, r in self.project.results.items()]
            hint = f"Known result IDs: {known}" if known else "No analyses have been run yet"
            raise SessionError(
                f"No statistics result with ID '{result_id}'. {hint}. Result IDs come from run_analysis; "
                "they cannot be invented."
            )
        return self._result_obj(res)

    @staticmethod
    def _result_obj(res: Any) -> StatResult:
        return StatResult.model_validate(res) if isinstance(res, dict) else res

    def _figure(self, name: Any) -> FigureSpec:
        spec = self.project.figures.get(name) if isinstance(name, str) else None
        if spec is None:
            raise SessionError(f"Figure '{name}' not found. Figures: {list(self.project.figures)}")
        return FigureSpec.model_validate(spec) if isinstance(spec, dict) else spec

    def _composition(self, name: Any) -> CompositionSpec:
        comp = self.project.compositions.get(name) if isinstance(name, str) else None
        if comp is None:
            raise SessionError(f"Composition '{name}' not found. Compositions: {list(self.project.compositions)}")
        return CompositionSpec.model_validate(comp) if isinstance(comp, dict) else comp

    def _figure_or_composition(self, name: Any) -> tuple[str, FigureSpec | CompositionSpec]:
        if isinstance(name, str) and name in self.project.figures:
            return "figure", self._figure(name)
        if isinstance(name, str) and name in self.project.compositions:
            return "composition", self._composition(name)
        raise SessionError(
            f"No figure or composition named '{name}'. Figures: {list(self.project.figures)}; "
            f"compositions: {list(self.project.compositions)}"
        )

    def _result_ids_of(self, kind: str, spec: FigureSpec | CompositionSpec) -> list[str]:
        if kind == "figure":
            return spec.stat_result_ids()  # type: ignore[union-attr]
        ids: list[str] = []
        for panel in spec.panels:  # type: ignore[union-attr]
            if panel.figure in self.project.figures:
                ids += self._figure(panel.figure).stat_result_ids()
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------ file safety
    def _ensure_output_dir(self) -> Path:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(
                f"The output directory '{self.output_dir}' cannot be created ({exc}). Start the server with "
                f"--output-dir or set {OUTPUT_DIR_ENV} to a folder you can write to."
            ) from exc
        return self.output_dir

    def _output_path(self, filename: Any, what: str = "File name") -> Path:
        """Resolve a user-supplied file name to a path that is guaranteed to be inside the output directory."""
        if not isinstance(filename, str) or not filename.strip() or "\x00" in filename:
            raise SessionError(f"{what} must be a non-empty text value.")
        raw = filename.strip()
        norm = raw.replace("\\", "/")
        outside = (
            f"{what} '{raw}' points outside the output directory. Files are only written inside "
            f"'{self.output_dir}'; give a plain file name such as 'figure1'."
        )
        if Path(raw).is_absolute() or norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
            target = Path(raw)
        else:
            parts = [seg for seg in norm.split("/") if seg not in ("", ".")]
            if not parts or any(seg == ".." or seg.startswith("~") for seg in parts):
                raise SessionError(outside)
            last = Path(parts[-1])
            suffix = re.sub(r"[^\w.]+", "_", last.suffix, flags=re.UNICODE)
            parts = [safe_stem(seg, "folder") for seg in parts[:-1]] + [safe_stem(last.stem, "file") + suffix]
            target = self.output_dir.joinpath(*parts)
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:
            raise SessionError(outside) from exc
        if resolved != self.output_dir and self.output_dir not in resolved.parents:
            raise SessionError(outside)
        if resolved == self.output_dir:
            raise SessionError(f"{what} '{raw}' is the output directory itself; give a file name.")
        return resolved

    @staticmethod
    def _readable_table_file(path: Any) -> Path:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise SessionError("path must be a non-empty text value.")
        allowed = ", ".join(TABLE_FILE_SUFFIXES)
        candidate = Path(path.strip()).expanduser()
        if candidate.suffix.lower() not in TABLE_FILE_SUFFIXES:
            raise SessionError(
                f"Only {allowed} files can be loaded as tables (got '{candidate.suffix or 'no extension'}')."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SessionError(f"File not found: {candidate}") from exc
        if not resolved.is_file():
            raise SessionError(f"Not a file: {candidate}")
        if resolved.suffix.lower() not in TABLE_FILE_SUFFIXES:  # a link to a different kind of file
            raise SessionError(f"Only {allowed} files can be loaded as tables.")
        size = resolved.stat().st_size
        if size > MAX_TABLE_FILE_BYTES:
            raise SessionError(
                f"'{resolved.name}' is {size / 1e6:.0f} MB; the limit for table files is "
                f"{MAX_TABLE_FILE_BYTES // (1024 * 1024)} MB."
            )
        return resolved

    # ------------------------------------------------------------------ statistics registry
    @staticmethod
    def _analyses_for(kind: str) -> list[Any]:
        try:
            return _registry.analyses_for(kind)
        except StatsError:
            raise
        except Exception as exc:  # noqa: BLE001 - an analysis module failed to import
            raise SessionError(
                f"The statistics engine could not be loaded ({type(exc).__name__}: {exc})."
            ) from exc

    def _analysis_info(self, key: Any, table: DataTable) -> Any:
        valid = self._analyses_for(table.kind)
        listing = ", ".join(f"{a.key} ({a.label})" for a in valid)
        if not isinstance(key, str) or key not in _registry.REGISTRY:
            raise SessionError(
                f"Unknown analysis '{key}'. Analyses valid for the {table.kind_label} table "
                f"'{table.name}': {listing}. Call list_analyses for their options."
            )
        info = _registry.REGISTRY[key]
        if table.kind not in info.tables:
            raise SessionError(
                f"'{info.label}' ({key}) needs a table of type {list(info.tables)}, but '{table.name}' is a "
                f"{table.kind_label} table. Analyses valid for this table: {listing}."
            )
        return info

    @staticmethod
    def _param_specs(info: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "label": p.label,
                "kind": p.kind,
                "default": p.default,
                "choices": list(p.choices),
                "optional": p.optional,
                "help": p.help,
            }
            for p in info.params
        ]

    @staticmethod
    def _param_help(info: Any) -> str:
        if not info.params:
            return f"'{info.key}' has no options."
        parts = []
        for p in info.params:
            text = f"{p.name} ({p.kind}, default {p.default!r}"
            if p.choices:
                text += f", choices {list(p.choices)}"
            parts.append(text + ")")
        return f"Valid options for '{info.key}': " + "; ".join(parts) + "."

    @staticmethod
    def _coerce_param(spec: Any, value: Any, info: Any) -> Any:
        """Check one analysis option against its ParamSpec (lenient about JSON number/text forms)."""
        where = f"Option '{spec.name}' of '{info.key}'"
        if value is None:
            return None
        kind = spec.kind
        if kind == "choice" and spec.choices:
            for choice in spec.choices:
                if choice == value and type(choice) is type(value):
                    return choice
            for choice in spec.choices:
                if str(choice).lower() == str(value).strip().lower():
                    return choice
            raise SessionError(f"{where} must be one of {list(spec.choices)} (got {value!r}).")
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1", "false", "no", "0"}:
                return value.strip().lower() in {"true", "yes", "1"}
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            raise SessionError(f"{where} must be true or false (got {value!r}).")
        if kind in {"float", "int"}:
            number: Any = value
            if isinstance(value, str):
                try:
                    number = float(value)
                except ValueError:
                    number = None
            if isinstance(number, bool) or not isinstance(number, int | float) or not math.isfinite(number):
                raise SessionError(f"{where} must be a number (got {value!r}).")
            if kind == "int":
                if float(number) != int(number):
                    raise SessionError(f"{where} must be a whole number (got {value!r}).")
                return int(number)
            return float(number)
        if kind in {"groups", "columns"}:
            return _as_list(value, where)
        if kind in {"group", "column", "text"}:
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                raise SessionError(f"{where} must be a text value (got {value!r}).")
            return value if isinstance(value, str) else str(value)
        return value

    # ------------------------------------------------------------------ summaries
    @staticmethod
    def _table_names(table: DataTable) -> dict[str, Any]:
        """Group / series / column names of a table (what analyses and figures refer to)."""
        if isinstance(table, ColumnTable):
            return {"groups": table.group_names}
        if isinstance(table, XYTable):
            return {"x": table.x_name, "series": table.series_names}
        if isinstance(table, GroupedTable):
            return {
                "row_label": table.row_label,
                "row_levels": table.row_levels,
                "column_label": table.column_label,
                "groups": table.column_levels,
            }
        if isinstance(table, MultipleVariablesTable):
            return {"numeric_columns": table.numeric_columns, "categorical_columns": table.categorical_columns}
        if isinstance(table, ContingencyTable):
            return {
                "row_label": table.row_label,
                "row_levels": table.row_levels,
                "column_label": table.column_label,
                "column_levels": table.column_levels,
            }
        return {}

    def _table_summary(self, table: DataTable) -> dict[str, Any]:
        rows, cols = table.data.shape
        out: dict[str, Any] = {
            "name": table.name,
            "kind": table.kind,
            "kind_description": KIND_DESCRIPTIONS.get(table.kind, ""),
            "shape": {"rows": rows, "columns": cols},
            "columns": list(table.data.columns),
            **self._table_names(table),
        }
        if isinstance(table, ColumnTable):
            out["n_per_group"] = {g: int(v.size) for g, v in table.groups().items()}
        elif isinstance(table, XYTable):
            out["replicates_per_series"] = {s: len(c) for s, c in table.series_map.items()}
            out["n_per_group"] = {s: int(table.xy_points(s)[0].size) for s in table.series_names}
            out["x_values"] = int(np.isfinite(table.x).sum())
        elif isinstance(table, GroupedTable):
            out["replicates_per_group"] = {g: len(c) for g, c in table.group_map.items()}
            out["n_per_group"] = {
                g: int(sum(table.cell(r, g).size for r in table.row_levels)) for g in table.column_levels
            }
            out["n_per_cell"] = {
                r: {g: int(table.cell(r, g).size) for g in table.column_levels} for r in table.row_levels
            }
        elif isinstance(table, MultipleVariablesTable):
            frame = table.analysis_frame()
            out["n_per_column"] = {c: int(frame[c].notna().sum()) for c in frame.columns}
            levels = {}
            for col in table.categorical_columns:
                counts = frame[col].dropna().astype(str).value_counts(sort=False)
                levels[col] = {str(k): int(v) for k, v in list(counts.items())[:30]}
            out["levels"] = levels
        elif isinstance(table, ContingencyTable):
            counts = table.counts
            out["row_totals"] = {str(k): float(v) for k, v in counts.sum(axis=1).items()}
            out["column_totals"] = {str(k): float(v) for k, v in counts.sum(axis=0).items()}
            out["n_total"] = float(counts.to_numpy().sum())
        preview = table.data.head(5)
        out["preview"] = [
            {c: (None if _is_missing(v) else v) for c, v in row.items()}
            for row in preview.to_dict(orient="records")
        ]
        out["approximate"] = table.approximate
        if table.notes:
            out["notes"] = table.notes
        return out

    @staticmethod
    def _check_table_names(table: DataTable) -> None:
        """Refuse tables whose headers / levels would smuggle a typed statistic onto a figure."""
        names: list[str] = [str(c) for c in table.data.columns]
        if isinstance(table, XYTable):
            names += table.series_names
        elif isinstance(table, GroupedTable):
            names += table.row_levels + table.column_levels
        elif isinstance(table, ContingencyTable):
            names += table.row_levels
        elif isinstance(table, MultipleVariablesTable):
            for col in table.categorical_columns:
                names += [str(v) for v in table.data[col].dropna().unique()[:500]]
        bad = [n for n in dict.fromkeys(names) if looks_like_statistic(n, data_name=True)]
        if bad:
            raise SessionError(
                f"Column or group name(s) {bad[:5]} look like statistical results. Names become axis and "
                "legend labels, and statistics may only come from run_analysis; rename them in the data."
            )

    @staticmethod
    def _result_line(res: StatResult) -> str:
        return summary_text(res)

    def _result_dict(self, res: StatResult) -> dict[str, Any]:
        """Compact, verbatim dump of a StatResult (bulky plot arrays in ``extra`` are left out)."""
        data = jsonable(res.model_dump(mode="json"))
        extra = data.pop("extra", {}) or {}
        small = {k: v for k, v in extra.items() if len(json.dumps(v, default=str)) <= 800}
        for key in ("effect_sizes", "parameters", "comparisons", "assumptions"):
            data[key] = _drop_none(data.get(key, []))
        data["summary"] = self._result_line(res)
        data["extra"] = small
        data["extra_omitted"] = sorted(set(extra) - set(small))
        if res.id in self._stale_results:
            data["warnings"] = [
                f"Table '{res.table}' was replaced after this analysis ran; re-run it on the current data.",
                *data.get("warnings", []),
            ]
        return data

    def _log(self, message: str) -> None:
        self.project.log.append(f"MCP: {message}")

    # ====================================================================== tools: data
    @_tool
    def load_table(
        self,
        name: str,
        kind: str | None = None,
        text: str | None = None,
        path: str | None = None,
        sample: str | None = None,
        replicates: int | None = None,
        row_label: str | None = None,
        column_label: str | None = None,
        sheet: str | None = None,
    ) -> dict[str, Any]:
        """Create a table from pasted text, a file, or a bundled synthetic sample."""
        name = _clean_name(name, "Table name")
        _reject_stat_text(name, "Table name")  # the table name becomes the default axis title
        given = [k for k, v in (("text", text), ("path", path), ("sample", sample)) if v not in (None, "")]
        if len(given) != 1:
            got = f"got {given}" if given else "got none"
            raise SessionError(f"Give exactly one of text, path or sample ({got}).")
        options: dict[str, Any] = {}
        if replicates is not None:
            if isinstance(replicates, bool) or not isinstance(replicates, int) or not 1 <= replicates <= 1000:
                raise SessionError("replicates must be a whole number of side-by-side replicate columns (1 or more).")
            options["replicates"] = replicates
        for key, value in (("row_label", row_label), ("column_label", column_label)):
            if value is not None:
                _reject_stat_text(value, key)
                options[key] = _clean_name(value, key)

        if given == ["sample"]:
            if not isinstance(sample, str) or sample not in SAMPLES:
                listing = "; ".join(f"{k} ({v['kind']})" for k, v in SAMPLES.items())
                raise SessionError(f"Unknown sample '{sample}'. Bundled synthetic samples: {listing}.")
            info = SAMPLES[sample]
            if kind not in (None, "", info["kind"]):
                raise SessionError(
                    f"Sample '{sample}' is a {info['kind']} table; pass kind='{info['kind']}' (got '{kind}')."
                )
            kind = info["kind"]
            table = build_table(kind, sample_frame(sample), name, **{**info["options"], **options})
            table.notes = info["description"]
            source = f"sample '{sample}'"
        else:
            if kind not in TABLE_KINDS:
                raise SessionError(f"kind must be one of {list(TABLE_KINDS)} (got {kind!r}).")
            if given == ["text"]:
                table = table_from_text(kind, str(text), name, **options)
                source = "pasted text"
            else:
                file = self._readable_table_file(path)
                try:
                    table = import_table(file, kind, name=name, sheet=sheet, **options)
                except (SessionError, *_DOMAIN_ERRORS):
                    raise
                except Exception as exc:  # noqa: BLE001 - unreadable / corrupt file
                    raise SessionError(
                        f"'{file.name}' could not be read as a table ({type(exc).__name__}: {exc})."
                    ) from exc
                source = f"file '{file.name}'"
        if table.data.shape[0] == 0:
            raise SessionError("No data rows were found. Paste the cells including the header row.")
        if table.kind != "multiple_variables":
            numeric = table.data.select_dtypes(include="number")
            if numeric.shape[1] == 0 or not numeric.notna().to_numpy().any():
                raise SessionError(
                    "No numbers were found in the data. Include the header row, and separate the columns with "
                    "tabs, commas or semicolons."
                )
        self._check_table_names(table)

        warnings: list[str] = []
        replaced = name in self.project.tables
        if replaced:
            stale = [rid for rid, r in self.project.results.items() if self._result_obj(r).table == name]
            self._stale_results.update(stale)
            figures = [f for f in self.project.figures if self._figure(f).data.table == name]
            warnings.append(
                f"Table '{name}' already existed and was replaced. {len(stale)} earlier result(s) and "
                f"{len(figures)} figure(s) refer to it: re-run the analyses and re-make the figures."
            )
        self.project.add_table(table)
        if given == ["sample"]:
            self._default_labels[name] = SAMPLES[str(sample)]["title"]
        else:
            self._default_labels.pop(name, None)
        self._log(f"loaded {table.kind} table '{name}' from {source}")

        summary = self._table_summary(table)
        try:
            summary["analyses"] = [
                {"key": a.key, "label": a.label, "category": a.category} for a in self._analyses_for(table.kind)
            ]
        except (SessionError, StatsError) as exc:
            summary["analyses"] = []
            warnings.append(str(exc))
        summary["plot_types"] = plot_types_for(table.kind)
        summary["source"] = source
        summary["replaced"] = replaced
        summary["warnings"] = warnings
        summary["next"] = (
            f"Call recommend_test(table='{name}') for a suggested analysis, list_analyses(table='{name}') "
            f"for every option, or make_figure(name=..., table='{name}', plot_type=...) to draw it."
        )
        return summary

    @_tool
    def list_state(self) -> dict[str, Any]:
        """Everything in the session: tables, results, figures, compositions."""
        tables = [
            {"name": t.name, "kind": t.kind, "rows": t.data.shape[0], "columns": t.data.shape[1],
             "summary": t.summary()}
            for t in self.project.tables.values()
        ]
        results = []
        for rid, raw in self.project.results.items():
            res = self._result_obj(raw)
            results.append({
                "id": rid, "analysis": res.analysis, "test_name": res.test_name, "table": res.table,
                "summary": self._result_line(res), "significant": res.significant,
                "comparisons": len(res.comparisons), "stale": rid in self._stale_results,
            })
        figures = []
        for fname in self.project.figures:
            spec = self._figure(fname)
            figures.append({
                "name": fname, "plot_type": spec.plot_type, "table": spec.data.table,
                "stat_result_ids": spec.stat_result_ids(),
                "summary": f"{PLOT_TYPES[spec.plot_type]['label']} of table '{spec.data.table}'",
            })
        compositions = []
        for cname in self.project.compositions:
            comp = self._composition(cname)
            compositions.append({
                "name": cname, "rows": comp.rows, "cols": comp.cols, "panels": [p.figure for p in comp.panels],
                "summary": f"{comp.rows} x {comp.cols} grid, {len(comp.panels)} panel(s), "
                           f"{comp.width_mm:g} x {comp.height_mm:g} mm",
            })
        return {
            "project": self.project.name,
            "output_dir": str(self.output_dir),
            "tables": tables,
            "results": results,
            "figures": figures,
            "compositions": compositions,
            "samples": {k: {"kind": v["kind"], "description": v["description"]} for k, v in SAMPLES.items()},
        }

    # ================================================================ tools: statistics
    @_tool
    def list_analyses(self, table: str, analysis: str | None = None) -> dict[str, Any]:
        """Every analysis valid for a table (or just one of them), with its options."""
        tbl = self._table(table)
        infos = [self._analysis_info(analysis, tbl)] if analysis else self._analyses_for(tbl.kind)
        analyses = [
            {
                "key": a.key,
                "label": a.label,
                "category": a.category,
                "description": a.description,
                "params": self._param_specs(a),
            }
            for a in infos
        ]
        return {
            "table": tbl.name,
            "kind": tbl.kind,
            "names": self._table_names(tbl),
            "analyses": analyses,
            "param_kinds": {
                "group / groups": "one / several group, series or column-group names of the table (omit = all)",
                "column / columns": "one / several column names of a multiple_variables table",
                "choice": "one of the listed choices",
                "float / int / bool / text": "plain values",
            },
            "next": f"Call run_analysis(table='{tbl.name}', analysis=<key>, params={{...}}).",
        }

    @_tool
    def recommend_test(
        self,
        table: str,
        paired: bool | None = None,
        control: str | None = None,
        groups: list[str] | None = None,
        goal: str | None = None,
    ) -> dict[str, Any]:
        """Recommend an analysis for a table (never runs it)."""
        from ..stats.recommend import recommend_test as _recommend

        tbl = self._table(table)
        wanted = _as_list(groups, "groups") or None
        if isinstance(tbl, ColumnTable):
            unknown = [g for g in [*(wanted or []), *([control] if control else [])] if g not in tbl.group_names]
            if unknown:
                raise SessionError(f"Unknown group(s) {unknown}. Groups of '{tbl.name}': {tbl.group_names}")
        try:
            rec = _recommend(tbl, paired=paired, control=control, groups=wanted, goal=goal)
        except _DOMAIN_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SessionError(
                f"No recommendation could be made for '{tbl.name}' ({type(exc).__name__}: {exc})."
            ) from exc
        out = jsonable(rec.model_dump(mode="json"))
        out["checks"] = _drop_none(out.get("checks", []))
        out["table"] = tbl.name
        out["explanation"] = rec.explain()
        out["next"] = (
            f"Call run_analysis(table='{tbl.name}', analysis='{rec.analysis}', "
            f"params={json.dumps(jsonable(rec.params))}) to compute it."
        )
        return out

    @_tool
    def run_analysis(
        self,
        table: str,
        analysis: str,
        params: dict[str, Any] | None = None,
        allow_approximate: bool = False,
    ) -> dict[str, Any]:
        """Run a registered analysis with the TorusPlot statistics engine and store its result."""
        tbl = self._table(table)
        info = self._analysis_info(analysis, tbl)
        options = _as_dict(params, "params")
        specs = {p.name: p for p in info.params}
        unknown = sorted(set(options) - set(specs))
        if unknown:
            raise SessionError(f"Unknown option(s) {unknown} for '{info.key}'. {self._param_help(info)}")
        clean = {key: self._coerce_param(specs[key], value, info) for key, value in options.items()}
        try:
            result = _registry.run_analysis(tbl, info.key, allow_approximate=bool(allow_approximate), **clean)
        except _DOMAIN_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. an option of the wrong shape
            raise SessionError(
                f"'{info.label}' could not be computed with these options ({type(exc).__name__}: {exc}). "
                f"{self._param_help(info)}"
            ) from exc
        self.project.add_result(result)
        self._log(f"ran {info.key} on table '{tbl.name}' -> {result.id}")
        out = self._result_dict(result)
        out["next"] = self._next_after_analysis(result, tbl)
        return out

    @staticmethod
    def _next_after_analysis(result: StatResult, table: DataTable) -> str:
        if (result.extra.get("curve") or {}).get("x"):
            ptype = "dose_response" if "dose" in str(result.extra.get("model", "")) else "xy"
            return (
                f"Draw the fit with make_figure(name=..., table='{table.name}', plot_type='{ptype}', "
                f"fit_result_ids=['{result.id}'])."
            )
        if result.extra.get("curves"):
            return (
                f"Draw the curves with make_figure(name=..., table='{table.name}', plot_type='survival', "
                f"data_result_id='{result.id}', annotate_result_id='{result.id}')."
            )
        what = "comparison brackets" if result.comparisons else "the test summary"
        return (
            f"Show {what} on a figure with make_figure(name=..., table='{table.name}', plot_type=..., "
            f"annotate_result_id='{result.id}'); get_stats_report(result_ids=['{result.id}']) gives the "
            "report and methods paragraph."
        )

    @_tool
    def get_result(self, result_id: str) -> dict[str, Any]:
        """The stored StatResult with this ID (same payload as run_analysis)."""
        return self._result_dict(self._result(result_id))

    # =================================================================== tools: figures
    def _check_spec_text(self, spec: FigureSpec) -> None:
        """Defence in depth: no typed statistics in any text the figure draws."""
        _reject_stat_text(spec.title, "Figure title")
        _reject_stat_text(spec.legend.title, "Legend title")
        for axis_name, axis in (("X", spec.axes.x), ("Y", spec.axes.y)):
            _reject_stat_text(axis.label, f"{axis_name} axis label")
            _reject_stat_text(axis.tick_format, f"{axis_name} tick format")
            for label in axis.tick_labels or []:
                _reject_stat_text(label, f"{axis_name} tick label")
        for layer in spec.layers:
            _reject_stat_text(layer.label, "Layer label")
        for text in spec.texts:
            _reject_stat_text(text.text, "Text annotation")

    def _check_spec_references(self, spec: FigureSpec) -> list[str]:
        """Every referenced StatResult must exist; returns warnings about questionable links."""
        table = self._table(spec.data.table)
        valid = plot_types_for(table.kind)
        if spec.plot_type not in valid:
            raise SessionError(
                f"Plot type '{spec.plot_type}' cannot draw a {table.kind_label} table. Plot types valid for "
                f"'{table.name}': {valid}"
            )
        warnings = []
        for rid in spec.stat_result_ids():
            res = self._result(rid)
            if res.table and res.table != spec.data.table:
                warnings.append(
                    f"Result '{rid}' ({res.test_name}) was computed on table '{res.table}', but this figure "
                    f"plots table '{spec.data.table}'. Make sure both hold the same data."
                )
            if rid in self._stale_results:
                warnings.append(
                    f"Result '{rid}' was computed before table '{res.table}' was replaced; re-run the analysis."
                )
        return warnings

    def _render(self, kind: str, spec: FigureSpec | CompositionSpec, warnings: list[str]) -> Any:
        """Render a figure or composition; rendering notes are appended to ``warnings``."""
        try:
            with _captured_warnings(warnings):
                if kind == "figure":
                    fig = render_figure(spec, self.project)  # type: ignore[arg-type]
                else:
                    fig = render_composition(spec, self.project)  # type: ignore[arg-type]
            for note in figure_warnings(fig):
                if note not in warnings:
                    warnings.append(note)
            return fig
        except (SessionError, ValidationError, *_DOMAIN_ERRORS):
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. a colour or marker matplotlib rejects
            raise SessionError(
                f"The {kind} could not be rendered ({type(exc).__name__}: {exc}). Check the spec values "
                "(colours, markers, scales, column roles)."
            ) from exc

    def _preview(self, name: str, fig: Any, warnings: list[str]) -> dict[str, Any]:
        width_in, height_in = fig.get_size_inches()
        dpi = int(max(40, min(_PREVIEW_DPI, _PREVIEW_MAX_PX / max(width_in, height_in))))
        with _captured_warnings(warnings):
            png = figure_bytes(fig, "png", dpi=dpi)
        path = self._ensure_output_dir() / f"{safe_stem(name)}_preview.png"
        try:
            path.write_bytes(png)
        except OSError as exc:
            raise SessionError(f"The preview could not be written to '{path}': {exc}") from exc
        self._previews[name] = png
        return {
            "preview_path": str(path),
            "preview_size_px": [int(round(width_in * dpi)), int(round(height_in * dpi))],
        }

    def _figure_payload(self, spec: FigureSpec, warnings: list[str], replaced: bool) -> dict[str, Any]:
        warnings = list(warnings)
        fig = self._render("figure", spec, warnings)
        preview = self._preview(spec.name, fig, warnings)
        self.project.figures[spec.name] = spec
        ctx = getattr(fig, "_torusplot_context", None)
        brackets = [
            {"comparison_id": b.comparison_id, "label": b.label, "significant": b.significant}
            for b in getattr(ctx, "brackets_drawn", [])
        ]
        summaries = [
            {"stat_result_id": a.stat_result_id, "text": summary_text(self._result(a.stat_result_id))}
            for a in spec.annotations
            if a.kind == "summary_text"
        ]
        width, height = figure_size_mm(fig)
        payload: dict[str, Any] = {
            "name": spec.name,
            "kind": "figure",
            "replaced": replaced,
            "plot_type": spec.plot_type,
            "table": spec.data.table,
            "size_mm": [round(width, 2), round(height, 2)],
            "stat_result_ids": spec.stat_result_ids(),
            "brackets_drawn": len(brackets),
            "brackets": brackets,
            "summary_texts": summaries,
            "warnings": warnings,
            "spec": spec.model_dump(mode="json", exclude_none=True),
            **preview,
        }
        payload["next"] = (
            f"Adjust the look with restyle_figure(name='{spec.name}', ...), combine figures with "
            f"compose_panels, or write files with export_figure(name='{spec.name}')."
        )
        return payload

    @staticmethod
    def _comparison_selection(res: StatResult, comparisons: Any) -> Any:
        if comparisons in (None, "all", "significant"):
            return comparisons or "all"
        if isinstance(comparisons, str):
            comparisons = [comparisons]
        if not isinstance(comparisons, list | tuple) or not all(isinstance(c, str) for c in comparisons):
            raise SessionError("comparisons must be 'all', 'significant' or a list of comparison IDs.")
        known = {c.id: c for c in res.comparisons}
        unknown = [c for c in comparisons if c not in known]
        if unknown:
            listing = [f"{c.id} ({c.group_a} vs {c.group_b})" for c in res.comparisons]
            raise SessionError(
                f"Unknown comparison ID(s) {unknown} for result '{res.id}'. Its comparisons: {listing}"
            )
        return list(comparisons)

    @_tool
    def make_figure(
        self,
        name: str,
        table: str | None = None,
        plot_type: str | None = None,
        spec: dict[str, Any] | None = None,
        annotate_result_id: str | None = None,
        comparisons: str | list[str] = "all",
        annotation_format: str = "stars",
        fit_result_ids: list[str] | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        style_preset: str | None = None,
        palette: str | list[str] | None = None,
        width_mm: float | None = None,
        height_mm: float | None = None,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        replicate: str | None = None,
        subject: str | None = None,
        groups: list[str] | None = None,
        data_result_id: str | None = None,
        annotation_kind: str = "auto",
    ) -> dict[str, Any]:
        """Build, store and render a figure; statistics are attached by StatResult ID only."""
        name = _clean_name(name, "Figure name")
        if name in self.project.compositions:
            raise SessionError(f"'{name}' is already the name of a composition; choose another figure name.")
        for value, where in ((title, "Figure title"), (x_label, "X axis label"), (y_label, "Y axis label")):
            _reject_stat_text(value, where)
        d = _as_dict(spec, "spec")
        d["name"] = name
        data = _sub(d, "data", "spec")
        if table is not None:
            data["table"] = table
        if plot_type is not None:
            d["plot_type"] = plot_type
        if not data.get("table"):
            raise SessionError("Give the table to plot (table=... or spec.data.table).")
        if not d.get("plot_type"):
            raise SessionError("Give the plot type (plot_type=... or spec.plot_type).")
        tbl = self._table(data["table"])
        ptype = d["plot_type"]
        valid = plot_types_for(tbl.kind)
        if ptype not in valid:
            known = "is not a plot type" if ptype not in PLOT_TYPES else f"cannot draw a {tbl.kind_label} table"
            raise SessionError(f"Plot type '{ptype}' {known}. Plot types valid for '{tbl.name}': {valid}")

        for key, value in (("x", x), ("y", y), ("hue", hue), ("replicate", replicate), ("subject", subject),
                           ("stat_result_id", data_result_id)):
            if value is not None:
                data[key] = value
        if groups is not None:
            data["groups"] = _as_list(groups, "groups")
        if title is not None:
            d["title"] = title or None
        axes = _sub(d, "axes", "spec")
        for axis, label in (("x", x_label), ("y", y_label)):
            if label is not None:
                _sub(axes, axis, "spec.axes")["label"] = label or None
        if (y_label is None and tbl.name in self._default_labels and tbl.kind in {"column", "grouped", "xy"}
                and ptype in CATEGORICAL | _XY_TYPES):
            y_axis = _sub(axes, "y", "spec.axes")
            if not y_axis.get("label"):
                y_axis["label"] = self._default_labels[tbl.name]
        style = _sub(d, "style", "spec")
        if style_preset is not None:
            style["preset"] = get_preset(style_preset)["name"]
        if palette is not None:
            get_palette(palette, 3)
            style["palette"] = palette
        layout = _sub(d, "layout", "spec")
        if width_mm is not None:
            layout["width_mm"] = width_mm
        if height_mm is not None:
            layout["height_mm"] = height_mm

        fit_ids = _as_list(fit_result_ids, "fit_result_ids")
        if fit_ids:
            if ptype not in _XY_TYPES:
                raise SessionError(
                    f"fit_result_ids draws fitted curves and needs plot_type 'xy' or 'dose_response' (got '{ptype}')."
                )
            for rid in fit_ids:
                res = self._result(rid)
                if not (res.extra.get("curve") or {}).get("x"):
                    raise SessionError(
                        f"Result '{rid}' ({res.test_name}) has no fitted curve. Use the ID of a "
                        "linear_regression or nonlinear_regression result."
                    )
            layers = d.get("layers")
            if not layers:
                layers = [lay.model_dump(mode="json", exclude_none=True) for lay in default_layers(ptype)]
                layers = [lay for lay in layers if lay["type"] != "mean_line"]
            if not isinstance(layers, list):
                raise SessionError("spec.layers must be a list.")
            layers = [
                lay for lay in layers
                if not (isinstance(lay, dict) and lay.get("type") == "curve_fit"
                        and lay.get("stat_result_id") in (None, "", *fit_ids))
            ]
            d["layers"] = layers + [{"type": "curve_fit", "stat_result_id": rid} for rid in fit_ids]

        if annotate_result_id is not None:
            res = self._result(annotate_result_id)
            kind = _choice(annotation_kind, ANNOTATION_KINDS, "annotation_kind")
            fmt = _choice(annotation_format, ANNOTATION_FORMATS, "annotation_format")
            if kind == "auto":
                kind = "brackets" if res.comparisons and ptype in CATEGORICAL else "summary_text"
            elif kind == "brackets" and not res.comparisons:
                raise SessionError(
                    f"Result '{res.id}' ({res.test_name}) has no pairwise comparisons to draw as brackets; "
                    "use annotation_kind='summary_text' (or 'auto')."
                )
            elif kind == "brackets" and ptype not in CATEGORICAL:
                raise SessionError(
                    f"Comparison brackets are only drawn on categorical plots ({sorted(CATEGORICAL)}); "
                    "use annotation_kind='summary_text' (or 'auto')."
                )
            annotation: dict[str, Any] = {"stat_result_id": res.id, "kind": kind, "format": fmt}
            if kind == "brackets":
                annotation["comparisons"] = self._comparison_selection(res, comparisons)
            existing = d.get("annotations") or []
            if not isinstance(existing, list):
                raise SessionError("spec.annotations must be a list.")
            d["annotations"] = [
                a for a in existing if not (isinstance(a, dict) and a.get("stat_result_id") == res.id)
            ] + [annotation]

        try:
            fig_spec = FigureSpec.model_validate(d)
        except ValidationError as exc:
            raise SessionError(
                validation_message(exc, "figure spec") + "\nCall get_figure_schema for the full schema."
            ) from exc
        self._check_spec_text(fig_spec)
        warnings = self._check_spec_references(fig_spec)
        replaced = name in self.project.figures
        payload = self._figure_payload(fig_spec, warnings, replaced)
        self._log(f"{'re-made' if replaced else 'made'} figure '{name}' ({fig_spec.plot_type}, table '{tbl.name}')")
        return payload

    @_tool
    def restyle_figure(
        self,
        name: str,
        style_preset: str | None = None,
        palette: str | list[str] | None = None,
        colors: dict[str, str] | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        width_mm: float | None = None,
        height_mm: float | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        y_scale: str | None = None,
        x_scale: str | None = None,
        error_bars: str | None = None,
        points: str | None = None,
        legend_position: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Change only the styling/layout of an existing figure, then re-render it."""
        spec = self._figure(name)
        ptype = spec.plot_type
        if error_bars is not None:
            kind = _choice(error_bars, ERROR_BAR_KINDS, "error_bars")
            if ptype not in CATEGORICAL | _XY_TYPES or ptype in {"stacked_bar", "paired"}:
                raise SessionError(f"error_bars does not apply to plot type '{ptype}'.")
            spec = spec.with_error_bars(None if kind == "none" else kind)
        if points is not None:
            kind = _choice(points, POINT_KINDS, "points")
            if ptype not in CATEGORICAL | _XY_TYPES or ptype == "stacked_bar":
                raise SessionError(f"points does not apply to plot type '{ptype}'.")
            spec = spec.with_points(None if kind == "none" else kind)

        d = spec.model_dump(mode="json")
        style = d["style"]
        overrides = _as_dict(style_overrides, "style_overrides")
        unknown = sorted(set(overrides) - set(StyleSpec.model_fields))
        if unknown:
            raise SessionError(
                f"Unknown style option(s) {unknown}. Valid style options: {sorted(StyleSpec.model_fields)}"
            )
        style.update(overrides)
        if style_preset is not None:
            style["preset"] = style_preset
        style["preset"] = get_preset(style["preset"])["name"]
        if palette is not None:
            style["palette"] = palette
        if style.get("palette") is not None:
            get_palette(style["palette"], 3)
        if colors is not None:
            merged = dict(style.get("colors") or {})
            for group, color in _as_dict(colors, "colors").items():
                merged[str(group)] = normalize_color(color)
            style["colors"] = merged
        if font_family is not None:
            style["font_family"] = font_family or None
        if font_size is not None:
            style["font_size"] = font_size
        if width_mm is not None:
            d["layout"]["width_mm"] = width_mm
        if height_mm is not None:
            d["layout"]["height_mm"] = height_mm
        if title is not None:
            d["title"] = title or None
        if x_label is not None:
            d["axes"]["x"]["label"] = x_label or None
        if y_label is not None:
            d["axes"]["y"]["label"] = y_label or None
        if y_min is not None:
            d["axes"]["y"]["min"] = y_min
        if y_max is not None:
            d["axes"]["y"]["max"] = y_max
        if y_scale is not None:
            d["axes"]["y"]["scale"] = _choice(y_scale, AXIS_SCALES, "y_scale")
        if x_scale is not None:
            d["axes"]["x"]["scale"] = _choice(x_scale, AXIS_SCALES, "x_scale")
        if legend_position is not None:
            d["legend"]["position"] = _choice(legend_position, LEGEND_POSITIONS, "legend_position")

        try:
            new_spec = FigureSpec.model_validate(d)
        except ValidationError as exc:
            raise SessionError(validation_message(exc, "style")) from exc
        self._check_spec_text(new_spec)
        # restyling never touches data or statistics
        if new_spec.data != spec.data or new_spec.annotations != spec.annotations:  # pragma: no cover
            raise SessionError("restyle_figure cannot change data or statistics.")
        warnings = self._check_spec_references(new_spec)
        payload = self._figure_payload(new_spec, warnings, replaced=True)
        self._log(f"restyled figure '{name}'")
        return payload

    @_tool
    def compose_panels(
        self,
        name: str,
        figures: list[str] | None = None,
        rows: int | None = None,
        cols: int | None = None,
        width_mm: float = 183.0,
        height_mm: float | None = None,
        panel_labels: str = "A",
        share_x: str = "none",
        share_y: str = "none",
        shared_legend: bool = False,
        style_preset: str | None = None,
        layout: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Arrange existing figures as panels of one multi-panel figure."""
        name = _clean_name(name, "Composition name")
        if name in self.project.figures:
            raise SessionError(f"'{name}' is already the name of a figure; choose another composition name.")
        for value, what in ((rows, "rows"), (cols, "cols")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 12):
                raise SessionError(f"{what} must be a whole number from 1 to 12.")
        width = _number(width_mm, "width_mm", 20, 2000)

        panels: list[dict[str, Any]] = []
        if layout:
            if not isinstance(layout, list) or not all(isinstance(p, dict) for p in layout):
                raise SessionError("layout must be a list of {figure, row, col, rowspan, colspan} objects.")
            panels = [copy.deepcopy(p) for p in layout]
            for p in panels:
                if "figure" not in p or "row" not in p or "col" not in p:
                    raise SessionError("Every layout entry needs 'figure', 'row' and 'col' (0-based).")
            try:
                need_rows = max(int(p["row"]) + int(p.get("rowspan", 1)) for p in panels)
                need_cols = max(int(p["col"]) + int(p.get("colspan", 1)) for p in panels)
            except (TypeError, ValueError) as exc:
                raise SessionError("layout row/col/rowspan/colspan must be whole numbers.") from exc
            rows, cols = rows or max(need_rows, 1), cols or max(need_cols, 1)
        else:
            names = _as_list(figures, "figures")
            if not names:
                raise SessionError("Give the figures to combine (figures=[...]) or an explicit layout.")
            n = len(names)
            if rows is None and cols is None:
                cols = n if n <= 3 else (2 if n == 4 else 3)
            if cols is None:
                cols = math.ceil(n / rows)  # type: ignore[arg-type]
            if rows is None:
                rows = math.ceil(n / cols)
            if rows * cols < n:
                raise SessionError(f"{n} figures do not fit in a {rows} x {cols} grid.")
            if rows > 12 or cols > 12:
                raise SessionError("A composition can have at most 12 rows and 12 columns.")
            panels = [{"figure": fig, "row": i // cols, "col": i % cols} for i, fig in enumerate(names)]

        missing = [p["figure"] for p in panels if p["figure"] not in self.project.figures]
        if missing:
            raise SessionError(
                f"Figure(s) not found: {missing}. Figures: {list(self.project.figures)}. "
                "Create them with make_figure first."
            )
        for p in panels:
            label = p.get("label")
            if label is not None:
                _reject_stat_text(label, "Panel label")
                if not isinstance(label, str) or len(label) > 12:
                    raise SessionError("A panel label must be a short text value (at most 12 characters).")
        if height_mm is None:
            height = min(max(rows * (width / cols) * 0.8, 40.0), 250.0)
        else:
            height = _number(height_mm, "height_mm", 20, 2000)
        d: dict[str, Any] = {
            "name": name,
            "rows": rows,
            "cols": cols,
            "panels": panels,
            "width_mm": width,
            "height_mm": round(height, 1),
            "panel_labels": _choice(panel_labels, PANEL_LABEL_STYLES, "panel_labels"),
            "share_x": _choice(share_x, SHARE_MODES, "share_x"),
            "share_y": _choice(share_y, SHARE_MODES, "share_y"),
            "shared_legend": bool(shared_legend),
        }
        if style_preset is not None:
            d["style_preset"] = get_preset(style_preset)["name"]
        try:
            comp = CompositionSpec.model_validate(d)
        except ValidationError as exc:
            raise SessionError(validation_message(exc, "composition")) from exc

        warnings: list[str] = []
        for panel in comp.panels:
            warnings += self._check_spec_references(self._figure(panel.figure))
        fig = self._render("composition", comp, warnings)
        preview = self._preview(name, fig, warnings)
        replaced = name in self.project.compositions
        self.project.compositions[name] = comp
        contexts = getattr(fig, "_torusplot_contexts", [])
        letters = [getattr(c, "_letter", "") for c in contexts]
        width_out, height_out = figure_size_mm(fig)
        payload: dict[str, Any] = {
            "name": name,
            "kind": "composition",
            "replaced": replaced,
            "grid": {"rows": comp.rows, "cols": comp.cols},
            "panels": [
                {"label": letters[i] if i < len(letters) else "", "figure": p.figure, "row": p.row, "col": p.col,
                 "rowspan": p.rowspan, "colspan": p.colspan}
                for i, p in enumerate(comp.panels)
            ],
            "size_mm": [round(width_out, 2), round(height_out, 2)],
            "stat_result_ids": self._result_ids_of("composition", comp),
            "warnings": list(dict.fromkeys(warnings)),
            "spec": comp.model_dump(mode="json", exclude_none=True),
            **preview,
            "next": f"Write files with export_figure(name='{name}').",
        }
        self._log(f"composed '{name}' from {[p.figure for p in comp.panels]}")
        return payload

    # ==================================================================== tools: export
    @_tool
    def export_figure(
        self,
        name: str,
        formats: list[str] | None = None,
        dpi: int = 300,
        transparent: bool = False,
        with_stats_report: bool = True,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Export a figure or composition (and its statistics report) into the output directory."""
        kind, spec = self._figure_or_composition(name)
        wanted = _as_list(["svg", "pdf", "png"] if formats is None else formats, "formats")
        fmts = list(dict.fromkeys({"tif": "tiff"}.get(f.lower().lstrip("."), f.lower().lstrip(".")) for f in wanted))
        bad = [f for f in fmts if f not in EXPORT_FORMATS]
        if bad or not fmts:
            raise SessionError(f"formats must be a non-empty list drawn from {list(EXPORT_FORMATS)} (got {wanted}).")
        if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 1200:
            raise SessionError("dpi must be a whole number from 72 to 1200 (journals usually ask for 300 or 600).")

        if filename is None:
            base = self._ensure_output_dir() / safe_stem(name)
        else:
            target = self._output_path(filename, "filename")
            if target.suffix.lower().lstrip(".") in {"svg", "pdf", "png", "tif", "tiff"}:
                target = target.with_suffix("")
            base = target
        warnings: list[str] = []
        if kind == "figure":
            warnings += self._check_spec_references(spec)  # type: ignore[arg-type]
        else:
            for panel in spec.panels:  # type: ignore[union-attr]
                warnings += self._check_spec_references(self._figure(panel.figure))
        fig = self._render(kind, spec, warnings)
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
            with _captured_warnings(warnings):
                written = export_all(fig, base, formats=fmts, dpi=dpi, transparent=bool(transparent))
        except OSError as exc:
            raise SessionError(f"The files could not be written to '{base.parent}': {exc}") from exc
        files = [
            {"format": fmt, "path": str(path), "size_bytes": path.stat().st_size} for fmt, path in written.items()
        ]
        width, height = figure_size_mm(fig)
        out: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "output_dir": str(self.output_dir),
            "files": files,
            "size_mm": [round(width, 2), round(height, 2)],
            "dpi": dpi,
            "transparent": bool(transparent),
            "warnings": list(dict.fromkeys(warnings)),
        }
        if "svg" in written:
            check = validate_svg(written["svg"])
            out["svg_check"] = {
                "ok": check["ok"], "editable_text_elements": check["text_elements"], "problems": check["problems"],
            }
        ids = self._result_ids_of(kind, spec)
        out["stat_result_ids"] = ids
        if with_stats_report:
            if ids:
                results = [self._result(rid) for rid in ids]
                report = write_stats_report(
                    results, base, title=f"Statistics report: {name}",
                    spec=spec if kind == "figure" else None,  # type: ignore[arg-type]
                )
                out["stats_report"] = [
                    {"format": fmt, "path": str(path), "size_bytes": path.stat().st_size}
                    for fmt, path in report.items()
                ]
            else:
                out["stats_report"] = []
                out["stats_report_note"] = "No statistics results are attached to this figure, so no report was written."
        self._log(f"exported {kind} '{name}' as {fmts}")
        return out

    @_tool
    def get_stats_report(
        self,
        figure: str | None = None,
        result_ids: list[str] | None = None,
        format: str = "markdown",
    ) -> dict[str, Any]:
        """Statistics report + methods paragraph for a figure's results or explicit result IDs."""
        fmt = _choice(str(format).lower(), ("markdown", "csv"), "format")
        ids: list[str] = []
        spec: FigureSpec | None = None
        title = "Statistics report"
        if figure is not None:
            kind, found = self._figure_or_composition(figure)
            ids += self._result_ids_of(kind, found)
            spec = found if kind == "figure" else None  # type: ignore[assignment]
            title = f"Statistics report: {figure}"
        ids += _as_list(result_ids, "result_ids")
        if figure is None and result_ids is None:
            ids = list(self.project.results)
        ids = list(dict.fromkeys(ids))
        results = [self._result(rid) for rid in ids]
        if fmt == "csv":
            report = stats_report_csv(results)
        else:
            report = stats_report_markdown(results, title=title, spec=spec)
        return {
            "format": fmt,
            "result_ids": ids,
            "report": report,
            "methods_paragraph": methods_paragraph(results, spec) if results else "",
            "significance_legend": significance_legend(),
            "note": "" if results else "No statistics results are attached; run run_analysis first.",
        }

    # =================================================================== tools: project
    @_tool
    def save_project(self, filename: str = "project.tplot") -> dict[str, Any]:
        """Save the whole session as a .tplot project file inside the output directory."""
        target = self._output_path(filename, "filename")
        if target.suffix.lower() != ".tplot":
            target = target.with_suffix(".tplot")
        try:
            path = self.project.save(target)
        except OSError as exc:
            raise SessionError(f"The project could not be written to '{target}': {exc}") from exc
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "tables": len(self.project.tables),
            "results": len(self.project.results),
            "figures": len(self.project.figures),
            "compositions": len(self.project.compositions),
        }

    @_tool
    def load_project(self, path: str) -> dict[str, Any]:
        """Replace the session with a saved .tplot project."""
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise SessionError("path must be a non-empty text value.")
        candidate = Path(path.strip()).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = self.output_dir / candidate  # a name returned by save_project
        if candidate.suffix.lower() != ".tplot":
            raise SessionError("Only .tplot project files can be loaded.")
        if not candidate.is_file():
            raise SessionError(f"Project file not found: {candidate}")
        try:
            project = Project.load(candidate)
        except (ProjectError, ValidationError):
            raise
        except Exception as exc:  # noqa: BLE001 - damaged archive
            raise SessionError(f"'{candidate.name}' is not a readable TorusPlot project ({exc}).") from exc
        self.project = project
        self._previews.clear()
        self._default_labels.clear()
        self._stale_results.clear()
        self._log(f"loaded project '{candidate.name}'")
        state = self.list_state()
        state["loaded_from"] = str(candidate.resolve())
        return state

    # =============================================================== tools: catalogues
    @_tool
    def list_styles(self) -> dict[str, Any]:
        """Style presets, palettes and plot types (with descriptions)."""
        presets = []
        for pname in preset_names():
            preset = get_preset(pname)
            presets.append({
                "name": pname,
                "description": " ".join(str(preset.get("description", "")).split()),
                "width_mm": preset["width_mm"],
                "height_mm": preset["height_mm"],
                "font_family": preset["font_family"],
                "font_size": preset["font_size"],
                "palette": preset["palette"],
            })
        palettes = [
            {"name": p, "colors": PALETTES[p]} if p in PALETTES
            else {"name": p, "colors": get_palette(p, 5), "note": "sequential colormap, sampled to the number of groups"}
            for p in palette_names()
        ]
        plot_types = [
            {
                "key": key,
                "label": info["label"],
                "description": info["description"],
                "tables": list(info["tables"]),
                "default_layers": [lay.type for lay in default_layers(key)],
            }
            for key, info in PLOT_TYPES.items()
        ]
        return {
            "presets": presets,
            "palettes": palettes,
            "plot_types": plot_types,
            "error_bars": list(ERROR_BAR_KINDS),
            "points": list(POINT_KINDS),
            "annotation_formats": list(ANNOTATION_FORMATS),
            "axis_scales": list(AXIS_SCALES),
            "legend_positions": list(LEGEND_POSITIONS),
            "panel_labels": list(PANEL_LABEL_STYLES),
            "style_options": sorted(StyleSpec.model_fields),
        }

    @_tool
    def get_figure_schema(self) -> dict[str, Any]:
        """The FigureSpec JSON schema (for agents that send full specs to make_figure)."""
        return {
            "schema": figure_spec_schema(),
            "example": minimal_example(),
            "notes": [
                "Pass a full or partial spec as make_figure(spec=...); the simple arguments override it.",
                "Empty 'layers' means the default layers of the plot type.",
                "Statistics are never typed: 'annotations' entries only hold a stat_result_id returned by "
                "run_analysis (plus which comparisons to show and how to format them).",
                "Text that looks like a p value or significance stars is rejected in titles, labels and texts.",
            ],
        }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN / NA
    except (TypeError, ValueError):
        return True
