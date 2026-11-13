"""Analysis registry: which analyses exist, which table types they accept, their options.

Each analysis module registers table-level entry points with :func:`register`. The UI,
CLI and MCP server discover analyses here, so an analysis is only ever offered for the
table types it is valid for.

    result = run_analysis(table, "ttest_unpaired", group_a="Control", group_b="Drug")
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..data.tables import DataTable
from ._utils import ApproximateDataError, StatsError, jsonable
from .result import StatResult

#: Modules that register analyses when imported (order = menu order).
ANALYSIS_MODULES = (
    "descriptives",
    "assumptions",
    "two_group",
    "oneway",
    "factorial",
    "repeated",
    "correlation",
    "regression",
    "nonlinear",
    "survival",
    "contingency",
    "outliers",
    "nested",
)

CATEGORIES = (
    "Descriptive statistics",
    "Assumption checks",
    "Compare two groups",
    "Compare three or more groups",
    "Factorial designs",
    "Repeated measures",
    "Correlation",
    "Regression",
    "Nonlinear regression",
    "Survival",
    "Contingency tables",
    "Outliers",
    "Replicate-aware (nested)",
)


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing option of an analysis (drives the UI widgets and tooltips).

    ``kind`` is one of:
      ``float`` ``int`` ``bool`` ``text``          plain values
      ``choice``                                   one of ``choices``
      ``group`` / ``groups``                       one / several column-or-series names
                                                   of the table (``None``/empty = all)
      ``column`` / ``columns``                     one / several variable names of a
                                                   Multiple-variables table
    """

    name: str
    label: str
    kind: str = "float"
    default: Any = None
    choices: tuple[Any, ...] = ()
    help: str = ""
    optional: bool = False


@dataclass(frozen=True)
class AnalysisInfo:
    key: str
    label: str
    category: str
    tables: tuple[str, ...]
    func: Callable[..., StatResult]
    params: tuple[ParamSpec, ...] = ()
    description: str = ""
    citation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


REGISTRY: dict[str, AnalysisInfo] = {}
_loaded = False


def register(
    key: str,
    label: str,
    category: str,
    tables: tuple[str, ...] | list[str],
    params: tuple[ParamSpec, ...] | list[ParamSpec] = (),
    description: str = "",
    citation: str = "",
) -> Callable[[Callable[..., StatResult]], Callable[..., StatResult]]:
    """Decorator registering ``func(table, **params) -> StatResult`` as an analysis."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown analysis category '{category}'.")

    def decorator(func: Callable[..., StatResult]) -> Callable[..., StatResult]:
        REGISTRY[key] = AnalysisInfo(
            key=key,
            label=label,
            category=category,
            tables=tuple(tables),
            func=func,
            params=tuple(params),
            description=description,
            citation=citation,
        )
        return func

    return decorator


def load_all() -> dict[str, AnalysisInfo]:
    """Import every analysis module so the registry is complete."""
    global _loaded
    if not _loaded:
        for module in ANALYSIS_MODULES:
            importlib.import_module(f"{__package__}.{module}")
        _loaded = True
    return REGISTRY


def get_analysis(key: str) -> AnalysisInfo:
    load_all()
    if key not in REGISTRY:
        raise StatsError(f"Unknown analysis '{key}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]


def analyses_for(kind: str) -> list[AnalysisInfo]:
    """Analyses valid for a table kind, in menu order."""
    load_all()
    infos = [a for a in REGISTRY.values() if kind in a.tables]
    return sorted(infos, key=lambda a: CATEGORIES.index(a.category))


def run_analysis(
    table: DataTable, key: str, *, allow_approximate: bool = False, **params: Any
) -> StatResult:
    """Run a registered analysis on a table and return a finalized :class:`StatResult`.

    Digitized ("approximate") tables are refused unless ``allow_approximate=True`` is
    passed explicitly; the result is then flagged and carries a warning.
    """
    info = get_analysis(key)
    if table.kind not in info.tables:
        valid = ", ".join(info.tables)
        raise StatsError(
            f"'{info.label}' is not available for a {table.kind_label} table "
            f"(valid table types: {valid})."
        )
    if table.approximate and not allow_approximate:
        raise ApproximateDataError(
            f"Table '{table.name}' holds APPROXIMATE values digitized from an image (for reference "
            "only), so statistical tests are switched off for it. If a rough reference value is all "
            "you need, tick 'I understand these are estimates read off an image. Allow statistics "
            "anyway.' on page 2 Analyze; the result is then marked APPROXIMATE. (In a script, pass "
            "allow_approximate=True.)"
        )
    unknown = set(params) - {p.name for p in info.params}
    if unknown:
        raise StatsError(
            f"Unknown option(s) for '{info.label}': {sorted(unknown)}. "
            f"Valid options: {[p.name for p in info.params]}"
        )
    merged = {**info.defaults(), **params}
    result = info.func(table, **merged)
    result.analysis = key
    result.table = table.name
    result.params = jsonable(merged)
    for note in table.exclusion_notes():
        if note not in result.data_notes:
            result.data_notes.append(note)
    if table.approximate:
        result.approximate_data = True
        result.warnings.insert(
            0,
            "APPROXIMATE DATA: this analysis was run on digitized values read off an image. "
            "Treat every number as an estimate; do not report it as a measured result.",
        )
    result.extra = jsonable(result.extra)
    result.descriptives = jsonable(result.descriptives)
    result.anova_table = jsonable(result.anova_table)
    return result.finalize()
