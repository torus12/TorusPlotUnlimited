""""Describe it": plain-language request -> validated FigureSpec.

THE HARD RULE: AI may only produce or modify a FigureSpec. It never computes, edits or
invents statistics. This module enforces that in code, not by trusting the prompt:

1. The prompt contains table/column/group *names* and, for statistics, only result IDs,
   test names and comparison group names. No P value or other number is sent.
2. Every model answer goes through ``FigureSpec.model_validate`` (the schema has no field
   that can hold a P value or star string and rejects such text in titles, texts and
   layer labels), plus :func:`check_ai_text`, which applies the same idea to the remaining
   free-text fields (axis labels, tick labels, legend title).
3. :func:`sanitize_spec` then removes every reference to a statistics result that was not
   provided (never kept, never invented), fixes an unknown table and drops unknown group
   names, each with a warning.

Retry policy of :func:`describe_to_spec`: the provider is called once; each time the
answer is not valid the provider is called again with the previous answer and the exact
validation error appended. With ``max_retries=N`` there are at most ``1 + N`` provider
calls. ``SpecOutcome.attempts`` is the number of provider calls that were made.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..data.tables import (
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    XYTable,
)
from ..spec.figure import FigureSpec, TextAnnotation, plot_types_for
from ..stats.result import StatResult
from .jsonutil import JSONExtractError, extract_json
from .prompts import (
    STATS_RULE,
    describe_system_prompt,
    describe_user_prompt,
    normalize_results,
    normalize_tables,
    retry_prompt,
)
from .providers.base import Provider

#: Text that reads like a typed statistic but is not caught by the schema's own rule.
_STAT_TEXT = re.compile(
    r"(\bp\s*[<=>≤≥]\s*[\d.])"  # p < 1e-5, P=.05
    r"|(\bp[\s\-]?val)"  # p value, p-value
    r"|((?<![\w*])\*{2,4}(?![\w*]))"  # ** / *** / **** standing alone
    r"|(\bsignifican)"  # significant, significance
    r"|(\bn\.s\.)",
    re.IGNORECASE,
)

_MV_ROLES = ("x", "y", "hue", "replicate", "subject", "time", "event")


class SpecCheckError(ValueError):
    """A model answer is a valid FigureSpec but cannot be used (wrong table kind, typed statistics...)."""


@dataclass
class SpecOutcome:
    """Result of an AI or outside-instructions request.

    ``spec`` is None when no valid figure could be produced; ``errors`` then says why (one
    entry per failed attempt). ``warnings`` lists what the sanitiser changed. ``attempts``
    counts provider calls (0 when no provider was needed). ``raw`` holds the raw answers.
    """

    spec: FigureSpec | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.spec is not None

    @property
    def message(self) -> str:
        """One user-facing sentence about the outcome."""
        if self.spec is not None:
            return "Figure updated." if not self.warnings else f"Figure updated with {len(self.warnings)} note(s)."
        return self.errors[-1] if self.errors else "No figure could be produced."


# ---------------------------------------------------------------------- validation helpers
def format_validation_error(exc: ValidationError, limit: int = 12) -> str:
    """Compact ``path: message`` list of a pydantic error (what is fed back to the model)."""
    lines = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}")
    extra = len(exc.errors()) - limit
    if extra > 0:
        lines.append(f"... and {extra} more problem(s)")
    return "; ".join(lines)


def looks_like_statistic(text: Any) -> bool:
    """True if ``text`` reads like a typed statistic (P value, stars, 'significant' ...)."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        TextAnnotation(text=text, x=0.0, y=0.0)  # the schema's own rule
    except ValidationError:
        return True
    return bool(_STAT_TEXT.search(text))


def free_text_fields(spec: FigureSpec) -> list[tuple[str, str]]:
    """``(path, text)`` of every free-text field a model could write into."""
    out: list[tuple[str, str]] = []
    if spec.title:
        out.append(("title", spec.title))
    for i, text in enumerate(spec.texts):
        out.append((f"texts.{i}.text", text.text))
    for i, layer in enumerate(spec.layers):
        if layer.label:
            out.append((f"layers.{i}.label", layer.label))
    for axis_name in ("x", "y"):
        axis = getattr(spec.axes, axis_name)
        if axis.label:
            out.append((f"axes.{axis_name}.label", axis.label))
        for j, label in enumerate(axis.tick_labels or []):
            out.append((f"axes.{axis_name}.tick_labels.{j}", label))
    if spec.legend.title:
        out.append(("legend.title", spec.legend.title))
    return out


def check_ai_text(spec: FigureSpec, base_spec: FigureSpec | None = None) -> None:
    """Raise :class:`SpecCheckError` if any free text written by a model looks like a statistic.

    Text that is unchanged from ``base_spec`` (the user's own figure) is not re-checked.
    """
    existing = dict(free_text_fields(base_spec)) if base_spec is not None else {}
    kept = set(existing.values())
    for path, text in free_text_fields(spec):
        if text in kept:
            continue
        if looks_like_statistic(text):
            raise SpecCheckError(
                f"{path} looks like a statistical result ('{text}'). Statistics cannot be typed into a "
                "figure: remove it, and to show significance add an item to \"annotations\" that "
                "references a computed result by its \"stat_result_id\"."
            )


def check_plot_type(spec: FigureSpec, table: DataTable) -> None:
    """Raise :class:`SpecCheckError` if the plot type cannot be drawn from this kind of table."""
    valid = plot_types_for(table.kind)
    if spec.plot_type not in valid:
        raise SpecCheckError(
            f"plot_type '{spec.plot_type}' cannot be drawn from table '{table.name}', which is a "
            f"{table.kind_label} table. Valid plot types for this table: {', '.join(valid)}."
        )


def _group_names(table: DataTable, spec: FigureSpec) -> tuple[list[str] | None, list[str] | None]:
    """(valid ``data.groups`` names, valid ``data.rows`` names); None = cannot be checked."""
    if isinstance(table, ColumnTable):
        return table.group_names, None
    if isinstance(table, XYTable):
        return table.series_names, None
    if isinstance(table, GroupedTable | ContingencyTable):
        return table.column_levels, table.row_levels
    if isinstance(table, MultipleVariablesTable):
        column = spec.data.hue if spec.plot_type in {"xy", "dose_response"} else spec.data.x
        if column and column in table.data.columns:
            levels = [str(v) for v in table.data[column].tolist() if v is not None and str(v) != "nan"]
            return list(dict.fromkeys(levels)), None
    return None, None


def _color_names(table: DataTable, groups: list[str] | None, rows: list[str] | None) -> set[str] | None:
    """Names that ``style.colors`` may refer to (None = cannot be checked)."""
    if isinstance(table, MultipleVariablesTable):
        names: set[str] = set(table.numeric_columns)
        for col in table.categorical_columns:
            names.update(str(v) for v in table.data[col].tolist() if v is not None)
        return names
    if groups is None:
        return None
    return set(groups) | set(rows or [])


def resolve_data_table(
    name: str | None, tables: Mapping[str, DataTable], current_table: str | None
) -> tuple[str, str | None]:
    """Return ``(table name to use, warning or None)``; raise :class:`SpecCheckError` if undecidable."""
    if name and name in tables:
        return name, None
    fallback = current_table if current_table in tables else None
    if fallback is None and len(tables) == 1:
        fallback = next(iter(tables))
    if fallback is None:
        available = ", ".join(f"'{t}'" for t in tables) or "(none)"
        if name:
            raise SpecCheckError(f"data.table '{name}' is not one of the available tables: {available}.")
        raise SpecCheckError(f"data.table is missing. Available tables: {available}.")
    if name:
        return fallback, f"Table '{name}' does not exist; the figure uses the current table '{fallback}' instead."
    return fallback, f"No table was named; the figure uses the current table '{fallback}'."


def sanitize_spec(
    spec: FigureSpec,
    tables: Any,
    results: Any = None,
    *,
    current_table: str | None = None,
) -> tuple[FigureSpec, list[str]]:
    """Make a validated spec safe to use with the given tables and computed results.

    * ``data.table`` that does not exist -> the current table (warning), or
      :class:`SpecCheckError` when there is no current table to fall back to.
    * Any ``stat_result_id`` (annotations, layers, data) that is not in ``results`` is
      **removed** with a warning. Comparison IDs and group pairs that the referenced result
      does not contain are dropped the same way. Nothing is ever invented or guessed.
    * Group, row and column names that do not exist in the table are dropped with a warning.

    Returns ``(new spec, warnings)``; the input spec is not modified.
    """
    tables = normalize_tables(tables)
    known: dict[str, StatResult] = normalize_results(results)
    warnings: list[str] = []
    data = spec.model_dump(mode="python")

    table_name, note = resolve_data_table(spec.data.table, tables, current_table)
    if note:
        warnings.append(note)
    data["data"]["table"] = table_name
    table = tables[table_name]
    binding = data["data"]

    # --- column roles of a Multiple-variables table
    if isinstance(table, MultipleVariablesTable):
        columns = set(table.data.columns)
        for role in _MV_ROLES:
            value = binding.get(role)
            if value is not None and value not in columns:
                warnings.append(f"Column '{value}' ({role}) is not in table '{table_name}' and was ignored.")
                binding[role] = None
        if binding.get("columns"):
            kept = [c for c in binding["columns"] if c in columns]
            for c in binding["columns"]:
                if c not in columns:
                    warnings.append(f"Column '{c}' is not in table '{table_name}' and was ignored.")
            binding["columns"] = kept or None

    # --- group / row names
    probe = spec.model_copy(update={"data": spec.data.model_copy(update={k: binding.get(k) for k in _MV_ROLES})})
    groups, rows = _group_names(table, probe)
    for key, valid, what in (("groups", groups, "Group"), ("rows", rows, "Row")):
        wanted = binding.get(key)
        if wanted and valid is not None:
            kept = [g for g in wanted if g in valid]
            for g in wanted:
                if g not in valid:
                    warnings.append(f"{what} '{g}' does not exist in table '{table_name}' and was dropped.")
            binding[key] = kept or None
    color_names = _color_names(table, groups, rows)
    if color_names is not None:
        for name in list(data["style"]["colors"]):
            if name not in color_names:
                warnings.append(f"Colour override for '{name}' was dropped: there is no such group.")
                del data["style"]["colors"][name]
    if groups is not None:
        kept_layers = []
        for i, layer in enumerate(data["layers"]):
            if layer.get("series") is not None and layer["series"] not in groups:
                warnings.append(
                    f"Layer {i} ({layer['type']}) was dropped: series '{layer['series']}' does not exist."
                )
                continue
            kept_layers.append(layer)
        data["layers"] = kept_layers

    # --- statistics references: remove what was not computed
    def missing(rid: str) -> str:
        return f"refers to statistics result '{rid}', which has not been computed. {STATS_RULE}"

    if binding.get("stat_result_id") and binding["stat_result_id"] not in known:
        warnings.append(f"The figure's data binding {missing(binding['stat_result_id'])} The reference was removed.")
        binding["stat_result_id"] = None
    for i, layer in enumerate(data["layers"]):
        rid = layer.get("stat_result_id")
        if rid and rid not in known:
            warnings.append(f"Layer {i} ({layer['type']}) {missing(rid)} The reference was removed.")
            layer["stat_result_id"] = None
    kept_annotations = []
    for i, ann in enumerate(data["annotations"]):
        rid = ann["stat_result_id"]
        result = known.get(rid)
        if result is None:
            warnings.append(f"Annotation {i + 1} {missing(rid)} The annotation was removed.")
            continue
        if result.table and result.table != table_name:
            warnings.append(
                f"Annotation {i + 1} uses result '{rid}', which was computed on table '{result.table}', "
                f"not on '{table_name}'. Check that it belongs on this figure."
            )
        emptied = False
        if isinstance(ann["comparisons"], list):
            ids = {c.id for c in result.comparisons}
            kept = [c for c in ann["comparisons"] if c in ids]
            for c in ann["comparisons"]:
                if c not in ids:
                    warnings.append(f"Comparison '{c}' is not part of result '{rid}' and was dropped.")
            emptied = bool(ann["comparisons"]) and not kept
            ann["comparisons"] = kept
        if ann.get("pairs"):
            pairs = {frozenset((c.group_a, c.group_b)) for c in result.comparisons}
            kept_pairs = [tuple(p) for p in ann["pairs"] if frozenset(p) in pairs]
            for p in ann["pairs"]:
                if frozenset(p) not in pairs:
                    warnings.append(
                        f"Result '{rid}' has no comparison between '{p[0]}' and '{p[1]}'; that pair was dropped."
                    )
            emptied = emptied or not kept_pairs
            ann["pairs"] = kept_pairs or None
        if emptied:
            warnings.append(f"Annotation {i + 1} was removed: none of its comparisons exist in result '{rid}'.")
            continue
        kept_annotations.append(ann)
    data["annotations"] = kept_annotations

    return FigureSpec.model_validate(data), warnings


def validate_answer(
    raw: str | Mapping[str, Any],
    tables: Any,
    results: Any = None,
    *,
    current_table: str | None = None,
    base_spec: FigureSpec | None = None,
) -> tuple[FigureSpec, list[str]]:
    """Raw model text (or a parsed dict) -> ``(safe FigureSpec, warnings)``.

    Raises :class:`JSONExtractError`, :class:`pydantic.ValidationError` or
    :class:`SpecCheckError`; all three are problems to feed back to the model.
    """
    tables = normalize_tables(tables)
    payload = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else extract_json(raw)
    if base_spec is not None and "name" not in payload:
        payload["name"] = base_spec.name
    binding = payload.get("data")
    if isinstance(binding, dict) and not binding.get("table") and current_table:
        binding["table"] = current_table  # the prompt says: unless told otherwise use the current table
    spec = FigureSpec.model_validate(payload)
    check_ai_text(spec, base_spec)
    name, _ = resolve_data_table(spec.data.table, tables, current_table)
    check_plot_type(spec, tables[name])
    return sanitize_spec(spec, tables, results, current_table=current_table)


def error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "The JSON is not a valid FigureSpec: " + format_validation_error(exc)
    return str(exc)


# ---------------------------------------------------------------------- public entry point
def describe_to_spec(
    text: str,
    tables: Any,
    results: Any = None,
    *,
    current_table: str | None = None,
    provider: Provider,
    max_retries: int = 3,
    base_spec: FigureSpec | None = None,
) -> SpecOutcome:
    """Turn a plain-language request into a validated, sanitised :class:`FigureSpec`.

    ``tables`` is a mapping ``{name: table}`` (or a table, a list, or a Project);
    ``results`` the computed ``StatResult`` objects the figure may reference.
    With ``base_spec`` the request is an edit of that figure ("make the bars blue").

    The provider is called at most ``1 + max_retries`` times (see the module docstring).
    A :class:`~torusplot.ai.providers.base.ProviderError` (service down, key missing...)
    is not retried and propagates to the caller.
    """
    tables = normalize_tables(tables)
    known = normalize_results(results)
    outcome = SpecOutcome()
    if not text or not text.strip():
        outcome.errors.append("Describe the figure you want in a sentence or two.")
        return outcome
    if not tables:
        outcome.errors.append("Add a data table first: a figure is always drawn from one of your tables.")
        return outcome
    if current_table is None and base_spec is not None and base_spec.data.table in tables:
        current_table = base_spec.data.table
    if current_table is None and len(tables) == 1:
        current_table = next(iter(tables))

    system = describe_system_prompt()
    first_prompt = describe_user_prompt(text, tables, known, current_table=current_table, base_spec=base_spec)
    prompt = first_prompt
    for _ in range(1 + max(0, int(max_retries))):
        raw = provider.complete(system, prompt, json_mode=True)
        outcome.attempts += 1
        outcome.raw.append(raw)
        try:
            spec, warnings = validate_answer(
                raw, tables, known, current_table=current_table, base_spec=base_spec
            )
        except (JSONExtractError, ValidationError, SpecCheckError) as exc:
            message = error_text(exc)
            outcome.errors.append(message)
            prompt = retry_prompt(first_prompt, raw, message)
            continue
        outcome.spec = spec
        outcome.warnings = warnings
        return outcome
    return outcome
