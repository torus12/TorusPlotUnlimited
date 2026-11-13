"""Prompt builders for the AI layer.

What a model is told is deliberately narrow:

* **Tables**: names, kinds, column / series / group / row names and counts. Never a data value.
* **Statistics**: result IDs, test names and the group names / IDs of computed comparisons,
  so the model can choose *which* computed comparisons to annotate. No P value, statistic,
  effect size or significance flag is ever put in a prompt.
* **Schema**: a compact field reference generated from the pydantic models (there is no
  hand-maintained second copy of the schema).

Every user prompt starts with a ``TASK: <name>`` line; the offline mock provider uses it
to pick a recorded answer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..data.tables import (
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    XYTable,
)
from ..spec.figure import PLOT_TYPES, FigureSpec
from ..spec.schema import figure_spec_schema, minimal_example, plot_type_guide
from ..stats.result import StatResult

TASK_DESCRIBE = "describe_figure"
TASK_STYLE = "extract_style"
TASK_CRITIQUE = "critique_figure"

#: Categorical columns with more levels than this are summarised by a count only
#: (long lists are usually subject IDs, i.e. data rather than group names).
MAX_LEVELS_LISTED = 12

STATS_RULE = (
    "Statistics must be computed in TorusPlot and attached to a figure by result ID; "
    "they can never be typed or pasted in."
)

RULES = """\
1. Answer with ONE JSON object that is a valid FigureSpec. JSON only: no prose, no markdown, no comments.
2. You only describe how the figure LOOKS. You never compute, estimate, write or change statistics. Never put P values, significance stars, "ns", test statistics, effect sizes, confidence limits or claims such as "significant" anywhere (title, labels, legend, tick labels, texts).
3. Significance brackets are drawn by TorusPlot from a result it has already computed. To show them, add an item to "annotations" whose "stat_result_id" is one of the listed result IDs (optionally choose comparisons by ID or by group pairs, or use "comparisons": "significant"). If no result is listed, leave "annotations" empty. Never invent an ID.
4. Use only the table, column, series, group and row names that are listed, spelled exactly as given.
5. "data.table" must be the name of a listed table; unless the request says otherwise use the current table.
6. Choose a "plot_type" that is valid for the kind of that table (see PLOT TYPES).
7. Leave out every field you do not need: unset fields use the style preset. "layers": [] draws the default layers of the plot type.
8. Colours are hex strings such as "#1F77B4". Sizes of the figure are in millimetres."""


# ---------------------------------------------------------------------- inputs
def normalize_tables(tables: Any) -> dict[str, DataTable]:
    """Accept a mapping, a single table, a list of tables or a Project; return ``{name: table}``."""
    if tables is None:
        return {}
    if isinstance(tables, DataTable):
        return {tables.name: tables}
    if isinstance(tables, Mapping):
        return {str(k): v for k, v in tables.items()}
    inner = getattr(tables, "tables", None)
    if isinstance(inner, Mapping):
        return {str(k): v for k, v in inner.items()}
    if isinstance(tables, Iterable):
        return {t.name: t for t in tables}
    raise TypeError("tables must be a mapping of name -> table, a table, a list of tables or a Project.")


def normalize_results(results: Any) -> dict[str, StatResult]:
    """Accept a mapping, a list of results, a Project or None; return ``{id: StatResult}``."""
    if results is None:
        return {}
    inner = getattr(results, "results", None)
    if inner is not None and not isinstance(results, Mapping):
        results = inner
    items = results.values() if isinstance(results, Mapping) else results
    out: dict[str, StatResult] = {}
    for item in items:
        res = item if isinstance(item, StatResult) else StatResult.model_validate(item)
        out[res.id] = res
    return out


def resolve_table(table: DataTable | str | None, tables: Mapping[str, DataTable]) -> DataTable | None:
    """A table given as an object or a name (None when it cannot be resolved)."""
    if isinstance(table, DataTable):
        return table
    if isinstance(table, str) and table in tables:
        return tables[table]
    return None


# ---------------------------------------------------------------------- summaries (names + counts only)
def table_summary(table: DataTable) -> dict[str, Any]:
    """Names and counts describing a table. Contains no data value."""
    out: dict[str, Any] = {"name": table.name, "kind": table.kind, "rows": int(len(table.data))}
    if isinstance(table, ColumnTable):
        counts = {g: int(v.size) for g, v in table.groups().items()}
        out.update(groups=table.group_names, n_per_group=counts)
    elif isinstance(table, XYTable):
        out.update(
            x=table.x_name,
            series=table.series_names,
            replicates={s: len(cols) for s, cols in table.series_map.items()},
        )
    elif isinstance(table, GroupedTable):
        out.update(
            row_label=table.row_label,
            row_levels=table.row_levels,
            column_label=table.column_label,
            groups=table.column_levels,
            replicates={g: len(cols) for g, cols in table.group_map.items()},
        )
    elif isinstance(table, MultipleVariablesTable):
        categorical: dict[str, Any] = {}
        for col in table.categorical_columns:
            levels = list(dict.fromkeys(str(v) for v in table.data[col].tolist() if v is not None))
            entry: dict[str, Any] = {"n_levels": len(levels)}
            if len(levels) <= MAX_LEVELS_LISTED:
                entry["levels"] = levels
            categorical[col] = entry
        out.update(numeric_columns=table.numeric_columns, categorical_columns=categorical)
    elif isinstance(table, ContingencyTable):
        out.update(
            row_label=table.row_label,
            row_levels=table.row_levels,
            column_label=table.column_label,
            groups=table.column_levels,
        )
    if table.approximate:
        out["approximate"] = True
    return out


def results_summary(results: Any) -> list[dict[str, Any]]:
    """IDs, test names and comparison group names only. No number is included."""
    out = []
    for res in normalize_results(results).values():
        entry: dict[str, Any] = {"id": res.id, "test": res.test_name}
        if res.table:
            entry["table"] = res.table
        comps = []
        for comp in res.comparisons:
            item: dict[str, Any] = {"id": comp.id, "groups": [comp.group_a, comp.group_b]}
            if comp.label:
                item["within"] = comp.label
            comps.append(item)
        if comps:
            entry["comparisons"] = comps
        out.append(entry)
    return out


def default_binding(summary: Mapping[str, Any]) -> dict[str, Any]:
    """A sensible default ``{"plot_type", "data"}`` for a table summary (no AI involved)."""
    name, kind = summary["name"], summary["kind"]
    data: dict[str, Any] = {"table": name}
    if kind == "xy":
        return {"plot_type": "xy", "data": data}
    if kind == "contingency":
        return {"plot_type": "stacked_bar", "data": data}
    if kind == "multiple_variables":
        numeric = list(summary.get("numeric_columns") or [])
        categorical = summary.get("categorical_columns") or {}
        listed = [c for c, info in categorical.items() if "levels" in info]
        if numeric and listed:
            data.update(x=listed[0], y=numeric[0])
            return {"plot_type": "bar", "data": data}
        if numeric:
            data.update(y=numeric[0])
            return {"plot_type": "histogram", "data": data}
        return {"plot_type": "bar", "data": data}
    return {"plot_type": "bar", "data": data}


# ---------------------------------------------------------------------- schema-derived reference
def _type_text(node: Mapping[str, Any]) -> str:
    if "$ref" in node:
        return str(node["$ref"]).rsplit("/", 1)[-1]
    if "const" in node:
        return json.dumps(node["const"])
    if "enum" in node:
        return " | ".join(json.dumps(v) for v in node["enum"])
    for key in ("anyOf", "oneOf"):
        if key in node:
            parts = [_type_text(p) for p in node[key] if p.get("type") != "null"]
            return " | ".join(dict.fromkeys(parts)) or "null"
    kind = node.get("type")
    if kind == "array":
        if "prefixItems" in node:
            return "[" + ", ".join(_type_text(p) for p in node["prefixItems"]) + "]"
        return "[" + _type_text(node.get("items", {})) + ", ...]"
    if kind == "object":
        extra = node.get("additionalProperties")
        return "{name: " + (_type_text(extra) if isinstance(extra, Mapping) else "any") + "}"
    text = {"number": "number", "integer": "integer", "string": "string", "boolean": "true | false"}.get(
        str(kind), str(kind or "any")
    )
    low = node.get("minimum", node.get("exclusiveMinimum"))
    high = node.get("maximum", node.get("exclusiveMaximum"))
    if low is not None or high is not None:
        text += f" ({'' if low is None else low}..{'' if high is None else high})"
    return text


def _is_optional(node: Mapping[str, Any]) -> bool:
    return any(p.get("type") == "null" for p in node.get("anyOf", []))


def _range_of(node: Mapping[str, Any]) -> Mapping[str, Any]:
    """For ``anyOf [number, null]`` return the non-null branch (to show its limits)."""
    branches = [p for p in node.get("anyOf", []) if p.get("type") != "null"]
    return branches[0] if len(branches) == 1 else node


def field_reference() -> str:
    """Compact, human-readable field list generated from ``FigureSpec.model_json_schema()``."""
    schema = figure_spec_schema()
    defs: Mapping[str, Any] = schema.get("$defs", {})
    models: dict[str, Mapping[str, Any]] = {"FigureSpec": schema}

    def collect(node: Any) -> None:
        if isinstance(node, Mapping):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name not in models and name in defs:
                    models[name] = defs[name]
                    collect(defs[name])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(schema.get("properties", {}))
    lines: list[str] = []
    for name, model in models.items():
        required = set(model.get("required", []))
        lines.append(f"{name}:")
        for field, node in model.get("properties", {}).items():
            text = _type_text(_range_of(node))
            flags = []
            if field in required:
                flags.append("REQUIRED")
            elif _is_optional(node):
                flags.append("optional")
            if "default" in node and node["default"] is not None and field not in required:
                default = json.dumps(node["default"])
                if len(default) <= 40:
                    flags.append(f"default {default}")
            line = f"  {field}: {text}"
            if flags:
                line += f"  [{', '.join(flags)}]"
            if node.get("description"):
                line += f"  - {node['description']}"
            lines.append(line)
    return "\n".join(lines)


def style_catalog() -> str:
    """Style presets and palettes by name (imported lazily: this pulls in matplotlib)."""
    from ..plot.palettes import DIVERGING, PALETTES, SEQUENTIAL
    from ..plot.styles import get_preset, preset_names

    lines = ["STYLE PRESETS (style.preset):"]
    for name in preset_names():
        preset = get_preset(name)
        description = " ".join(str(preset.get("description", "")).split())
        lines.append(f"- {name}: {description} ({preset['width_mm']:g} x {preset['height_mm']:g} mm)")
    lines.append("")
    lines.append("PALETTES (style.palette; or give your own list of hex colours):")
    for name, colors in PALETTES.items():
        lines.append(f"- {name}: {', '.join(colors)}")
    lines.append(f"- Sequential colormaps sampled as palettes: {', '.join(SEQUENTIAL)}")
    lines.append(f"- Diverging colormaps (style.colormap, for heat maps): {', '.join(DIVERGING)}")
    return "\n".join(lines)


def reference_text() -> str:
    """Rules + plot types + presets/palettes + field reference + example (shared by all prompts)."""
    return "\n\n".join(
        [
            "RULES\n" + RULES,
            "PLOT TYPES (plot_type: what it is. Tables = kinds of table it can draw)\n" + plot_type_guide(),
            style_catalog(),
            "FIELD REFERENCE (generated from the FigureSpec schema; unknown fields are rejected)\n"
            + field_reference(),
            "EXAMPLE OF A VALID ANSWER\n" + json.dumps(minimal_example(), indent=1),
        ]
    )


# ---------------------------------------------------------------------- describe
def describe_system_prompt() -> str:
    return (
        "You turn a scientist's plain-language request into a FigureSpec: the JSON description of a "
        "figure in the TorusPlot graphing program. TorusPlot draws the figure and computes every "
        "number itself.\n\n" + reference_text()
    )


def tables_block(tables: Mapping[str, DataTable]) -> str:
    if not tables:
        return "(no tables)"
    return "\n".join(json.dumps(table_summary(t), ensure_ascii=False) for t in tables.values())


def results_block(results: Any) -> str:
    summary = results_summary(results)
    if not summary:
        return "(none: leave \"annotations\" empty)"
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in summary)


def describe_user_prompt(
    text: str,
    tables: Mapping[str, DataTable],
    results: Any = None,
    *,
    current_table: str | None = None,
    base_spec: FigureSpec | None = None,
) -> str:
    """User prompt of "Describe it" (and of edits of an existing figure when ``base_spec`` is given)."""
    parts = [f"TASK: {TASK_DESCRIBE}"]
    if current_table:
        parts.append(f"CURRENT_TABLE: {current_table}")
    parts.append("")
    parts.append("TABLES (one JSON object per line; names and counts only, the data stay in TorusPlot):")
    parts.append(tables_block(tables))
    parts.append("")
    parts.append(
        "COMPUTED STATISTICS RESULTS (IDs you may reference in \"annotations\"; the numbers stay in TorusPlot):"
    )
    parts.append(results_block(results))
    if base_spec is not None:
        parts.append("")
        parts.append(
            "CURRENT FIGURE (the request below is a change to this figure; keep everything that is "
            "not mentioned and return the COMPLETE updated FigureSpec):"
        )
        parts.append(base_spec.model_dump_json(exclude_defaults=True))
    parts.append("")
    parts.append("REQUEST:")
    parts.append("<<<")
    parts.append(text.strip())
    parts.append(">>>")
    parts.append("")
    parts.append("Return one JSON object (the FigureSpec). JSON only.")
    return "\n".join(parts)


def retry_prompt(prompt: str, previous_output: str, error: str, *, max_chars: int = 6000) -> str:
    """The original prompt plus the model's invalid answer and the exact validation error."""
    previous = previous_output.strip()
    if len(previous) > max_chars:
        previous = previous[:max_chars] + "\n... (truncated)"
    return (
        f"{prompt}\n\nYOUR PREVIOUS ANSWER:\n<<<\n{previous}\n>>>\n"
        f"Your previous answer was invalid: {error}\nReturn the corrected JSON only."
    )


# ---------------------------------------------------------------------- match this figure (vision)
def style_system_prompt() -> str:
    return (
        "You look at a picture of a scientific figure and describe its visual STYLE only, so that "
        "another program can style a different figure the same way. Do not read, estimate or report "
        "any data values, numbers, axis values, statistics, P values, stars or the wording of labels. "
        "Answer with one JSON object and nothing else."
    )


def style_user_prompt() -> str:
    plot_types = " | ".join(f'"{k}"' for k in PLOT_TYPES)
    return "\n".join(
        [
            f"TASK: {TASK_STYLE}",
            "",
            "Describe the STYLE of the attached figure. Do NOT read data values. Use null when unsure.",
            "Return exactly these keys:",
            "{",
            f' "plot_type": {plot_types} | null,',
            ' "colors": ["#RRGGBB", ...]   (fill/line colours of the data series, in legend order),',
            ' "background": "#RRGGBB" | null,',
            ' "text_color": "#RRGGBB" | null,',
            ' "has_top_right_spines": true | false | null   (is the plot area boxed on all four sides?),',
            ' "grid": true | false | null,',
            ' "font_category": "sans" | "serif" | null,',
            ' "bar_edge": "none" | "black" | "group" | "#RRGGBB" | null   (outline of bars/boxes),',
            ' "error_bars": "sem" | "sd" | "ci95" | "range" | "iqr" | "none" | null   (only if the figure says so),',
            ' "points_shown": true | false | null   (are individual data points drawn?),',
            ' "legend_position": "best" | "upper right" | "upper left" | "lower left" | "lower right" | "right" '
            '| "outside right" | "top" | "bottom" | "none" | null',
            "}",
            "JSON only.",
        ]
    )


def critique_system_prompt() -> str:
    return (
        "You compare two images. Image 1 is a reference figure whose LOOK the user likes. Image 2 is "
        "TorusPlot's rendering of the user's OWN data, which is different data on purpose. Suggest "
        "only style changes that make image 2 look more like image 1. Never comment on, read or "
        "change data values or statistics. Answer with one JSON object and nothing else."
    )


def critique_user_prompt(spec: FigureSpec, allowed_paths: Iterable[str]) -> str:
    current = spec.model_dump(mode="json", include={"plot_type", "style", "layout", "legend", "axes"},
                              exclude_none=True)
    current["layers"] = [
        {"index": i, **lay.model_dump(mode="json", exclude_none=True, exclude={"stat_result_id", "label"})}
        for i, lay in enumerate(spec.effective_layers())
    ]
    return "\n".join(
        [
            f"TASK: {TASK_CRITIQUE}",
            "",
            "CURRENT STYLE OF IMAGE 2 (JSON):",
            json.dumps(current, ensure_ascii=False),
            "",
            "You may change ONLY these settings (dot-paths; <i> is a layer index; anything else is ignored):",
            ", ".join(allowed_paths),
            "",
            "Return:",
            '{"patch": {"<dot-path>": <new value>, ...}, "done": true | false, "comment": "<one short sentence>"}',
            'Use "done": true (and an empty patch) when the styles already match well. JSON only.',
        ]
    )
