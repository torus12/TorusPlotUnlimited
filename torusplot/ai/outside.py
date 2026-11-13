""""Outside AI instructions": import a figure description written by any chatbot.

The user copies the prompt from ``docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md`` into a chatbot of
their choice and pastes the answer back. When the pasted text contains JSON, this path
needs **no provider and no network**: the JSON is cleaned up leniently, validated against
the FigureSpec schema and sanitised.

Lenient clean-up (each change is reported as a warning):

* unknown keys are dropped (recursively, by comparing with the pydantic model fields);
* a few common aliases are understood (``type``/``kind``/``chart`` -> ``plot_type``,
  ``table``/``dataset`` -> ``data.table``, ``palette``/``preset``/``font`` -> ``style``,
  ``width``/``height`` in mm -> ``layout``, ``xlabel``/``ylabel`` -> axis labels);
* a missing or unknown ``data.table`` becomes the current table;
* **statistics cannot be pasted in**: annotations without a valid ``stat_result_id``,
  keys such as ``p``/``p_value``/``stars``/``significance`` and text that reads like a
  statistic are dropped, with a warning that statistics must be computed in TorusPlot and
  attached by result ID.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ..spec.figure import PLOT_TYPES, FigureSpec, StatAnnotation
from ..spec.schema import figure_spec_schema_json, minimal_example, plot_type_guide
from .describe import (
    SpecCheckError,
    SpecOutcome,
    check_plot_type,
    describe_to_spec,
    error_text,
    looks_like_statistic,
    resolve_data_table,
    sanitize_spec,
)
from .jsonutil import JSONExtractError, extract_json
from .prompts import (
    RULES,
    STATS_RULE,
    normalize_results,
    normalize_tables,
    style_catalog,
    table_summary,
)
from .providers.base import Provider

TEMPLATE_DOC = "docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md"

NO_PROVIDER_MESSAGE = (
    "This text contains no JSON figure description. Either ask your chatbot to answer with JSON "
    f"using TorusPlot's prompt template ({TEMPLATE_DOC}), or enable an AI provider in Settings so "
    "TorusPlot can interpret plain-language instructions."
)

_STAT_KEY = re.compile(
    r"^(p|p_?val(ue)?s?|p_?adj(usted)?|adj(usted)?_?p(_?val(ue)?)?|q_?val(ue)?s?|stars?|asterisks?"
    r"|signific\w*|sig|ns|stat(istic)?s?|test_?(result|statistic)s?|effect_?sizes?"
    r"|ci(_?(low|high|lower|upper))?|confidence_?intervals?|results?)$",
    re.IGNORECASE,
)

_WRAPPERS = ("figure", "spec", "figure_spec", "figurespec", "FigureSpec")
_PLOT_TYPE_ALIASES = ("type", "kind", "chart", "chart_type", "plot")
_PLOT_TYPE_SYNONYMS = {
    "bar_chart": "bar",
    "bar_plot": "bar",
    "barplot": "bar",
    "column": "bar",
    "box_plot": "box",
    "boxplot": "box",
    "violin_plot": "violin",
    "swarm": "beeswarm",
    "line": "xy",
    "line_plot": "xy",
    "line_chart": "xy",
    "scatter_plot": "scatter",
    "kaplan_meier": "survival",
    "km": "survival",
    "heat_map": "heatmap",
    "hist": "histogram",
    "grouped_bars": "grouped_bar",
    "stacked_bars": "stacked_bar",
}
_STYLE_ALIASES = {"palette": "palette", "preset": "preset", "font": "font_family", "font_family": "font_family",
                  "font_size": "font_size"}
_LAYOUT_ALIASES = {"width": "width_mm", "height": "height_mm", "width_mm": "width_mm", "height_mm": "height_mm"}
_LABEL_ALIASES = {"xlabel": "x", "x_label": "x", "ylabel": "y", "y_label": "y"}


# ---------------------------------------------------------------------- clean-up
def _nested_model(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """``(pydantic model inside the annotation, is it a list of that model)``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False
    origin = get_origin(annotation)
    if origin is list:
        model, _ = _nested_model(get_args(annotation)[0])
        return model, model is not None
    if origin in (Union, UnionType):
        for arg in get_args(annotation):
            if arg is NoneType:
                continue
            model, is_list = _nested_model(arg)
            if model is not None:
                return model, is_list
    return None, False


def _prune(node: Any, model: type[BaseModel], path: str, warnings: list[str]) -> Any:
    """Drop keys that the model does not define (recursively)."""
    if not isinstance(node, dict):
        return node
    for key in list(node):
        if key not in model.model_fields:
            if model is StatAnnotation or _STAT_KEY.match(str(key)):
                warnings.append(f"'{path}{key}' was dropped: {STATS_RULE}")
            else:
                warnings.append(f"Unknown setting '{path}{key}' was ignored.")
            del node[key]
            continue
        sub, is_list = _nested_model(model.model_fields[key].annotation)
        if sub is None:
            continue
        if is_list and isinstance(node[key], list):
            node[key] = [_prune(item, sub, f"{path}{key}[{i}].", warnings) for i, item in enumerate(node[key])]
        elif not is_list:
            _prune(node[key], sub, f"{path}{key}.", warnings)
    return node


def _mm(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mm)?\s*", value)
        if match:
            return float(match.group(1))
    return None


def _normalize_plot_type(value: Any) -> Any:
    """'Box plot' -> 'box', 'bar chart' -> 'bar' ... (unknown values are returned unchanged)."""
    if not isinstance(value, str):
        return value
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if key in PLOT_TYPES:
        return key
    if key in _PLOT_TYPE_SYNONYMS:
        return _PLOT_TYPE_SYNONYMS[key]
    stripped = re.sub(r"_(plot|chart|graph)$", "", key)
    if stripped in PLOT_TYPES:
        return stripped
    return _PLOT_TYPE_SYNONYMS.get(stripped, value)


def _sub(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if not isinstance(payload.get(key), dict):
        payload[key] = {}
    return payload[key]


def _strip_stat_text(payload: dict[str, Any], warnings: list[str]) -> None:
    """Remove free text that reads like a typed statistic."""

    def bad(where: str, text: Any) -> bool:
        if looks_like_statistic(text):
            warnings.append(f"{where} ('{text}') was removed: it reads like a statistical result. {STATS_RULE}")
            return True
        return False

    if bad("The title", payload.get("title")):
        payload["title"] = None
    if isinstance(payload.get("texts"), list):
        payload["texts"] = [
            t for t in payload["texts"] if not (isinstance(t, dict) and bad("A text annotation", t.get("text")))
        ]
    for i, layer in enumerate(payload.get("layers") if isinstance(payload.get("layers"), list) else []):
        if isinstance(layer, dict) and bad(f"The label of layer {i}", layer.get("label")):
            layer["label"] = None
    axes = payload.get("axes")
    for name in ("x", "y"):
        axis = axes.get(name) if isinstance(axes, dict) else None
        if not isinstance(axis, dict):
            continue
        if bad(f"The {name.upper()} axis label", axis.get("label")):
            axis["label"] = None
        labels = axis.get("tick_labels")
        if isinstance(labels, list) and any(bad(f"A tick label of the {name.upper()} axis", t) for t in labels):
            axis["tick_labels"] = None
            axis["ticks"] = None
    legend = payload.get("legend")
    if isinstance(legend, dict) and bad("The legend title", legend.get("title")):
        legend["title"] = None


def clean_outside_json(
    payload: Mapping[str, Any],
    tables: Any,
    results: Any = None,
    *,
    current_table: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Lenient clean-up of a pasted JSON object. Returns ``(cleaned dict, warnings)``.

    Raises :class:`~torusplot.ai.describe.SpecCheckError` when no table can be chosen.
    """
    tables = normalize_tables(tables)
    known = normalize_results(results)
    warnings: list[str] = []
    data: dict[str, Any] = copy.deepcopy(dict(payload))

    # a spec wrapped in {"figure": {...}}
    for wrapper in _WRAPPERS:
        if "plot_type" not in data and isinstance(data.get(wrapper), dict) and len(data) <= 2:
            data = dict(data[wrapper])
            break

    # --- aliases
    for alias in _PLOT_TYPE_ALIASES:
        if alias in data:
            value = data.pop(alias)
            if "plot_type" not in data and isinstance(value, str):
                data["plot_type"] = value
                warnings.append(f"'{alias}' was read as 'plot_type'.")
    if "plot_type" in data:
        data["plot_type"] = _normalize_plot_type(data["plot_type"])
    if isinstance(data.get("data"), str):
        data["data"] = {"table": data["data"]}
    for alias in ("table", "dataset"):
        if alias in data:
            value = data.pop(alias)
            binding = _sub(data, "data")
            if "table" not in binding and isinstance(value, str):
                binding["table"] = value
                warnings.append(f"'{alias}' was read as 'data.table'.")
    if isinstance(data.get("style"), str):
        data["style"] = {"preset": data["style"]}
    for alias, target in _STYLE_ALIASES.items():
        if alias in data:
            value = data.pop(alias)
            style = _sub(data, "style")
            if target not in style:
                style[target] = value
                warnings.append(f"'{alias}' was read as 'style.{target}'.")
    for alias, target in _LAYOUT_ALIASES.items():
        if alias in data:
            value = _mm(data.pop(alias))
            layout = _sub(data, "layout")
            if value is None:
                warnings.append(f"'{alias}' was ignored: give the size as a number of millimetres.")
            elif target not in layout:
                layout[target] = value
                warnings.append(f"'{alias}' was read as 'layout.{target}' (millimetres).")
    for alias, axis in _LABEL_ALIASES.items():
        if alias in data:
            value = data.pop(alias)
            axes = _sub(data, "axes")
            target = _sub(axes, axis)
            if "label" not in target and isinstance(value, str):
                target["label"] = value
                warnings.append(f"'{alias}' was read as 'axes.{axis}.label'.")

    # --- statistics cannot be pasted in
    annotations = data.get("annotations")
    if annotations is not None:
        kept = []
        for i, ann in enumerate(annotations if isinstance(annotations, list) else [annotations]):
            rid = ann.get("stat_result_id") if isinstance(ann, dict) else None
            if not isinstance(rid, str) or rid not in known:
                what = f"refers to result '{rid}', which has not been computed" if rid else "has no result ID"
                warnings.append(f"Annotation {i + 1} was dropped: it {what}. {STATS_RULE}")
                continue
            kept.append(ann)
        data["annotations"] = kept
    _strip_stat_text(data, warnings)

    # --- unknown keys
    _prune(data, FigureSpec, "", warnings)

    # --- table
    binding = _sub(data, "data")
    name = binding.get("table") if isinstance(binding.get("table"), str) else None
    binding["table"], note = resolve_data_table(name, tables, current_table)
    if note:
        warnings.append(note)
    return data, warnings


# ---------------------------------------------------------------------- public entry point
def import_outside_instructions(
    text: str,
    tables: Any,
    results: Any = None,
    *,
    current_table: str | None = None,
    provider: Provider | None = None,
) -> SpecOutcome:
    """Import instructions written by an outside chatbot (or by hand).

    * Text containing a JSON object: cleaned, validated and sanitised **without any
      provider or network** (``SpecOutcome.attempts == 0``).
    * Plain prose: delegated to :func:`~torusplot.ai.describe.describe_to_spec` when an
      (enabled) ``provider`` is passed; otherwise ``spec`` is None and ``errors`` tells the
      user to enable a provider or to ask their chatbot for JSON with the prompt template.
    * Pasted JSON that cannot be repaired is also handed to the provider when there is one.
    """
    tables = normalize_tables(tables)
    known = normalize_results(results)
    outcome = SpecOutcome()
    if not text or not text.strip():
        outcome.errors.append("Paste the instructions (ideally the JSON answer of your chatbot) first.")
        return outcome
    if not tables:
        outcome.errors.append("Add a data table first: a figure is always drawn from one of your tables.")
        return outcome
    if current_table is None and len(tables) == 1:
        current_table = next(iter(tables))

    try:
        payload = extract_json(text)
    except JSONExtractError:
        payload = None

    if payload:
        outcome.raw.append(json.dumps(payload, ensure_ascii=False))
        try:
            cleaned, warnings = clean_outside_json(payload, tables, known, current_table=current_table)
            spec = FigureSpec.model_validate(cleaned)
            check_plot_type(spec, tables[spec.data.table])
            spec, more = sanitize_spec(spec, tables, known, current_table=current_table)
        except (ValidationError, SpecCheckError) as exc:
            outcome.errors.append(error_text(exc))
        else:
            outcome.spec = spec
            outcome.warnings = warnings + more
            return outcome

    if provider is None:
        if not payload:
            outcome.errors.append(NO_PROVIDER_MESSAGE)
        return outcome
    delegated = describe_to_spec(text, tables, known, current_table=current_table, provider=provider)
    delegated.errors = outcome.errors + delegated.errors
    delegated.raw = outcome.raw + delegated.raw
    return delegated


# ---------------------------------------------------------------------- prompt template
def example_answer(table_name: str = "My table") -> dict[str, Any]:
    """The filled example shown in the template (a valid FigureSpec)."""
    example = minimal_example()
    example["data"] = {"table": table_name}
    example["legend"] = {"show": False}
    return example


def outside_prompt_template(tables: Any = None) -> str:
    """The copy-paste prompt for any chatbot, as Markdown.

    With ``tables`` the "My data table" part is filled in with the names (never the values)
    of those tables; without, it is a placeholder for the user to fill in. The committed
    ``docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md`` is ``outside_prompt_template()`` written by
    ``tools/build_prompt_template.py``; a test keeps the two identical.
    """
    known = normalize_tables(tables)
    if known:
        table_lines = [
            "These are my tables (names and sizes only):",
            "",
            "```json",
            *[json.dumps(table_summary(t), ensure_ascii=False) for t in known.values()],
            "```",
        ]
        first = next(iter(known))
    else:
        table_lines = [
            "[DESCRIBE YOUR TABLE HERE: its name in TorusPlot, its kind (column, xy, grouped, "
            "multiple_variables or contingency) and the names of its columns / groups / series. "
            "Names only: do not paste confidential data.]",
        ]
        first = "My table"
    parts = [
        "# Prompt template for an outside AI chatbot",
        "",
        "<!-- GENERATED FILE: do not edit by hand. Regenerate with `python tools/build_prompt_template.py`. -->",
        "",
        "TorusPlot can build a figure from instructions written by any chatbot, with no API key and "
        "no AI provider configured:",
        "",
        "1. Copy everything below the line **PROMPT STARTS HERE** into your chatbot.",
        "2. Replace the parts in [SQUARE BRACKETS] with your table, your result ID (if any) and your wish.",
        "3. Paste the chatbot's JSON answer into TorusPlot's \"Outside AI instructions\" box.",
        "",
        "TorusPlot checks the answer before using it: settings it does not know are ignored, a wrong "
        "table name falls back to your current table, and statistics can never be pasted in. "
        "P values and significance stars always come from an analysis that TorusPlot itself has run.",
        "",
        "---",
        "",
        "**PROMPT STARTS HERE**",
        "",
        "You are helping me make a scientific figure in TorusPlot, a graphing and statistics program. "
        "TorusPlot draws figures from a JSON description called a FigureSpec. Write the FigureSpec for "
        "the figure I describe. TorusPlot computes every number itself.",
        "",
        "## Rules",
        "",
        RULES,
        "",
        "## My data table",
        "",
        *table_lines,
        "",
        "## Statistics results already computed in TorusPlot (optional)",
        "",
        "[IF YOU RAN AN ANALYSIS IN TORUSPLOT AND WANT ITS BRACKETS ON THE FIGURE, PASTE THE RESULT ID "
        "HERE, e.g. sr_3f9a1c2b. OTHERWISE WRITE: none.]",
        "",
        "To show a computed result, add for example "
        '`{"stat_result_id": "<the ID above>", "comparisons": "significant"}` to "annotations". '
        "Without an ID, leave \"annotations\" empty.",
        "",
        "## The figure I want",
        "",
        "[DESCRIBE THE FIGURE HERE: for example \"a box plot of all groups with a log Y axis, the "
        "Okabe-Ito palette and a single-column journal size\".]",
        "",
        "## Plot types",
        "",
        "`plot_type`: what it is. \"Tables\" lists the kinds of table it can be drawn from.",
        "",
        plot_type_guide(),
        "",
        "## Style presets and palettes",
        "",
        style_catalog(),
        "",
        "## Example of a valid answer",
        "",
        "```json",
        json.dumps(example_answer(first), indent=1, ensure_ascii=False),
        "```",
        "",
        "## Full JSON schema of a FigureSpec",
        "",
        "Fields that are not in this schema are ignored. Leave out everything you do not need.",
        "",
        "```json",
        figure_spec_schema_json(indent=1),
        "```",
        "",
        "Answer with the JSON object only.",
        "",
    ]
    return "\n".join(parts)
