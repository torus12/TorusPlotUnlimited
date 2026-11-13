"""JSON-schema export of the FigureSpec, for AI features and outside chatbots."""

from __future__ import annotations

import json
from typing import Any

from .figure import PLOT_TYPES, CompositionSpec, FigureSpec, default_layers


def figure_spec_schema() -> dict[str, Any]:
    """The full JSON Schema (draft 2020-12) of :class:`FigureSpec`."""
    return FigureSpec.model_json_schema()


def composition_spec_schema() -> dict[str, Any]:
    return CompositionSpec.model_json_schema()


def figure_spec_schema_json(indent: int | None = 1) -> str:
    return json.dumps(figure_spec_schema(), indent=indent, sort_keys=False)


def plot_type_guide() -> str:
    """Short text catalogue of plot types, their tables and default layers (for prompts/docs)."""
    lines = []
    for key, info in PLOT_TYPES.items():
        layers = ", ".join(lay.type for lay in default_layers(key))
        lines.append(
            f"- {key}: {info['label']}. {info['description']} "
            f"Tables: {', '.join(info['tables'])}. Default layers: {layers}."
        )
    return "\n".join(lines)


def minimal_example() -> dict[str, Any]:
    """A small valid FigureSpec (used in prompts and docs)."""
    return {
        "plot_type": "bar",
        "data": {"table": "My table"},
        "layers": [{"type": "mean_bar"}, {"type": "sem_errorbar"}, {"type": "data_points_jitter"}],
        "title": "Particle size by condition",
        "axes": {"y": {"label": "Hydrodynamic diameter (nm)"}},
        "style": {"preset": "Torus Signature", "palette": "Okabe-Ito"},
        "layout": {"width_mm": 89, "height_mm": 70},
    }
