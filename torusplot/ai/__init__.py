"""TorusPlot's optional AI layer. OFF by default; everything works offline without it.

THE HARD RULE: AI may only produce or modify a FigureSpec (plot type, layers, layout,
styling, and which *already computed* comparisons to annotate). AI never computes, edits
or invents statistics. This is enforced in code (see :mod:`torusplot.ai.describe` and
:mod:`torusplot.ai.feedback`), not left to prompts.

Features
--------
* **Describe it**: :func:`describe_to_spec` (plain language -> validated FigureSpec).
* **Outside AI instructions**: :func:`import_outside_instructions` (paste a chatbot's JSON;
  needs no provider) and :func:`outside_prompt_template`.
* **Match this figure**: :func:`match_figure`, :func:`extract_style`, :func:`style_to_spec`
  (style only, measured deterministically; a vision model is optional).
* **Digitize**: :func:`digitize_image`, :func:`to_table` (deterministic; approximate data).
  ``digitize_image`` is :func:`torusplot.ai.digitize.digitize`; it has another name here
  because ``torusplot.ai.digitize`` is the module.

Importing this package is cheap: names are resolved lazily (PEP 562), so no network
module, SDK, matplotlib or scikit-image is imported until a feature is used.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # settings
    "AISettings": "settings",
    "PROVIDERS": "settings",
    "RECOMMENDED_OLLAMA_MODELS": "settings",
    "load_settings": "settings",
    # providers
    "MockProvider": "providers.mock",
    "Provider": "providers.base",
    "ProviderError": "providers.base",
    "active_provider": "providers.base",
    "get_provider": "providers.base",
    # json
    "JSONExtractError": "jsonutil",
    "extract_json": "jsonutil",
    # describe
    "SpecCheckError": "describe",
    "SpecOutcome": "describe",
    "describe_to_spec": "describe",
    "looks_like_statistic": "describe",
    "sanitize_spec": "describe",
    # outside
    "clean_outside_json": "outside",
    "import_outside_instructions": "outside",
    "outside_prompt_template": "outside",
    # match
    "MatchOutcome": "match",
    "StyleExtraction": "match",
    "analyze_image": "match",
    "dominant_colors": "match",
    "extract_style": "match",
    "match_figure": "match",
    "style_to_spec": "match",
    # feedback
    "allowed_patch_paths": "feedback",
    "apply_style_patch": "feedback",
    "critique_patch": "feedback",
    "ssim_score": "feedback",
    # digitize
    "AxisCalibration": "digitize",
    "DigitizeError": "digitize",
    "DigitizeResult": "digitize",
    "calibration_from_plot_area": "digitize",
    "candidate_colors": "digitize",
    "detect_plot_area": "digitize",
    "digitize_image": "digitize:digitize",
    "pixel_to_data": "digitize",
    "to_table": "digitize",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, _, attribute = module.partition(":")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), attribute or name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))

