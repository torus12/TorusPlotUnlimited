"""Methods-paragraph generator.

Builds a paste-ready paragraph for a paper's "Statistical analysis" section from the
StatResults behind a figure. Every statement comes from what was actually run.
"""

from __future__ import annotations

from collections.abc import Iterable

from .. import __version__
from ..spec.figure import FigureSpec
from ..stats.result import StatResult

_ERROR_TEXT = {
    "sem_errorbar": "mean \u00b1 SEM",
    "sd_errorbar": "mean \u00b1 SD",
    "ci95_errorbar": "mean with 95% confidence interval",
    "range_errorbar": "mean with range",
    "iqr_errorbar": "median with interquartile range",
}


def data_presentation_sentence(spec: FigureSpec | None) -> str:
    """'Data are presented as mean ± SEM with individual data points.' from the figure's layers."""
    if spec is None:
        return ""
    types = [lay.type for lay in spec.effective_layers()]
    summary = next((_ERROR_TEXT[t] for t in types if t in _ERROR_TEXT), None)
    if summary is None:
        if "boxplot" in types:
            summary = "median, interquartile range (box) and whiskers extending to the most extreme values within 1.5 \u00d7 IQR"
        elif "median_dash" in types or "median_bar" in types:
            summary = "median"
        elif any(t in types for t in ("mean_bar", "mean_dash", "mean_dot", "mean_line")):
            summary = "mean"
    points = any(t.startswith("data_points") for t in types)
    if spec.plot_type == "superplot":
        return (
            "Data are shown as SuperPlots: small symbols are individual measurements (technical replicates), "
            "large symbols are the mean of each independent biological replicate, and summary statistics "
            f"({summary or 'mean'}) and tests were computed on the replicate means."
        )
    if spec.plot_type == "survival":
        return "Survival is shown as Kaplan-Meier curves; tick marks indicate censored subjects."
    if summary and points:
        return f"Data are presented as {summary}, with individual data points shown."
    if summary:
        return f"Data are presented as {summary}."
    if points:
        return "Individual data points are shown."
    return ""


_COMMON_FIRST_WORDS = {
    "one-way", "two-way", "three-way", "unpaired", "paired", "repeated", "repeated-measures", "mixed", "simple",
    "multiple", "nonlinear", "linear", "chi-square", "log-rank", "descriptive", "ordinary", "one-sample",
    "nested", "survival", "correlation", "normality", "equal-variance",
}


def _fallback_sentence(res: StatResult) -> str:
    name = res.test_name.strip()
    first = name.split(" ", 1)[0]
    if first.lower() in _COMMON_FIRST_WORDS:
        name = name[:1].lower() + name[1:]
    text = f"Data were analyzed using {name}"
    if res.correction and res.comparisons:
        text += f" followed by {res.correction} multiple-comparisons testing"
    return text + "."


def methods_paragraph(
    results: Iterable[StatResult],
    spec: FigureSpec | None = None,
    *,
    include_software: bool = True,
) -> str:
    """Compose the statistics paragraph for the given results (and optionally their figure)."""
    results = [r for r in results if r is not None]
    sentences: list[str] = []
    first = data_presentation_sentence(spec)
    if first:
        sentences.append(first)
    seen: set[str] = set()
    for res in results:
        sentence = (res.methods_text or "").strip() or _fallback_sentence(res)
        if not sentence.endswith("."):
            sentence += "."
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    alphas = sorted({r.alpha for r in results if r.p_value is not None or r.comparisons})
    if alphas:
        if len(alphas) == 1:
            sentences.append(f"P < {alphas[0]:g} was considered statistically significant.")
        else:
            sentences.append("Significance thresholds (alpha) are given with each test.")
    exclusions = [n for r in results for n in r.data_notes if "excluded" in n.lower()]
    for note in dict.fromkeys(exclusions):
        sentences.append(note if note.endswith(".") else note + ".")
    if any(r.approximate_data for r in results):
        sentences.append(
            "Note: at least one analysis used values digitized from a published image; these are "
            "approximate and reported for reference only."
        )
    n_notes = _n_sentence(results)
    if n_notes:
        sentences.append(n_notes)
    if include_software and results:
        sentences.append(
            f"Analyses and graphs were produced with TorusPlot {__version__} (Torus Washington LLC), "
            "whose statistics engine is built on SciPy and statsmodels."
        )
    return " ".join(sentences)


def _n_sentence(results: list[StatResult]) -> str:
    for res in results:
        if res.n:
            parts = ", ".join(f"{name}: n = {n}" for name, n in res.n.items())
            if len(parts) <= 220:
                return f"Sample sizes: {parts}."
            sizes = sorted(set(res.n.values()))
            return f"Sample sizes ranged from n = {sizes[0]} to n = {sizes[-1]} per group."
    return ""


def significance_legend(star_style: str = "four") -> str:
    """Text for a figure legend explaining the stars."""
    if star_style == "three":
        return "ns, not significant; *P < 0.05; **P < 0.01; ***P < 0.001."
    return "ns, not significant; *P < 0.05; **P < 0.01; ***P < 0.001; ****P < 0.0001."
