#!/usr/bin/env python
"""Write docs/USER_GUIDE.md, running every worked example on synthetic data.

    python tools/build_user_guide.py           # (re)write the guide
    python tools/build_user_guide.py --check   # exit 1 if the committed guide is out of date

The prose of the guide lives in this script. Every number in the guide is produced here by
calling TorusPlot's own statistics engine (``run_analysis``) on the bundled synthetic sample
datasets, or on small synthetic tables built below with fixed random seeds. Nothing is typed
by hand, so the guide cannot drift from what the app computes: ``tests/test_docs.py`` runs
this script with ``--check``.

Result IDs are random, so they are never printed (a fixed placeholder is shown instead).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "USER_GUIDE.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from torusplot.data import (  # noqa: E402
    KIND_DESCRIPTIONS,
    KIND_LABELS,
    TABLE_KINDS,
    ColumnTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    TableError,
    build_table,
    table_from_meta,
)
from torusplot.data.samples import SAMPLES, load_sample, sample_frame  # noqa: E402
from torusplot.export import (  # noqa: E402
    data_presentation_sentence,
    methods_paragraph,
    significance_legend,
)
from torusplot.plot import (  # noqa: E402
    CVD_LABELS,
    PALETTES,
    comparison_label,
    get_palette,
    get_preset,
    palette_names,
    palette_report,
    preset_names,
    summary_text,
)
from torusplot.plot.fonts import BUNDLED_FAMILIES  # noqa: E402
from torusplot.spec import PLOT_TYPES, FigureSpec, default_layers  # noqa: E402
from torusplot.stats import (  # noqa: E402
    CATEGORIES,
    AnalysisInfo,
    StatResult,
    StatsError,
    analyses_for,
    format_p,
    load_all,
    p_to_stars,
    run_analysis,
)
from torusplot.stats._utils import fmt, p_phrase  # noqa: E402
from torusplot.stats.nonlinear import MODELS as CURVE_MODELS  # noqa: E402
from torusplot.stats.power import power_anova, power_ttest  # noqa: E402
from torusplot.stats.recommend import recommend_test  # noqa: E402

ID_PLACEHOLDER = "sr_xxxxxxxx"
BADGES = {"pass": "PASS ", "warn": "CHECK", "fail": "FAIL ", "info": "INFO "}


# =============================================================================================
# Synthetic example tables (fixed seeds; none of this is real experimental data)
# =============================================================================================
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def table_repeated() -> DataTable:
    """Column table, rows = animals: tumour volume in 8 animals on four days."""
    rng = _rng(101)
    base = rng.normal(150.0, 20.0, 8)
    data = {}
    for day, growth, noise in [("Day 0", 0.0, 6.0), ("Day 7", 45.0, 9.0), ("Day 14", 110.0, 16.0), ("Day 21", 190.0, 30.0)]:
        data[day] = np.round(base + growth * rng.normal(1.0, 0.10, 8) + rng.normal(0.0, noise, 8), 0)
    return ColumnTable.from_dict("Tumour volume over time (mm3)", data)


def table_scores() -> DataTable:
    """Column table, rows = animals: an ordinal 0-10 histology score under three treatments."""
    rng = _rng(102)
    base = rng.integers(4, 8, 10)
    data = {
        "Vehicle": np.clip(base + rng.integers(0, 3, 10), 0, 10),
        "Low dose": np.clip(base - rng.integers(0, 3, 10), 0, 10),
        "High dose": np.clip(base - rng.integers(1, 5, 10), 0, 10),
    }
    return ColumnTable.from_dict("Histology score (0-10)", {k: v.astype(float) for k, v in data.items()})


def table_mixed() -> DataTable:
    """Grouped table: rows = days, column groups = treatments, each sub-column = one animal."""
    rng = _rng(103)
    days = [("Day 0", 0.0), ("Day 7", 1.0), ("Day 14", 2.0)]
    cells: dict[str, dict[str, list[float]]] = {day: {} for day, _ in days}
    for group, slope in [("Vehicle", 60.0), ("Treated", 22.0)]:
        base = rng.normal(150.0, 15.0, 5)
        gain = rng.normal(1.0, 0.12, 5)
        for day, step in days:
            cells[day][group] = [round(float(b + slope * step * g + rng.normal(0.0, 7.0)), 0) for b, g in zip(base, gain)]
    return GroupedTable.from_cells("Tumour volume by treatment (mm3)", cells, row_label="Day", column_label="Treatment")


def table_rm_twoway() -> DataTable:
    """Multiple-variables table: 8 subjects, each measured under 2 treatments x 3 time points."""
    rng = _rng(104)
    rows = []
    for s in range(1, 9):
        base = rng.normal(72.0, 6.0)
        for treatment, effect in [("Placebo", 0.0), ("Drug", -1.0)]:
            for time, step in [("0 h", 0.0), ("1 h", 6.0), ("2 h", 9.0)]:
                rows.append((f"S{s}", treatment, time, round(float(base + effect * step + rng.normal(0.0, 2.5)), 1)))
    frame = pd.DataFrame(rows, columns=["Subject", "Treatment", "Time", "Heart rate (bpm)"])
    return MultipleVariablesTable("Heart rate crossover", frame)


def table_factorial() -> DataTable:
    """Multiple-variables table: Treatment x Sex x Age, 4 animals per cell."""
    rng = _rng(105)
    rows = []
    for treatment, t_eff in [("Vehicle", 0.0), ("Drug", 2.4)]:
        for sex, s_eff in [("Female", 0.0), ("Male", 0.9)]:
            for age, a_eff in [("Young", 0.0), ("Old", -0.6)]:
                extra = 1.2 if (treatment == "Drug" and age == "Old") else 0.0
                for _ in range(4):
                    rows.append((treatment, sex, age, round(float(rng.normal(10.0 + t_eff + s_eff + a_eff - extra, 1.0)), 2)))
    return MultipleVariablesTable("Response by treatment, sex and age",
                                  pd.DataFrame(rows, columns=["Treatment", "Sex", "Age", "Response"]))


def table_ancova() -> DataTable:
    """Multiple-variables table: final tumour volume in three groups with a baseline covariate."""
    rng = _rng(106)
    rows = []
    for group, effect in [("Vehicle", 0.0), ("Low dose", -45.0), ("High dose", -95.0)]:
        for _ in range(8):
            baseline = float(rng.normal(150.0, 22.0))
            final = 60.0 + 1.5 * baseline + effect + float(rng.normal(0.0, 18.0))
            rows.append((group, round(baseline, 0), round(final, 0)))
    return MultipleVariablesTable("Final volume with baseline",
                                  pd.DataFrame(rows, columns=["Group", "Baseline (mm3)", "Final (mm3)"]))


def table_matrix() -> DataTable:
    """Multiple-variables table: four properties measured on 24 nanoparticle batches."""
    rng = _rng(107)
    size = rng.normal(105.0, 14.0, 24)
    pdi = 0.08 + 0.0022 * (size - 80.0) + rng.normal(0.0, 0.022, 24)
    zeta = rng.normal(-12.0, 3.0, 24)
    encapsulation = 96.0 - 0.28 * (size - 80.0) + rng.normal(0.0, 3.2, 24)
    frame = pd.DataFrame({
        "Size (nm)": np.round(size, 1),
        "PDI": np.round(pdi, 3),
        "Zeta (mV)": np.round(zeta, 1),
        "Encapsulation (%)": np.round(encapsulation, 1),
    })
    return MultipleVariablesTable("Batch properties", frame)


def table_linear() -> DataTable:
    """XY table: drug released (%) over time for two formulations, 3 replicates each."""
    rng = _rng(108)
    time = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    data: dict[str, Any] = {"Time (h)": time}
    for label, intercept, slope in [("Formulation A", 4.0, 8.2), ("Formulation B", 5.0, 5.1)]:
        for rep in range(3):
            data[f"{label}:{rep + 1}"] = np.round(intercept + slope * time + rng.normal(0.0, 2.6, time.size), 1)
    return build_table("xy", pd.DataFrame(data), "Drug released (%)", replicates=3)


def table_regression() -> DataTable:
    """Multiple-variables table: encapsulation explained by two numeric and one categorical predictor."""
    rng = _rng(109)
    rows = []
    for i in range(30):
        mixer = "Microfluidic" if i % 2 == 0 else "Bulk"
        ratio = float(rng.uniform(5.0, 30.0))
        flow = float(rng.uniform(1.0, 12.0))
        value = 55.0 + 0.9 * ratio + 0.15 * flow + (9.0 if mixer == "Microfluidic" else 0.0) + float(rng.normal(0.0, 4.0))
        rows.append((round(value, 1), round(ratio, 1), round(flow, 1), mixer))
    frame = pd.DataFrame(rows, columns=["Encapsulation (%)", "Lipid ratio", "Flow rate (mL/min)", "Mixer"])
    return MultipleVariablesTable("Encapsulation by process settings", frame)


def table_trend() -> DataTable:
    """Contingency table: responders in four ordered dose groups."""
    frame = pd.DataFrame({
        "Dose": ["0 mg/kg", "1 mg/kg", "3 mg/kg", "10 mg/kg"],
        "Responder": [3, 6, 10, 15],
        "Non-responder": [17, 14, 10, 5],
    })
    return table_from_meta(frame, {"kind": "contingency", "name": "Response by dose"})


def table_outlier() -> DataTable:
    """Column table with one planted, grossly high value in the second group."""
    rng = _rng(110)
    control = np.round(rng.normal(50.0, 4.0, 9), 1)
    treated = np.round(rng.normal(62.0, 4.5, 9), 1)
    treated[6] = round(float(treated.mean() + 9 * treated.std(ddof=1)), 1)
    return ColumnTable.from_dict("Signal (a.u.)", {"Control": control, "Treated": treated})


def _sample(name: str) -> Callable[[], DataTable]:
    return lambda: load_sample(name)


#: key -> (builder, how the guide describes the data)
TABLES: dict[str, tuple[Callable[[], DataTable], str]] = {
    "size": (_sample("nanoparticle_size"), "the bundled sample **Nanoparticle diameter (nm)** (`nanoparticle_size`; synthetic; 4 synthesis methods, 8 batches each)"),
    "paired": (_sample("paired"), "the bundled sample **Tumour volume (mm3)** (`paired`; synthetic; 10 animals measured before and after treatment, one animal per row)"),
    "dose": (_sample("dose_response"), "the bundled sample **Viability (%)** (`dose_response`; synthetic; 9 doses, two formulations, 3 replicates per dose)"),
    "uptake": (_sample("cell_uptake"), "the bundled sample **Uptake (a.u.)** (`cell_uptake`; synthetic; 3 time points x 3 formulations, 4 replicates per cell)"),
    "cells": (_sample("cell_uptake_cells"), "the bundled sample **Per-cell uptake** (`cell_uptake_cells`; synthetic; 2 treatments, 3 independent experiments, 20 cells in each)"),
    "survival": (_sample("survival"), "the bundled sample **Survival study** (`survival`; synthetic; 3 groups of 12 animals, time in days, event = 1 or 0)"),
    "contingency": (_sample("contingency"), "the bundled sample **Response by treatment** (`contingency`; synthetic; responders and non-responders in a treated and a control group)"),
    "repeated": (table_repeated, "a small **synthetic** Column table built by the guide script (seed 101): tumour volume (mm3) of 8 animals on Day 0, 7, 14 and 21, one animal per row"),
    "scores": (table_scores, "a small **synthetic** Column table built by the guide script (seed 102): a 0-10 histology score for 10 animals under Vehicle, Low dose and High dose, one animal per row"),
    "mixed": (table_mixed, "a small **synthetic** Grouped table built by the guide script (seed 103): tumour volume (mm3) on Day 0, 7 and 14 (rows) for Vehicle and Treated (column groups), 5 animals per group, each sub-column one animal"),
    "rm2": (table_rm_twoway, "a small **synthetic** Multiple variables table built by the guide script (seed 104): heart rate of 8 subjects, each measured under Placebo and Drug at 0, 1 and 2 h (48 rows)"),
    "factorial": (table_factorial, "a small **synthetic** Multiple variables table built by the guide script (seed 105): a response for Treatment x Sex x Age, 4 animals per combination (32 rows)"),
    "ancova": (table_ancova, "a small **synthetic** Multiple variables table built by the guide script (seed 106): final tumour volume and baseline volume for 8 animals in each of three groups"),
    "matrix": (table_matrix, "a small **synthetic** Multiple variables table built by the guide script (seed 107): size, PDI, zeta potential and encapsulation of 24 batches"),
    "linear": (table_linear, "a small **synthetic** XY table built by the guide script (seed 108): drug released (%) at 6 time points for two formulations, 3 replicates each"),
    "regression": (table_regression, "a small **synthetic** Multiple variables table built by the guide script (seed 109): encapsulation (%) of 30 batches with lipid ratio, flow rate and mixer type"),
    "trend": (table_trend, "a small **synthetic** Contingency table built by the guide script: responders and non-responders in four ordered dose groups of 20 animals"),
    "outlier": (table_outlier, "a small **synthetic** Column table built by the guide script (seed 110): two groups of 9 values with one grossly high value planted in the Treated group"),
}

_TABLE_CACHE: dict[str, DataTable] = {}


def get_table(key: str) -> DataTable:
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = TABLES[key][0]()
    return _TABLE_CACHE[key]


# =============================================================================================
# Formatting helpers (all numbers go through the engine's own ``fmt`` / ``format_p``)
# =============================================================================================
def num(value: Any, digits: int = 4) -> str:
    """The engine's own number format; large values are written out instead of in e-notation."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fmt(value, digits)
    if np.isfinite(v) and 1e4 <= abs(v) < 1e12:
        return f"{v:.0f}"
    return fmt(value, digits)


def md(text: str) -> str:
    """Engine prose as Markdown text: asterisks must not start emphasis."""
    return text.replace("*", "\\*")


def stars(p: float | None) -> str:
    return f"`{p_to_stars(p)}`"


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return out


def stat_text(result: StatResult) -> str:
    if result.statistic is None:
        return ""
    name = result.statistic_name or "Statistic"
    if result.df is not None and result.df2 is not None:
        return f"{name}({num(result.df)}, {num(result.df2)}) = {num(result.statistic)}"
    if result.df is not None:
        return f"{name}({num(result.df)}) = {num(result.statistic)}"
    return f"{name} = {num(result.statistic)}"


def ci_text(low: Any, high: Any, level: float = 0.95) -> str:
    if low is None and high is None:
        return ""
    lower = "(not available)" if low is None else num(low)
    upper = "(not reached)" if high is None else num(high)
    return f"{level * 100:g}% CI {lower} to {upper}"


def effect_text(es: Any) -> str:
    text = f"{es.name} = {num(es.value)}"
    extras = [t for t in (ci_text(es.ci_low, es.ci_high, es.ci_level), es.magnitude or "") if t]
    if extras:
        text += " (" + "; ".join(extras) + ")"
    if es.target:
        text += f" [{es.target}]"
    return text


def param_text(par: Any) -> str:
    text = f"{par.name} = {num(par.value)}"
    extras = []
    interval = ci_text(par.ci_low, par.ci_high, par.ci_level)
    if interval:
        extras.append(interval)
    if par.p_value is not None:
        extras.append(p_phrase(par.p_value))
    if extras:
        text += " (" + "; ".join(extras) + ")"
    return text


def run(table_key: str, analysis: str, **params: Any) -> StatResult:
    """Run one analysis quietly; Python-level warnings are not part of the guide."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_analysis(get_table(table_key), analysis, **params)


@dataclass
class Show:
    """What the result block of one example prints."""

    estimates: int = 3
    estimate_names: tuple[str, ...] = ()
    effects: int = 1
    comparisons: int = 6
    anova: bool = True
    checks: int = 4
    descriptives: bool = False
    interpretation: bool = True
    extra_table: Callable[[StatResult], list[str]] | None = None


def _labelled(label: str, lines: list[str]) -> list[str]:
    return [f"{(label if i == 0 else ''):<14}{line}" for i, line in enumerate(lines)]


def _pick_estimates(result: StatResult, show: Show) -> list[Any]:
    if not show.estimate_names:
        return list(result.parameters[: show.estimates])
    missing = [n for n in show.estimate_names if result.parameter(n) is None]
    if missing:
        raise SystemExit(f"{result.analysis}: estimate(s) not found: {missing}. "
                         f"Available: {[p.name for p in result.parameters]}")
    return [result.parameter(n) for n in show.estimate_names]


def result_block(result: StatResult, show: Show) -> list[str]:
    """Compact, stable excerpt of a StatResult as Markdown (never prints the random ID)."""
    box: list[str] = []
    box += _labelled("Test", [result.test_name])
    box += _labelled("Result ID", [f"{ID_PLACEHOLDER}  (random; yours will differ)"])
    if result.statistic is not None:
        box += _labelled("Statistic", [stat_text(result)])
    if result.p_value is not None:
        verdict = "significant" if result.significant else "not significant"
        info = load_all().get(result.analysis)
        tailed = info is not None and any(p.name == "alternative" for p in info.params)  # tests with a tails option
        tail = f"{result.alternative}; " if tailed else ""
        box += _labelled("P value", [f"{format_p(result.p_value)}  {p_to_stars(result.p_value)}  "
                                     f"({tail}{verdict} at alpha = {result.alpha:g})"])
    if result.p_adjusted is not None:
        box += _labelled("Adjusted P", [format_p(result.p_adjusted)])
    if result.effect_sizes and show.effects:
        first = primary_effect(result)
        effects = [first] + [e for e in result.effect_sizes if e is not first]
        box += _labelled("Effect size", [effect_text(e) for e in effects[: show.effects]])
    estimates = _pick_estimates(result, show)
    if estimates:
        box += _labelled("Estimates", [param_text(p) for p in estimates])
    if result.n:
        sizes = "; ".join(f"{name}: {n}" for name, n in result.n.items())
        if len(sizes) > 100:
            values = sorted(set(result.n.values()))
            span = f"{values[0]}" if len(values) == 1 else f"{values[0]} to {values[-1]}"
            sizes = f"{len(result.n)} groups or cells, n = {span} each"
        total = f"  (total {result.n_total})" if result.n_total is not None else ""
        box += _labelled("Sample size", [sizes + total])
    checked = [a for a in result.assumptions if a.status != "info"]
    if any(a.name.startswith("Normality of residuals") for a in checked):  # show the model-level checks first
        per_group = [a for a in checked if a.name.startswith("Normality within group")]
        checked = [a for a in checked if a not in per_group] + per_group
    stated = len(result.assumptions) - len(checked)
    lines = []
    for a in checked[: show.checks]:
        line = f"{BADGES[a.status]} {a.name}" + (f" [{a.target}]" if a.target else "")
        lines.append(line + (f", P = {format_p(a.p_value)}" if a.p_value is not None else ""))
    if len(checked) > show.checks:
        lines.append(f"... and {len(checked) - show.checks} more check(s) listed in the app")
    if stated:
        lines.append(f"plus {stated} design assumption(s) stated in the app (they cannot be checked from the numbers)")
    if lines and show.checks:
        box += _labelled("Checks", lines)

    out = ["```text", *box, "```", ""]
    if show.descriptives and result.descriptives:
        out += descriptives_table(result) + [""]
    if show.extra_table is not None:
        out += show.extra_table(result) + [""]
    if show.anova and result.anova_table:
        out += anova_markdown(result) + [""]
    if show.comparisons and result.comparisons:
        out += comparisons_markdown(result, show.comparisons) + [""]
    if show.interpretation and result.interpretation:
        out += ["What TorusPlot writes under the result:", "", "> " + md(result.interpretation), ""]
    if result.warnings:
        shown = result.warnings[:2]
        out += ["Warnings shown with this result:", ""] + [f"- {md(w)}" for w in shown]
        if len(result.warnings) > len(shown):
            out.append(f"- ... and {len(result.warnings) - len(shown)} more warning(s) of the same kind.")
        out.append("")
    return out


def descriptives_table(result: StatResult) -> list[str]:
    keys = [("group", "Group"), ("n", "n"), ("mean", "Mean"), ("sd", "SD"), ("sem", "SEM"),
            ("ci_low", "CI low"), ("ci_high", "CI high"), ("median", "Median"), ("q1", "Q1"), ("q3", "Q3")]
    present = [(k, label) for k, label in keys if any(k in row for row in result.descriptives)]
    rows = [[row.get(k) if k in {"group", "n"} else num(row.get(k)) for k, _ in present]
            for row in result.descriptives]
    return md_table([label for _, label in present], rows)


def _cell(value: Any) -> str:
    return "" if value is None else num(value)


def anova_markdown(result: StatResult) -> list[str]:
    corrected = any("p_gg" in row for row in result.anova_table)
    headers = ["Source", "SS", "df", "MS", "F", "P (sphericity assumed)" if corrected else "P"]
    if corrected:
        headers += ["P (Greenhouse-Geisser)", "P (Huynh-Feldt)"]
    rows = []
    for row in result.anova_table:
        line = [row.get("source", ""), _cell(row.get("ss")), _cell(row.get("df")), _cell(row.get("ms")),
                _cell(row.get("f")), format_p(row["p"]) if row.get("p") is not None else ""]
        if corrected:
            line += [format_p(row[k]) if row.get(k) is not None else "" for k in ("p_gg", "p_hf")]
        rows.append(line)
    return ["ANOVA table:", "", *md_table(headers, rows)]


def comparisons_markdown(result: StatResult, limit: int) -> list[str]:
    title = "Multiple comparisons" + (f" ({result.correction})" if result.correction else "")
    comps = result.comparisons[:limit]
    has_est = any(c.estimate is not None for c in comps)
    has_ci = any(c.ci_low is not None and c.ci_high is not None for c in comps)
    headers = ["Comparison"] + (["Difference"] if has_est else []) + (["CI of difference"] if has_ci else [])
    headers += ["P", "Adjusted P", "Stars"]
    rows = []
    for comp in comps:
        row = [f"{comp.group_a} vs {comp.group_b}" + (f" [{comp.label}]" if comp.label else "")]
        if has_est:
            row.append(_cell(comp.estimate))
        if has_ci:
            row.append(f"{num(comp.ci_low)} to {num(comp.ci_high)}" if comp.ci_low is not None and comp.ci_high is not None else "")
        row += [format_p(comp.p_value) if comp.p_value is not None else "",
                format_p(comp.p_adjusted) if comp.p_adjusted is not None else "",
                stars(comp.p_for_display)]
        rows.append(row)
    out = [title + ":", "", *md_table(headers, rows)]
    notes = []
    if result.comparison_family:
        notes.append("Family of comparisons: " + result.comparison_family.rstrip(". ") + ".")
    if len(result.comparisons) > limit:
        notes.append(f"First {limit} of {len(result.comparisons)} comparisons shown; the app lists all of them.")
    if notes:
        out += ["", " ".join(notes)]
    return out


def context(result: StatResult) -> dict[str, Any]:
    """Values that the hand-written 'How to read it' text may quote (always from the result)."""
    ctx: dict[str, Any] = {
        "stat": stat_text(result) or "(no test statistic)",
        "p": p_phrase(result.p_value),
        "verdict": "statistically significant" if result.significant else "not statistically significant",
        "alpha": f"{result.alpha:g}",
        "ncomp": len(result.comparisons),
        "nsig": sum(1 for c in result.comparisons if c.significant),
        "correction": result.correction or "",
        "ntotal": result.n_total,
        "test": result.test_name,
    }
    if result.effect_sizes:
        es = primary_effect(result)
        ctx["es"] = f"{es.name} = {num(es.value)}"
        ctx["es_ci"] = ci_text(es.ci_low, es.ci_high, es.ci_level) or "no confidence interval"
        ctx["es_size"] = es.magnitude or ""
    for i, par in enumerate(result.parameters):
        ctx[f"est{i}"] = param_text(par)
    return ctx


def primary_effect(result: StatResult) -> Any:
    """The effect size that belongs to the headline statistic (factorial results name their headline term)."""
    term = (result.extra or {}).get("headline_term")
    for es in result.effect_sizes:
        if term is not None and es.target == term:
            return es
    return result.effect_sizes[0]


def quote_line(result: StatResult, with_effect: bool, estimates: tuple[int | str, ...]) -> str:
    """The numbers a results sentence needs, assembled from the result."""
    parts = []
    if result.statistic is not None and result.statistic_name != "odds ratio":
        parts.append(stat_text(result))
    if result.p_value is not None:
        parts.append(p_phrase(result.p_value))
    for item in estimates:
        par = result.parameters[item] if isinstance(item, int) else result.parameter(item)
        if par is None:
            raise SystemExit(f"{result.analysis}: no estimate '{item}' to quote.")
        parts.append(param_text(par).replace(f"; {p_phrase(par.p_value)}", "") if par.p_value is not None else param_text(par))
    if with_effect and result.effect_sizes and primary_effect(result).value is not None:
        es = primary_effect(result)
        interval = ci_text(es.ci_low, es.ci_high, es.ci_level)
        parts.append(f"{es.name} = {num(es.value)}" + (f" ({interval})" if interval else ""))
    return "; ".join(parts)


# =============================================================================================
# Worked examples: one per registered analysis
# =============================================================================================
@dataclass
class Example:
    key: str
    table: str
    use_when: str
    read: list[str]
    dont: str
    params: dict[str, Any] = field(default_factory=dict)
    show: Show = field(default_factory=Show)
    more: Callable[[StatResult], dict[str, Any]] | None = None
    #: guard for the hand-written prose: the build stops if the outcome flips
    expect_significant: bool | None = None
    #: extra Markdown placed after "What not to conclude" (generated, never typed numbers)
    appendix: Callable[[], list[str]] | None = None
    #: what "Numbers to quote" lists besides the statistic and P value
    quote_effect: bool = True
    quote_estimates: tuple[int | str, ...] = ()


def _desc(result: StatResult, group: str, key: str) -> str:
    for row in result.descriptives:
        if row.get("group") == group:
            return num(row.get(key))
    raise SystemExit(f"{result.analysis}: no descriptive row for group '{group}'.")


def _normality_table(result: StatResult) -> list[str]:
    rows = [[r["group"], r["test"], f"{r['statistic_name']} = {num(r['statistic'])}", format_p(r["p_value"]),
             "pass" if r["passed"] else "FAIL"] for r in result.extra["table"]]
    return md_table(["Group", "Test", "Statistic", "P", "Outcome"], rows)


def _normality_more(result: StatResult) -> dict[str, Any]:
    rows = result.extra["table"]
    return {"n_rows": len(rows), "n_fail": sum(1 for r in rows if not r["passed"]),
            "n_each": sorted(set(result.n.values()))[0]}


def _variance_table(result: StatResult) -> list[str]:
    rows = []
    for r in result.extra["table"]:
        df = f"({num(r['df1'])}, {num(r['df2'])})" if r.get("df2") is not None else f"({num(r['df1'])})"
        rows.append([r["test"], f"{r['statistic_name']}{df} = {num(r['statistic'])}", format_p(r["p_value"])])
    return md_table(["Test", "Statistic", "P"], rows)


def _matrix_table(result: StatResult) -> list[str]:
    rows = [[f"{p['a']} vs {p['b']}", num(p["r"]), f"{num(p['ci_low'])} to {num(p['ci_high'])}", p["n"],
             format_p(p["p"]), format_p(p["p_adjusted"]), stars(p["p_adjusted"])] for p in result.extra["pairs"]]
    return md_table(["Pair", "r", "95% CI", "n", "P", "Adjusted P", "Stars"], rows)


def _matrix_more(result: StatResult) -> dict[str, Any]:
    pairs = result.extra["pairs"]
    return {"npairs": len(pairs), "nsig_pairs": sum(1 for p in pairs if p["significant"]),
            "nraw": sum(1 for p in pairs if p["p"] < result.alpha)}


def _grubbs_table(result: StatResult) -> list[str]:
    rows = [[r["group"], r["n"], num(r["most_extreme_value"]), num(r["g"]), num(r["g_critical"]),
             format_p(r["p_value"]), "flagged" if r["outlier"] else "not flagged"] for r in result.extra["table"]]
    return md_table(["Group", "n", "Most extreme value", "G", "Critical G", "P", "Outcome"], rows)


def _rout_table(result: StatResult) -> list[str]:
    rows = [[r["group"], r["n"], num(r["mean"]), num(r["robust_location"]), num(r["sd"]), num(r["rsdr"]),
             ", ".join(num(v) for v in r["flagged_values"]) or "none"] for r in result.extra["table"]]
    return md_table(["Group", "n", "Mean", "Robust centre", "SD", "Robust SD (RSDR)", "Flagged values"], rows)


def _nested_table(result: StatResult) -> list[str]:
    rows = [[r["group"], r["replicate"], r["n_technical"], num(r["mean"]), num(r["sd_technical"])]
            for r in result.extra["replicate_means"]]
    return md_table(["Group", "Biological replicate", "Cells measured", "Replicate mean", "SD of the cells"], rows)


def _nested_more(result: StatResult) -> dict[str, Any]:
    reps = result.extra["replicate_means"]
    return {"n_cells": sum(r["n_technical"] for r in reps), "n_reps": len(reps)}


def _rm_more(result: StatResult) -> dict[str, Any]:
    sph = result.extra["sphericity"]
    sph = sph[0] if isinstance(sph, list) else sph
    return {"mauchly_p": p_phrase(sph["p"]), "eps_gg": num(sph["eps_gg"]), "eps_hf": num(sph["eps_hf"])}


def _rank_more(result: StatResult) -> dict[str, Any]:
    return {"p_method": "an exact" if result.extra.get("exact") else "an approximate (normal approximation)"}


def _survival_table(result: StatResult) -> list[str]:
    rows = []
    for row in result.descriptives:
        median = "not reached" if row.get("median") is None else num(row["median"])
        low = "(not available)" if row.get("median_ci_low") is None else num(row["median_ci_low"])
        high = "(not reached)" if row.get("median_ci_high") is None else num(row["median_ci_high"])
        rows.append([row["group"], row["n"], row["events"], row["censored"], median, f"{low} to {high}"])
    out = md_table(["Group", "n", "Events", "Censored", "Median survival", "95% CI of the median"], rows)
    ratios = [[f"{p['group_a']} vs {p['group_b']}", num(p["hr"]), f"{num(p['hr_ci_low'])} to {num(p['hr_ci_high'])}"]
              for p in result.extra["pairwise"]]
    return out + ["", "Hazard ratios (first group relative to second):", "",
                  *md_table(["Pair", "Hazard ratio", "95% CI"], ratios)]


def _model_list() -> list[str]:
    rows = [[f"`{key}`", model.label, f"`{model.equation}`"] for key, model in CURVE_MODELS.items()]
    rows.append(["`user_defined`", "Your own equation", "typed under *Equation (user-defined model)*, for example "
                 "`Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)`"])
    return ["**Models you can choose.**", "", *md_table(["*Model*", "What it is", "Equation"], rows), "",
            "Other options: *Fixed parameters* holds a parameter constant (for example `Bottom=0`). *Weighting* "
            "changes how much each point counts. *Confidence interval method* = `profile` gives intervals that may "
            "be asymmetric and are more reliable when a parameter is poorly determined."]


def _welch_more(result: StatResult) -> dict[str, Any]:
    a, b = list(result.n)
    return {"sd_a": _desc(result, a, "sd"), "sd_b": _desc(result, b, "sd"), "a": a, "b": b}


EXAMPLES: list[Example] = [
    # ------------------------------------------------------------------ Descriptive statistics
    Example(
        key="descriptives", table="size",
        use_when="Always, before any test. You want the count, centre and spread of each group, and a first "
                 "look for typing errors or odd values.",
        show=Show(descriptives=True, estimates=0, checks=0),
        read=[
            "**n** is the number of values used. Empty cells are not counted.",
            "**Mean** and **SD** describe the data. The SD (standard deviation) is the typical distance of a value from the mean.",
            "**SEM** and the **95% CI** describe how precisely the mean is known. They shrink as n grows. The SD does not.",
            "**Median, Q1, Q3** are the middle value and the quartiles. Prefer them when the data are skewed.",
            "There is no P value. Nothing was compared.",
        ],
        dont="Do not decide that two groups differ because their confidence intervals do not overlap, or that they "
             "are the same because the intervals overlap. Run a test.",
    ),
    # ----------------------------------------------------------------------- Assumption checks
    Example(
        key="normality", table="size",
        use_when="You want to know whether a t test or ANOVA is reasonable. Those tests assume a normal (Gaussian) "
                 "distribution: the symmetric, bell-shaped curve.",
        show=Show(extra_table=_normality_table, estimates=0, checks=0),
        more=_normality_more,
        read=[
            "One row per group and test. A P value below alpha (0.05) is evidence that the group is **not** normal. "
            "The app then shows a FAIL badge.",
            "In this example {n_fail} of {n_rows} checks fail.",
            "Shapiro-Wilk is the most useful single test for small samples. D'Agostino-Pearson needs at least 8 values. "
            "The Kolmogorov-Smirnov P value always uses the Lilliefors correction; it is never reported below 0.001 "
            "or above 0.99.",
            "Each group has only {n_each} values. With so few values these tests rarely detect non-normality. "
            "A pass means \"no deviation detected\". It does not mean \"normal\".",
        ],
        dont="Do not treat a pass as proof of normality. Do not try one test after another until one passes. "
             "Also look at the Q-Q plot (page **3 Graph**, graph type *Q-Q plot (normality)*).",
    ),
    Example(
        key="equal_variance", table="size",
        use_when="Before an unpaired t test or an ordinary ANOVA. Both assume the groups have similar spread "
                 "(variance is the SD squared).",
        show=Show(extra_table=_variance_table, checks=0),
        expect_significant=False,
        read=[
            "The headline result is the Brown-Forsythe test: {stat}, {p}. Here it is {verdict}.",
            "Levene's and Bartlett's tests are listed for comparison. Bartlett's test is very sensitive to non-normal data.",
            "{est0}. A ratio above about 2 to 3 deserves attention even when the test passes, because the test has "
            "little power with small groups.",
        ],
        dont="A non-significant result does not prove that the variances are equal. If in doubt, use Welch's t test "
             "or Welch's ANOVA. They do not need this assumption.",
    ),
    # ---------------------------------------------------------------------- Compare two groups
    Example(
        key="ttest_unpaired", table="size", params={"group_a": "Bulk mixing", "group_b": "Sonication"},
        use_when="Two independent groups: different animals, batches or wells in each group. Both groups roughly "
                 "normal, with similar spread.",
        expect_significant=True,
        read=[
            "{est0}. This is first group minus second group. The 95% CI is the range of true differences that is "
            "compatible with the data.",
            "{stat}, {p}: the difference is {verdict} at alpha = {alpha}. The P value is the probability of a "
            "difference at least this large if the two population means were equal.",
            "{es} ({es_ci}). This is the difference in units of the pooled SD. The interval is wide because n is small.",
            "Checks: normality of each group and equal variance. Both pass here.",
        ],
        dont="Do not use it for before/after data from the same animals; use the paired t test. A significant P "
             "value does not say that the difference is large or important. Read the difference and its CI.",
    ),
    Example(
        key="ttest_welch", table="size", params={"group_a": "Bulk mixing", "group_b": "Microfluidic"},
        use_when="Two independent groups when the SDs or the group sizes differ. Many statisticians use it as the "
                 "routine default, because it does not assume equal variances.",
        more=_welch_more, expect_significant=True,
        read=[
            "Read it like the unpaired t test. The degrees of freedom are not a whole number ({stat}). "
            "Welch's method lowers them when the SDs differ.",
            "Here the SDs are {sd_a} ({a}) and {sd_b} ({b}).",
            "{est0}.",
            "{es} ({es_ci}). Cohen's d is standardised by the pooled SD of the two groups, as for the unpaired t test.",
        ],
        dont="Welch's test still assumes roughly normal data in each group. It does not protect against outliers "
             "or strong skew.",
    ),
    Example(
        key="ttest_paired", table="paired",
        use_when="Two measurements on the same subjects: before and after, left and right, treated and untreated "
                 "wells from the same donor. One subject per row.",
        expect_significant=True,
        read=[
            "{est0}. This is the mean of the within-animal differences (first column minus second column).",
            "{stat}, {p}: {verdict}.",
            "{est1}. Pairing helps when this correlation is positive. It was effective here.",
            "{es}. For paired data this is the mean difference divided by the SD of the differences (d_z). "
            "It is not comparable with Cohen's d from an unpaired test.",
            "The normality check is on the differences, not on each column.",
        ],
        dont="Do not use it when the rows are not the same subject. The test says that the average change is not "
             "zero. It does not say that every animal responded.",
    ),
    Example(
        key="ttest_one_sample", table="paired", params={"group": "Before", "hypothetical_value": 400.0},
        use_when="One group compared with a fixed value that you chose in advance: 0 (no change), 1 (a ratio of "
                 "one), 100 (percent of control), or a specification limit.",
        expect_significant=False,
        read=[
            "{est0}.",
            "{est1}. This is the mean minus the hypothetical value.",
            "{stat}, {p}: {verdict}. The 95% CI of the mean contains the hypothetical value; that always goes "
            "together with a two-sided P above 0.05.",
            "Not significant does not mean \"equal to 400\". The data are compatible with every value inside the CI.",
        ],
        dont="Do not pick the hypothetical value after looking at the data. If every control value was set to "
             "exactly 100%, the control group has no spread; compare the treated values with 100 using this test, "
             "not an unpaired test.",
    ),
    Example(
        key="mann_whitney", table="size", params={"group_a": "Bulk mixing", "group_b": "Sonication"},
        use_when="Two independent groups when the data are skewed, contain outliers, are scores or ranks, or you do "
                 "not want to assume a normal distribution.",
        more=_rank_more, expect_significant=True,
        read=[
            "The test ranks all values together and asks whether one group tends to have the larger values. "
            "{stat}, {p}, with {p_method} P value: {verdict}.",
            "{est0}. The Hodges-Lehmann estimate is the median of all pairwise differences between the groups. "
            "It is the shift that the test is about.",
            "{est1}. This is a different number and has no confidence interval.",
            "{es} ({es_ci}). A value of +1 means every value in the first group is larger than every value in "
            "the second; 0 means no tendency.",
        ],
        dont="It is not automatically a test of medians. It compares medians only if the two distributions have "
             "the same shape. With 3 values per group the smallest possible two-sided P value is 0.10, so the "
             "test cannot reach 0.05.",
    ),
    Example(
        key="wilcoxon_signed_rank", table="paired",
        use_when="Paired data when the within-pair differences are not normal, or are scores.",
        more=_rank_more, expect_significant=True,
        read=[
            "W is the sum of the signed ranks of the differences. {stat}, {p}, with {p_method} P value: {verdict}.",
            "{est0}. This is the typical within-pair difference (Hodges-Lehmann estimate).",
            "{es}. It is exactly 1 here because every animal changed in the same direction. Its confidence "
            "interval comes from an approximation that collapses to a single point at 1. Do not read that "
            "interval as certainty.",
        ],
        dont="The test assumes that the differences are roughly symmetric around their median. With fewer than 6 "
             "pairs a two-sided P value below 0.05 is impossible.",
    ),
    Example(
        key="wilcoxon_one_sample", table="paired", params={"group": "Before", "hypothetical_value": 400.0},
        use_when="One group compared with a fixed value, when the values are not normal or are scores.",
        more=_rank_more, expect_significant=False,
        read=[
            "The test ranks the distances of the values from the hypothetical value. {stat}, {p}, with "
            "{p_method} P value: {verdict}.",
            "{est0}. The pseudo-median equals the median when the distribution is symmetric.",
            "The warning explains why the P value is approximate here: some distances are tied.",
        ],
        dont="Like every rank test it needs enough values. With 5 or fewer values a two-sided P below 0.05 is "
             "impossible.",
    ),
    # ------------------------------------------------------------- Compare three or more groups
    Example(
        key="anova_oneway", table="size",
        use_when="Three or more independent groups defined by one factor. Roughly normal data with similar spread.",
        expect_significant=True,
        show=Show(estimates=0),
        read=[
            "The ANOVA table splits the variation. *Between groups* is the variation of the group means. "
            "*Within groups* is the scatter inside the groups. F is the ratio of the two mean squares.",
            "{stat}, {p}: at least one group mean differs from the others. This overall test does not say which.",
            "{es} ({es_ci}). This is the share of the total variation that is explained by the grouping.",
            "Multiple comparisons ({correction}): one row per pair. *Difference* is first group minus second "
            "group. Use the **Adjusted P** column. Here {nsig} of {ncomp} pairs are significant.",
            "If you only want to compare each group with a control, set *Post-hoc comparisons* to `dunnett` and "
            "choose the *Control group*. That makes fewer comparisons and has more power.",
        ],
        dont="Do not run a series of t tests instead. With 4 groups there are 6 pairs, and the chance of at least "
             "one false positive rises well above 5%. Do not use this analysis when the same subjects were "
             "measured in every group.",
    ),
    Example(
        key="welch_anova", table="size",
        use_when="Three or more independent groups whose SDs or sample sizes differ.",
        expect_significant=True,
        show=Show(estimates=0, anova=False),
        read=[
            "{stat}, {p}. The second number of degrees of freedom is not a whole number; that is normal for this test.",
            "The Games-Howell comparisons use each pair's own SDs. Here {nsig} of {ncomp} pairs are significant.",
            "{es}. The effect size is descriptive only. No confidence interval is given, because the usual "
            "interval assumes equal variances (see the warning).",
        ],
        dont="Welch's ANOVA still assumes roughly normal data within each group, and every group needs at least "
             "2 values and some spread.",
    ),
    Example(
        key="brown_forsythe_anova", table="size",
        use_when="The same situation as Welch's ANOVA: three or more independent groups with unequal spread.",
        expect_significant=True,
        show=Show(estimates=0, anova=False, comparisons=0),
        read=[
            "{stat}, {p}. F* uses the ordinary between-group sum of squares with adjusted degrees of freedom.",
            "Welch's ANOVA and the Brown-Forsythe ANOVA answer the same question and usually agree. Choose one "
            "before you see the results, and report which one you used.",
            "The follow-up comparisons are Games-Howell, as for Welch's ANOVA ({nsig} of {ncomp} significant here).",
        ],
        dont="Do not confuse it with the Brown-Forsythe test for equal variances (see *Equal variance tests*). "
             "The names are the same; the questions are different.",
    ),
    Example(
        key="kruskal_wallis", table="size",
        use_when="Three or more independent groups when the data are skewed, contain outliers, or are scores.",
        expect_significant=True,
        show=Show(estimates=0),
        read=[
            "All values are ranked together. {stat}, {p}: values in at least one group tend to be higher or lower.",
            "The P value comes from a chi-square approximation. It is rough when every group has 5 or fewer values. "
            "TorusPlot does not compute an exact Kruskal-Wallis P value.",
            "Dunn's comparisons: *Difference* is the difference in mean ranks, so there is no confidence interval. "
            "Here {nsig} of {ncomp} pairs are significant ({correction}).",
            "Compare with the one-way ANOVA example on the same data: the rank test with a Bonferroni correction "
            "finds fewer pairs. Rank tests have less power when the data really are normal.",
        ],
        dont="It is a test of medians only if all groups have the same distribution shape. Do not describe the "
             "groups by mean and SD next to a rank test without saying so.",
    ),
    # ------------------------------------------------------------------------ Factorial designs
    Example(
        key="anova_twoway", table="uptake", params={"posthoc": "tukey"},
        use_when="Two grouping factors, and every value comes from a different subject or independent sample. "
                 "Example: 3 time points x 3 formulations, separate wells for every combination.",
        expect_significant=True,
        show=Show(effects=0, comparisons=6),
        read=[
            "Read the **interaction** row first (*Time x Formulation*). A significant interaction means that the "
            "effect of one factor depends on the level of the other. The main effects then describe averages and "
            "can mislead.",
            "The statistic in the result box is the interaction: {stat}, {p}.",
            "Each other row of the ANOVA table has its own F and P value.",
            "Multiple comparisons: with *What to compare* left at `simple_b_within_a`, the formulations are compared "
            "within each time point. Each time point is its own Tukey family. Here {nsig} of {ncomp} comparisons "
            "are significant.",
        ],
        dont="Do not use it when the same animals or wells were measured at every time point; use the "
             "mixed-design ANOVA. With unequal group sizes, Type II and Type III sums of squares differ; "
             "TorusPlot uses Type III unless you change it.",
    ),
    Example(
        key="anova_threeway", table="factorial",
        params={"value": "Response", "factor_a": "Treatment", "factor_b": "Sex", "factor_c": "Age"},
        use_when="Three grouping factors, every value from a different subject. Needs replicates in every "
                 "combination of the three factors.",
        expect_significant=False,
        show=Show(effects=0),
        read=[
            "The table has seven effects: three main effects, three two-factor interactions and one three-factor "
            "interaction.",
            "Start at the bottom. The statistic in the result box is the three-factor interaction: {stat}, {p}, "
            "{verdict}.",
            "When the highest interaction is not significant, read the two-factor interactions, then the main effects.",
            "No follow-up comparisons are made unless you choose a method under *Multiple comparisons*.",
        ],
        dont="Seven tests are made on one data set. A single P value just under 0.05 among them is weak evidence. "
             "Plot the data before you interpret a three-way table.",
    ),
    Example(
        key="ancova", table="ancova",
        params={"value": "Final (mm3)", "factor": "Group", "covariates": ["Baseline (mm3)"], "posthoc": "tukey"},
        use_when="You compare groups, and a second numeric variable (a covariate, such as the baseline value, age "
                 "or body weight) also influences the measurement.",
        expect_significant=True,
        read=[
            "The group effect is tested after adjusting for the covariate: {stat}, {p}.",
            "{est0}. This is the common slope: the change in the outcome per unit of the covariate.",
            "Check the *Equal slopes* badge. ANCOVA assumes the covariate has the same slope in every group. "
            "It passes here.",
            "The interpretation lists the **adjusted means**: the group means estimated at the average covariate value.",
            "Comparisons of adjusted means with the Tukey-Kramer method are approximate. The label says so.",
        ],
        dont="The covariate must not be affected by the treatment. Measure it before treatment starts. ANCOVA "
             "cannot repair groups that differ strongly at baseline because allocation was not random.",
    ),
    # ----------------------------------------------------------------------- Repeated measures
    Example(
        key="friedman", table="scores",
        use_when="Three or more conditions measured on the same subjects, when the values are scores or clearly "
                 "not normal. One subject per row.",
        expect_significant=True,
        show=Show(estimates=3),
        read=[
            "Values are ranked within each subject. {stat}, {p}: at least one condition tends to give higher or "
            "lower values.",
            "{es}. Kendall's W runs from 0 (subjects do not agree on the order of the conditions) to 1 (complete agreement).",
            "Dunn-type comparisons use the same within-subject ranks. *Difference* is a difference in mean ranks. "
            "Here {nsig} of {ncomp} pairs are significant.",
        ],
        dont="A subject with a missing value in any condition is left out completely. The chi-square "
             "approximation is rough with few subjects.",
    ),
    Example(
        key="anova_rm_oneway", table="repeated",
        use_when="Three or more conditions or time points measured in the same subjects. One subject per row, one "
                 "column per condition. Roughly normal data.",
        more=_rm_more, expect_significant=True,
        show=Show(effects=1, checks=3),
        read=[
            "The degrees of freedom in {stat} are not whole numbers. With more than two conditions TorusPlot applies "
            "the Greenhouse-Geisser correction by default. {p}.",
            "Sphericity means that the differences between every pair of conditions vary equally. Mauchly's test "
            "gives {mauchly_p} here. The Greenhouse-Geisser epsilon is {eps_gg} (1 = no violation).",
            "The ANOVA table shows three P values for the condition effect: sphericity assumed, "
            "Greenhouse-Geisser and Huynh-Feldt. Report the corrected one.",
            "Follow-up tests are paired t tests with Holm adjustment. {nsig} of {ncomp} are significant.",
        ],
        dont="Do not use it when different animals were measured at each time point. A subject with one missing "
             "value is dropped from the whole analysis; TorusPlot has no mixed-effects model that keeps "
             "incomplete subjects.",
    ),
    Example(
        key="anova_mixed", table="mixed", params={"posthoc": "holm"},
        use_when="Separate groups of subjects, each subject measured repeatedly. Example: two treatment groups "
                 "followed over time.",
        expect_significant=True,
        show=Show(effects=0, checks=3),
        read=[
            "Three effects: the group factor (between subjects), the repeated factor (within subjects) and their "
            "interaction.",
            "Read the interaction first. It is the statistic in the result box: {stat}, {p}. A significant "
            "interaction means that the change over the repeated factor differs between the groups.",
            "The group effect is tested against the variation between subjects. Its degrees of freedom come from "
            "the number of animals, not from the number of measurements.",
            "With *What to compare* left at `between_per_level`, the groups are compared at each time point "
            "({nsig} of {ncomp} significant after Holm adjustment).",
        ],
        dont="The layout matters. In a Grouped table, rows are the repeated levels and each sub-column must be "
             "the same subject in every row. If the sub-columns are not the same subjects, use the two-way ANOVA.",
    ),
    Example(
        key="anova_rm_twoway", table="rm2",
        params={"value": "Heart rate (bpm)", "subject": "Subject", "within_a": "Treatment", "within_b": "Time"},
        use_when="Every subject is measured at every combination of two factors. Example: a crossover study in "
                 "which each subject receives both treatments and is measured at every time point.",
        expect_significant=True,
        show=Show(effects=0, checks=3),
        read=[
            "Each effect is tested against its own interaction with subjects, so the table has three error rows.",
            "The statistic in the result box is the interaction: {stat}, {p}.",
            "Effects with more than two levels get the Greenhouse-Geisser correction. Compare the P columns.",
            "This analysis does not produce multiple comparisons.",
        ],
        dont="It needs complete data: one value for every subject at every combination. It only accepts a "
             "Multiple variables (long) table.",
    ),
    # ----------------------------------------------------------------------------- Correlation
    Example(
        key="correlation", table="paired",
        use_when="Two measurements on each subject, and you ask how strongly they go together. You do not need to "
                 "call one of them X and the other Y.",
        expect_significant=True,
        show=Show(effects=2, estimates=0),
        read=[
            "{es} ({es_ci}). r runs from -1 (perfect decreasing line) through 0 (no straight-line relation) to +1.",
            "{stat}, {p}: tests whether the true correlation is zero.",
            "r squared is the share of variance that the two variables have in common.",
            "Set *Method* to `spearman` for skewed data, outliers, or a curved but steadily rising relation.",
        ],
        dont="Correlation does not show cause. It does not show agreement either: here the before and after "
             "volumes correlate strongly although every animal changed. Do not correlate repeated readings of "
             "the same sample as if they were independent.",
    ),
    Example(
        key="correlation_matrix", table="matrix", params={"adjust": "holm"},
        use_when="Several numeric variables measured on the same samples, and you want every pairwise correlation.",
        more=_matrix_more,
        show=Show(effects=0, estimates=0, extra_table=_matrix_table, checks=0),
        read=[
            "One row per pair of variables. Each pair uses the rows where both values are present.",
            "With {npairs} pairs, some would look significant by chance. *Adjusted P* corrects for that "
            "(Holm method here). {nsig_pairs} pairs remain significant; {nraw} had an unadjusted P below 0.05.",
            "Draw the matrix as a heat map on page **3 Graph** (graph type *Heat map*).",
        ],
        dont="Do not report only the pairs that came out significant without saying how many pairs were tested.",
    ),
    # ------------------------------------------------------------------------------ Regression
    Example(
        key="linear_regression", table="linear", params={"series": "Formulation A"},
        use_when="X is a quantity you set or measured (time, dose, concentration), Y is the response, and the "
                 "relation looks like a straight line.",
        expect_significant=True,
        read=[
            "{est0}. The slope is the change in Y per unit of X.",
            "{est1}. The Y-intercept is the predicted Y at X = 0.",
            "{stat}, {p}: tests whether the slope is zero.",
            "{es}: the share of the variation in Y that the line explains.",
            "Checks: normal residuals, constant scatter, lack of fit (needs replicates) and the runs test "
            "(points systematically above or below the line). All pass here.",
        ],
        dont="Do not extrapolate beyond the X range of the data. A high R squared does not prove that a straight "
             "line is the right model; look at the residual plot. By default every replicate is its own point. "
             "That is right only for independent samples; tick *Fit the mean of replicates* for repeat readings "
             "of one sample.",
    ),
    Example(
        key="compare_slopes", table="linear",
        use_when="Two or more straight lines in one XY table, and you ask whether they are equally steep.",
        expect_significant=True,
        read=[
            "The first test compares the slopes: {stat}, {p}, {verdict}.",
            "{est0}; {est1}.",
            "If the slopes do not differ, the second test (elevations) asks whether one line lies higher. If the "
            "slopes differ, as here, comparing elevations is not meaningful.",
        ],
        dont="The test assumes similar scatter around each line, and that each line really is straight.",
    ),
    Example(
        key="multiple_regression", table="regression",
        params={"y": "Encapsulation (%)", "predictors": ["Lipid ratio", "Flow rate (mL/min)", "Mixer"]},
        use_when="One numeric outcome and several possible explanatory variables measured on the same rows.",
        expect_significant=True,
        show=Show(estimates=4, effects=2, checks=3),
        read=[
            "Each coefficient is the expected change in the outcome for a one-unit increase in that predictor, "
            "with the other predictors held constant.",
            "A text column is treated as categories. *Mixer[Microfluidic]* is the difference from the reference "
            "level, which is the first level in alphabetical order (here *Bulk*).",
            "{stat}, {p}: tests the model as a whole. {es}.",
            "Each coefficient has its own CI and P value. A coefficient whose CI contains 0 is not distinguishable "
            "from no effect in this data set.",
            "The VIF check looks for predictors that largely repeat each other (multicollinearity).",
        ],
        dont="Coefficients describe association, not necessarily cause. Keep the number of predictors small "
             "compared with the number of rows; TorusPlot warns below about 10 rows per predictor. Do not add and "
             "remove predictors until something is significant.",
    ),
    # -------------------------------------------------------------------- Nonlinear regression
    Example(
        key="nonlinear_regression", table="dose",
        params={"series": "Free drug", "model": "dose_response_4pl", "potency_name": "IC50"},
        use_when="The relation between X and Y is a curve with a known shape: dose-response, enzyme kinetics, "
                 "exponential decay, binding. You want the parameters of the curve, such as the IC50.",
        show=Show(estimate_names=("IC50", "HillSlope", "Top", "Bottom"), effects=1, checks=5),
        read=[
            "There is no overall P value. The result is the set of best-fit parameters with their confidence intervals.",
            "{ic50}. The IC50 (or EC50) is the concentration that gives a response halfway between Bottom and Top. "
            "Its CI is not symmetric because the fit works on the logarithm of the IC50.",
            "{hill}. A negative Hill slope means the response falls as the dose rises.",
            "{top}; {bottom}. These are the plateaus of the curve.",
            "The CHECK badge and the warning say that the Bottom plateau is poorly defined by these doses. "
            "If the data are normalised, fix it under *Fixed parameters* (for example `Bottom=0`).",
            "{es} describes how close the points are to the curve. It is not a test and cannot choose between models.",
        ],
        more=lambda r: {"ic50": param_text(r.parameter("IC50")), "hill": param_text(r.parameter("HillSlope")),
                        "top": param_text(r.parameter("Top")), "bottom": param_text(r.parameter("Bottom"))},
        appendix=_model_list,
        dont="Do not report an IC50 that lies outside the tested dose range, or whose CI spans orders of magnitude. "
             "Do not fit means when the replicates are independent samples. Use the model with `_logx` in its "
             "name only when your X values are already log10(concentration).",
    ),
    Example(
        key="compare_fits", table="dose", params={"series": "Free drug"},
        use_when="You hesitate between two models for the same data, for example a dose-response curve with a "
                 "standard slope (3 parameters) or a variable slope (4 parameters).",
        expect_significant=False,
        show=Show(effects=2, estimates=0),
        read=[
            "The question is whether the more complex model fits enough better to justify its extra parameter. "
            "{stat}, {p}: {verdict}.",
            "The simpler model is kept unless P is below alpha. Here the simpler model is preferred.",
            "The interpretation also gives the AICc comparison. AICc is the only valid comparison when neither "
            "model is a special case of the other (not nested).",
        ],
        dont="A statistical preference is not proof. Check that the preferred curve is scientifically sensible. "
             "Both models must be fitted to exactly the same points.",
    ),
    Example(
        key="compare_curve_parameter", table="dose", params={"parameter": "EC50"},
        use_when="Two or more curves in one XY table, and you ask whether one parameter (usually the EC50 or IC50) "
                 "differs between them.",
        expect_significant=True,
        show=Show(estimate_names=("LogEC50 (Free drug)", "LogEC50 (Nanoformulation)",
                                  "EC50 ratio (Nanoformulation / Free drug)")),
        read=[
            "TorusPlot fits the curves twice: once with a separate EC50 for each data set, once with one shared "
            "EC50. {stat}, {p}: forcing a shared EC50 makes the fit significantly worse, so the EC50 values differ.",
            "The comparison is made on the log scale (LogEC50), which is the appropriate scale for potencies.",
            "{ratio}. A ratio below 1 means the second curve is more potent.",
        ],
        more=lambda r: {"ratio": param_text(r.parameter("EC50 ratio (Nanoformulation / Free drug)"))},
        dont="The other parameters (Top, Bottom, Hill slope) stay separate for each curve. If the curves have "
             "different plateaus, their EC50 values refer to different response ranges. The test assumes the "
             "same scatter in every data set.",
    ),
    # -------------------------------------------------------------------------------- Survival
    Example(
        key="survival", table="survival",
        use_when="Time-to-event data: time until death, relapse, tumour reaching a size limit, device failure. "
                 "Some subjects have not had the event when the study ends (they are censored).",
        expect_significant=True,
        show=Show(effects=0, estimates=0, extra_table=_survival_table),
        read=[
            "**Median survival** is the time at which the Kaplan-Meier curve first reaches 50%. *Not reached* "
            "means the curve or its confidence band never dropped to 50%.",
            "{stat}, {p}: the log-rank test compares the whole curves of all groups at once.",
            "With three or more groups every pair is compared and the P values are adjusted ({correction}). "
            "Significant pairs here: {nsig} of {ncomp}.",
            "The result also holds a hazard ratio for each pair. A hazard ratio above 1 means the event happens "
            "at a higher rate in the first-named group.",
            "A censored subject (event = 0) counts as at risk until its last time and is then removed. It is "
            "neither a death nor a survivor to the end.",
        ],
        dont="Do not delete censored animals, and do not code them as events. The hazard ratio assumes that the "
             "ratio of event rates is constant over time; if the curves cross, a single hazard ratio misleads. "
             "TorusPlot does not fit Cox regression models.",
    ),
    # ---------------------------------------------------------------------- Contingency tables
    Example(
        key="chi_square", table="contingency",
        use_when="Counts of subjects in categories, and you ask whether two categorical variables are related. "
                 "Suited to large counts and to tables larger than 2 x 2.",
        expect_significant=True,
        show=Show(effects=2, estimate_names=("Odds ratio", "Relative risk", "Risk difference")),
        read=[
            "{stat}, {p}. For a 2 x 2 table TorusPlot applies Yates' continuity correction by default, which "
            "makes the test more cautious.",
            "{es}. For a 2 x 2 table phi equals Cramér's V; 0 means no association, 1 a perfect one.",
            "For 2 x 2 tables the result adds the risk in each row, the relative risk, the odds ratio, the risk "
            "difference and the number needed to treat, each with a CI.",
            "Orientation: the first row is the exposed or treated group, the first column is \"outcome present\". "
            "All ratios are first row relative to second row.",
        ],
        dont="Enter counts of subjects, never percentages or means. Each subject must be counted once. For "
             "before/after data on the same subjects you need McNemar's test, which TorusPlot does not have.",
    ),
    Example(
        key="fisher_exact", table="contingency",
        use_when="A 2 x 2 table of counts, especially with small numbers. The P value is exact for any sample size.",
        expect_significant=True,
        show=Show(effects=1, estimate_names=("Odds ratio", "Odds ratio (conditional MLE)", "Relative risk")),
        read=[
            "{p}. This is the exact two-sided P value. The line *Statistic* shows the sample odds ratio, not a "
            "test statistic.",
            "Two odds ratios are listed. *Odds ratio* is the simple cross-product ratio with the Woolf (logit) CI. "
            "*Odds ratio (conditional MLE)* comes with an exact CI and is the one that belongs to Fisher's test.",
            "Compare with the chi-square example on the same table: the P values are close because the counts "
            "are not small.",
        ],
        dont="The same rules as for the chi-square test: counts only, each subject once. For tables larger than "
             "2 x 2 only the two-sided test exists, and very large tables fall back to a simulated P value "
             "(the result says so).",
    ),
    Example(
        key="chi_square_trend", table="trend",
        use_when="A yes/no outcome in three or more **ordered** groups, such as increasing doses. You ask whether "
                 "the proportion rises or falls steadily.",
        expect_significant=True,
        read=[
            "{stat}, {p}: tests a straight-line trend in the proportion across the ordered groups. It has 1 degree "
            "of freedom, so it is more powerful than the ordinary chi-square test when a trend exists.",
            "{est0}. The slope is the change in the proportion per step from one group to the next.",
            "The groups are scored 1, 2, 3, ... in the order of the table. The test treats the steps as equally large.",
        ],
        dont="Put the groups in their natural order before you run it. In the app the scores are always "
             "1, 2, 3, ...; the real spacing of the doses (here 0, 1, 3, 10 mg/kg) is not used.",
    ),
    # -------------------------------------------------------------------------------- Outliers
    Example(
        key="outliers_grubbs", table="outlier",
        use_when="You suspect a single extreme value in a group that is otherwise roughly normal.",
        show=Show(extra_table=_grubbs_table, estimates=0, checks=0),
        read=[
            "G is the distance of the most extreme value from the group mean, in SDs. The value is flagged when "
            "G exceeds the critical value for that sample size.",
            "Each group is tested on its own. One value is flagged in the Treated group; nothing in Control.",
            "The value is only flagged. The table is unchanged.",
        ],
        dont="Grubbs' test looks for one outlier. Two outliers can hide each other, because both inflate the SD. "
             "A flagged value is not automatically a mistake.",
    ),
    Example(
        key="outliers_rout", table="outlier",
        use_when="You suspect one or several extreme values. ROUT estimates the centre and spread robustly, so "
                 "outliers do not hide each other.",
        show=Show(extra_table=_rout_table, estimates=0, checks=0),
        read=[
            "The robust centre and the robust SD (RSDR) are barely moved by the extreme value. Compare them with "
            "the ordinary mean and SD of the Treated group.",
            "Q is the maximum share of flagged values you accept to be false alarms. The default is 1%. A larger "
            "Q flags more readily.",
            "The value is only flagged. The table is unchanged.",
        ],
        dont="Do not exclude a value because the exclusion makes a P value smaller. Exclude only with a "
             "scientific reason, decide the rule in advance, and report it. See section 5.",
    ),
    # ----------------------------------------------------------------- Replicate-aware (nested)
    Example(
        key="nested_analysis", table="cells",
        use_when="You measured many cells, wells or fields (technical replicates) inside a few independent "
                 "experiments, animals or donors (biological replicates).",
        more=_nested_more, expect_significant=True,
        show=Show(extra_table=_nested_table, effects=1, estimates=1),
        read=[
            "The {n_cells} cell measurements are first reduced to {n_reps} means, one per experiment and "
            "treatment. The test runs on those means.",
            "The sample size is the number of experiments (3 per group), not the number of cells.",
            "The same experiment labels occur in both groups, so TorusPlot pairs the experiments: {test}. "
            "{stat}, {p}.",
            "{est0}.",
            "The warning shows the P value that the wrong analysis (every cell as an independent sample) would "
            "give. It is far smaller. Do not report it.",
        ],
        dont="Do not use the per-cell P value. With 3 replicates per group, normality cannot be judged and the "
             "confidence intervals are wide; that is the honest state of the evidence. If equal labels mean "
             "different animals in each group, set *Match replicates across groups* to `no`.",
    ),
]

#: key -> (estimates to list under "Numbers to quote", by position or name; whether to add the effect size)
QUOTES: dict[str, tuple[tuple[int | str, ...], bool]] = {
    "ttest_unpaired": ((0,), True),
    "ttest_welch": ((0,), True),
    "ttest_paired": ((0,), True),
    "ttest_one_sample": ((1,), True),
    "mann_whitney": ((0,), True),
    "wilcoxon_signed_rank": ((0,), False),
    "wilcoxon_one_sample": ((0,), True),
    "correlation_matrix": ((), False),
    "linear_regression": ((0,), True),
    "compare_slopes": ((0, 1), False),
    "nonlinear_regression": (("IC50", "HillSlope"), True),
    "compare_fits": ((), False),
    "compare_curve_parameter": (("EC50 ratio (Nanoformulation / Free drug)",), False),
    "survival": ((), False),
    "chi_square": (("Relative risk",), True),
    "fisher_exact": (("Odds ratio (conditional MLE)",), False),
    "chi_square_trend": ((0,), True),
    "nested_analysis": ((0,), True),
}
for _example in EXAMPLES:
    if _example.key in QUOTES:
        _example.quote_estimates, _example.quote_effect = QUOTES[_example.key]


#: example table key -> bundled sample name (tables that a reader can load in the app)
SAMPLE_OF = {"size": "nanoparticle_size", "paired": "paired", "dose": "dose_response", "uptake": "cell_uptake",
             "cells": "cell_uptake_cells", "survival": "survival", "contingency": "contingency"}


def _option_value(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def example_section(ex: Example, info: AnalysisInfo) -> list[str]:
    table = get_table(ex.table)
    if table.kind not in info.tables:
        raise SystemExit(f"{ex.key}: example table '{ex.table}' is a {table.kind} table, not valid for this analysis.")
    result = run(ex.table, ex.key, **ex.params)
    if ex.expect_significant is not None and bool(result.significant) != ex.expect_significant:
        raise SystemExit(
            f"{ex.key}: the example result changed (significant = {result.significant}), so the hand-written "
            "'How to read it' text in tools/build_user_guide.py no longer matches. Update the text."
        )
    ctx = context(result)
    if ex.more is not None:
        ctx.update(ex.more(result))
    try:
        bullets = [line.format_map(ctx) for line in ex.read]
    except KeyError as exc:
        raise SystemExit(f"{ex.key}: 'How to read it' text needs a value the result does not provide: {exc}") from exc

    labels = {p.name: p.label for p in info.params}
    options = [f"*{labels[name]}* = `{_option_value(value)}`" for name, value in ex.params.items()]
    kinds = ", ".join(KIND_LABELS[k] for k in info.tables)
    if ex.table in SAMPLE_OF:
        title = SAMPLES[SAMPLE_OF[ex.table]]["title"]
        step1 = f"Page **1 Data** > *Load a sample dataset* > choose *{title}* > **Load sample**."
    else:
        step1 = (f"Page **1 Data** > paste or import your own data as a **{table.kind_label}** table. "
                 "(The table of this example is built by the guide script; it is not in the sample list.)")
    step2 = f"Page **2 Analyze** > *Category*: {info.category} > *Analysis*: {info.label}"
    step2 += (" > set " + "; ".join(options) if options else " > leave the options at their defaults")
    step2 += f" > **Run {info.label}**."

    out = [f"#### {info.label}", "",
           f"Registry key: `{ex.key}`. Works on these table types: {kinds}. Method reference: {info.citation}.", "",
           f"**Use it when.** {ex.use_when}", "",
           "**In the app.**", "", f"1. {step1}", f"2. {step2}", "",
           f"**Example.** Data: {TABLES[ex.table][1]}. "
           + ("Options: " + "; ".join(options) + "." if options else "Options: all defaults."), ""]
    out += result_block(result, ex.show)
    out += ["**How to read it.**", ""] + [f"- {b}" for b in bullets] + [""]
    out += [f"**What not to conclude.** {ex.dont}", ""]
    if ex.appendix is not None:
        out += ex.appendix() + [""]
    out += ["**Report it like this.** TorusPlot writes this methods sentence for you:", "",
            "> " + md(result.methods_text.strip()), ""]
    quote = quote_line(result, ex.quote_effect, ex.quote_estimates)
    if quote:
        out += [f"Numbers to quote with it: {quote}. Always add the sample size (n) of every group.", ""]
    return out


# =============================================================================================
# Sections of the guide
# =============================================================================================
HEADER = """\
<!-- GENERATED FILE: do not edit by hand. Edit tools/build_user_guide.py and run
     `python tools/build_user_guide.py`. Every number below is real TorusPlot output. -->

# TorusPlot user guide

TorusPlot is a graphing and statistics program that runs on your own computer.
This guide shows how to use it, with one worked example for every analysis.

- You do not need to know how to code.
- Every result block in this guide was produced by TorusPlot itself, on **synthetic**
  (simulated) data. None of the example data are real measurements.
- Result IDs are random. This guide prints `sr_xxxxxxxx` in their place.
- For the formulas, assumptions and references behind each method see
  [STATS_METHODS.md](STATS_METHODS.md). For installation see the [README](../README.md).

![A multi-panel figure made with TorusPlot](images/gallery_composition.png)

## Contents

1. [Start here](#1-start-here)
2. [Choosing a table type](#2-choosing-a-table-type)
3. [Which test should I use?](#3-which-test-should-i-use)
4. [Worked examples: one per analysis](#4-worked-examples-one-per-analysis)
5. [Outliers, exclusions and replicates](#5-outliers-exclusions-and-replicates)
6. [Making figures](#6-making-figures)
7. [Styling and exporting](#7-styling-and-exporting)
8. [Optional AI features and the MCP server](#8-optional-ai-features-and-the-mcp-server)
9. [Troubleshooting and FAQ](#9-troubleshooting-and-faq)
10. [Glossary](#10-glossary)
"""


def section_start() -> list[str]:
    rec = recommend_test(get_table("size"))
    result = run("size", rec.analysis, **rec.params)
    sample_title = SAMPLES["nanoparticle_size"]["title"]
    return f"""\
## 1. Start here

Start TorusPlot with `run_torusplot.bat` (Windows) or `./run_torusplot.sh` (macOS, Linux).
Your web browser opens the app. It runs on your computer, not on the internet.

The left sidebar lists the pages under **Workflow**. Work from top to bottom.

| Step | Page | What you do there |
|---|---|---|
| 1 | **1 Data** | Load a sample, paste cells from Excel, or import a file. Choose the table type. |
| 2 | **2 Analyze** | Ask for a test recommendation or pick an analysis. Read the result. |
| 3 | **3 Graph** | Create a figure from the table. Add error bars, points and significance brackets. |
| 4 | **4 Style** | Pick a style preset, colours, fonts and the exact size in mm. |
| 5 | **5 Compose** | Combine several figures into one multi-panel figure with letters A, B, C. |
| 6 | **6 Export** | Download figure files, the statistics report and the methods paragraph. Save the project. |

Two more pages, **AI Assistant** and **Settings**, hold the optional AI features (section 8).

### 1.1 Data

![The Data page](images/app_data.png)

The page has two tabs: **Add data** and **Tables**.

On **Add data** there are three ways to bring numbers in:

- **Load a sample dataset.** Choose a *Sample* and click **Load sample**. The samples are synthetic.
- **Paste from Excel.** Copy cells in Excel or Google Sheets, click in *Pasted cells* and paste.
  Check *Column titles*, choose the *Table type*, type a *Table name* and click
  **Create table from pasted cells**.
- **Import a file.** Drop a CSV, TSV, TXT or Excel (.xlsx) file. For Excel pick the *Worksheet*.
  Choose the *Table type* and click **Create table from file**.

On **Tables** you see the grid of the selected *Table*. You can type into it.

- **Apply edits** saves what you typed.
- **Download CSV** saves the table as a file.
- **Make long-format copy** creates a *Multiple variables* copy with one row per value.
- **Delete table** removes the table from the project.
- **Restore excluded values** appears when you have excluded values (section 5).

### 1.2 Analyze

![Test recommendation with assumption badges](images/app_analyze.png)

1. Choose the *Table*.
2. Under *Not sure which test? Ask for a recommendation*, click **Recommend a test**.
   - Tick *Same subjects measured repeatedly (paired)* if each row (Column table) or each replicate
     column (Grouped table) is the same animal, sample or person.
   - With three or more groups you can choose a *Control group (optional)*.
3. Read the recommendation, its numbered reasons and the badges. Then click **Run: ...** to run it.
4. Or go to *Choose an analysis*: pick a *Category* and an *Analysis*, set the options, and click
   **Run** followed by the analysis name. Only analyses that are valid for the table type are listed.
   Every option has a tooltip.
5. Read the result under *Results for this table*.

![The results panel](images/app_results.png)

The results panel shows, from top to bottom:

- the test name and its result ID;
- a plain-English reading of the result;
- the statistic with degrees of freedom, the P value, and the main effect size;
- **Assumption checks** with badges: PASS, CHECK (look at it), FAIL, INFO (cannot be checked from the numbers);
- **Why this test?** with the method references;
- warnings;
- tables: *Multiple comparisons*, *ANOVA table*, *Estimates*, *Effect sizes*, *Descriptive statistics*;
- **Methods sentence**: a sentence for your paper.

At the bottom of the page, *Power and sample size calculator* plans an experiment without any data.

### 1.3 Graph

![Graph page with live preview](images/app_graph.png)

1. Open *New figure*. Choose the *Graph type*, type a *Figure name*, click **Create figure**.
   If the table already has a result with pairwise comparisons, its brackets are attached for you.
2. Change the figure in the four panels on the left. The *Live preview* updates at once.
   - **Data**: which groups, series or columns to show, and in which order.
   - **Elements**: *Error bars* and *Individual data points*.
   - **Statistics on the graph**: *Significance brackets from*, *Test summary text in a corner*,
     *Fitted curves to draw*.
   - **Titles and axes**: *Title*, axis titles, *Y scale*, *X scale*, *Y minimum*, *Y maximum*.

### 1.4 Style

![Style page](images/app_style.png)

Choose the *Figure* and a *Style preset*. Fine-tune under **Colours**, **Text**, **Size and lines**
and **Legend**. Open *Colour-blind preview* to check the figure. Details are in section 7.

### 1.5 Compose

![Compose page](images/app_compose.png)

Choose the figures under *Panels (in reading order)*, set *Columns*, *Width (mm)*, *Height (mm)* and
*Panel letters*, then click **Save composition**.

### 1.6 Export

![Export page](images/app_export.png)

On the tab **Figures and reports**: choose *What to export*, tick the formats, choose the
*Resolution (dpi) for TIFF/PNG*, and click the **Download** buttons. Below them are the
*Methods paragraph* and the buttons **Download report (Markdown)** and **Download report (CSV)**.

On the tab **Project file**: **Save project (.tplot)** keeps everything; *Open a project* loads it again.

### 1.7 A first run in two minutes

1. **1 Data**: choose the sample *{sample_title}* and click **Load sample**.
2. **2 Analyze**: click **Recommend a test**. For this sample TorusPlot answers:
   *Recommended: {rec.label}*. Click **Run: {rec.label}**.
   The result reads: {stat_text(result)}, {p_phrase(result.p_value)}.
3. **3 Graph**: click **Create figure**. The brackets come from the result you just computed.
4. **4 Style**: pick *Journal Single-Column (89 mm)*.
5. **6 Export**: download the PDF and the statistics report.
""".splitlines()


def _frame_table(frame: pd.DataFrame, rows: int, max_cols: int = 7) -> list[str]:
    frame = frame.head(rows)
    cols = list(frame.columns)
    cut = len(cols) > max_cols
    cols = cols[:max_cols]
    headers = [str(c) for c in cols] + (["..."] if cut else [])
    body = []
    for _, row in frame.iterrows():
        cells = [f"{row[c]:g}" if isinstance(row[c], int | float | np.integer | np.floating) else str(row[c]) for c in cols]
        body.append(cells + (["..."] if cut else []))
    return md_table(headers, body)


def section_tables() -> list[str]:
    out = """\
## 2. Choosing a table type

The table type tells TorusPlot how your experiment is organised. It decides which analyses and
graphs are offered. There are five types.

### 2.1 Which one do I need?

- One grouping variable (Control, Drug A, Drug B)? Use **Column**. Each column is a group.
- The same subjects measured twice or more (before/after, several time points)? Use **Column**,
  with one subject per row.
- Every point has an X value (dose, time, concentration)? Use **XY**.
- Two grouping variables (for example time point and treatment)? Use **Grouped**.
- Counts of subjects in categories (responders, non-responders)? Use **Contingency**.
- One row per subject or measurement, with a column for each variable? Use **Multiple variables**.
  You need it for survival data, for data with technical and biological replicates, for three
  grouping factors, for covariates, and for multiple regression.

When you paste data, TorusPlot suggests a table type. You can always change it.

### 2.2 What each type looks like

The pictures below show the first rows of the bundled synthetic samples.

""".splitlines()

    pictures = [
        ("column", "nanoparticle_size", 3, "Each column is one group. Rows are replicates. The columns may have "
         "different lengths. For paired or repeated measures, each row is one subject."),
        ("xy", "dose_response", 3, "The first column is X. The other columns are Y values. Replicates of one data "
         "set sit side by side (`Free drug:1`, `Free drug:2`, ...)."),
        ("grouped", "cell_uptake", 3, "The first column names the rows (first factor, here time). The column groups "
         "are the second factor (here formulation), each with replicate sub-columns."),
        ("multiple_variables", "cell_uptake_cells", 3, "One row per measurement. One column per variable. Text "
         "columns are categories; numeric columns are measurements. You choose the role of each column when you "
         "analyse or graph."),
        ("contingency", "contingency", 2, "Each cell is a **count of subjects**. Rows are the groups; put the "
         "treated or exposed group first. Columns are the outcomes; put \"outcome present\" first."),
    ]
    for kind, sample, rows, text in pictures:
        out += [f"**{KIND_LABELS[kind]}.** {KIND_DESCRIPTIONS[kind]}", "",
                *_frame_table(sample_frame(sample), rows), "", text, ""]
    survival = sample_frame("survival")
    out += ["A second **Multiple variables** example, for survival data (time, event, group):", "",
            *_frame_table(survival, 3), "",
            "`Event` is 1 when the event happened and 0 when the subject was censored (still event-free when last seen).", ""]

    out += """\
### 2.3 Pasting from Excel

1. In Excel or Google Sheets select the cells, including the column titles, and copy (Ctrl+C).
2. In TorusPlot go to **1 Data** > **Add data** > *Paste from Excel*, click in *Pasted cells* and paste (Ctrl+V).
3. TorusPlot shows what it detected: the number of rows and columns and a suggested table type.
4. Check *Column titles*. *Automatic* treats a first row of text as titles. Choose
   *First row is column titles* or *No titles (numbers only)* if the guess is wrong.
5. Choose the *Table type*, type a *Table name*, and click **Create table from pasted cells**.

What the paste box understands:

- Tabs, semicolons, commas or spaces between cells.
- Decimal commas (`1,5`). They are converted, and the page says so.
- Empty cells and the markers `NA`, `N/A`, `nan`, `null`, `none`, `-`, `--`, `.`, `#N/A`, `#DIV/0!`,
  `#VALUE!` become missing values.
- A percent sign after a number is dropped (`12%` becomes 12).
- A column of numbers with a few stray text cells (such as "outlier?") stays numeric; the text cells
  become missing values. This applies when the column has at least 5 values and at least 80% are numbers.

The table name becomes the default axis title, so a name such as `Particle size (nm)` saves typing later.

### 2.4 Replicates side by side

XY and Grouped tables can hold replicates in neighbouring columns. There are two ways to say which
columns belong together:

- Type the number in *Replicates side by side* (for example 3). Every 3 neighbouring columns form one group.
  The number of data columns must be a multiple of it.
- Or leave it at 0 and name the columns so that TorusPlot can group them: `Drug:1, Drug:2`,
  `Drug_1, Drug_2`, `Drug rep 1`, or simply the same title repeated.
- Take care with titles that end in a number after `:`, `_`, `-` or `#`. `Day_7` and `Day_14` are
  read as two replicates of `Day`. Rename them (for example `Day 7`, `Day 14`) if they are separate groups.

How replicates are used:

- **XY tables**: regression fits every replicate as its own point by default. Tick
  *Fit the mean of replicates* if the replicates are repeat readings of one sample.
- **Grouped tables**: for a two-way ANOVA each replicate is an independent value. For a
  mixed-design ANOVA each replicate sub-column is one subject followed across the rows.
- Many cells or wells measured inside a few independent experiments do **not** belong in
  side-by-side replicate columns. Use a Multiple variables table with a replicate column (section 5).

### 2.5 Long-format data

Long format (also called tidy format) has one row per value and one column per variable.

- If your data are already long, paste them as a **Multiple variables** table.
- To see any table in long format, open **1 Data** > **Tables** > *Long-format view (what analyses see)*.
- **Make long-format copy** makes a Multiple variables copy of the selected table, one row per value.
- The app has no button for the opposite direction. If an analysis needs a Column table
  (for example a paired t test), paste the data in the wide layout: one column per group.

### 2.6 Which analyses each table type offers

The menu on **2 Analyze** only lists analyses that are valid for the selected table.
""".splitlines()
    rows = []
    for kind in TABLE_KINDS:
        infos = analyses_for(kind)
        rows.append([KIND_LABELS[kind], "; ".join(i.label for i in infos)])
    out += ["", *md_table(["Table type", "Analyses offered"], rows), ""]
    return out


def section_which_test() -> list[str]:
    rec = recommend_test(get_table("size"))
    tiny = ColumnTable.from_dict("Pilot", {"Control": [4.1, 5.0, 4.6], "Treated": [6.9, 7.0, 18.8]})
    small = recommend_test(tiny)
    out = """\
## 3. Which test should I use?

### 3.1 How *Recommend a test* decides

The button **Recommend a test** on page **2 Analyze** looks at the table and follows three steps.

1. **Design.** What kind of table is it? How many groups are there? Are the measurements paired
   (same subjects) or independent? You tell it about pairing with the tick box.
2. **Normality.** It runs the Shapiro-Wilk test at alpha = 0.05: on each group (two groups), on the
   paired differences (paired data), or on the residuals (three or more groups, grouped tables,
   regression). A residual is a value minus the mean of its group.
3. **Equal variance.** For independent groups it runs the Brown-Forsythe test at alpha = 0.05.

It then names one analysis, lists its reasons, shows the checks as badges, adds cautions, and lists
alternatives. It does not run the test until you click **Run: ...**, and it never changes your data.

### 3.2 Decision table for Column tables

| Groups | Design | Normality check | Equal-variance check | Recommended analysis |
|---|---|---|---|---|
| 1 | one group vs a fixed value | passes | not needed | One-sample t test |
| 1 | one group vs a fixed value | fails | not needed | Wilcoxon signed-rank test (one sample) |
| 2 | unpaired | passes | passes | Unpaired t test |
| 2 | unpaired | passes | fails | Welch's t test |
| 2 | unpaired | fails | not used | Mann-Whitney test |
| 2 | paired | passes (differences) | not needed | Paired t test |
| 2 | paired | fails (differences) | not needed | Wilcoxon matched-pairs signed-rank test |
| 3 or more | unpaired | passes (residuals) | passes | One-way ANOVA with Tukey's test (Dunnett's test if you chose a control) |
| 3 or more | unpaired | passes (residuals) | fails | Welch's ANOVA with Games-Howell comparisons |
| 3 or more | unpaired | fails (residuals) | not used | Kruskal-Wallis test with Dunn's comparisons |
| 3 or more | paired | passes (residuals) | not needed | Repeated-measures one-way ANOVA |
| 3 or more | paired | fails (residuals) | not needed | Friedman test |

For the other table types:

| Table | What TorusPlot sees | Recommended analysis |
|---|---|---|
| Grouped | paired box not ticked | Two-way ANOVA with Tukey's comparisons |
| Grouped | paired box ticked (each sub-column is one subject) | Mixed-design ANOVA |
| XY | at least 4 positive X values that span a factor of 100 or more (dose-like), or an X title that contains "log" | Nonlinear regression, four-parameter dose-response |
| XY | otherwise | Simple linear regression |
| Contingency | 2 x 2 table | Fisher's exact test |
| Contingency | larger table | Chi-square test |
| Multiple variables | a time column and an event column | Survival analysis |
| Multiple variables | a replicate-like column (replicate, experiment, batch, animal, donor ...) and a group column | Replicate-aware comparison |
| Multiple variables | one value and three text columns | Three-way ANOVA |
| Multiple variables | one value and two text columns | Two-way ANOVA |
| Multiple variables | one text column and two or more numeric columns | ANCOVA |
| Multiple variables | three or more numeric columns | Multiple linear regression |
| Multiple variables | two numeric columns | Correlation |

For Multiple variables tables the choice depends on the column names and types. Check that the
proposed roles (value, group, replicate, time, event) are what you mean.

There is no rank-based two-way test. If the residuals of a grouped table are not normal, TorusPlot
suggests a log transform of the values.

### 3.3 What a recommendation looks like

This is the real recommendation for the bundled sample *Nanoparticle diameter (nm)*:

```text
""".splitlines()
    out += rec.explain().splitlines() + ["```", ""]
    out += """\
### 3.4 The caveats it prints, and why they matter

The recommendation is honest about what a pre-test can and cannot show.

- **Small samples.** Normality and variance tests have little power with few values. A pass then
  means little. With fewer than 5 values in a group TorusPlot says so.
- **Rank tests need enough values.** With 3 values per group a two-sided Mann-Whitney test cannot
  give P below 0.05. With fewer than 6 pairs the Wilcoxon test cannot either.
- **Pairing.** If two columns have the same number of rows and you did not tick the paired box,
  TorusPlot reminds you to check whether the data are paired.
- **Choosing a test from a pre-test is a convention, not a guarantee.** Welch's t test and Welch's
  ANOVA are always listed as alternatives because they do not need equal variances.
  The safest plan is to choose the test from the design before you collect the data.

Example: a synthetic pilot table with 3 values per group, one of them extreme. TorusPlot recommends
""".splitlines()
    out[-1] += f" *{small.label}* and prints these cautions:"
    out += [""] + [f"- {c}" for c in small.cautions] + [""]
    out += """\
The check results in a recommendation can differ slightly from the checks printed with the analysis
itself. For three or more groups the recommendation tests residuals that are first standardised
within each group, so that unequal spread is not mistaken for non-normality. The ANOVA result tests
the plain residuals.
""".splitlines()
    return out


def section_examples(registry: dict[str, AnalysisInfo]) -> list[str]:
    by_key = {ex.key: ex for ex in EXAMPLES}
    if set(by_key) != set(registry):
        raise SystemExit(f"Examples and registry differ: {sorted(set(by_key) ^ set(registry))}")
    legend = significance_legend().rstrip(".")
    out = f"""\
## 4. Worked examples: one per analysis

There is one example for each of the {len(registry)} analyses in the menu of page **2 Analyze**,
grouped as in that menu. Three tools that are not in the menu follow at the end.

Every example has the same parts:

- **Use it when**: the situation the analysis is made for.
- **In the app**: the table type and the clicks.
- **Example**: the data and the options. All data are synthetic.
- A result block: what TorusPlot computed. It is a short excerpt; the app shows more.
- **How to read it**: what the numbers mean.
- **What not to conclude**: the usual mistakes.
- **Report it like this**: the methods sentence TorusPlot writes, and the numbers to quote.

How to read a result block:

- *Statistic* gives the test statistic with its degrees of freedom in brackets.
- *P value* is followed by the star code (`{legend}`) and by the verdict at the chosen alpha.
- *Effect size* says how large the effect is, with a 95% confidence interval (CI) where one exists.
- *Checks* lists assumption checks with their badge: PASS, CHECK or FAIL.
- Numbers are shown with 4 significant digits. P values below 0.0001 are written `<0.0001`.
""".splitlines()
    for category in CATEGORIES:
        infos = [info for info in registry.values() if info.category == category]
        if not infos:
            continue
        out += ["", f"### 4.{CATEGORIES.index(category) + 1} {category}", ""]
        for info in infos:
            out += example_section(by_key[info.key], info)
    out += section_extra_tools(len(CATEGORIES) + 1)
    return out


def _power_block(result: StatResult) -> list[str]:
    lines = [f"{p.name:<40}{num(p.value)}" for p in result.parameters]
    out = ["```text", f"{result.test_name}", *lines, "```", "", "> " + result.interpretation, ""]
    out += [f"- Warning: {w}" for w in result.warnings]
    return out + ([""] if result.warnings else [])


def section_extra_tools(number: int) -> list[str]:
    out = [f"### 4.{number} Tools that are not in the analysis menu", "",
           "These three tools do not appear under *Choose an analysis*. They have their own place in the app.", ""]

    solve_n = power_ttest(effect_size=0.8, n=None, power=0.8)
    low = power_ttest(effect_size=0.8, n=5, power=None)
    anova = power_anova(effect_size=0.25, n=None, power=0.8, k_groups=4)
    out += """\
#### Power and sample size

**Use it when.** You plan an experiment and ask how many subjects you need, or you ask what a
finished experiment could have detected. Power is the probability that the study finds a
statistically significant result if the true effect has the size you assume.

**In the app.** Page **2 Analyze** > open *Power and sample size calculator* at the bottom of the
page. Choose the *Design*. Fill in two of *Effect size*, *n per group* and *Power*, and leave the
third one empty. Set *Alpha*. Click **Calculate**. No data table is needed.

The effect size for t tests is Cohen's d: the expected difference divided by the SD.
For a one-way ANOVA it is Cohen's f: the SD of the group means divided by the SD within groups.

**Example 1.** Two independent groups, d = 0.8, power 0.8, alpha 0.05, *n per group* left empty.
""".splitlines()
    out += [""] + _power_block(solve_n)
    out += ["**Example 2.** The same design with only 5 subjects per group, *Power* left empty.", ""] + _power_block(low)
    out += ["**Example 3.** One-way ANOVA with 4 groups, f = 0.25, power 0.8, *n per group* left empty.", ""] + _power_block(anova)
    out += """\
**How to read it.**

- The required n is per group. For a paired design it is the number of pairs.
- TorusPlot gives the exact (fractional) solution and the next whole number.
- The power of an ANOVA is the power of the overall F test only, not of each pairwise comparison.

**What not to conclude.** The answer is only as good as the effect size you put in. Take the effect
size from earlier data or from the smallest effect that would matter, not from a wish. Do not compute
power from the effect you observed in the same experiment to explain a non-significant result; that
"observed power" adds nothing to the P value.

**Report it like this.**
""".splitlines()
    out += ["", "> " + solve_n.methods_text.strip(), ""]

    rec = recommend_test(get_table("paired"), paired=True)
    out += """\
#### Recommend a test

**Use it when.** You have a table and do not know which analysis fits. Section 3 explains the rules.

**In the app.** Page **2 Analyze** > **Recommend a test**. Tick
*Same subjects measured repeatedly (paired)* first if it applies.

**Example.** The bundled sample **Tumour volume (mm3)** (`paired`; synthetic), with the paired box ticked:

```text
""".splitlines()
    out += rec.explain().splitlines() + ["```", ""]
    out += """\
**How to read it.**

- The first line is the recommended analysis. The numbered lines are the reasons.
- *Alternatives* lists what you would choose instead, and when.
- The recommendation is a starting point. You decide, and you know things about the experiment
  that the numbers do not show.

**What not to conclude.** A recommendation is not a result. Nothing has been tested yet.

**Report it like this.** Report the test you ran and why it fits the design. Do not write that the
software chose the test.

#### Descriptive graphs

**Use it when.** You want to look at the data before, or instead of, testing: the spread, the
shape of the distribution, single odd values.

**In the app.** Page **3 Graph** > *New figure* > choose a *Graph type* > **Create figure**. No analysis
is needed for these graph types:

- *Histogram*: how often each range of values occurs.
- *Box plot + points*: median, quartiles and whiskers with the raw points.
- *Violin plot + points*: the smoothed shape of each distribution with the raw points.
- *Beeswarm* and *Scatter dot plot*: every value as a dot.
- *Q-Q plot (normality)*: points close to the line suggest a normal distribution.

**Example.** For each graph type, TorusPlot writes how the data are presented. These sentences are
real output for the default elements of each type:
""".splitlines()
    rows = []
    for ptype in ("bar", "scatter", "box", "violin", "beeswarm", "histogram", "qq"):
        spec = FigureSpec(plot_type=ptype, data={"table": "any"})
        rows.append([PLOT_TYPES[ptype]["label"], data_presentation_sentence(spec) or "(no sentence: nothing is summarised)"])
    out += ["", *md_table(["Graph type", "Sentence in the methods paragraph"], rows), ""]
    out += """\
**How to read it.** A graph that shows every data point lets the reader judge the spread and the
sample size directly. Bars with error bars alone hide both.

**What not to conclude.** A graph is not a test. Error bars that do not overlap are not proof of a
difference, and overlapping error bars are not proof of no difference.

**Report it like this.** Say in the figure legend what the bars, lines and error bars show (mean or
median; SD, SEM or CI) and give n. The methods paragraph on page **6 Export** starts with that sentence.
""".splitlines()
    return out


def _error_text(func: Callable[[], Any]) -> str:
    """The text of the error a failing action shows (taken from the running code)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func()
    except ValidationError as exc:
        return str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
    except (StatsError, TableError, ValueError) as exc:
        return str(exc)
    raise SystemExit("An example that should fail did not fail; update tools/build_user_guide.py.")


def section_outliers() -> list[str]:
    table = table_outlier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grubbs = run_analysis(table, "outliers_grubbs")
        before = run_analysis(table, "ttest_welch")
        cells = [(int(f["row"]), str(f["group"])) for f in grubbs.extra["flagged"]]
        if len(cells) != 1:
            raise SystemExit("The outlier example should flag exactly one value; update tools/build_user_guide.py.")
        table.exclude(cells)
        after = run_analysis(table, "ttest_welch")
    flagged = grubbs.extra["flagged"][0]
    flag_value, flag_group, flag_row = num(flagged["value"]), flagged["group"], int(flagged["row"]) + 1
    super_sentence = data_presentation_sentence(FigureSpec(plot_type="superplot", data={"table": "any"}))
    nested = run("cells", "nested_analysis")
    naive = next(w for w in nested.warnings if "Pseudoreplicated" in w)
    out = f"""\
## 5. Outliers, exclusions and replicates

### 5.1 Outliers are flagged, never removed

TorusPlot never deletes or changes a value on its own.

- The outlier analyses (*Identify outliers (Grubbs)*, *Identify outliers (ROUT)*) only **flag** values.
- The raw table always keeps every value you entered.
- A value leaves an analysis only when you exclude it yourself, with a confirmation.
- Every exclusion is written into each later result, the statistics report and the methods paragraph.

### 5.2 How to exclude a value

1. Page **2 Analyze**: run *Identify outliers (Grubbs)* or *Identify outliers (ROUT)* on the table.
2. Under the result, *Flagged values* lists what was flagged.
3. Choose the values under *Values to exclude from analysis*.
4. Tick *I have a scientific reason to exclude these values and will report it.*
5. Click **Exclude selected values**.
6. Run your analysis again.

To undo it: page **1 Data** > **Tables** > **Restore excluded values**.

### 5.3 Example: what an exclusion changes, and how it is reported

Data: the synthetic outlier table of section 4 (two groups of 9 values; one value planted).
Grubbs' test flags the value {flag_value} in the group {flag_group} (row {flag_row}).
The same Welch's t test was run before and after excluding that value:

""".splitlines()
    rows = [["Before the exclusion", "; ".join(f"{k}: {v}" for k, v in before.n.items()), stat_text(before),
             format_p(before.p_value), param_text(before.parameters[0])],
            ["After the exclusion", "; ".join(f"{k}: {v}" for k, v in after.n.items()), stat_text(after),
             format_p(after.p_value), param_text(after.parameters[0])]]
    out += md_table(["", "n", "Statistic", "P", "Estimate"], rows)
    out += ["", "After the exclusion every result for this table carries this data note:", ""]
    out += [f"> {md(note)}" for note in after.data_notes if "excluded" in note]
    out += ["", "The methods paragraph (page **6 Export**) then reads:", "",
            "> " + methods_paragraph([after], include_software=False), ""]
    out += f"""\
The excluded value changes the result. That is exactly why the exclusion must be decided on scientific
grounds and reported. Good practice:

- Decide the rule for exclusions before you see the results.
- Exclude a value only for a reason you can name (a pipetting error noted in the lab book, a dead
  well), not because a test flagged it.
- Report how many values were excluded, from which groups, and by which rule.
- If you can, show that the conclusion holds with and without the excluded values.

### 5.4 Technical and biological replicates

- A **biological replicate** is an independent repeat of the whole experiment: another animal,
  another donor, another experiment on another day.
- A **technical replicate** is a repeated measurement inside one biological replicate: several
  cells, wells or fields of view.

Technical replicates from the same experiment are not independent of each other. Counting each
cell as n makes P values far too small. This mistake is called pseudoreplication.

In the nested example of section 4, TorusPlot prints the size of the mistake as a warning:

> {naive}

### 5.5 The SuperPlot workflow

A SuperPlot shows every cell and, on top, the mean of each experiment. The statistics use the
experiment means.

1. Arrange the data as a **Multiple variables** table: one row per cell, with a column for the
   measured value, a column for the group, and a column that names the biological replicate
   (for example `Experiment`). The bundled sample *Per-cell uptake* has this layout.
2. Page **2 Analyze** > *Category*: Replicate-aware (nested) > *Analysis*:
   Replicate-aware comparison (SuperPlot analysis) > **Run**. Check that the *Group column* and the
   *Biological replicate column* are the right ones.
3. Page **3 Graph** > *Graph type*: SuperPlot (replicate-coloured) > **Create figure**.
   In the **Data** panel, check *Biological replicate*.
4. Small dots are the cells. Large symbols are the experiment means, one colour per experiment.
   The mean line and the error bar are computed from the experiment means.

The methods paragraph for a SuperPlot starts with this sentence:

> {super_sentence}

If the experiments are matched (experiment 1 contains every condition), TorusPlot pairs them. Set
*Match replicates across groups* to `no` when equal labels mean different animals in each group.

![A SuperPlot](images/plot_superplot.png)
""".splitlines()
    return out


LAYER_WORDS = {
    "mean_bar": "bar at the mean", "stack_bar": "stacked bar", "mean_dash": "short line at the mean",
    "median_dash": "short line at the median", "mean_dot": "dot at the mean", "mean_line": "line through the means",
    "sem_errorbar": "SEM error bar", "sd_errorbar": "SD error bar", "ci95_errorbar": "95% CI error bar",
    "data_points": "points in a line", "data_points_jitter": "jittered points", "data_points_beeswarm": "beeswarm points",
    "paired_lines": "one line per subject", "replicate_means": "replicate means", "boxplot": "box and whiskers",
    "violin": "violin outline", "histogram": "histogram bars", "curve_fit": "fitted curve", "km_curve": "survival steps",
    "censor_marks": "ticks at censored times", "effect_axis": "axis with the mean difference and its CI",
    "qq_points": "quantile points", "qq_line": "reference line", "residual_points": "residual points",
    "reference_line": "zero line", "heatmap": "coloured grid",
}


def section_figures() -> list[str]:
    result = run("size", "anova_oneway")
    title_error = _error_text(lambda: FigureSpec(plot_type="bar", data={"table": "t"}, title="P < 0.05 vs control"))
    out = """\
## 6. Making figures

### 6.1 Graph types

Page **3 Graph** lists only the graph types that suit the selected table.
*Default elements* are what a new figure of that type shows; you can change them in the **Elements** panel.

""".splitlines()
    rows = []
    for key, info in PLOT_TYPES.items():
        layers = ", ".join(LAYER_WORDS[lay.type] for lay in default_layers(key))
        rows.append([info["label"], info["description"], ", ".join(KIND_LABELS[k] for k in info["tables"]), layers])
    out += md_table(["Graph type", "What it shows", "Table types", "Default elements"], rows)
    out += """

![Bar graph with points and brackets](images/plot_bar.png)

![Violin plot](images/plot_violin.png)

![Grouped bars](images/plot_grouped_bar.png)

For four graph types you can choose a computed result under *Draw from result*:
*Survival curve (Kaplan-Meier)*, *Residual plot*, *Heat map* (for a correlation matrix) and
*Q-Q plot (normality)*. A residual plot always needs a regression result. The other three can
also be drawn straight from the table.

### 6.2 Error bars: SD, SEM or confidence interval

Choose them under **Elements** > *Error bars*.

| Choice | What is drawn | What it tells the reader |
|---|---|---|
| SD | mean plus and minus the standard deviation | The spread of the data. It does not shrink with more data. |
| SEM | mean plus and minus SD / square root of n | How precisely the mean is known. It shrinks as n grows. It says little about the spread. |
| 95% CI | the 95% confidence interval of the mean (t distribution) | The range that is likely to contain the true mean. About twice as wide as the SEM bar for large n, wider for small n. |
| Range | from the smallest to the largest value, around the mean | The extremes. |
| IQR | from the first to the third quartile, around the median | The middle half of the data. Suited to skewed data. |
| None | no error bar | Use it when the points speak for themselves. |

- Always say in the legend which one you used, and give n.
- SEM bars look small. Do not choose them for that reason.
- The IQR bar is centred on the median. The bar or line of a bar or dot plot still marks the mean, so
  use IQR with a box plot or a violin plot.
- Keep *Individual data points* on. Most journals ask for them. Choices: *Jittered*,
  *Beeswarm (no overlap)*, *In a line*, *Hidden*.

### 6.3 Significance brackets come only from computed results

You cannot type a P value or stars onto a figure.

- In **Statistics on the graph**, choose a result under *Significance brackets from*.
  Only results of this table that contain pairwise comparisons are listed.
- *Show*: *All comparisons* or *Only significant ones*.
- *Label*: stars, the P value, or both. The text is read from the adjusted P value of each comparison.
- *Test summary text in a corner* writes the test name and its P value, again from a computed result.
""".splitlines()
    comp = result.comparisons[0]
    labels = [[name, f"`{comparison_label(comp, kind)}`"] for name, kind in
              [("Stars", "stars"), ("P = 0.0123", "p"), ("0.0123", "p_short"), ("Stars and P value", "stars_p")]]
    out += ["", f"Example (one-way ANOVA with Tukey's test on the sample *Nanoparticle diameter (nm)*, comparison "
                f"{comp.group_a} vs {comp.group_b}):", "", *md_table(["*Label* choice", "Text on the bracket"], labels), "",
            "The corner text for the same result is:", "", f"> {summary_text(result)}", "",
            f"Stars follow this code: `{significance_legend().rstrip('.')}`.", "",
            "If you type something that looks like a statistic into a title or a text label, TorusPlot refuses it:", "",
            f"> {title_error}", "",
            "The same rule holds for the AI features: they can choose which computed comparisons to show, "
            "but they cannot write a number.", "",
            "If the result behind a bracket is deleted, the bracket is not drawn and a note appears under the preview.", ""]
    out += """\
### 6.4 Fitted curves

1. Run the regression on page **2 Analyze** (linear or nonlinear).
2. On page **3 Graph**, create an *XY* or *Dose-response curve* figure.
3. Under **Statistics on the graph**, choose the result under *Fitted curves to draw*.
   Tick *Show 95% confidence band* to shade the uncertainty of the curve.

- The curve is drawn from the stored result. It is not refitted by the graph.
- Dose-response graphs use a logarithmic X axis automatically. Points with X of 0 or less cannot be
  shown on a log axis; a note under the preview says how many were skipped.
- A new dose-response figure attaches the latest curve fit of each data set by itself.

![Dose-response curves](images/plot_dose_response.png)

### 6.5 Estimation plots and before-after plots

An **estimation plot** shows the raw data on the left and the difference between the group means
with its 95% CI on a second axis on the right. It puts the size of the effect in front of the reader,
not only a P value.

- If a result is attached, the difference and CI come from it.
- If none is attached, TorusPlot computes the difference between the first two groups with Welch's
  t test (second group minus first group).

![Estimation plot](images/plot_estimation.png)

A **before-after plot** draws one line per subject. Use it for paired data, with one subject per row.

![Before-after plot](images/plot_paired.png)

### 6.6 Survival curves

Run *Survival analysis (Kaplan-Meier, log-rank)* first, then create a figure of type
*Survival curve (Kaplan-Meier)*. The curve steps down at each event. Tick marks show censored subjects.

![Survival curves](images/plot_survival.png)
""".splitlines()
    return out


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def section_style() -> list[str]:
    out = """\
## 7. Styling and exporting

### 7.1 Style presets

A preset sets fonts, line widths, colours and the figure size in one step. Choose it on page
**4 Style** under *Style preset*. Every value can then be changed by hand; an empty box means
"use the preset".

""".splitlines()
    rows = []
    for name in preset_names():
        p = get_preset(name)
        rows.append([name, f"{p['width_mm']:g} x {p['height_mm']:g} mm", p["font_family"], f"{p['font_size']:g} pt",
                     p["palette"], p["description"]])
    out += md_table(["Preset", "Default size", "Font", "Base text size", "Palette", "Description"], rows)
    out += """

![The style presets](images/style_presets.png)

The three journal presets share the same font, text sizes, line widths and palette. They differ in
size: single column, one and a half columns, double column.
Check the author guidelines of your journal for its exact widths and minimum font size.

### 7.2 Colours and the colour-blind preview

Under **Colours** on page **4 Style**:

- *Palette*: a named palette, the preset default, or *Custom colours* (hex codes separated by commas).
- *Colour groups along a gradient*: for ordered groups such as doses or time points.
- *Set a colour for individual groups*: one colour picker per group.

Open *Colour-blind preview* under the live preview. It shows the figure as seen with:
""".splitlines()
    out += [f"- {label}" for label in CVD_LABELS.values()]
    out += """
The table below is computed by TorusPlot's own palette check. Two colours count as distinguishable
when their colour distance (CIE76 Delta-E) is at least 10 after the simulation.

""".splitlines()
    rows = []
    for name in palette_names():
        four = palette_report(get_palette(name, 4))
        cvd4 = all(four[k]["ok"] for k in ("deuteranopia", "protanopia", "tritanopia"))
        if name in PALETTES:
            full = palette_report(PALETTES[name])
            cvd_all = _yes(all(full[k]["ok"] for k in ("deuteranopia", "protanopia", "tritanopia")))
            count = str(len(PALETTES[name]))
        else:
            cvd_all, count = "(gradient)", "any"
        rows.append([name, count, _yes(cvd4), cvd_all, _yes(bool(four["grayscale"]["ok"]))])
    out += md_table(["Palette", "Colours", "Colour-blind safe with 4 groups", "Colour-blind safe with all its colours",
                     "Distinguishable in grayscale with 4 groups"], rows)
    out += """

- Okabe-Ito is the safest named palette for categories. The journal presets use it.
- No palette of saturated colours survives grayscale printing. If the figure may be printed in
  black and white, use a gradient palette (viridis family), *Classic Lab*, or different symbols.
- With many groups, even a good palette runs out of distinguishable colours. Check the preview.

### 7.3 Fonts

Under **Text**: *Font*, *Base font size (pt)*, *Tick label size (pt)*, *Axis title size (pt)*.
""".splitlines()
    out += ["", f"- TorusPlot ships with {len(BUNDLED_FAMILIES)} fonts: {', '.join(BUNDLED_FAMILIES)}. "
                "They render the same on every computer.",
            "- Liberation Sans has the same letter widths as Arial. A figure laid out with it keeps its "
            "proportions if a journal replaces the font with Arial.",
            "- Fonts installed on your computer are listed after the bundled ones. Arial and Helvetica are not "
            "bundled; you can select them if they are installed.",
            "- If a project asks for a font that is not installed, TorusPlot uses Inter and says so in a note.", ""]
    out += """\
### 7.4 Exact size in millimetres

Under **Size and lines**: *Width (mm)* and *Height (mm)*.

- The exported file has exactly this size. Nothing is cropped or rescaled.
- Page **6 Export** prints the size under the preview (*Exact size: ... mm*).
- Text sizes are in points and stay as set. Make the figure at its final printed size, so that 7 pt
  text is 7 pt in the journal.

### 7.5 Multi-panel figures

Page **5 Compose** arranges finished figures on a grid.

1. *Panels (in reading order)*: choose the figures. They are placed left to right, top to bottom.
2. *Columns*: the number of columns of the grid.
3. *Width (mm)* and *Height (mm)*: the size of the whole composition. The default width is 183 mm.
4. *Panel letters*: `A`, `a`, `(A)`, `(a)`, `1` or `none`.
5. *Same Y range* and *Same X range*: give panels in a row, in a column, or all panels the same axis range.
6. *One shared legend*: one legend for the whole figure.
7. *Apply one style to all panels*: one preset for every panel, or keep each figure's own style.
8. Click **Save composition**. It then appears on page **6 Export**.

### 7.6 Export formats

| Format | What you get | Use it for |
|---|---|---|
| SVG | Vector file. Text stays text and every element is in a named group. | Touching up in a vector editor (Inkscape, Illustrator). |
| PDF | Vector file with embedded fonts. Text stays selectable. | Journal submission of line art and graphs; inserting into documents. |
| TIFF | Flattened image with LZW compression at the chosen resolution. | Journal systems that ask for TIFF. |
| PNG | Image at the chosen resolution. | Slides, documents, the web. |

Resolution (*Resolution (dpi) for TIFF/PNG*: 150, 300, 600 or 1200):

- 300 dpi for figures with colour or shading.
- 600 dpi or more for line art (graphs with thin lines and small text).
- 150 dpi only for drafts and slides.
- Vector formats (SVG, PDF) have no resolution. Prefer them when the journal accepts them.

*Transparent background* removes the white background (useful for slides).
Under the SVG button, *SVG check* confirms that the file has editable text and named groups.

### 7.7 The statistics report

Page **6 Export** > *Statistics report and methods paragraph*.

- Results that are drawn on the figure are always included. Add others under
  *Also include these results for the same data*.
- **Download report (Markdown)** is a readable report: for each result the interpretation, why the
  test was chosen, the statistic, P value, effect sizes, estimates, the ANOVA table, every comparison
  with its ID, descriptive statistics, assumption checks, warnings, data notes (exclusions) and the
  method references.
- **Download report (CSV)** has the same numbers as a flat table, one row per number, for spreadsheets.
- Nothing in the report is typed by hand. Keep it with the figure; reviewers ask for these numbers.

### 7.8 The methods paragraph

The box *Methods paragraph* holds text you can paste into the statistics section of a manuscript.
It is assembled from what was actually run:

1. how the data are presented (from the elements of the figure);
2. the methods sentence of each result;
3. the significance threshold;
4. any exclusions;
5. the sample sizes;
6. a last sentence that names the TorusPlot version and the numerical libraries.

Example for a bar graph of the sample *Nanoparticle diameter (nm)* with the one-way ANOVA result
(the last sentence is left out here):
""".splitlines()
    result = run("size", "anova_oneway")
    spec = FigureSpec(plot_type="bar", data={"table": get_table("size").name})
    out += ["", "> " + methods_paragraph([result], spec, include_software=False), "",
            "Read the paragraph before you paste it. Add what the software cannot know: how subjects were "
            "allocated, what n counts (animals, experiments, cells), and whether the analysis was planned in advance.", ""]
    out += """\
### 7.9 Project files (.tplot)

Page **6 Export** > tab **Project file**.

- **Save project (.tplot)** downloads one file with your tables, figures, compositions, statistics
  results and the activity log.
- *Open a project* and **Open this project (replaces the current one)** load it again.
- Opening a project re-creates every figure exactly. The saved results are used as they are;
  nothing is recalculated.
- A `.tplot` file is an ordinary zip archive: a manifest (`project.yaml`), one CSV per table
  (`data/`) and the results (`results.json`). You can open it with any zip tool.
- The *Activity log* lists what was done: tables added, analyses run, values excluded.
- Save before you close the browser tab or stop TorusPlot. The project is kept in memory only.
""".splitlines()
    return out


def section_ai() -> list[str]:
    return """\
## 8. Optional AI features and the MCP server

### 8.1 Optional AI features

AI features are **off by default**. TorusPlot works completely without them, and offline.

**The hard rule: AI can change how a figure looks, never the statistics.** An AI feature can
produce or change a figure description (graph type, elements, colours, size, and which already
computed comparisons to show). It cannot compute, type or alter a P value, a star or any other
number. This is enforced by the program, not by trusting the AI: a figure description has no place
to put a number, and text that looks like a statistic is rejected (section 6.3).

**Turning AI on.** Page **Settings** > switch on *Enable AI features* > choose a *Provider* >
click **Test the provider**.

| Provider | What it means |
|---|---|
| Demo | Offline, canned example answers. For trying the screens. |
| Ollama | A free program that runs AI models on your own computer. Nothing leaves the computer. You download a model yourself. |
| Google Gemini, Anthropic, OpenAI | Cloud services, with your own API key. The key is read from a file named `.env` next to the app (copy `.env.example`), or pasted for the current session only. |

With a cloud provider, the text you type into AI boxes and the images you upload there are sent to
that company under your account. Its terms, pricing and data policies apply. Your data values are
not included in AI prompts; table and column names are.

**Using AI.** Page **AI Assistant** has four tabs.

| Tab | What it does | Needs a provider? |
|---|---|---|
| Describe it | Type what you want under *Describe the figure* and click **Create figure from description**. Choose an existing figure under *Start from* to change it. | Yes |
| Match this figure | Upload an image under *Reference figure (PNG or JPG)* and click **Match this figure**. The style (colours, proportions, look of the axes) is copied onto your data. The other figure's data are never copied. | No. Without a provider, colours and proportions are measured from the image; a vision model adds the graph type and finer details. |
| Digitize | Upload an image of a plot, mark the plot area, the axis values and the series, and click **Digitize**. **Add as an APPROXIMATE table** stores the values. | No |
| Outside AI instructions | Copy the prompt under *Prompt to give your chatbot* into any chatbot, paste its answer under *Paste the chatbot's answer here*, and click **Turn into a figure**. | No, when the answer is in TorusPlot's format |

![The AI Assistant page](images/app_assistant.png)

- A figure made by AI is a normal figure. Refine it on **3 Graph** and **4 Style**.
- Digitized values are estimates read off an image. The table is marked APPROXIMATE on every page
  and in every report, and statistics are blocked on it unless you override (section 9).
- The full prompt for chatbots is also in
  [PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md](PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md).

### 8.2 MCP server

The Model Context Protocol (MCP) lets an AI assistant call TorusPlot's tools: load a table, run an
analysis, make and export a figure. The server runs on your computer and makes no network calls
itself. The same hard rule applies: every number comes from TorusPlot's statistics engine.

Setup, the list of tools and the privacy notes are in [MCP.md](MCP.md).
""".splitlines()


def section_troubleshooting() -> list[str]:
    size = get_table("size")
    digitized = ColumnTable.from_dict("Digitized", {"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]}, approximate=True)
    xy_frame = pd.DataFrame({"x": [1, 2, 3], "a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "d": [1, 2, 3], "e": [1, 2, 3]})
    percents = build_table("contingency", pd.DataFrame({"Group": ["Treated", "Control"], "Yes": [12.5, 40.1], "No": [87.5, 59.9]}), "Percent")
    coded = build_table("multiple_variables", pd.DataFrame({"time": [1, 2, 3, 4], "event": [1, 2, 0, 1], "group": list("aabb")}), "Coded")
    one_rep = build_table("multiple_variables", pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "group": list("aabb"), "replicate": ["e1"] * 4}), "One")
    few_xy = build_table("xy", pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}), "Few")
    cases: list[tuple[str, Callable[[], Any], str]] = [
        ("The analysis could not be run:", lambda: run_analysis(digitized, "ttest_welch"),
         "The table was digitized from an image. Its numbers are estimates. Use measured data. If you only want "
         "a rough reference value, tick *I understand these are estimates read off an image. Allow statistics "
         "anyway.* on page **2 Analyze**; the result is then marked APPROXIMATE."),
        ("The analysis could not be run:", lambda: run_analysis(size, "fisher_exact"),
         "The analysis does not fit the table type. In the app the menu already hides such analyses. Re-create "
         "the table with the right type (section 2)."),
        ("The analysis could not be run:", lambda: run_analysis(ColumnTable.from_dict("t", {"A": [1.0], "B": [2.0, 3.0]}), "ttest_unpaired"),
         "A group has too few values. Add replicates. Check that numbers were not read as text (a stray letter or "
         "unit in a cell turns it into a missing value)."),
        ("The analysis could not be run:", lambda: run_analysis(size, "ttest_unpaired", group_a="Bulk"),
         "The group name does not exist in the table. Pick the group from the list; names are case-sensitive."),
        ("The analysis could not be run:", lambda: run_analysis(ColumnTable.from_dict("t", {"A": [1.0, np.nan, 3.0], "B": [np.nan, 3.0, np.nan]}), "ttest_paired"),
         "Paired tests use only rows that have a value in both columns. Put each subject on one row."),
        ("The analysis could not be run:", lambda: run_analysis(get_table("paired"), "friedman"),
         "Use the test the message names."),
        ("The analysis could not be run:", lambda: run_analysis(ColumnTable.from_dict("t", {"A": [1.0], "B": [2.0], "C": [3.0]}), "anova_oneway"),
         "Each group needs replicates."),
        ("The analysis could not be run:", lambda: run_analysis(percents, "chi_square"),
         "Enter the number of subjects in each cell, not percentages."),
        ("The analysis could not be run:", lambda: run_analysis(coded, "survival"),
         "Recode the event column to 1 (event) and 0 (censored)."),
        ("The analysis could not be run:", lambda: run_analysis(one_rep, "nested_analysis"),
         "The replicate column must name at least two independent experiments or animals in each group, and "
         "three or more are needed for a useful test."),
        ("The analysis could not be run:", lambda: run_analysis(few_xy, "nonlinear_regression"),
         "Add more X values, or fix some parameters under *Fixed parameters* (for example `Bottom=0, Top=100`)."),
        ("The analysis could not be run:", lambda: run_analysis(get_table("dose"), "nonlinear_regression", model="user_defined", equation="Y = open('x')"),
         "A user-defined equation may only contain numbers, X, parameter names, + - * / ^, brackets and the listed "
         "functions."),
        ("Could not create the table:", lambda: build_table("xy", xy_frame, "t", replicates=3),
         "The number of Y columns is not a multiple of *Replicates side by side*. Correct the number, or set it "
         "to 0 and name the columns `Name:1, Name:2`."),
        ("That setting is not valid:", lambda: FigureSpec(plot_type="bar", data={"table": "t"}, title="P < 0.05 vs control"),
         "Remove the statistic from the title. Add brackets or the corner summary from a computed result instead "
         "(section 6.3)."),
    ]
    out = """\
## 9. Troubleshooting and FAQ

### 9.1 Messages and what to do

The messages below are the real texts TorusPlot shows. In the app they follow the words in the first column.

""".splitlines()
    rows = [[prefix, _error_text(func), advice] for prefix, func, advice in cases]
    out += md_table(["The app says", "Message", "What to do"], rows)
    out += """

Other notes you may see under a figure preview:

- *This figure cannot be drawn yet: ...* The figure lacks something, usually a result. A residual
  plot, for example, needs a regression result chosen under *Draw from result*.
- *Note: ... statistics result ... was not found, so it was not drawn.* The result behind a bracket
  or curve was deleted. Run the analysis again and choose the new result.
- *Note: ... comparison(s) ... do not match groups on this plot and were not drawn.* The result
  compares groups that the figure does not show. Show those groups or choose another result.

### 9.2 Frequently asked questions

**The numbers differ slightly from another statistics package. Which is right?**
Small differences usually come from conventions, not errors. Common ones:
quartiles (TorusPlot uses linear interpolation, definition 7 of Hyndman and Fan);
exact or approximate P values for rank tests; Yates' correction for 2 x 2 tables;
Type II or Type III sums of squares in unbalanced designs; which sphericity correction is the headline.
[STATS_METHODS.md](STATS_METHODS.md) lists the convention for each method.

**The normality test passes. Can I trust the t test?**
A pass means no deviation was detected. With small samples that is weak evidence. Judge from what is
known about the measurement, look at the points, and consider Welch's test or a rank test.

**My result is not significant. Is there no effect?**
No. Look at the confidence interval of the difference. If it is wide, the experiment could not tell.
Use the power calculator before the next experiment.

**I have 3 values per group. What can I do?**
Report the individual values and the difference with its confidence interval. A rank test cannot
reach P below 0.05 with 3 per group. A t test can, but it rests on assumptions that 3 values cannot check.

**Can I compare more than two groups with several t tests?**
Use a one-way ANOVA (or Kruskal-Wallis test) with multiple comparisons. They control the chance of
a false positive across all comparisons.

**Where did my brackets go?**
Brackets belong to a result. If you deleted the result or changed the groups shown, they are not
drawn. Choose a result again under *Significance brackets from*.

**Can I type "P < 0.05" on the figure myself?**
No. See section 6.3.

**Nothing happens when I close the browser. Is my work saved?**
No. Save the project on page **6 Export** > **Project file** before you close TorusPlot.

**Is my data sent anywhere?**
No, unless you turn on a cloud AI provider (section 8).

**Who checks my statistics?**
TorusPlot is a tool, not a statistician. For clinical data, regulatory submissions and other
high-stakes work, have a statistician review the plan and the results.
""".splitlines()
    return out


def section_glossary() -> list[str]:
    terms = [
        ("Alpha", "The threshold you set in advance for calling a P value significant. Usually 0.05. It is the "
                  "chance of a false positive that you accept when there is truly no effect."),
        ("Censoring", "In survival data, a subject is censored when it was still event-free when last seen "
                      "(the study ended, or the subject was lost). Its time still counts until then."),
        ("Confidence interval (CI)", "A range computed from the data that is likely to contain the true value. "
                                     "For a 95% CI, the method captures the true value in 95% of repeated experiments."),
        ("EC50 / IC50", "The concentration that gives a response halfway between the bottom and the top of a "
                        "dose-response curve. EC50 is used for stimulation, IC50 for inhibition."),
        ("Effect size", "A number that says how large an effect is, separately from how sure you are. "
                        "**Cohen's d**: the difference between two means divided by the SD (0.2 small, 0.5 medium, "
                        "0.8 large by Cohen's benchmarks). **Hedges' g**: Cohen's d with a correction for small "
                        "samples. **Eta squared**: the share of the total variation explained by a factor, from 0 "
                        "to 1 (0.01 small, 0.06 medium, 0.14 large). The benchmarks are rough conventions, not rules."),
        ("Multiple comparisons", "When you make several comparisons, the chance that at least one is a false "
                                 "positive grows. Procedures such as Tukey, Dunnett, Bonferroni and Holm adjust the "
                                 "P values to keep that chance at alpha. Read the adjusted P value."),
        ("P value", "The probability of data at least as extreme as yours if there were truly no effect. A small "
                    "P value is evidence against \"no effect\". It is not the probability that the hypothesis is "
                    "true, and it does not measure the size of the effect."),
        ("Paired vs unpaired", "Paired: the same subject (or a matched pair) is measured under both conditions, "
                               "and the test looks at the within-subject differences. Unpaired: different subjects "
                               "in each group."),
        ("Parametric vs nonparametric", "Parametric tests (t test, ANOVA) assume a distribution, usually the "
                                        "normal one, and compare means. Nonparametric tests (Mann-Whitney, "
                                        "Wilcoxon, Kruskal-Wallis, Friedman) use ranks and assume less, at the cost "
                                        "of some power when the data are normal."),
        ("Pseudoreplication", "Treating measurements that are not independent (cells from one dish, repeated "
                              "readings of one sample) as if each were an independent sample. It makes P values "
                              "too small. See section 5.4."),
        ("SD vs SEM", "The standard deviation (SD) describes how much individual values scatter. The standard "
                      "error of the mean (SEM = SD divided by the square root of n) describes how precisely the "
                      "mean is known. The SEM gets smaller with more data; the SD does not."),
        ("Sphericity", "An assumption of repeated-measures ANOVA: the differences between every pair of "
                       "conditions have the same variance. When it is violated, the Greenhouse-Geisser or "
                       "Huynh-Feldt correction lowers the degrees of freedom."),
    ]
    out = ["## 10. Glossary", ""]
    for term, text in terms:
        out += [f"**{term}.** {text}", ""]
    return out


#: Widget and button names that the guide quotes, by app source file. The build stops when one of
#: them disappears from the app, so the guide cannot keep describing a button that no longer exists.
UI_LABELS: dict[str, tuple[str, ...]] = {
    "state.py": ("1 Data", "2 Analyze", "3 Graph", "4 Style", "5 Compose", "6 Export", "AI Assistant", "Settings"),
    "streamlit_app.py": ("Workflow",),
    "ui_helpers.py": ("Assumption checks", "Why this test?", "Methods sentence", "Multiple comparisons", "ANOVA table",
                      "Estimates", "Effect sizes", "Descriptive statistics"),
    "views/data.py": ("Add data", "Load a sample dataset", "Load sample", "Paste from Excel", "Pasted cells",
                      "Column titles", "First row is column titles", "No titles (numbers only)", "Table type",
                      "Table name", "Create table from pasted cells", "Import a file", "Worksheet",
                      "Create table from file", "Apply edits", "Download CSV", "Make long-format copy", "Delete table",
                      "Restore excluded values", "Replicates side by side", "Long-format view (what analyses see)"),
    "views/analyze.py": ("Not sure which test? Ask for a recommendation", "Recommend a test",
                         "Same subjects measured repeatedly (paired)", "Control group (optional)", "Choose an analysis",
                         "Category", "Analysis", "Results for this table", "Flagged values",
                         "Values to exclude from analysis",
                         "I have a scientific reason to exclude these values and will report it.",
                         "Exclude selected values", "Power and sample size calculator", "n per group", "Calculate",
                         "I understand these are estimates read off an image. Allow statistics anyway."),
    "views/graph.py": ("New figure", "Graph type", "Figure name", "Create figure", "Live preview", "Elements", "Error bars",
                       "Individual data points", "Jittered", "Beeswarm (no overlap)", "In a line", "Hidden",
                       "Statistics on the graph", "Significance brackets from", "All comparisons",
                       "Only significant ones", "Test summary text in a corner", "Fitted curves to draw",
                       "Show 95% confidence band", "Draw from result", "Titles and axes", "Y scale", "X scale",
                       "Y minimum", "Y maximum", "Biological replicate"),
    "views/style.py": ("Style preset", "Colours", "Palette", "Custom colours", "Colour groups along a gradient",
                       "Set a colour for individual groups", "Font", "Base font size (pt)", "Tick label size (pt)",
                       "Axis title size (pt)", "Size and lines", "Width (mm)", "Height (mm)", "Legend",
                       "Colour-blind preview"),
    "views/compose.py": ("Panels (in reading order)", "Columns", "Width (mm)", "Height (mm)", "Panel letters",
                         "Same Y range", "Same X range", "One shared legend", "Apply one style to all panels",
                         "Save composition"),
    "views/export.py": ("Figures and reports", "Project file", "What to export", "Resolution (dpi) for TIFF/PNG",
                        "Transparent background", "Exact size", "SVG check",
                        "Also include these results for the same data", "Methods paragraph",
                        "Download report (Markdown)", "Download report (CSV)", "Save project (.tplot)",
                        "Open a project", "Open this project (replaces the current one)", "Activity log"),
    "views/settings.py": ("Enable AI features", "Provider", "Test the provider"),
    "views/assistant.py": ("Describe it", "Match this figure", "Digitize", "Outside AI instructions",
                           "Describe the figure", "Start from", "Create figure from description",
                           "Reference figure (PNG or JPG)", "Add as an APPROXIMATE table",
                           "Prompt to give your chatbot", "Paste the chatbot's answer here", "Turn into a figure"),
}


def check_ui_labels(text: str) -> None:
    """Every quoted widget name must exist in the app source and be used in the guide."""
    for rel, labels in UI_LABELS.items():
        source = (ROOT / "app" / rel).read_text(encoding="utf-8")
        for label in labels:
            if label not in source:
                raise SystemExit(f"The guide quotes '{label}', which is no longer in app/{rel}. "
                                 "Update the text in tools/build_user_guide.py.")
            if label not in text:
                raise SystemExit(f"UI_LABELS lists '{label}' (app/{rel}), but the guide does not mention it.")


def build() -> str:
    registry = load_all()
    lines: list[str] = HEADER.splitlines() + [""]
    for section in (section_start(), section_tables(), section_which_test(), section_examples(registry),
                    section_outliers(), section_figures(), section_style(), section_ai(),
                    section_troubleshooting(), section_glossary()):
        lines += section + [""]
    text = "\n".join(line.rstrip() for line in lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    text = text.rstrip() + "\n"
    for banned in ("graphpad", "prism"):
        if banned in text.lower():
            raise SystemExit(f"The guide must not contain the word '{banned}'.")
    if re.search(r"sr_[0-9a-f]{8}", text):
        raise SystemExit("A random result ID leaked into the guide.")
    check_ui_labels(text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Only check that the committed guide is up to date")
    parser.add_argument("--output", type=Path, default=DOC, help="Where to write the guide")
    args = parser.parse_args(argv)

    text = build()
    current = None
    if args.output.is_file():
        current = args.output.read_text(encoding="utf-8").replace("\r\n", "\n")
    if args.check:
        if current != text:
            print(f"{args.output} is out of date. Run: python tools/build_user_guide.py")
            return 1
        print(f"{args.output} is up to date.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"Wrote {args.output} ({len(text.splitlines())} lines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
