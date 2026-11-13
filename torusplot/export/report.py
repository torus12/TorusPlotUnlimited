"""Statistics report that travels with each figure: Markdown (to read) + CSV (to reuse)."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .. import __version__
from ..spec.figure import FigureSpec
from ..stats.result import StatResult, format_p, p_to_stars
from .methods import methods_paragraph, significance_legend

_BADGE = {"pass": "PASS", "warn": "CHECK", "fail": "FAIL", "info": "INFO"}  # same words as the app's badges


def _g(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        v = float(value)
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(value, int) or (v.is_integer() and abs(v) < 1e9):
            return str(int(v))
        return f"{v:.{digits}g}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    return out + [""]


def _result_markdown(res: StatResult) -> list[str]:
    lines = [f"## {res.test_name}", ""]
    meta = [f"Result ID: `{res.id}`"]
    if res.table:
        meta.append(f"Table: {res.table}")
    lines += [" · ".join(meta), ""]
    if res.approximate_data:
        lines += ["> **APPROXIMATE DATA: for reference only.** This analysis used digitized values.", ""]
    if res.interpretation:
        lines += [res.interpretation, ""]
    if res.why_chosen:
        lines += [f"**Why this test?** {res.why_chosen}", ""]
    main = []
    if res.statistic is not None:
        df = ""
        if res.df is not None and res.df2 is not None:
            df = f"({_g(res.df)}, {_g(res.df2)})"
        elif res.df is not None:
            df = f"({_g(res.df)})"
        main.append([f"{res.statistic_name or 'Statistic'}{df}", _g(res.statistic, 5)])
    if res.p_value is not None:
        main.append([f"P value ({res.alternative})", f"{format_p(res.p_value)}  {p_to_stars(res.p_value)}"])
    if res.p_adjusted is not None:
        main.append(["Adjusted P value", format_p(res.p_adjusted)])
    if main:
        main.append(["Alpha", _g(res.alpha)])
        if res.significant is not None:
            main.append(["Significant?", "yes" if res.significant else "no"])
        lines += _md_table(["Quantity", "Value"], main)
    if res.effect_sizes:
        lines += ["**Effect sizes**", ""]
        lines += _md_table(
            ["Effect size", "Term", "Value", "CI", "Magnitude"],
            [[e.name, e.target or "", _g(e.value),
              f"{e.ci_level * 100:g}% {_g(e.ci_low)} to {_g(e.ci_high)}" if e.ci_low is not None else "", e.magnitude or ""]
             for e in res.effect_sizes],
        )
    if res.parameters:
        lines += ["**Estimates**", ""]
        lines += _md_table(
            ["Parameter", "Value", "SE", "CI", "P"],
            [[p.name, _g(p.value, 5), _g(p.se),
              f"{p.ci_level * 100:g}% {_g(p.ci_low, 5)} to {_g(p.ci_high, 5)}" if p.ci_low is not None else "",
              format_p(p.p_value) if p.p_value is not None else ""] for p in res.parameters],
        )
    if res.anova_table:
        keys = list(dict.fromkeys(k for row in res.anova_table for k in row))
        lines += ["**ANOVA table**", ""]
        lines += _md_table(keys, [[format_p(row.get(k)) if k.startswith("p") and isinstance(row.get(k), float) and k != "partial_eta_sq" and k != "percent_of_variation"
                                   else _g(row.get(k), 5) for k in keys] for row in res.anova_table])
    if res.comparisons:
        title = "**Multiple comparisons**"
        if res.correction:
            title += f" ({res.correction})"
        lines += [title, ""]
        if res.comparison_family:
            lines += [f"Family: {res.comparison_family}", ""]
        lines += _md_table(
            ["ID", "Comparison", "Difference", "CI", "Statistic", "P", "Adjusted P", "Summary"],
            [[f"`{c.id}`", f"{c.group_a} vs {c.group_b}" + (f" [{c.label}]" if c.label else ""), _g(c.estimate),
              f"{_g(c.ci_low)} to {_g(c.ci_high)}" if c.ci_low is not None else "",
              f"{c.statistic_name or ''} {_g(c.statistic)}".strip() if c.statistic is not None else "",
              format_p(c.p_value) if c.p_value is not None else "",
              format_p(c.p_adjusted) if c.p_adjusted is not None else "", p_to_stars(c.p_for_display)]
             for c in res.comparisons],
        )
    if res.descriptives:
        keys = list(dict.fromkeys(k for row in res.descriptives for k in row))
        lines += ["**Descriptive statistics**", ""]
        lines += _md_table(keys, [[_g(row.get(k), 5) for k in keys] for row in res.descriptives])
    elif res.n:
        lines += ["**Sample sizes**", ""] + _md_table(["Group", "n"], [[k, v] for k, v in res.n.items()])
    if res.assumptions:
        lines += ["**Assumption checks**", ""]
        lines += _md_table(
            ["Status", "Check", "Applies to", "Statistic", "P", "Detail"],
            [[_BADGE[a.status], a.name, a.target or "", _g(a.statistic), format_p(a.p_value) if a.p_value is not None else "", a.detail]
             for a in res.assumptions],
        )
    if res.warnings:
        lines += ["**Warnings**", ""] + [f"- {w}" for w in res.warnings] + [""]
    if res.data_notes:
        lines += ["**Data notes**", ""] + [f"- {w}" for w in res.data_notes] + [""]
    if res.citations:
        lines += ["**Method references**", ""] + [f"- {c}" for c in res.citations] + [""]
    return lines


def stats_report_markdown(results: Iterable[StatResult], title: str = "Statistics report",
                          spec: FigureSpec | None = None) -> str:
    """Human-readable report for the results behind a figure."""
    results = [r for r in results if r is not None]
    lines = [f"# {title}", "", f"Generated by TorusPlot {__version__}. All numbers come from the TorusPlot "
             "statistics engine; nothing in this report was typed or edited by hand or by AI.", ""]
    if spec is not None:
        lines += [f"Figure: **{spec.name}** ({spec.plot_type}), data table: {spec.data.table}", ""]
    if not results:
        lines += ["No statistical analyses are attached to this figure.", ""]
        return "\n".join(lines)
    lines += ["## Methods paragraph", "", methods_paragraph(results, spec), "", significance_legend(), ""]
    for res in results:
        lines += _result_markdown(res)
    return "\n".join(lines).rstrip() + "\n"


CSV_COLUMNS = [
    "result_id", "test", "table", "section", "name", "group_a", "group_b", "label", "value", "se", "ci_low",
    "ci_high", "statistic_name", "statistic", "df", "df2", "p_value", "p_adjusted", "significant", "status", "detail",
]


def stats_report_rows(results: Iterable[StatResult]) -> list[dict[str, Any]]:
    """Flat, tidy rows (one per number) for spreadsheets."""
    rows: list[dict[str, Any]] = []

    def add(res: StatResult, section: str, **kw: Any) -> None:
        row = dict.fromkeys(CSV_COLUMNS, "")
        row.update({"result_id": res.id, "test": res.test_name, "table": res.table or "", "section": section})
        row.update({k: ("" if v is None else v) for k, v in kw.items()})
        rows.append(row)

    for res in results:
        if res is None:
            continue
        add(res, "test", name=res.test_name, statistic_name=res.statistic_name, statistic=res.statistic, df=res.df,
            df2=res.df2, p_value=res.p_value, p_adjusted=res.p_adjusted, significant=res.significant,
            detail=res.correction or "")
        for e in res.effect_sizes:
            add(res, "effect_size", name=e.name, label=e.target, value=e.value, ci_low=e.ci_low, ci_high=e.ci_high,
                detail=e.magnitude or "")
        for p in res.parameters:
            add(res, "parameter", name=p.name, value=p.value, se=p.se, ci_low=p.ci_low, ci_high=p.ci_high,
                statistic=p.statistic, p_value=p.p_value)
        for row in res.anova_table:
            # repeated-measures rows carry the sphericity-corrected P that the result reports
            reported = row.get("p_reported", row.get("p"))
            detail = f"MS={_g(row.get('ms'))}"
            if row.get("p_reported") is not None and row.get("p") is not None and row.get("p_reported") != row.get("p"):
                detail += f"; uncorrected P={_g(row.get('p'))}"
            add(res, "anova", name=row.get("source", ""), value=row.get("ss"), statistic_name="F", statistic=row.get("f"),
                df=row.get("df"), p_value=reported, detail=detail)
        for c in res.comparisons:
            add(res, "comparison", name=c.id, group_a=c.group_a, group_b=c.group_b, label=c.label, value=c.estimate,
                se=c.se, ci_low=c.ci_low, ci_high=c.ci_high, statistic_name=c.statistic_name, statistic=c.statistic,
                df=c.df, p_value=c.p_value, p_adjusted=c.p_adjusted, significant=c.significant, detail=res.correction or "")
        for g, n in res.n.items():
            add(res, "n", name=g, value=n)
        for a in res.assumptions:
            add(res, "assumption", name=a.name, label=a.target, statistic=a.statistic, p_value=a.p_value,
                status=a.status, detail=a.detail)
        for w in res.warnings:
            add(res, "warning", detail=w)
        for w in res.data_notes:
            add(res, "data_note", detail=w)
    return rows


def stats_report_csv(results: Iterable[StatResult]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in stats_report_rows(results):
        writer.writerow(row)
    return buf.getvalue()


def write_stats_report(results: Iterable[StatResult], base: str | Path, title: str = "Statistics report",
                       spec: FigureSpec | None = None) -> dict[str, Path]:
    """Write ``<base>_stats.md`` and ``<base>_stats.csv`` next to an exported figure."""
    results = list(results)
    base = Path(base)
    if base.suffix.lower() in {".svg", ".pdf", ".png", ".tif", ".tiff", ".md", ".csv"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    md = base.parent / f"{base.name}_stats.md"
    cs = base.parent / f"{base.name}_stats.csv"
    md.write_text(stats_report_markdown(results, title, spec), encoding="utf-8")
    cs.write_text(stats_report_csv(results), encoding="utf-8", newline="")
    return {"markdown": md, "csv": cs}
