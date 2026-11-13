"""Tests of the TorusPlot MCP server (all offline) plus a few shared helpers."""

from __future__ import annotations

import json
from typing import Any

from torusplot.stats.result import Comparison, StatResult

NANO_GROUPS = ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def strict_json(obj: Any) -> str:
    """Serialise the way a strict MCP client parses: fails on NaN / Infinity / non-JSON types."""
    return json.dumps(obj, allow_nan=False)


def synthetic_result(table: str = "np", groups: list[str] | None = None, with_comparisons: bool = True) -> StatResult:
    """A hand-built StatResult used as a TEST DOUBLE for annotation plumbing.

    The numbers are arbitrary fixtures, not statistics: tests that check real numbers
    go through ``run_analysis`` (see test_session_stats.py). It lets the figure tests run
    while the analysis modules are unavailable.
    """
    groups = groups or NANO_GROUPS
    p_values = [0.00002, 0.0004, 0.2, 0.03, 0.0071, 0.6]
    comparisons = []
    if with_comparisons:
        pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
        for k, (a, b) in enumerate(pairs):
            p = p_values[k % len(p_values)]
            comparisons.append(Comparison(id="", group_a=a, group_b=b, p_value=p / 3, p_adjusted=p))
    return StatResult(
        analysis="anova_oneway",
        test_name="One-way ANOVA (test double)",
        table=table,
        statistic_name="F",
        statistic=12.5,
        df=3,
        df2=28,
        p_value=0.00003,
        comparisons=comparisons,
        correction="Tukey HSD" if with_comparisons else None,
        methods_text="Test double: not a computed statistic.",
    ).finalize()
