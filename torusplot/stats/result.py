"""Typed result objects returned by every analysis in :mod:`torusplot.stats`.

A :class:`StatResult` is the *only* source of statistical numbers in TorusPlot.
Figures, reports and AI features refer to results by ``id`` (and to individual
pairwise comparisons by ``Comparison.id``); they never carry their own p-values.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["pass", "warn", "fail", "info"]


def new_result_id() -> str:
    """Short unique ID such as ``sr_3f9a1c2b``."""
    return "sr_" + uuid.uuid4().hex[:8]


def _clean_float(value: Any) -> Any:
    """Convert numpy scalars to Python floats; keep None; map non-finite to None-safe floats."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return f


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class AssumptionCheck(_Model):
    """One assumption that was checked before/with a test, and how it came out."""

    name: str = Field(description="e.g. 'Normality (Shapiro-Wilk)'")
    target: str | None = Field(default=None, description="Group / term the check applies to")
    statistic: float | None = None
    p_value: float | None = None
    status: Status = Field(description="pass / warn / fail / info (badge shown in the UI)")
    detail: str = Field(default="", description="Plain-English explanation of the outcome")

    @field_validator("statistic", "p_value", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _clean_float(v)


class EffectSize(_Model):
    """An effect size with its confidence interval."""

    name: str = Field(description="e.g. \"Cohen's d\", \"Hedges' g\", 'eta squared'")
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    ci_method: str | None = Field(default=None, description="How the CI was computed")
    magnitude: str | None = Field(default=None, description="Conventional label (small/medium/large)")
    target: str | None = Field(default=None, description="Term or comparison it belongs to")

    @field_validator("value", "ci_low", "ci_high", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _clean_float(v)


class Parameter(_Model):
    """An estimated quantity: a mean difference, regression coefficient, EC50, median survival..."""

    name: str
    value: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    statistic: float | None = None
    p_value: float | None = None
    unit_note: str | None = None

    @field_validator("value", "se", "ci_low", "ci_high", "statistic", "p_value", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _clean_float(v)


class Comparison(_Model):
    """One pairwise comparison inside a family of multiple comparisons."""

    id: str = Field(description="'<result id>:c<k>'; what figure annotations point to")
    group_a: str
    group_b: str
    label: str | None = Field(default=None, description="Context, e.g. the row level for simple effects")
    cell_a: dict[str, str] | None = Field(
        default=None,
        description="Factorial designs: {factor name: level} locating side A (lets figures find the bars)",
    )
    cell_b: dict[str, str] | None = Field(default=None, description="Same for side B")
    estimate: float | None = Field(default=None, description="Difference (A - B) in means / mean ranks")
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    statistic_name: str | None = None
    statistic: float | None = None
    df: float | None = None
    p_value: float | None = Field(default=None, description="Unadjusted p value (if defined)")
    p_adjusted: float | None = Field(default=None, description="Multiplicity-adjusted p value")
    significant: bool | None = None
    effect_size: EffectSize | None = None

    @field_validator("estimate", "se", "ci_low", "ci_high", "statistic", "df", "p_value", "p_adjusted", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _clean_float(v)

    @property
    def p_for_display(self) -> float | None:
        """Adjusted p when available, otherwise the raw p."""
        return self.p_adjusted if self.p_adjusted is not None else self.p_value


class StatResult(_Model):
    """Everything one analysis produced."""

    id: str = Field(default_factory=new_result_id)
    analysis: str = Field(description="Registry key, e.g. 'ttest_unpaired'")
    test_name: str = Field(description="Human-readable name, e.g. \"Welch's t test\"")
    table: str | None = Field(default=None, description="Name of the data table analysed")
    why_chosen: str = Field(default="", description="Plain-English reason this test fits the design")
    assumptions: list[AssumptionCheck] = Field(default_factory=list)

    statistic_name: str | None = Field(default=None, description="'t', 'F', 'U', 'H', 'chi2' ...")
    statistic: float | None = None
    df: float | None = Field(default=None, description="Degrees of freedom (numerator for F)")
    df2: float | None = Field(default=None, description="Denominator degrees of freedom for F")
    p_value: float | None = None
    p_adjusted: float | None = Field(default=None, description="Adjusted p, when the test is one of a family")
    alpha: float = 0.05
    alternative: str = "two-sided"
    significant: bool | None = None

    effect_sizes: list[EffectSize] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    n: dict[str, int] = Field(default_factory=dict, description="Sample size per group")
    n_total: int | None = None
    descriptives: list[dict[str, Any]] = Field(default_factory=list, description="Per-group summary rows")
    anova_table: list[dict[str, Any]] = Field(default_factory=list, description="Source/SS/df/MS/F/p rows")
    comparisons: list[Comparison] = Field(default_factory=list)
    comparison_family: str | None = Field(default=None, description="What the family of comparisons is")
    correction: str | None = Field(default=None, description="Multiplicity method, e.g. 'Tukey HSD'")

    warnings: list[str] = Field(default_factory=list)
    data_notes: list[str] = Field(default_factory=list, description="Exclusions, missing values, replicate handling")
    interpretation: str = Field(default="", description="Plain-English reading of the result")
    methods_text: str = Field(default="", description="Sentence for a paper's methods section")
    citations: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict, description="Options the analysis was run with")
    extra: dict[str, Any] = Field(default_factory=dict, description="Curves, residuals, tables for plots")
    approximate_data: bool = Field(default=False, description="True if run on digitized/approximate data")

    @field_validator("statistic", "df", "df2", "p_value", "p_adjusted", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _clean_float(v)

    # ------------------------------------------------------------------ helpers
    def finalize(self) -> StatResult:
        """Assign comparison IDs, set ``significant`` flags, and return self."""
        for k, comp in enumerate(self.comparisons, start=1):
            comp.id = f"{self.id}:c{k}"
            p = comp.p_for_display
            if comp.significant is None and p is not None and not math.isnan(p):
                comp.significant = bool(p < self.alpha)
        if self.significant is None and self.p_value is not None and not math.isnan(self.p_value):
            p = self.p_adjusted if self.p_adjusted is not None else self.p_value
            self.significant = bool(p < self.alpha)
        if self.n and self.n_total is None:
            self.n_total = int(sum(self.n.values()))
        return self

    def comparison(self, group_a: str, group_b: str, label: str | None = None) -> Comparison | None:
        """Find a comparison by its two group names (order-insensitive)."""
        for comp in self.comparisons:
            if {comp.group_a, comp.group_b} == {group_a, group_b} and (label is None or comp.label == label):
                return comp
        return None

    def comparison_by_id(self, comparison_id: str) -> Comparison | None:
        for comp in self.comparisons:
            if comp.id == comparison_id:
                return comp
        return None

    def effect(self, name: str) -> EffectSize | None:
        for es in self.effect_sizes:
            if es.name.lower() == name.lower():
                return es
        return None

    def parameter(self, name: str) -> Parameter | None:
        for par in self.parameters:
            if par.name.lower() == name.lower():
                return par
        return None

    def worst_assumption_status(self) -> Status:
        order = {"info": 0, "pass": 1, "warn": 2, "fail": 3}
        worst: Status = "pass"
        for a in self.assumptions:
            if order[a.status] > order[worst]:
                worst = a.status
        return worst


def format_p(p: float | None, digits: int = 4) -> str:
    """Format a p value the way journals expect (``<0.0001`` floor, no ``p = 0``)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    floor = 10.0 ** (-digits)
    if p < floor:
        return f"<{floor:.{digits}f}"
    return f"{p:.{digits}f}"


def p_to_stars(p: float | None, style: str = "four") -> str:
    """Significance stars.

    ``style='four'``:  ns (>= 0.05), * (< 0.05), ** (< 0.01), *** (< 0.001), **** (< 0.0001)
    ``style='three'``: ns, *, **, *** (three-star maximum)
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.0001 and style == "four":
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
