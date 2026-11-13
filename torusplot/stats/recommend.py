"""``recommend_test``: inspect the data and design, then recommend an analysis in plain English.

The recommendation follows the decision path taught in biostatistics courses:

1. *What kind of table is it?* (one grouping variable, two, XY, counts, ...)
2. *How many groups / factors, and are the measurements paired or independent?*
3. *Do the data look roughly normal?* (Shapiro-Wilk on each group, on the paired
   differences, or on the model residuals)
4. *Are the group variances similar?* (Brown-Forsythe test)

It returns the registry key of the recommended analysis, ready-to-use parameters, the
checks that led there (with pass/warn/fail badges) and sensible alternatives. It never
runs the recommended test itself and never changes the data.

Notes on rigor
--------------
* Normality tests have little power with small samples and too much with huge ones;
  the explanation says so whenever n is small, instead of pretending the check is decisive.
* With very small groups a rank test cannot reach significance at all. The smallest
  two-sided exact P value of the Mann-Whitney test is 2 / C(n1 + n2, n1) (complete
  separation of the groups: 0.10 for 3 vs 3, 0.057 for 3 vs 4) and that of the Wilcoxon
  signed-rank test is 2 / 2^n (0.0625 for 5 pairs); whenever that minimum is not below
  0.05 the recommendation says so.
* Columns are recognised by whole words in their names (never by substrings: "Exposure"
  is not an "exp" replicate column) and, above all, by the structure of the data: a
  replicate column must hold several rows per group, a subject column exactly one row per
  repeated level. Repeated measurements are never recommended a factorial ANOVA that
  would treat the subject identifier as a fixed factor.
* Choosing between Student's and Welch's t test from a variance pre-test is the
  conventional path and is what is recommended here; Welch's test is always listed as the
  safe alternative because it does not need the equal-variance assumption
  (Welch 1947; Ruxton 2006 Behav Ecol 17:688-690).
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from ..data.tables import (
    ColumnTable,
    ContingencyTable,
    DataTable,
    GroupedTable,
    MultipleVariablesTable,
    XYTable,
)
from ._utils import StatsError, fmt
from .factorial import _no_variation, _unit_scale
from .result import AssumptionCheck

ALPHA_CHECK = 0.05


class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(description="Registry key of the alternative analysis")
    label: str
    when: str = Field(description="When you would choose it instead")
    params: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """What to run, with which options, and why."""

    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(description="Registry key, usable with run_analysis(table, analysis, **params)")
    label: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(description="Plain-English explanation of why this test fits")
    steps: list[str] = Field(default_factory=list, description="The decision path, one sentence per step")
    checks: list[AssumptionCheck] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    design: dict[str, Any] = Field(default_factory=dict, description="What was detected about the design")
    cautions: list[str] = Field(default_factory=list)

    def explain(self) -> str:
        """Multi-line 'Why this test?' text for the UI."""
        lines = [f"Recommended: {self.label}", "", self.reason, ""]
        lines += [f"{i}. {s}" for i, s in enumerate(self.steps, start=1)]
        if self.cautions:
            lines += ["", "Keep in mind:"] + [f"- {c}" for c in self.cautions]
        if self.alternatives:
            lines += ["", "Alternatives:"] + [f"- {a.label}: {a.when}" for a in self.alternatives]
        return "\n".join(lines)


# ------------------------------------------------------------------------- checks
def _shapiro(values: np.ndarray, target: str) -> AssumptionCheck:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    name = "Normality (Shapiro-Wilk)"
    if v.size < 3:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail=f"Only {v.size} value(s): normality cannot be tested.")
    if _no_variation(v):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="All values are identical: normality cannot be tested.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, p = stats.shapiro(v / _unit_scale(v))  # exact rescaling: the check is unit-free
    if p < ALPHA_CHECK:
        return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="fail",
                               detail=f"The values depart from a normal distribution (P = {fmt(p, 3)}).")
    detail = f"No evidence against normality (P = {fmt(p, 3)})."
    if v.size < 8:
        detail += f" With only n = {v.size} this test has little power, so it cannot confirm normality."
    return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="pass", detail=detail)


def _brown_forsythe(groups: dict[str, np.ndarray]) -> AssumptionCheck:
    name = "Equal variances (Brown-Forsythe)"
    usable = [g for g in groups.values() if g.size >= 2]
    if len(usable) < 2 or len(usable) < len(groups):
        return AssumptionCheck(name=name, status="warn", detail="Needs at least 2 values in every group.")
    if all(_no_variation(g) for g in usable):
        return AssumptionCheck(name=name, status="warn", detail="No variability within the groups.")
    unit = _unit_scale(np.concatenate(usable))  # exact common rescaling: the test is unit-free
    usable = [g / unit for g in usable]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.levene(*usable, center="median")
    if not np.isfinite(p):
        return AssumptionCheck(name=name, status="warn", detail="The test could not be computed.")
    sds = [float(g.std(ddof=1)) for g in usable]
    ratio = max(sds) / min(sds) if min(sds) > 0 else math.inf
    if p < ALPHA_CHECK:
        return AssumptionCheck(name=name, statistic=stat, p_value=p, status="fail",
                               detail=f"The spreads differ between groups (P = {fmt(p, 3)}; largest SD is {fmt(ratio, 3)}x the smallest).")
    return AssumptionCheck(name=name, statistic=stat, p_value=p, status="pass",
                           detail=f"The spreads are similar (P = {fmt(p, 3)}; SD ratio {fmt(ratio, 3)}).")


def _normal(checks: list[AssumptionCheck]) -> bool:
    return not any(c.status == "fail" for c in checks if c.name.startswith("Normality"))


def _small_n_note(ns: list[int]) -> str | None:
    if ns and min(ns) < 5:
        return (
            f"The smallest group has n = {min(ns)}. With so few values no test can check its own assumptions "
            "reliably; rely on what is known about this kind of measurement and report individual points."
        )
    return None


def _mann_whitney_min_p(n1: int, n2: int) -> float:
    """Smallest attainable two-sided exact P of the Mann-Whitney test: the two groups do
    not overlap at all, which has probability 2 / C(n1 + n2, n1) under the null hypothesis."""
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def _signed_rank_min_p(n: int) -> float:
    """Smallest attainable two-sided exact P of the Wilcoxon signed-rank test: all n
    differences have the same sign, probability 2 / 2^n under the null hypothesis."""
    return min(1.0, 2.0 / 2.0**n)


def _mann_whitney_power_note(n1: int, n2: int) -> str | None:
    p_min = _mann_whitney_min_p(n1, n2)
    if p_min < ALPHA_CHECK:
        return None
    sizes = f"n = {n1} per group" if n1 == n2 else f"n = {n1} and {n2}"
    return (f"With {sizes} a two-sided Mann-Whitney test cannot give P < 0.05: even when the two groups "
            f"do not overlap at all the exact P value is {fmt(p_min, 3)}. The test is recommended for "
            "validity, but it has no power here; collect more data if you can.")


def _signed_rank_power_note(n: int, what: str = "pairs") -> str | None:
    p_min = _signed_rank_min_p(n)
    if p_min < ALPHA_CHECK:
        return None
    return (f"With {n} {what} the Wilcoxon test cannot give a two-sided P below 0.05 "
            f"(the smallest possible P is {fmt(p_min, 3)}; at least 6 {what} are needed).")


# ---------------------------------------------------------------------- per table
def _recommend_column(table: ColumnTable, paired: bool | None, control: str | None,
                      groups: list[str] | None) -> Recommendation:
    data = table.groups()
    names = groups or [n for n in table.group_names if data[n].size > 0]
    missing = [n for n in names if n not in data]
    if missing:
        raise StatsError(f"Unknown group(s): {missing}. Available: {table.group_names}")
    data = {n: data[n] for n in names}
    k = len(names)
    if k == 0:
        raise StatsError("The table has no data yet.")
    ns = [int(v.size) for v in data.values()]
    design: dict[str, Any] = {"table": "column", "groups": k, "n": dict(zip(names, ns)), "paired": bool(paired)}
    cautions = [c for c in [_small_n_note(ns)] if c]
    gsel = {"groups": names} if groups else {}

    if k == 1:
        check = _shapiro(data[names[0]], names[0])
        if _normal([check]):
            return Recommendation(
                analysis="ttest_one_sample", label="One-sample t test", params={"group": names[0]},
                reason="There is one group, so the question is whether its mean differs from a hypothetical "
                       "value you choose (for example 0 or 100%). The values look compatible with a normal "
                       "distribution, so the one-sample t test applies.",
                steps=["One group of values: compare its mean with a hypothetical value.", check.detail],
                checks=[check], design=design, cautions=cautions,
                alternatives=[Alternative(analysis="wilcoxon_one_sample", label="Wilcoxon signed-rank test (one sample)",
                                          when="if the values are clearly not normal (skewed, outliers)", params={"group": names[0]}),
                              Alternative(analysis="descriptives", label="Descriptive statistics", when="to summarise the group")],
            )
        one_note = _signed_rank_power_note(ns[0], "values")
        return Recommendation(
            analysis="wilcoxon_one_sample", label="Wilcoxon signed-rank test (one sample)", params={"group": names[0]},
            reason="There is one group and its values are not normally distributed, so the median is compared "
                   "with a hypothetical value using the rank-based Wilcoxon signed-rank test.",
            steps=["One group of values: compare it with a hypothetical value.", check.detail],
            checks=[check], design=design, cautions=cautions + ([one_note] if one_note else []),
            alternatives=[Alternative(analysis="ttest_one_sample", label="One-sample t test",
                                      when="if you have reason to believe the population is normal", params={"group": names[0]})],
        )

    if k == 2:
        a, b = names
        pair = {"group_a": a, "group_b": b}
        if paired:
            frame = table.paired_frame([a, b])
            diff = frame[a].to_numpy(float) - frame[b].to_numpy(float)
            if diff.size < 2:
                raise StatsError("A paired comparison needs at least 2 rows that have a value in both columns.")
            check = _shapiro(diff, "paired differences")
            design["pairs"] = int(diff.size)
            steps = ["Two groups, and each row is the same subject measured twice (paired).",
                     "For paired data the assumption is about the differences between pairs. " + check.detail]
            if _normal([check]):
                return Recommendation(
                    analysis="ttest_paired", label="Paired t test", params=pair,
                    reason="Each row is one subject measured under both conditions, so the test looks at the "
                           "within-subject differences. Those differences look compatible with a normal "
                           "distribution, so the paired t test applies.",
                    steps=steps, checks=[check], design=design, cautions=cautions,
                    alternatives=[Alternative(analysis="wilcoxon_signed_rank", label="Wilcoxon matched-pairs signed-rank test",
                                              when="if the differences are skewed or contain outliers", params=pair)],
                )
            pair_note = _signed_rank_power_note(int(diff.size))
            extra = [pair_note] if pair_note else []
            return Recommendation(
                analysis="wilcoxon_signed_rank", label="Wilcoxon matched-pairs signed-rank test", params=pair,
                reason="The data are paired but the differences between pairs are not normally distributed, so "
                       "the rank-based Wilcoxon signed-rank test is recommended.",
                steps=steps, checks=[check], design=design, cautions=cautions + extra,
                alternatives=[Alternative(analysis="ttest_paired", label="Paired t test",
                                          when="if the non-normality is mild and n is reasonably large", params=pair)],
            )
        norm = [_shapiro(data[n], n) for n in names]
        var = _brown_forsythe(data)
        checks = [*norm, var]
        steps = ["Two groups of different subjects (independent / unpaired).",
                 "Normality of each group: " + " ".join(f"{c.target}: {c.detail}" for c in norm)]
        if paired is None and ns[0] == ns[1]:
            cautions.append("If each row is the same subject measured twice, the data are paired: "
                            "ask for the recommendation again with 'paired' switched on.")
        if not _normal(norm):
            mw_note = _mann_whitney_power_note(ns[0], ns[1])
            extra = [mw_note] if mw_note else []
            return Recommendation(
                analysis="mann_whitney", label="Mann-Whitney test", params=pair,
                reason="The two groups are independent and at least one of them is not normally distributed, so "
                       "the rank-based Mann-Whitney test is recommended. It asks whether values in one group "
                       "tend to be larger than in the other.",
                steps=steps + ["At least one group fails the normality check, so a rank-based test is used."],
                checks=checks, design=design, cautions=cautions + extra,
                alternatives=[Alternative(analysis="ttest_welch", label="Welch's t test",
                                          when="if the departure from normality is mild and groups are large (n > 30)", params=pair)],
            )
        steps.append("Equal spread: " + var.detail)
        if var.status == "fail":
            return Recommendation(
                analysis="ttest_welch", label="Welch's t test (unequal variances)", params=pair,
                reason="The two groups are independent and look normally distributed, but their spreads differ. "
                       "Welch's t test compares the means without assuming equal variances.",
                steps=steps, checks=checks, design=design, cautions=cautions,
                alternatives=[Alternative(analysis="mann_whitney", label="Mann-Whitney test",
                                          when="if you prefer a rank-based test that makes no normality assumption", params=pair)],
            )
        return Recommendation(
            analysis="ttest_unpaired", label="Unpaired t test", params=pair,
            reason="The two groups are independent, look compatible with a normal distribution and have similar "
                   "spread, so the unpaired (Student's) t test compares their means.",
            steps=steps, checks=checks, design=design, cautions=cautions,
            alternatives=[Alternative(analysis="ttest_welch", label="Welch's t test",
                                      when="a safe default that does not need equal variances; many statisticians prefer it routinely", params=pair),
                          Alternative(analysis="mann_whitney", label="Mann-Whitney test",
                                      when="if you doubt normality and cannot verify it"
                                           + ("" if _mann_whitney_min_p(ns[0], ns[1]) < ALPHA_CHECK else
                                              f" (but with these group sizes it cannot give P < 0.05: its "
                                              f"smallest possible P is {fmt(_mann_whitney_min_p(ns[0], ns[1]), 3)})"),
                                      params=pair)],
        )

    # three or more groups
    if paired:
        frame = table.paired_frame(names)
        if len(frame) < 2:
            raise StatsError("Repeated-measures analysis needs at least 2 rows with a value in every column.")
        centred = frame.to_numpy(float)
        resid = centred - centred.mean(axis=1, keepdims=True) - centred.mean(axis=0, keepdims=True) + centred.mean()
        check = _shapiro(resid.ravel(), "model residuals")
        design["subjects"] = int(len(frame))
        steps = [f"{k} conditions measured on the same subjects (each row is one subject): repeated measures.",
                 check.detail]
        if _normal([check]):
            return Recommendation(
                analysis="anova_rm_oneway", label="Repeated-measures one-way ANOVA", params=gsel,
                reason="Every subject was measured under all conditions, so subject-to-subject variability can be "
                       "removed with a repeated-measures ANOVA. Sphericity is checked and corrected "
                       "(Greenhouse-Geisser) automatically.",
                steps=steps, checks=[check], design=design, cautions=cautions,
                alternatives=[Alternative(analysis="friedman", label="Friedman test",
                                          when="if the residuals are not normal or the data are ranks/scores", params=gsel)],
            )
        return Recommendation(
            analysis="friedman", label="Friedman test", params=gsel,
            reason="Every subject was measured under all conditions but the residuals are not normally "
                   "distributed, so the rank-based Friedman test is recommended, followed by Dunn's comparisons.",
            steps=steps, checks=[check], design=design, cautions=cautions,
            alternatives=[Alternative(analysis="anova_rm_oneway", label="Repeated-measures one-way ANOVA",
                                      when="if the non-normality is mild, or after a log transform", params=gsel)],
        )
    # Residuals are standardised within each group before pooling: otherwise groups with
    # different spreads form a heavy-tailed mixture that fails a normality test even when
    # every group is perfectly normal (which would wrongly steer users away from Welch's ANOVA).
    parts = []
    for v in data.values():
        if v.size >= 3 and v.std(ddof=1) > 0:
            parts.append((v - v.mean()) / v.std(ddof=1))
        elif v.size:
            parts.append(v - v.mean())
    resid = np.concatenate(parts)
    norm_resid = _shapiro(resid, "residuals (all groups, standardised per group)")
    per_group = [_shapiro(v, n) for n, v in data.items()]
    for c in per_group:  # per-group results are informative only; the residual test decides
        if c.status == "fail":
            c.status = "warn"
    var = _brown_forsythe(data)
    checks = [norm_resid, *per_group, var]
    posthoc = {"posthoc": "dunnett", "control": control} if control else {}
    if control and control not in names:
        raise StatsError(f"Control group '{control}' is not one of {names}.")
    steps = [f"{k} independent groups defined by one factor.", "Normality of residuals: " + norm_resid.detail]
    follow = f"Dunnett's test against '{control}'" if control else "Tukey's test for all pairs"
    if not _normal([norm_resid]):
        return Recommendation(
            analysis="kruskal_wallis", label="Kruskal-Wallis test with Dunn's multiple comparisons",
            params={**gsel, **({"control": control} if control else {})},
            reason="There are three or more independent groups and the residuals are not normally distributed, "
                   "so the rank-based Kruskal-Wallis test is recommended, followed by Dunn's comparisons.",
            steps=steps, checks=checks, design=design, cautions=cautions,
            alternatives=[Alternative(analysis="anova_oneway", label="One-way ANOVA",
                                      when="after a transformation (often log) that makes the residuals normal", params={**gsel, **posthoc}),
                          Alternative(analysis="welch_anova", label="Welch's ANOVA", when="if groups are large and only variances differ", params=gsel)],
        )
    steps.append("Equal spread: " + var.detail)
    if var.status == "fail":
        return Recommendation(
            analysis="welch_anova", label="Welch's ANOVA with Games-Howell comparisons", params=gsel,
            reason="There are three or more independent, roughly normal groups whose spreads differ. Welch's "
                   "ANOVA does not assume equal variances; Games-Howell comparisons follow the same idea.",
            steps=steps, checks=checks, design=design, cautions=cautions,
            alternatives=[Alternative(analysis="brown_forsythe_anova", label="Brown-Forsythe ANOVA", when="an equally valid alternative for unequal variances", params=gsel),
                          Alternative(analysis="kruskal_wallis", label="Kruskal-Wallis test", when="if you prefer a rank-based test", params=gsel)],
        )
    return Recommendation(
        analysis="anova_oneway", label=f"One-way ANOVA with {follow}", params={**gsel, **posthoc},
        reason="There are three or more independent groups, the residuals look normal and the spreads are "
               f"similar, so a one-way ANOVA compares the means. {follow} then shows which groups differ while "
               "controlling the family-wise error rate.",
        steps=steps, checks=checks, design=design, cautions=cautions,
        alternatives=[Alternative(analysis="welch_anova", label="Welch's ANOVA", when="if you do not want to assume equal variances", params=gsel),
                      Alternative(analysis="kruskal_wallis", label="Kruskal-Wallis test", when="if the data are skewed or ordinal", params=gsel)],
    )


def _recommend_grouped(table: GroupedTable, repeated: bool | None) -> Recommendation:
    long = table.to_long()
    if long.empty:
        raise StatsError("The table has no data yet.")
    counts = long.groupby(["row", "column"], observed=False)["value"].count()
    n_min, n_max = int(counts.min()), int(counts.max())
    resid = long["value"] - long.groupby(["row", "column"], observed=False)["value"].transform("mean")
    design = {"table": "grouped", "rows": len(table.row_levels), "columns": len(table.column_levels),
              "replicates_per_cell": [n_min, n_max], "repeated": bool(repeated)}
    steps = [f"Two grouping variables: {table.row_label} ({len(table.row_levels)} levels, rows) and "
             f"{table.column_label} ({len(table.column_levels)} levels, column groups)."]
    cautions = []
    if n_min == 0:
        cautions.append("At least one row x column cell is empty; a factorial ANOVA needs data in every cell.")
    if n_max <= 1:
        cautions.append("There is only one value per cell, so the interaction cannot be separated from random "
                        "error: the ANOVA will be run without an interaction term.")
        check = _shapiro(np.array([]), "residuals")
    else:
        check = _shapiro(resid.to_numpy(float), "model residuals")
        steps.append("Normality of residuals: " + check.detail)
    if n_min != n_max and n_min > 0:
        cautions.append("Cell sizes are unequal (unbalanced design); Type III sums of squares are used.")
    if check.status == "fail":
        cautions.append("The residuals are not normal. There is no standard rank-based two-way test; "
                        "consider a log transform of the values, then re-check.")
    if repeated:
        return Recommendation(
            analysis="anova_mixed", label="Mixed-design (repeated-measures) two-way ANOVA", params={},
            reason=f"Each sub-column is one subject followed across the rows ({table.row_label}), and the column "
                   f"groups ({table.column_label}) contain different subjects. A mixed ANOVA separates "
                   "between-subject from within-subject variability, with sphericity correction.",
            steps=steps + ["Subjects are matched across rows (repeated measures)."], checks=[check], design=design,
            cautions=cautions,
            alternatives=[Alternative(analysis="anova_twoway", label="Ordinary two-way ANOVA",
                                      when="if every value comes from a different subject")],
        )
    return Recommendation(
        analysis="anova_twoway", label="Two-way ANOVA with Tukey's multiple comparisons",
        params={"posthoc": "tukey"},
        reason=f"The data are classified by two factors ({table.row_label} and {table.column_label}). A two-way "
               "ANOVA tests each factor and their interaction (whether the effect of one depends on the "
               "other); Tukey's comparisons then compare the column groups within each row.",
        steps=steps, checks=[check], design=design, cautions=cautions,
        alternatives=[Alternative(analysis="anova_mixed", label="Mixed-design (repeated-measures) ANOVA",
                                  when="if the same subjects were measured in every row (e.g. over time)")],
    )


def _recommend_xy(table: XYTable, goal: str | None) -> Recommendation:
    name = table.series_names[0]
    x, y = table.xy_points(name)
    if x.size < 3:
        raise StatsError("At least 3 XY points are needed.")
    ux = np.unique(x)
    positive = ux[ux > 0]
    wide = positive.size >= 4 and positive.max() / positive.min() >= 100
    log_like = np.all(ux > -15) and np.all(ux < 3) and ux.size >= 5 and "log" in table.x_name.lower()
    x_already_log = bool(log_like and not wide)
    if x_already_log:
        # X is log10(dose): one unit of X is one decade of dose, so the span is max - min
        decades = float(ux.max() - ux.min())
    else:
        decades = float(np.log10(positive.max() / positive.min())) if positive.size > 1 else 0.0
    design = {"table": "xy", "series": len(table.series_names), "points": int(x.size), "x_levels": int(ux.size),
              "x_spans_decades": decades, "x_is_log10": x_already_log}
    dose = goal == "dose_response" or (goal is None and (wide or log_like))
    alt_lin = Alternative(analysis="linear_regression", label="Simple linear regression",
                          when="if Y changes in a straight line with X", params={"series": name})
    alt_corr = Alternative(analysis="correlation", label="Correlation (Pearson or Spearman)",
                           when="if you only want to know how strongly X and Y go together", params={"series": name})
    if dose:
        model = "dose_response_4pl_logx" if x_already_log else "dose_response_4pl"
        cautions = []
        if x_already_log:
            span_step = (f"X is already on a log10 scale ('{table.x_name}' runs from {fmt(float(ux.min()), 3)} to "
                         f"{fmt(float(ux.max()), 3)}), so the doses span about {fmt(decades, 2)} decades "
                         "(orders of magnitude): dose-like.")
        elif positive.size > 1:
            span_step = (f"X values span about {fmt(decades, 2)} decades (orders of magnitude; from "
                         f"{fmt(float(positive.min()), 3)} to {fmt(float(positive.max()), 3)}): dose-like.")
        else:
            span_step = "X was analysed as a dose (concentration) series, as you asked."
        if wide or log_like:
            why_dose = ("X covers several orders of magnitude, which is typical of a dose or concentration "
                        "series. ")
        else:
            why_dose = "You asked for a dose-response analysis. "
        if ux.size < 5:
            cautions.append(f"Only {ux.size} dose levels: a four-parameter curve needs at least 5 to be well defined; "
                            "consider fixing Top/Bottom or using the three-parameter model.")
        return Recommendation(
            analysis="nonlinear_regression", label="Nonlinear regression: four-parameter dose-response (EC50/IC50)",
            params={"series": name, "model": model},
            reason=why_dose
                   + "A four-parameter logistic (sigmoidal) curve estimates the top and bottom plateaus, the "
                     "EC50/IC50 and the Hill slope with confidence intervals.",
            steps=[span_step,
                   "Individual replicates are fitted (not just the means) so the uncertainty is honest."],
            design=design, cautions=cautions,
            alternatives=[Alternative(analysis="nonlinear_regression", label="Three-parameter dose-response (Hill slope fixed at 1)",
                                      when="with few dose levels or a poorly defined slope",
                                      params={"series": name, "model": model.replace("4pl", "3pl")}), alt_lin, alt_corr],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, *_ = stats.linregress(x, y) if np.ptp(x) > 0 else (0.0, float(np.mean(y)), 0, 0, 0)
    check = _shapiro(y - (intercept + slope * x), "regression residuals")
    return Recommendation(
        analysis="linear_regression", label="Simple linear regression", params={"series": name},
        reason="X is a measured or controlled quantity and Y is the response. Linear regression finds the best "
               "straight line, tests whether the slope differs from zero and gives confidence and prediction bands.",
        steps=["XY data on a linear X scale.", "Residuals from a straight line: " + check.detail], checks=[check],
        design=design,
        cautions=["Look at the residual plot: a curve in the residuals means a straight line is the wrong model."],
        alternatives=[alt_corr, Alternative(analysis="nonlinear_regression", label="Nonlinear regression",
                                            when="if the relationship is curved (saturation, decay, sigmoid)", params={"series": name})],
    )


def _recommend_contingency(table: ContingencyTable) -> Recommendation:
    counts = table.counts.to_numpy(float)
    r, c = counts.shape
    total = counts.sum()
    if total <= 0:
        raise StatsError("The table has no counts yet.")
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    low = float((expected < 5).mean())
    design = {"table": "contingency", "rows": r, "columns": c, "total": float(total), "min_expected": float(expected.min())}
    check = AssumptionCheck(
        name="Expected counts (Cochran's rule)", statistic=float(expected.min()),
        status="fail" if (low > 0.2 or expected.min() < 1) else "pass",
        detail=f"Smallest expected count is {fmt(expected.min(), 3)}; {low * 100:.0f}% of cells are below 5.",
    )
    if (r, c) == (2, 2):
        return Recommendation(
            analysis="fisher_exact", label="Fisher's exact test (with odds ratio and relative risk)", params={},
            reason="A 2 x 2 table of counts is best analysed with Fisher's exact test: its P value is exact for any "
                   "sample size, so it is valid even when some cells are small.",
            steps=["Counts in a 2 x 2 table.", check.detail], checks=[check], design=design,
            alternatives=[Alternative(analysis="chi_square", label="Chi-square test",
                                      when="with large counts (all expected counts at least 5); gives almost the same answer")],
        )
    trend_alt = [Alternative(
        analysis="chi_square_trend", label="Chi-square test for trend",
        when="if the categories of the longer side are ordered (doses, stages) and you want to know whether "
             "the proportion rises or falls steadily",
    )] if 2 in (r, c) else []
    if check.status == "fail":
        # Cochran's rule fails: the chi-square approximation is not trustworthy, and the exact
        # (Fisher-Freeman-Halton) test is available for r x c tables.
        return Recommendation(
            analysis="fisher_exact", label=f"Fisher's exact test ({r} x {c} table, Fisher-Freeman-Halton)",
            params={},
            reason=f"The table has {r} rows and {c} columns of counts, but too many expected counts are small "
                   "for the chi-square approximation (Cochran's rule: no expected count below 1 and at most "
                   "20% below 5). Fisher's exact test, in its extension to larger tables, gives a P value "
                   "that is valid whatever the counts.",
            steps=[f"Counts in a {r} x {c} table.", check.detail,
                   "Cochran's rule fails, so an exact test is used instead of the chi-square approximation."],
            checks=[check], design=design,
            cautions=["Some expected counts are small, so the chi-square approximation may be inaccurate; "
                      "the exact test avoids it. Merging sparse categories is another option if it makes "
                      "scientific sense.",
                      "For large tables the exact P value is estimated by simulation (the result says so)."],
            alternatives=[Alternative(analysis="chi_square", label="Chi-square test of independence",
                                      when="only after merging sparse categories so that the expected counts "
                                           "are large enough"), *trend_alt],
        )
    return Recommendation(
        analysis="chi_square", label="Chi-square test of independence", params={},
        reason=f"The table has {r} rows and {c} columns of counts. The chi-square test asks whether the "
               "distribution across columns differs between rows.",
        steps=[f"Counts in a {r} x {c} table.", check.detail], checks=[check], design=design, cautions=[],
        alternatives=[Alternative(analysis="fisher_exact", label="Fisher's exact test (r x c)",
                                  when="if you prefer an exact P value (it matters when counts are small)"),
                      *trend_alt],
    )


def _find(columns: list[str], *words: str) -> str | None:
    for col in columns:
        low = col.lower()
        if any(w in low for w in words):
            return col
    return None


# Whole-word name hints for a column that identifies the experimental UNIT (never matched as
# substrings: "Exposure", "Expression", "Runtime" or "Acid" are ordinary factors). A name is
# only a hint; what the column is used for is decided from the structure of the data.
_REPLICATE_TOKENS = frozenset({
    "replicate", "replicates", "rep", "reps", "biorep", "bioreps", "experiment", "experiments",
    "expt", "exp", "batch", "batches", "run", "runs", "plate", "plates", "donor", "donors",
    "litter", "litters", "culture", "cultures", "prep", "preps", "preparation",
})
_SUBJECT_TOKENS = frozenset({
    "subject", "subjects", "subj", "id", "ids", "animal", "animals", "mouse", "mice", "rat", "rats",
    "patient", "patients", "participant", "participants", "individual", "volunteer", "person",
})
_UNIT_TOKENS = _REPLICATE_TOKENS | _SUBJECT_TOKENS


def _name_tokens(name: str) -> set[str]:
    """Lower-case words of a column name ('Mouse_ID' / 'MouseID' -> {'mouse', 'id'})."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))
    return set(re.findall(r"[a-z]+|[0-9]+", spaced.lower()))


def _is_unit_name(name: str) -> bool:
    tokens = _name_tokens(name)
    if tokens & _UNIT_TOKENS:
        return True
    # 'subjectid', 'mouseid': a unit word directly followed by 'id'
    return any(t.endswith("id") and t[:-2] in _UNIT_TOKENS for t in tokens)


def _integer_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return bool(values.size) and bool(np.all(values == np.round(values)))


def _unit_structure(frame: pd.DataFrame, unit: str, factors: list[str], value: str) -> dict[str, Any] | None:
    """How the rows are arranged with respect to a candidate unit column.

    Returns ``None`` when nothing can be said (no complete rows, a single unit). Otherwise:
    ``within`` = factors that change inside a unit (at least half of the units see two or more
    levels), ``between`` = factors that are constant inside every unit, ``other`` = the rest,
    ``rows_per_cell`` = (median, max) number of rows per unit x within-level combination and
    ``n_units``.
    """
    work = frame[[unit, *factors, value]].dropna()
    if work.empty:
        return None
    n_units = int(work[unit].nunique())
    if n_units < 2:
        return None
    within, between, other = [], [], []
    for f in factors:
        per_unit = work.groupby(unit, observed=True)[f].nunique()
        if bool((per_unit == 1).all()):
            between.append(f)
        elif float((per_unit >= 2).mean()) >= 0.5:
            within.append(f)
        else:
            other.append(f)
    cells = work.groupby([unit, *within], observed=True).size()
    return {"within": within, "between": between, "other": other, "n_units": n_units,
            "rows_per_cell": (float(cells.median()), int(cells.max())), "rows": int(len(work))}


def _unit_design(table: MultipleVariablesTable, goal: str | None) -> dict[str, Any] | None:
    """Find a column that identifies the experimental unit (subject, animal, experiment) and
    classify the design from the structure of the data.

    ``kind`` is one of

    * ``"nested"``: several rows per unit (and per condition): technical replicates inside
      biological replicates;
    * ``"mixed"``: one row per unit x level of one factor, units nested in another factor;
    * ``"rm_oneway"``: one row per unit x level of the only factor that varies in a unit;
    * ``"rm_twoway"``: one row per unit x combination of two factors;
    * ``"rm_unsupported"``: repeated measures in a layout no registered analysis covers;
    * ``"id_only"``: one row per unit and nothing varies inside a unit: the column is just a
      row label and must not be used as a factor.
    """
    frame = table.frame(dropna=False)
    cats, numeric = list(table.categorical_columns), list(table.numeric_columns)
    named = [c for c in table.data.columns
             if _is_unit_name(c) and (c in cats or (_integer_like(frame[c]) and len(numeric) >= 2))]
    candidates = [(c, True) for c in named]
    if not candidates and len(cats) >= 3:
        # no telling name: a text column can still be recognised by structure alone, but only in
        # the unmistakable split-plot layout (see below)
        candidates = [(c, False) for c in sorted(cats, key=lambda c: -frame[c].nunique())]
    for unit, by_name in candidates:
        outcomes = [c for c in numeric if c != unit]
        if not outcomes:
            continue
        # numeric columns with a few repeated levels (time points, doses) may be factors, as
        # long as another numeric column is left to be the outcome
        discrete = [c for c in outcomes
                    if 2 <= frame[c].nunique() <= 12 and frame[c].nunique() * 2 <= frame[c].notna().sum()]
        if len(discrete) == len(outcomes):
            # every numeric column has few levels: keep the richest one as the outcome
            keep = max(reversed(discrete), key=lambda c: frame[c].nunique())
            discrete = [c for c in discrete if c != keep]
        value_pool = [c for c in outcomes if c not in discrete]
        factor_numeric = discrete
        # the outcome of a repeated / replicated design varies inside a unit; a numeric column that
        # is constant within every unit (age, body weight at entry) is a subject attribute
        varying = [c for c in value_pool
                   if float((frame.groupby(unit, observed=True)[c].nunique() >= 2).mean()) >= 0.5]
        value = (varying or value_pool)[0]
        factors = [c for c in cats if c != unit]
        info = _unit_structure(frame, unit, factors + factor_numeric, value)
        if info is None:
            continue
        # a numeric column only counts as a factor when it is a repeated (within-unit) level
        numeric_within = [f for f in info["within"] if f in factor_numeric]
        if len(numeric_within) != len(factor_numeric):
            info = _unit_structure(frame, unit, factors + numeric_within, value)
            if info is None:
                continue
        within, between = info["within"], info["between"]
        median_rows, max_rows = info["rows_per_cell"]
        out = {"unit": unit, "by_name": by_name, "value": value, **info}
        if not by_name:
            # structure-only detection: units nested in one factor, crossed with exactly one
            # other, one value per unit x level, and at least two units per group
            groups = frame[between[0]].nunique() if between else 0
            if (len(within) == 1 and len(between) >= 1 and not info["other"] and max_rows == 1
                    and info["n_units"] >= 2 * groups):
                return {**out, "kind": "mixed"}
            continue
        if goal == "nested":
            if within or between:
                return {**out, "kind": "nested"}
            continue
        if within and median_rows < 2:
            # (typically) one value per unit and repeated level: repeated measures
            if info["other"] or len(within) > 2 or (len(within) == 2 and between):
                return {**out, "kind": "rm_unsupported"}
            if len(within) == 2:
                return {**out, "kind": "rm_twoway"}
            return {**out, "kind": "mixed" if between else "rm_oneway"}
        if median_rows >= 2 and (within or between):
            return {**out, "kind": "nested"}
        if not within and median_rows < 2:
            return {**out, "kind": "id_only"}
    return None


def _recommend_units(found: dict[str, Any], design: dict[str, Any]) -> Recommendation | None:
    """Recommendation for a design with an identified unit column (``None`` = not handled)."""
    unit, value, within, between = found["unit"], found["value"], found["within"], found["between"]
    how = ("its name" if found["by_name"] else "the layout of the data (each of its levels belongs to one "
           "group and has exactly one value per repeated level)")
    design = {**design, "unit_column": unit, "unit_design": found["kind"], "within": within,
              "between": between, "units": found["n_units"]}
    kind = found["kind"]
    duplicates = ([f"Some levels of '{unit}' have more than one value at the same repeated level. Each subject "
                   "must contribute exactly one value per level: average or remove the duplicates first (the "
                   "analysis will say which subject is affected)."]
                  if kind in {"mixed", "rm_twoway"} and found["rows_per_cell"][1] > 1 else [])
    if kind == "nested":
        group = (within or between)[0]
        extra_factors = [f for f in (within + between) if f != group]
        cautions = []
        if extra_factors:
            cautions.append(
                f"This analysis compares the levels of one grouping column ('{group}'). The table also has "
                f"{', '.join(repr(f) for f in extra_factors)}: analyse one factor at a time, or average the "
                "technical replicates first and run a two-way ANOVA on the replicate means."
            )
        return Recommendation(
            analysis="nested_analysis", label="Replicate-aware analysis (statistics on biological-replicate means)",
            params={"value": value, "group": group, "replicate": unit},
            reason=f"'{unit}' identifies independent biological replicates that each contain several technical "
                   "measurements. Technical replicates are not independent, so they are averaged within each "
                   "biological replicate and the test is run on those means (n = number of replicates).",
            steps=[f"Value: {value}; group: {group}; biological replicate: {unit}.",
                   f"'{unit}' was recognised by {how}, and there are several rows for each replicate "
                   f"(typically {found['rows_per_cell'][0]:g} per replicate and group), as expected for "
                   "technical replicates.",
                   "Averaging first avoids pseudoreplication, which would make P values far too small."],
            design=design, cautions=cautions,
            alternatives=[Alternative(analysis="anova_twoway", label="Two-way ANOVA",
                                      when=f"if '{unit}' is a real experimental factor, not a replicate")],
        )
    if kind == "mixed":
        b, w = between[0], within[0]
        cautions = []
        if len(between) > 1:
            cautions.append(
                f"Only one between-subjects factor can be analysed; '{b}' was chosen. The table also has "
                f"{', '.join(repr(f) for f in between[1:])}."
            )
        return Recommendation(
            analysis="anova_mixed", label="Mixed-design ANOVA (repeated measures by group)",
            params={"value": value, "subject": unit, "between": b, "within": w},
            reason=f"Each level of '{unit}' is one subject that belongs to a single group of '{b}' and was "
                   f"measured once at every level of '{w}'. Repeated measurements of the same subject are not "
                   "independent, so the subject column must not be used as an ordinary factor: a mixed-design "
                   f"ANOVA tests '{b}' between subjects, '{w}' within subjects and their interaction, with a "
                   "sphericity correction.",
            steps=[f"Value: {value}; subject: {unit}; between-subjects factor: {b}; within-subjects factor: {w}.",
                   f"'{unit}' was recognised as the subject identifier by {how}: {found['n_units']} subjects, "
                   f"one value per subject and level of '{w}'."],
            design=design, cautions=cautions + duplicates,
            alternatives=[Alternative(analysis="anova_twoway", label="Ordinary two-way ANOVA",
                                      when="only if every value comes from a different subject (then the "
                                           "subject column is not needed)",
                                      params={"value": value, "factor_a": b, "factor_b": w})],
        )
    if kind == "rm_twoway":
        a, b = within
        return Recommendation(
            analysis="anova_rm_twoway", label="Two-way repeated-measures ANOVA",
            params={"value": value, "subject": unit, "within_a": a, "within_b": b},
            reason=f"Every level of '{unit}' (one subject) was measured once at every combination of '{a}' and "
                   f"'{b}', so both factors are compared within subjects. Each effect is tested against its own "
                   "interaction with subjects, with a sphericity correction.",
            steps=[f"Value: {value}; subject: {unit}; repeated factors: {a} and {b}.",
                   f"'{unit}' was recognised as the subject identifier by {how}: {found['n_units']} subjects, "
                   "one value per subject and combination."],
            design=design, cautions=duplicates,
        )
    if kind == "rm_oneway":
        w = within[0]
        levels = int(found.get("n_within_levels", 0))
        return Recommendation(
            analysis="nested_analysis",
            label=("Paired t test on matched subjects" if levels == 2
                   else "Repeated-measures (randomized-block) one-way ANOVA on matched subjects"),
            params={"value": value, "group": w, "replicate": unit, "paired": "yes"},
            reason=f"Every level of '{unit}' (one subject) was measured once under each level of '{w}', so the "
                   "conditions are compared within subjects. The replicate-aware analysis with matched "
                   "replicates runs exactly that: a paired t test for two conditions, a repeated-measures "
                   "(randomized-block) one-way ANOVA with paired comparisons for three or more.",
            steps=[f"Value: {value}; condition: {w}; subject: {unit} (matched across conditions).",
                   f"'{unit}' was recognised as the subject identifier by {how}: {found['n_units']} subjects, "
                   f"one value per subject and level of '{w}'."],
            design=design,
            cautions=["This blocked ANOVA does not test or correct for sphericity. For Mauchly's test and "
                      "Greenhouse-Geisser corrected P values, arrange the data as a Column table (one row per "
                      "subject, one column per condition) and run the repeated-measures one-way ANOVA."],
        )
    if kind == "rm_unsupported":
        return Recommendation(
            analysis="descriptives", label="Descriptive statistics (no ready-made test for this design)", params={},
            reason=f"'{unit}' identifies subjects that were measured repeatedly "
                   f"(within-subject factors: {', '.join(within) or 'none'}; between-subject factors: "
                   f"{', '.join(between) or 'none'}). Repeated measurements are not independent, so a factorial "
                   "ANOVA with the subject column as a factor would be wrong, and no built-in analysis covers "
                   "this combination of repeated and grouping factors.",
            steps=[f"Value: {value}; subject: {unit}.",
                   "Simplify the design: analyse one repeated factor at a time (mixed-design ANOVA), average "
                   "over one of the factors, or analyse each group separately."],
            design=design,
            cautions=["Do not run a three-way ANOVA on these data: it would treat repeated measurements of "
                      "the same subject as independent values."],
        )
    return None


def _recommend_mv(table: MultipleVariablesTable, goal: str | None) -> Recommendation:
    numeric, cats = table.numeric_columns, table.categorical_columns
    cols = list(table.data.columns)
    design = {"table": "multiple_variables", "numeric": numeric, "categorical": cats, "rows": int(len(table.data))}
    time_col = _find(numeric, "time", "day", "week", "month", "survival")
    event_col = _find(numeric, "event", "status", "death", "died", "censor")
    if goal == "survival" or (goal is None and time_col and event_col):
        if not (time_col and event_col):
            raise StatsError("Survival analysis needs a time column and an event column (1 = event, 0 = censored).")
        group = cats[0] if cats else None
        params = {"time": time_col, "event": event_col, **({"group": group} if group else {})}
        return Recommendation(
            analysis="survival", label="Kaplan-Meier survival analysis with log-rank test", params=params,
            reason="The table has a time column and an event (1 = event, 0 = censored) column. Kaplan-Meier curves "
                   "handle censored subjects correctly, and the log-rank test compares the curves between groups.",
            steps=[f"Time column: {time_col}; event column: {event_col}" + (f"; groups: {group}." if group else "; no grouping column.")],
            design=design,
            cautions=["Check the event coding: 1 must mean the event happened, 0 that the subject was censored."],
        )
    # ---- is there a column that identifies the experimental unit (subject, animal, experiment)?
    found = _unit_design(table, goal) if numeric else None
    if goal == "nested" and found is None:
        # asked for explicitly: fall back on a looser (substring) look at the column names
        rep_col = _find(cats, "replicate", "experiment", "exp", "batch", "animal", "mouse", "donor", "run", "day")
        others = [c for c in cats if c != rep_col]
        if rep_col and others and numeric:
            info = _unit_structure(table.frame(dropna=False), rep_col, others, numeric[0])
            if info is not None:
                if not (info["within"] or info["between"]):
                    info = {**info, "between": others[:1]}
                found = {"unit": rep_col, "by_name": True, "value": numeric[0], **info, "kind": "nested"}
        if found is None:
            raise StatsError(
                "Replicate-aware analysis needs a value column, a group column and a replicate column "
                "(named like 'Experiment', 'Replicate', 'Animal' or 'Donor')."
            )
    if found is not None and found["kind"] in {"rm_oneway", "mixed", "rm_twoway"}:
        found["n_within_levels"] = int(table.frame(dropna=False)[found["within"][0]].nunique())
    if found is not None and goal in (None, "nested", "compare") and found["kind"] != "id_only":
        rec = _recommend_units(found, design)
        if rec is not None:
            return rec
    if found is not None and found["kind"] == "id_only":
        # one row per unit and nothing repeated: the column is a row label, never a factor or predictor
        cats = [c for c in cats if c != found["unit"]]
        numeric = [c for c in numeric if c != found["unit"]]
        cols = [c for c in cols if c != found["unit"]]
        design = {**design, "id_column": found["unit"]}
    replicate_hint = [Alternative(
        analysis="nested_analysis", label="Replicate-aware analysis",
        when="if one of the grouping columns identifies independent experiments or animals (biological "
             "replicates) rather than an experimental factor",
    )]
    if goal in (None, "compare") and numeric and len(cats) >= 3:
        return Recommendation(
            analysis="anova_threeway", label="Three-way ANOVA",
            params={"value": numeric[0], "factor_a": cats[0], "factor_b": cats[1], "factor_c": cats[2]},
            reason="One measured value is classified by three grouping variables; a three-way ANOVA tests each "
                   "factor and their interactions.",
            steps=[f"Value: {numeric[0]}; factors: {cats[0]}, {cats[1]}, {cats[2]}."], design=design,
            cautions=["Three-way designs need replication in every cell and are hard to interpret; plot the data first.",
                      "Every value must come from a different subject. If one of the columns identifies subjects "
                      "that were measured repeatedly, use a repeated-measures or mixed-design ANOVA instead."],
            alternatives=replicate_hint,
        )
    if goal in (None, "compare") and numeric and len(cats) == 2:
        return Recommendation(
            analysis="anova_twoway", label="Two-way ANOVA",
            params={"value": numeric[0], "factor_a": cats[0], "factor_b": cats[1]},
            reason="One measured value is classified by two grouping variables; a two-way ANOVA tests each factor "
                   "and whether the effect of one depends on the other (interaction).",
            steps=[f"Value: {numeric[0]}; factors: {cats[0]} and {cats[1]}."], design=design,
            alternatives=replicate_hint,
        )
    if goal in (None, "compare", "ancova") and len(numeric) >= 2 and len(cats) == 1:
        return Recommendation(
            analysis="ancova", label="ANCOVA (group comparison adjusted for a covariate)",
            params={"value": numeric[0], "factor": cats[0], "covariates": numeric[1:2]},
            reason=f"There is one grouping variable ({cats[0]}) and more than one numeric column. ANCOVA compares "
                   f"the groups on {numeric[0]} after adjusting for {numeric[1]}.",
            steps=[f"Outcome: {numeric[0]}; group: {cats[0]}; covariate: {numeric[1]}."], design=design,
            alternatives=[Alternative(analysis="multiple_regression", label="Multiple regression", when="to model the outcome from several predictors")],
        )
    if len(numeric) >= 3 or goal == "regression":
        if len(numeric) < 2:
            raise StatsError("Regression needs at least two numeric columns.")
        return Recommendation(
            analysis="multiple_regression", label="Multiple linear regression",
            params={"y": numeric[0], "predictors": [c for c in cols if c != numeric[0]]},
            reason=f"Several variables were measured on each row. Multiple regression models {numeric[0]} from the "
                   "other columns and shows which predictors matter after accounting for the others.",
            steps=[f"Outcome: {numeric[0]}; predictors: all other columns."], design=design,
            alternatives=[Alternative(analysis="correlation_matrix", label="Correlation matrix", when="to explore pairwise relationships first")],
        )
    if len(numeric) == 2:
        return Recommendation(
            analysis="correlation", label="Correlation", params={"x": numeric[0], "y": numeric[1]},
            reason="Two numeric variables were measured on each row; correlation quantifies how strongly they go together.",
            steps=[f"Variables: {numeric[0]} and {numeric[1]}."], design=design,
        )
    if numeric:
        return Recommendation(
            analysis="descriptives", label="Descriptive statistics", params={},
            reason="There is not enough structure to choose a test automatically (one numeric column, no usable "
                   "grouping). Start with descriptive statistics, or convert the data to a Column table.",
            steps=["One numeric column."], design=design,
        )
    raise StatsError("The table has no numeric column to analyse.")


# ------------------------------------------------------------------------ public
def recommend_test(
    table: DataTable,
    *,
    paired: bool | None = None,
    control: str | None = None,
    groups: list[str] | None = None,
    goal: str | None = None,
) -> Recommendation:
    """Recommend an analysis for ``table`` and explain why.

    Parameters
    ----------
    paired:
        ``True`` if rows (Column table) or replicate sub-columns (Grouped table) are the
        same subject measured repeatedly; ``False``/``None`` for independent groups.
    control:
        Name of a control group. With 3+ groups this selects Dunnett's comparisons.
    groups:
        Subset of a Column table's groups to compare (default: all non-empty columns).
    goal:
        Optional hint: ``"dose_response"``, ``"relationship"`` (XY tables);
        ``"survival"``, ``"nested"``, ``"compare"``, ``"regression"``, ``"ancova"``
        (Multiple-variables tables).
    """
    if isinstance(table, ColumnTable):
        return _recommend_column(table, paired, control, groups)
    if isinstance(table, GroupedTable):
        return _recommend_grouped(table, paired)
    if isinstance(table, XYTable):
        return _recommend_xy(table, goal)
    if isinstance(table, ContingencyTable):
        return _recommend_contingency(table)
    if isinstance(table, MultipleVariablesTable):
        return _recommend_mv(table, goal)
    raise StatsError(f"No recommendation is available for table type '{type(table).__name__}'.")
