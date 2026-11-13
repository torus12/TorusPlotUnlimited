"""Replicate-aware ("SuperPlot") analysis of nested data.

No published worked example with reported results is available for this workflow, so
validation rests on (sources in tests/fixtures/sources/survival_contingency_nested.md):
* cross-implementation: TorusPlot's tests on replicate means vs scipy on hand-aggregated
  data, the blocked ANOVA vs a statsmodels OLS model with a block factor, and the
  replicate-means test vs a linear mixed model (balanced designs);
* from-first-principles computations written out in this file;
* property-based edge cases (hypothesis).
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from torusplot.data.tables import MultipleVariablesTable
from torusplot.stats import StatsError, multiplicity, registry, run_analysis
from torusplot.stats import effect_sizes as es
from torusplot.stats import nested as ns


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    """The module under test registered itself on import; do not require sibling analysis
    modules (written by other people, possibly not importable yet) to load."""
    monkeypatch.setattr(registry, "_loaded", True)


def _make(seed=0, groups=("Control", "Drug"), shifts=(0.0, 0.7), reps=("exp1", "exp2", "exp3", "exp4"),
          rep_effects=(0.0, 0.9, -0.6, 0.4), cells=30, noise=1.0, distinct_reps=False, interaction=0.25):
    """Cells nested in experiments; ``distinct_reps`` gives every group its own replicate labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi, (g, shift) in enumerate(zip(groups, shifts)):
        for ri, (r, effect) in enumerate(zip(reps, rep_effects)):
            label = f"{g[:1]}{ri + 1}" if distinct_reps else r
            centre = 10 + shift + effect + rng.normal(0, interaction)
            n_cells = cells if isinstance(cells, int) else cells[(gi + ri) % len(cells)]
            for v in rng.normal(centre, noise, n_cells):
                rows.append({"condition": g, "experiment": label, "intensity": v})
    return pd.DataFrame(rows)


def _means(df, summary="mean"):
    agg = df.groupby(["condition", "experiment"], sort=False)["intensity"].agg(summary).reset_index()
    return {g: sub.set_index("experiment")["intensity"] for g, sub in agg.groupby("condition", sort=False)}


def _run(df, **kwargs):
    kwargs.setdefault("mixed_model", False)
    return ns.nested_analysis(df, "intensity", "condition", "experiment", **kwargs).finalize()


# ================================================================ replicate means
def test_replicate_means_table():
    df = _make(1, cells=(5, 8, 3))
    out = ns.replicate_means(df, "condition", "experiment", "intensity")
    assert list(out.columns) == ["group", "replicate", "mean", "n_technical", "sd_technical"]
    assert out["group"].tolist() == ["Control"] * 4 + ["Drug"] * 4
    assert out["replicate"].tolist() == ["exp1", "exp2", "exp3", "exp4"] * 2
    ref = df.groupby(["condition", "experiment"], sort=False)["intensity"].agg(["mean", "std", "size"]).reset_index()
    assert np.allclose(out["mean"], ref["mean"], rtol=1e-12)
    assert np.allclose(out["sd_technical"], ref["std"], rtol=1e-12)
    assert out["n_technical"].tolist() == ref["size"].tolist()
    med = ns.replicate_means(df, "condition", "experiment", "intensity", summary="median")
    assert "median" in med.columns and "mean" not in med.columns
    assert np.allclose(med["median"], df.groupby(["condition", "experiment"], sort=False)["intensity"].median())
    with pytest.raises(StatsError, match="summary"):
        ns.replicate_means(df, "condition", "experiment", "intensity", summary="mode")


def test_replicate_means_single_values_missing_rows_and_labels():
    df = pd.DataFrame({
        "g": ["a", "a", "a", "b", "b", None, "b", "a"],
        "r": [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, np.nan, 2.0],
        "y": [1.0, 3.0, 5.0, 2.0, np.nan, 9.0, 4.0, "oops"],
    })
    out = ns.replicate_means(df, "g", "r", "y")
    assert out.to_dict("records")[0] == {"group": "a", "replicate": "1", "mean": 2.0, "n_technical": 2,
                                         "sd_technical": pytest.approx(math.sqrt(2.0))}
    single = out[(out["group"] == "a") & (out["replicate"] == "2")].iloc[0]
    assert single["n_technical"] == 1 and math.isnan(single["sd_technical"])
    assert len(out) == 3  # (b, 2) had no usable value; rows with missing labels are ignored


# ================================================================ two groups
def test_unpaired_welch_matches_scipy_on_aggregated_means():
    df = _make(2, distinct_reps=True)
    means = _means(df)
    a, b = means["Control"].to_numpy(), means["Drug"].to_numpy()
    res = _run(df)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res.test_name == "Welch's t test on replicate means" and res.statistic_name == "t"
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.df == pytest.approx(ref.df, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert res.n == {"Control": 4, "Drug": 4} and res.n_total == 8  # n = biological replicates
    diff = res.parameter("Mean difference (Control - Drug)")
    ci = ref.confidence_interval(0.95)
    assert diff.value == pytest.approx(a.mean() - b.mean(), rel=1e-12)
    assert (diff.ci_low, diff.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    assert diff.se == pytest.approx(math.sqrt(a.var(ddof=1) / 4 + b.var(ddof=1) / 4), rel=1e-12)
    g = es.cohen_d_independent(a, b, hedges=True)
    assert res.effect_sizes[0].name == "Hedges' g" and res.effect_sizes[0].value == pytest.approx(g.value, rel=1e-12)
    assert res.effect_sizes[0].ci_low == pytest.approx(g.ci_low, rel=1e-8)
    assert res.effect_sizes[1].name == "Cohen's d"
    assert res.extra["design"] == "unpaired" and "do not share the same replicate labels" in res.why_chosen
    assert "Welch" in res.methods_text and "Lord et al. 2020" in res.methods_text
    desc = res.descriptives[0]
    assert desc["n"] == 4 and desc["n_technical_total"] == 120 and desc["n_technical_per_replicate"] == [30] * 4
    assert desc["mean"] == pytest.approx(a.mean()) and desc["sd"] == pytest.approx(a.std(ddof=1))
    assert desc["sem"] == pytest.approx(stats.sem(a))
    mean_par = res.parameter("Mean of replicate means: Control")
    lo, hi = stats.t.interval(0.95, 3, loc=a.mean(), scale=stats.sem(a))
    assert (mean_par.ci_low, mean_par.ci_high) == pytest.approx((lo, hi), rel=1e-10)
    names = [c.name for c in res.assumptions]
    assert names.count("Normality (Shapiro-Wilk)") == 2 and "Equal variances" in names
    shapiro = next(c for c in res.assumptions if c.name.startswith("Normality") and c.target == "Drug")
    assert shapiro.p_value == pytest.approx(stats.shapiro(b).pvalue, rel=1e-10)


def test_student_t_option():
    df = _make(3, distinct_reps=True)
    means = _means(df)
    a, b = means["Control"].to_numpy(), means["Drug"].to_numpy()
    res = _run(df, test="t")
    ref = stats.ttest_ind(a, b, equal_var=True)
    assert res.test_name == "Unpaired t test on replicate means"
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10) and res.df == 6
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    ci = ref.confidence_interval(0.95)
    diff = res.parameter("Mean difference (Control - Drug)")
    assert (diff.ci_low, diff.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    bf = next(c for c in res.assumptions if c.name.startswith("Equal variances"))
    assert bf.p_value == pytest.approx(stats.levene(a, b, center="median").pvalue, rel=1e-10)
    # a 99% interval is wider
    wide = _run(df, test="t", ci_level=0.99).parameter("Mean difference (Control - Drug)")
    assert wide.ci_low < diff.ci_low and wide.ci_high > diff.ci_high


def test_paired_by_replicate_is_detected_automatically():
    df = _make(4)
    means = _means(df)
    a, b = means["Control"].to_numpy(), means["Drug"].to_numpy()
    res = _run(df)
    ref = stats.ttest_rel(a, b)
    assert res.test_name == "Paired t test on replicate means"
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10) and res.df == 3
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    ci = ref.confidence_interval(0.95)
    diff = res.parameter("Mean difference (Control - Drug)")
    assert (diff.ci_low, diff.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    dz = es.cohen_d_paired(a, b)
    assert res.effect_sizes[0].name == "Cohen's d_z (paired)"
    assert res.effect_sizes[0].value == pytest.approx(dz.value, rel=1e-12)
    assert res.effect_sizes[1].name == "Hedges' g_z (paired)"
    assert "same replicate labels (exp1, exp2, exp3, exp4) occur in every group" in res.why_chosen
    assert any("matched automatically" in n for n in res.data_notes)
    assert res.extra["design"] == "paired" and res.extra["matched_replicates"] == ["exp1", "exp2", "exp3", "exp4"]
    shapiro = next(c for c in res.assumptions if c.name.startswith("Normality"))
    assert shapiro.target == "Control - Drug"
    assert shapiro.p_value == pytest.approx(stats.shapiro(a - b).pvalue, rel=1e-10)
    # switching pairing off gives Welch's test on the same means
    unpaired = _run(df, paired=False)
    assert unpaired.test_name.startswith("Welch") and "pairing switched off" in unpaired.why_chosen
    assert unpaired.statistic == pytest.approx(stats.ttest_ind(a, b, equal_var=False).statistic, rel=1e-10)
    # explicit pairing gives the paired test without the "automatic" note
    explicit = _run(df, paired=True)
    assert explicit.statistic == pytest.approx(res.statistic) and not any("automatically" in n for n in explicit.data_notes)
    assert _run(df, test="paired_t").statistic == pytest.approx(res.statistic)


def test_paired_with_partially_matching_replicates():
    df = _make(5)
    df = df[~((df["condition"] == "Drug") & (df["experiment"] == "exp4"))]
    auto = _run(df)  # labels differ between groups -> not paired automatically
    assert auto.test_name.startswith("Welch") and auto.n == {"Control": 4, "Drug": 3}
    forced = _run(df, paired=True)
    means = _means(df)
    common = ["exp1", "exp2", "exp3"]
    ref = stats.ttest_rel(means["Control"][common], means["Drug"][common])
    assert forced.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert forced.n == {"Control": 3, "Drug": 3}
    assert any("exp4 do not occur in every group" in w for w in forced.warnings)
    assert forced.descriptives[0]["n_biological"] == 4 and forced.descriptives[0]["n"] == 3
    with pytest.raises(StatsError, match="shared by all groups"):
        _run(_make(5, distinct_reps=True), paired=True)


# ============================================================ three or more groups
def _three(seed=6, **kwargs):
    return _make(seed, groups=("Control", "Low", "High"), shifts=(0.0, 0.5, 1.2), **kwargs)


def test_one_way_anova_on_replicate_means():
    df = _three(distinct_reps=True)
    means = {g: m.to_numpy() for g, m in _means(df).items()}
    res = _run(df)
    ref = stats.f_oneway(*means.values())
    assert res.test_name == "One-way ANOVA on replicate means" and res.statistic_name == "F"
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert (res.df, res.df2) == (2, 9)
    table = {row["source"]: row for row in res.anova_table}
    assert table["condition"]["ss"] + table["Residual"]["ss"] == pytest.approx(table["Total"]["ss"], rel=1e-12)
    allv = np.concatenate(list(means.values()))
    assert table["Total"]["ss"] == pytest.approx(((allv - allv.mean()) ** 2).sum(), rel=1e-10)
    assert table["condition"]["f"] == pytest.approx(ref.statistic, rel=1e-10)
    eta = res.effect("eta squared")
    assert eta.value == pytest.approx(table["condition"]["ss"] / table["Total"]["ss"], rel=1e-12)
    assert eta.ci_low is not None and res.effect("omega squared").value <= eta.value
    # pairwise Welch t tests, Holm-adjusted
    assert len(res.comparisons) == 3 and res.correction == "Holm"
    assert "3 pairwise comparisons among 3 groups" in res.comparison_family
    raw = []
    for comp in res.comparisons:
        a, b = means[comp.group_a], means[comp.group_b]
        pair = stats.ttest_ind(a, b, equal_var=False)
        assert comp.statistic == pytest.approx(pair.statistic, rel=1e-10)
        assert comp.p_value == pytest.approx(pair.pvalue, rel=1e-8)
        assert comp.df == pytest.approx(pair.df, rel=1e-10)
        assert comp.estimate == pytest.approx(a.mean() - b.mean(), rel=1e-10)
        ci = pair.confidence_interval(0.95)
        assert (comp.ci_low, comp.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
        assert comp.effect_size.name == "Hedges' g" and comp.effect_size.target == f"{comp.group_a} vs {comp.group_b}"
        raw.append(pair.pvalue)
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(multiplicity.adjust_pvalues(raw, "holm"))
    assert all(c.significant == (c.p_adjusted < 0.05) for c in res.comparisons)
    assert [c.name for c in res.assumptions].count("Normality (Shapiro-Wilk)") == 3
    assert "F(2, 9)" in res.interpretation and "Holm" in res.interpretation


def test_blocked_anova_matches_statsmodels_block_model():
    df = _three()
    res = _run(df)
    assert res.test_name.startswith("Randomized-block (repeated-measures) one-way ANOVA")
    agg = df.groupby(["condition", "experiment"], sort=False)["intensity"].mean().reset_index()
    fit = smf.ols("intensity ~ C(condition) + C(experiment)", agg).fit()
    aov = sm.stats.anova_lm(fit, typ=2)
    table = {row["source"]: row for row in res.anova_table}
    assert res.statistic == pytest.approx(aov.loc["C(condition)", "F"], rel=1e-9)
    assert res.p_value == pytest.approx(aov.loc["C(condition)", "PR(>F)"], rel=1e-7)
    assert (res.df, res.df2) == (2, 6)
    assert table["condition"]["ss"] == pytest.approx(aov.loc["C(condition)", "sum_sq"], rel=1e-9)
    assert table["experiment (blocks)"]["ss"] == pytest.approx(aov.loc["C(experiment)", "sum_sq"], rel=1e-9)
    assert table["experiment (blocks)"]["f"] == pytest.approx(aov.loc["C(experiment)", "F"], rel=1e-9)
    assert table["Residual"]["ss"] == pytest.approx(aov.loc["Residual", "sum_sq"], rel=1e-9)
    assert sum(table[s]["ss"] for s in ("condition", "experiment (blocks)", "Residual")) == pytest.approx(
        table["Total"]["ss"], rel=1e-12)
    pes = res.effect("partial eta squared")
    assert pes.value == pytest.approx(table["condition"]["ss"] / (table["condition"]["ss"] + table["Residual"]["ss"]))
    assert table["condition"]["partial_eta_sq"] == pytest.approx(pes.value)
    # pairwise paired t tests
    means = _means(df)
    raw = []
    for comp in res.comparisons:
        pair = stats.ttest_rel(means[comp.group_a].to_numpy(), means[comp.group_b].to_numpy())
        assert comp.statistic == pytest.approx(pair.statistic, rel=1e-10) and comp.df == 3
        assert comp.p_value == pytest.approx(pair.pvalue, rel=1e-8)
        assert comp.effect_size.name.startswith("Cohen's d_z")
        raw.append(pair.pvalue)
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(multiplicity.adjust_pvalues(raw, "holm"))
    assert any(c.name == "Sphericity" and c.status == "info" for c in res.assumptions)
    assert "paired t tests" in res.comparison_family
    # unpaired analysis of the same data is the plain one-way ANOVA
    plain = _run(df, paired=False)
    assert plain.statistic == pytest.approx(stats.f_oneway(*[m.to_numpy() for m in means.values()]).statistic)
    assert _run(df, test="anova").statistic == pytest.approx(plain.statistic)
    assert _run(df, test="rm_anova").statistic == pytest.approx(res.statistic)


def test_blocked_anova_with_two_columns_is_the_paired_t_test():
    rng = np.random.default_rng(7)
    matrix = rng.normal(5, 1, (6, 2))
    res = ns._blocked_anova(matrix)
    t = stats.ttest_rel(matrix[:, 0], matrix[:, 1])
    assert res["f"] == pytest.approx(t.statistic**2, rel=1e-10)
    assert res["p_value"] == pytest.approx(t.pvalue, rel=1e-8)
    assert (res["df1"], res["df2"]) == (1, 5)


# ============================================================ warnings and messages
def test_pseudoreplication_warning_shows_the_naive_p_value():
    df = _make(8, shifts=(0.0, 0.35), cells=60, distinct_reps=True)
    res = _run(df)
    cells_a = df.loc[df["condition"] == "Control", "intensity"]
    cells_b = df.loc[df["condition"] == "Drug", "intensity"]
    naive = stats.ttest_ind(cells_a, cells_b, equal_var=False).pvalue
    warning = next(w for w in res.warnings if "Pseudoreplicated, do not report" in w)
    assert ns._fmt_p_any(naive) in warning and ns._fmt_p_any(res.p_value) in warning
    assert "n = 240 vs 240" in warning and "n = 4 vs 4" in warning
    assert naive < res.p_value and "overstates the evidence" in warning
    # the naive P value lives only in the warning text, never in the machine-readable result
    payload = json.loads(res.model_dump_json())
    assert "naive" not in json.dumps(payload["extra"]).lower()
    assert payload["p_value"] == pytest.approx(res.p_value)
    assert any("480 measurements" in n and "8 biological replicate means" in n for n in res.data_notes)
    # three groups: the naive analysis is a one-way ANOVA on all cells
    df3 = _three(9, distinct_reps=True)
    res3 = _run(df3)
    naive3 = stats.f_oneway(*[sub["intensity"] for _, sub in df3.groupby("condition", sort=False)]).pvalue
    warning3 = next(w for w in res3.warnings if "Pseudoreplicated" in w)
    assert ns._fmt_p_any(naive3) in warning3 and "one-way ANOVA" in warning3


def test_no_technical_replication_means_no_pseudoreplication_warning():
    df = _make(10, cells=1, distinct_reps=True)
    res = _run(df)
    assert not any("Pseudoreplicated" in w for w in res.warnings)
    assert any("single measurement" in n for n in res.data_notes)
    assert res.extra["replicate_means"][0]["sd_technical"] is None
    assert res.extra["replicate_means"][0]["n_technical"] == 1
    mixed = ns.nested_analysis(df, "intensity", "condition", "experiment")
    assert mixed.extra["mixed_model"]["p_value"] is None and "Not fitted" in mixed.extra["mixed_model"]["note"]


def test_few_biological_replicates():
    two = _make(11, reps=("e1", "e2"), rep_effects=(0.0, 0.5), distinct_reps=True)
    res = _run(two)
    assert res.n == {"Control": 2, "Drug": 2}
    assert any("Fewer than 3 biological replicates" in w for w in res.warnings)
    assert all(c.status == "warn" for c in res.assumptions if c.name.startswith("Normality"))
    student = _run(two, test="t")
    assert next(c for c in student.assumptions if c.name.startswith("Equal variances")).status == "warn"
    one = _make(12, reps=("e1",), rep_effects=(0.0,), cells=200)
    with pytest.raises(StatsError, match=r"only one biological replicate \(n = 1\)"):
        _run(one)
    lopsided = pd.concat([_make(13, distinct_reps=True).query("condition == 'Control'"),
                          one.query("condition == 'Drug'")])
    with pytest.raises(StatsError, match="Drug have only one biological replicate"):
        _run(lopsided)


def test_extra_replicate_rows_for_superplots_and_median_summary():
    df = _make(14, cells=(4, 6))
    res = _run(df, summary="median")
    rows = res.extra["replicate_means"]
    assert len(rows) == 8 and set(rows[0]) == {"group", "replicate", "mean", "median", "value", "n_technical",
                                              "sd_technical"}
    ref = df.groupby(["condition", "experiment"], sort=False)["intensity"].agg(["mean", "median", "std", "size"])
    for row in rows:
        r = ref.loc[(row["group"], row["replicate"])]
        assert row["mean"] == pytest.approx(r["mean"]) and row["median"] == pytest.approx(r["median"])
        assert row["value"] == row["median"] and row["n_technical"] == r["size"]
        assert row["sd_technical"] == pytest.approx(r["std"])
    assert res.extra["summary"] == "median"
    med = _means(df, "median")
    assert res.statistic == pytest.approx(stats.ttest_rel(med["Control"], med["Drug"]).statistic, rel=1e-10)
    assert "median" in res.methods_text and res.parameter("Mean of replicate medians: Control") is not None
    assert res.extra["columns"] == {"value": "intensity", "group": "condition", "replicate": "experiment"}


def test_assumption_failures_are_reported():
    # strongly skewed replicate means in one group -> Shapiro-Wilk fails
    rows = []
    for g, values in (("A", [1.0, 1.1, 1.05, 1.02, 1.08, 30.0]), ("B", [2.0, 2.5, 3.1, 3.6, 4.2, 4.9])):
        for i, v in enumerate(values):
            rows += [{"condition": g, "experiment": f"{g}{i}", "intensity": v + d} for d in (-0.01, 0.01)]
    res = _run(pd.DataFrame(rows), test="t")
    normal_a = next(c for c in res.assumptions if c.name.startswith("Normality") and c.target == "A")
    assert normal_a.status == "fail" and "log-transformed" in normal_a.detail
    assert next(c for c in res.assumptions if c.name.startswith("Normality") and c.target == "B").status == "pass"
    # very different spreads -> Brown-Forsythe fails and points to Welch
    rows = []
    for g, values in (("A", [5.0, 5.01, 4.99, 5.02, 4.98, 5.0]), ("B", [1.0, 9.0, 3.0, 12.0, -4.0, 7.5])):
        for i, v in enumerate(values):
            rows.append({"condition": g, "experiment": f"{g}{i}", "intensity": v})
    res = _run(pd.DataFrame(rows), test="t")
    bf = next(c for c in res.assumptions if c.name.startswith("Equal variances"))
    assert bf.status == "fail" and "Welch" in bf.detail


def test_degenerate_data_gives_warnings_not_crashes():
    flat = pd.DataFrame({"condition": ["a"] * 6 + ["b"] * 6, "experiment": ["1", "1", "2", "2", "3", "3"] * 2,
                         "intensity": [5.0] * 12})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for kwargs in ({}, {"paired": False}, {"test": "t"}):
            res = _run(flat, **kwargs)
            assert res.statistic is None and res.p_value is None and res.significant is None
            assert any("no variability" in w for w in res.warnings)
            assert all(e.value is None for e in res.effect_sizes)
            assert "could not be evaluated" in res.interpretation
            json.loads(res.model_dump_json())
        # constant, non-zero paired differences are just as degenerate
        shifted = flat.copy()
        shifted["intensity"] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        res = _run(shifted)
        assert res.test_name.startswith("Paired") and res.statistic is None
        assert res.parameter("Mean difference (a - b)").value == pytest.approx(-1.0)
        # three flat groups
        flat3 = pd.concat([flat, flat.assign(condition=lambda d: d["condition"].map({"a": "c", "b": "d"}))])
        for kwargs in ({}, {"paired": False}):
            res = _run(flat3, **kwargs)
            assert res.statistic is None and all(c.p_value is None for c in res.comparisons)
            assert all(c.effect_size.value is None for c in res.comparisons)
            json.loads(res.model_dump_json())
        # mixed model on degenerate data must not raise either
        res = ns.nested_analysis(flat, "intensity", "condition", "experiment")
        assert "mixed_model" in res.extra


def test_input_validation():
    df = _make(15)
    with pytest.raises(StatsError, match="value column 'nope' is not in the data"):
        ns.nested_analysis(df, "nope", "condition", "experiment")
    with pytest.raises(StatsError, match="three different columns"):
        ns.nested_analysis(df, "intensity", "condition", "condition")
    with pytest.raises(StatsError, match="does not contain numbers"):
        ns.nested_analysis(df.assign(intensity="high"), "intensity", "condition", "experiment")
    with pytest.raises(StatsError, match="No complete rows"):
        ns.nested_analysis(df.assign(intensity=np.nan), "intensity", "condition", "experiment")
    with pytest.raises(StatsError, match="Only one group"):
        _run(df[df["condition"] == "Drug"])
    with pytest.raises(StatsError, match="test must be"):
        _run(df, test="mannwhitney")
    with pytest.raises(StatsError, match="paired must be"):
        _run(df, paired="yes")
    with pytest.raises(StatsError, match="needs matched replicates"):
        _run(df, test="paired_t", paired=False)
    with pytest.raises(StatsError, match="unpaired test"):
        _run(df, test="welch", paired=True)
    with pytest.raises(StatsError, match="three or more groups"):
        _run(df, test="anova")
    with pytest.raises(StatsError, match="compares two groups"):
        _run(_three(), test="welch")
    with pytest.raises(StatsError, match="alpha"):
        _run(df, alpha=1.5)
    with pytest.raises(StatsError, match="ci_level"):
        _run(df, ci_level=95)
    with pytest.raises(TypeError):
        ns.nested_analysis([[1, 2, 3]], "a", "b", "c")
    # missing values are dropped and reported
    holes = df.copy()
    holes.loc[[0, 5, 9], "intensity"] = np.nan
    res = _run(holes)
    assert any("3 row(s)" in n for n in res.data_notes) and res.descriptives[0]["n_technical_total"] == 117


# ============================================================ mixed-model cross-check
def test_mixed_model_agrees_with_replicate_means_in_balanced_designs():
    # unmatched, balanced: the REML Wald chi-square equals the squared pooled-variance t
    df = _make(16, distinct_reps=True, interaction=0.6)
    res = ns.nested_analysis(df, "intensity", "condition", "experiment", test="t")
    mm = res.extra["mixed_model"]
    assert mm["converged"] is True and mm["df"] == 1 and mm["df_design"] == 6
    assert mm["statistic"] == pytest.approx(res.statistic**2, rel=2e-2)
    assert mm["p_value_design_df"] == pytest.approx(res.p_value, rel=5e-2)
    assert mm["p_value"] < res.p_value  # the large-sample P value is anti-conservative
    assert "Cross-check only" in mm["note"] and "replicate within group" in mm["model"]
    # matched, balanced: equals the squared paired t when the interaction variance is estimable
    df = _make(17, interaction=0.8)
    res = ns.nested_analysis(df, "intensity", "condition", "experiment")
    mm = res.extra["mixed_model"]
    assert res.test_name.startswith("Paired")
    assert mm["statistic"] == pytest.approx(res.statistic**2, rel=5e-2)
    assert mm["p_value_design_df"] == pytest.approx(res.p_value, rel=1e-1)
    assert "replicate:group" in mm["model"]
    # three groups: chi-square / (k - 1) is the ANOVA F
    df = _three(18, distinct_reps=True, interaction=0.6)
    res = ns.nested_analysis(df, "intensity", "condition", "experiment")
    mm = res.extra["mixed_model"]
    assert mm["df"] == 2 and mm["statistic"] / 2 == pytest.approx(res.statistic, rel=2e-2)
    assert "mixed_model" not in _run(df).extra  # switched off


def test_mixed_model_failure_is_contained(monkeypatch):
    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(smf, "mixedlm", boom)
    res = ns.nested_analysis(_make(19), "intensity", "condition", "experiment")
    mm = res.extra["mixed_model"]
    assert mm["converged"] is False and mm["p_value"] is None and "could not be fitted (LinAlgError)" in mm["note"]
    assert res.p_value is not None


# ======================================================================= registry
def _table(df=None, **rename):
    df = _make(20) if df is None else df
    df = df.rename(columns=rename)
    return MultipleVariablesTable("Cells", df)


def test_registry_defaults_with_name_matching():
    info = registry.REGISTRY["nested_analysis"]
    assert info.category == "Replicate-aware (nested)" and info.tables == ("multiple_variables",)
    assert all(p.help for p in info.params)
    assert "technical replicates" in next(p for p in info.params if p.name == "replicate").help
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run_analysis(_table(), "nested_analysis")
    assert res.analysis == "nested_analysis" and res.table == "Cells"
    assert res.extra["columns"] == {"value": "intensity", "group": "condition", "replicate": "experiment"}
    assert res.test_name.startswith("Paired") and res.n == {"Control": 4, "Drug": 4}
    assert res.interpretation and res.why_chosen and len(res.assumptions) >= 1 and res.significant is not None
    assert any("Group column chosen automatically: 'condition'" in n for n in res.data_notes)
    assert any("Biological replicate column chosen automatically: 'experiment'" in n for n in res.data_notes)
    assert any("Value column chosen automatically: 'intensity'" in n for n in res.data_notes)
    payload = json.loads(res.model_dump_json())
    assert len(payload["extra"]["replicate_means"]) == 8 and "mixed_model" in payload["extra"]
    assert res.params["paired"] == "auto"


def test_registry_options_fallbacks_and_errors():
    assert run_analysis(_table(), "nested_analysis", paired="no").test_name.startswith("Welch")
    assert run_analysis(_table(), "nested_analysis", paired="yes").test_name.startswith("Paired")
    assert run_analysis(_table(), "nested_analysis", paired=False, test="t").test_name.startswith("Unpaired")
    assert run_analysis(_table(), "nested_analysis", paired=None, summary="median").extra["summary"] == "median"
    with pytest.raises(StatsError, match="paired must be"):
        run_analysis(_table(), "nested_analysis", paired="sometimes")
    # no recognisable names: first numeric + first two text columns
    plain = _table(condition="colA", experiment="colB", intensity="colC")
    res = run_analysis(plain, "nested_analysis")
    assert res.extra["columns"] == {"value": "colC", "group": "colA", "replicate": "colB"}
    # numeric replicate column found by name; the value column is not confused with it
    df = _make(21)
    df["experiment"] = df["experiment"].str.replace("exp", "").astype(int)
    df = df[["experiment", "intensity", "condition"]]
    res = run_analysis(MultipleVariablesTable("Cells", df), "nested_analysis")
    assert res.extra["columns"] == {"value": "intensity", "group": "condition", "replicate": "experiment"}
    assert res.extra["matched_replicates"] == ["1", "2", "3", "4"]
    # explicit columns
    res = run_analysis(plain, "nested_analysis", value="colC", group="colA", replicate="colB")
    assert res.n == {"Control": 4, "Drug": 4}
    with pytest.raises(StatsError, match="not in the table"):
        run_analysis(plain, "nested_analysis", group="missing")
    only_numbers = MultipleVariablesTable("n", pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    with pytest.raises(StatsError, match="group"):
        run_analysis(only_numbers, "nested_analysis")
    one_text = MultipleVariablesTable("n", pd.DataFrame({"a": [1.0, 2.0], "who": ["x", "y"]}))
    with pytest.raises(StatsError, match="biological replicate"):
        run_analysis(one_text, "nested_analysis")
    no_values = MultipleVariablesTable("n", pd.DataFrame({"g": ["x", "y"], "r": ["p", "q"]}))
    with pytest.raises(StatsError, match="No numeric column"):
        run_analysis(no_values, "nested_analysis")


# ================================================================ property-based
_value = st.one_of(st.floats(-1e3, 1e3, allow_nan=False, width=32), st.sampled_from([0.0, 1.0, 5.0]))


@st.composite
def _nested_frames(draw):
    n_groups = draw(st.integers(2, 3))
    matched = draw(st.booleans())
    rows = []
    for g in range(n_groups):
        n_reps = draw(st.integers(1, 4))
        for r in range(n_reps):
            label = f"r{r}" if matched else f"g{g}r{r}"
            for v in draw(st.lists(_value, min_size=1, max_size=4)):
                rows.append({"condition": f"g{g}", "experiment": label, "intensity": float(v)})
    if draw(st.booleans()):
        rows.append({"condition": "g0", "experiment": "r0" if matched else "g0r0", "intensity": float("nan")})
    return pd.DataFrame(rows)


@settings(max_examples=60, deadline=None)
@given(_nested_frames(), st.sampled_from(["mean", "median"]), st.sampled_from([None, False]))
def test_property_nested_analysis_is_robust(df, summary, paired):
    reps = df.dropna().groupby("condition")["experiment"].nunique()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if (reps < 2).any() or len(reps) < 2:
            with pytest.raises(StatsError):
                _run(df, summary=summary, paired=paired)
            return
        res = _run(df, summary=summary, paired=paired)
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    assert (res.statistic is None) == (res.p_value is None)
    if res.statistic is None:
        assert any("no variability" in w for w in res.warnings)
    for comp in res.comparisons:
        assert comp.p_value is None or 0.0 <= comp.p_value <= comp.p_adjusted <= 1.0
    assert all(1 <= n <= 4 for n in res.n.values())
    text = res.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    # n is the number of biological replicates, never the number of measurements
    assert sum(res.n.values()) <= df.dropna().groupby(["condition", "experiment"]).ngroups
    # location / scale invariance of the test statistic
    if res.statistic is not None and abs(res.statistic) < 1e6:
        moved = _run(df.assign(intensity=df["intensity"] * 3.0 + 7.0), summary=summary, paired=paired)
        if moved.statistic is not None:
            assert moved.statistic == pytest.approx(res.statistic, rel=1e-4, abs=1e-6)


@settings(max_examples=12, deadline=None)
@given(_nested_frames())
def test_property_mixed_model_never_breaks_the_analysis(df):
    reps = df.dropna().groupby("condition")["experiment"].nunique()
    if (reps < 2).any() or len(reps) < 2:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ns.nested_analysis(df, "intensity", "condition", "experiment").finalize()
    mm = res.extra["mixed_model"]
    assert mm["p_value"] is None or 0.0 <= mm["p_value"] <= 1.0
    assert isinstance(mm["note"], str) and mm["note"]
    json.loads(res.model_dump_json())


# =============================================================================
# Audit part 2: unit of measurement, replicate counts in the text (D2)
# =============================================================================
def _nested_unit_free(res):
    return {
        "stat": res.statistic, "p": res.p_value, "df": res.df, "df2": res.df2,
        "effects": [(e.value, e.ci_low, e.ci_high) for e in res.effect_sizes],
        "assumption_p": [a.p_value for a in res.assumptions],
        "comp": [(c.statistic, c.p_value, c.p_adjusted, c.effect_size.value if c.effect_size else None)
                 for c in res.comparisons],
        "table": [(r["f"], r["p"]) for r in res.anova_table],
        "mixed": res.extra.get("mixed_model", {}).get("statistic"),
    }


def _nested_in_units(res, scale):
    return {
        "desc": [(d["mean"] / scale, d["sd"] / scale, d["sem"] / scale, d["ci_low"] / scale, d["ci_high"] / scale)
                 for d in res.descriptives],
        "params": [(p.value / scale, None if p.se is None else p.se / scale, p.ci_low / scale, p.ci_high / scale)
                   for p in res.parameters],
        "comp": [(c.estimate / scale, c.se / scale, c.ci_low / scale, c.ci_high / scale) for c in res.comparisons],
        "ss": [r["ss"] / scale / scale for r in res.anova_table],
        "reps": [(r["mean"] / scale, r["value"] / scale, r["sd_technical"] / scale)
                 for r in res.extra["replicate_means"]],
    }


def _deep_same(a, b, rel=1e-8, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _deep_same(a[key], b[key], rel, f"{path}/{key}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b)):
            _deep_same(u, v, rel, f"{path}[{i}]")
    elif a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} vs {b!r}"
    else:
        assert a == pytest.approx(b, rel=rel, abs=1e-10), f"{path}: {a!r} vs {b!r}"


@pytest.mark.parametrize("scale", [1e-13, 1.0, 1e9, 1e150])
@pytest.mark.parametrize("design", ["paired_t", "welch", "t", "rm_anova", "anova"])
def test_nested_analysis_does_not_depend_on_the_unit_of_measurement(scale, design):
    three = design in {"rm_anova", "anova"}
    kwargs = {"groups": ("Control", "Low", "High"), "shifts": (0.0, 0.5, 1.2)} if three else {}
    kwargs["distinct_reps"] = design in {"welch", "t", "anova"}
    df = _make(5, cells=6, **kwargs)
    scaled = df.assign(intensity=df["intensity"] * scale)
    base = _run(df, test=design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _run(scaled, test=design)
    assert res.statistic is not None and res.test_name == base.test_name
    _deep_same(_nested_unit_free(res), _nested_unit_free(base))
    _deep_same(_nested_in_units(res, scale), _nested_in_units(base, 1.0))
    json.loads(res.model_dump_json())


@pytest.mark.parametrize("scale", [1e-12, 1e9])
def test_mixed_model_cross_check_is_fitted_on_rescaled_values(scale):
    df = _make(2, cells=8)
    base = ns.nested_analysis(df, "intensity", "condition", "experiment")
    res = ns.nested_analysis(df.assign(intensity=df["intensity"] * scale), "intensity", "condition", "experiment")
    assert res.extra["mixed_model"]["converged"] == base.extra["mixed_model"]["converged"]
    assert res.extra["mixed_model"]["statistic"] == pytest.approx(base.extra["mixed_model"]["statistic"], rel=1e-6)
    assert res.p_value == pytest.approx(base.p_value, rel=1e-9)


@pytest.mark.parametrize("scale", [1e-13, 1.0, 1e9])
def test_constant_paired_differences_are_recognised_at_every_scale(scale):
    """Drug = Control + 0.1 in every experiment: the differences 10.1 - 10.0, 20.2 - 20.1, ...
    are equal only up to the rounding of the two columns (far more than an ulp of 0.1)."""
    rows = []
    for i, base in enumerate([10.0, 20.1, 30.3, 40.7]):
        rows += [("Control", f"exp{i}", base * scale), ("Drug", f"exp{i}", (base + 0.1) * scale)]
    df = pd.DataFrame(rows, columns=["condition", "experiment", "intensity"])
    res = _run(df, paired=True)
    assert res.statistic is None and res.p_value is None
    assert any("no variability" in w for w in res.warnings)
    check = next(a for a in res.assumptions if a.name.startswith("Normality"))
    assert check.p_value is None and "identical" in check.detail


def test_paired_summary_counts_only_the_replicates_that_entered_the_test():
    rng = np.random.default_rng(8)
    rows = []
    for gi, g in enumerate(["ctl", "drug"]):
        for r in range(4):
            if g == "drug" and r == 3:
                continue
            base = rng.normal() + 1.5 * gi
            rows += [(g, f"exp{r + 1}", base + rng.normal(0, 0.5)) for _ in range(5)]
    df = pd.DataFrame(rows, columns=["condition", "experiment", "intensity"])
    res = _run(df, paired=True)
    assert res.n == {"ctl": 3, "drug": 3} and res.df == 2
    assert "the 6 biological replicate means were compared between ctl (n = 3), drug (n = 3)" in res.interpretation
    assert any("reduced to 7 biological replicate means" in n and "the test used 6 of them" in n
               for n in res.data_notes)
    assert any("exp4" in w and "left out" in w for w in res.warnings)
    # all seven replicate means are still listed for the SuperPlot
    assert len(res.extra["replicate_means"]) == 7
    # nothing left out: the count is simply the total
    full = _run(_make(3, cells=4))
    assert "the 8 biological replicate means were compared" in full.interpretation
    assert any(n.endswith("the test used the replicate means.") for n in full.data_notes)


def test_median_summary_is_not_described_as_averaging():
    res = _run(_make(14, cells=(4, 6)), summary="median")
    assert res.interpretation.startswith(
        "Technical replicates were summarised by their median within each biological replicate")
    assert "averaged" not in res.interpretation
    assert "averaged" not in next(a for a in res.assumptions if a.name.startswith("Independent")).detail
    mean = _run(_make(14, cells=(4, 6)))
    assert mean.interpretation.startswith("Technical replicates were averaged within each biological replicate")
