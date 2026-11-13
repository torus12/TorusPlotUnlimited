"""Independent statistical audit, part 2 of 2.

Scope: ``factorial``, ``repeated``, ``correlation``, ``regression``, ``nonlinear``,
``equation``, ``survival``, ``contingency``, ``nested``, ``recommend`` and the NIST
regression fixtures.

SECTION 1 holds one FAILING test per defect found (each fails on the audited code for the
reason given in its docstring and should pass once the defect is fixed). Severity codes:

    A  wrong numbers / wrong statistical conclusion
    B  misleading text, label or recommendation
    C  crash / valid input refused
    D  documentation / minor

SECTION 2 holds a few PASSING guard tests for important behaviour that was verified against
independent implementations during the audit (they are not defects).

Expected values are derived inside each test from first principles or from an independent
implementation (statsmodels, scipy, lifelines); nothing is taken from ``tests/stats``.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from torusplot.data.tables import ColumnTable, ContingencyTable, MultipleVariablesTable, XYTable
from torusplot.stats import run_analysis
from torusplot.stats._utils import StatsError
from torusplot.stats.contingency import fisher_exact_test
from torusplot.stats.correlation import pearson, spearman
from torusplot.stats.factorial import anova_factorial
from torusplot.stats.nested import nested_analysis
from torusplot.stats.nonlinear import compare_fits, fit_curve
from torusplot.stats.recommend import recommend_test
from torusplot.stats.regression import linear_regression
from torusplot.stats.repeated import anova_mixed, anova_rm_oneway, anova_rm_twoway
from torusplot.stats.survival import kaplan_meier, logrank_test, survival_analysis

pytestmark = pytest.mark.filterwarnings("ignore")


# ======================================================================================
# helpers (independent reference computations)
# ======================================================================================
def _rm_oneway_f(y: np.ndarray) -> tuple[float, float]:
    """Classical one-way repeated-measures F and its uncorrected P (sums of squares)."""
    n, k = y.shape
    grand = y.mean()
    ss_cond = n * ((y.mean(axis=0) - grand) ** 2).sum()
    resid = y - y.mean(axis=0, keepdims=True) - y.mean(axis=1, keepdims=True) + grand
    ss_err = (resid**2).sum()
    f = (ss_cond / (k - 1)) / (ss_err / ((k - 1) * (n - 1)))
    return float(f), float(stats.f.sf(f, k - 1, (k - 1) * (n - 1)))


def _rm_matrix() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(5.0, 1.0, (8, 4)) + np.array([0.0, 0.5, 1.0, 2.0])


def _mixed_long(scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for s in range(10):
        group = "g1" if s < 5 else "g2"
        subject_effect = rng.normal(0.0, 1.0)
        for w in range(3):
            y = 5.0 + subject_effect + (1.0 * w if group == "g1" else 0.0) + rng.normal(0.0, 0.7)
            rows.append((f"s{s}", group, f"t{w}", y * scale))
    return pd.DataFrame(rows, columns=["subject", "group", "time", "y"])


def _split_plot_interaction_f(df: pd.DataFrame) -> float:
    """Group x time F of a balanced split-plot design from the classical formulas
    (Winer, Brown & Michels 1991): SS_AB from the cell means, error = B x subjects within A."""
    wide = df.pivot(index="subject", columns="time", values="y")
    groups = df.drop_duplicates("subject").set_index("subject")["group"].loc[wide.index]
    y = wide.to_numpy(float)
    codes = pd.factorize(groups)[0]
    g, p = len(np.unique(codes)), y.shape[1]
    n = y.shape[0]
    grand = y.mean()
    ss_ab = 0.0
    ss_err = 0.0
    time_mean = y.mean(axis=0)
    for j in range(g):
        block = y[codes == j]
        cell = block.mean(axis=0)
        ss_ab += block.shape[0] * ((cell - block.mean() - time_mean + grand) ** 2).sum()
        ss_err += ((block - cell - block.mean(axis=1, keepdims=True) + block.mean()) ** 2).sum()
    return float((ss_ab / ((g - 1) * (p - 1))) / (ss_err / ((n - g) * (p - 1))))


# ======================================================================================
# SECTION 1: FAILING TESTS, ONE PER DEFECT
# ======================================================================================

# ------------------------------------------------------------------ repeated.py (A)
def test_A1_rm_oneway_anova_is_invariant_to_the_unit_of_measurement():
    """DEFECT (A), repeated.py ``anova_rm_oneway`` (ss_total guard,
    ``np.ptp(y) <= 1e-12 * max(1.0, max|y|)``): the "all values identical" guard uses an
    ABSOLUTE floor of 1e-12, so data expressed in small SI units (currents in ampere,
    concentrations in mol/L: values around 1e-13) are declared "identical" and no F / P
    value is returned.

    Expected: F and P do not depend on the unit. For the fixed 8 x 4 matrix below the
    classical sums of squares give F(3, 21) = 13.944 (also what the function itself
    returns for the unscaled data); multiplying every value by 1e-13 must give the same F.
    Derivation: first-principles SS formula in ``_rm_oneway_f`` applied to the scaled data.
    """
    y = _rm_matrix() * 1e-13
    f_expected, p_expected = _rm_oneway_f(y)
    res = anova_rm_oneway(y, list("abcd"), correction="none", posthoc="none")
    assert res.statistic is not None, f"no F test for small-magnitude data: {res.warnings}"
    assert res.statistic == pytest.approx(f_expected, rel=1e-8)
    assert res.p_value == pytest.approx(p_expected, rel=1e-8)


def test_A2_rm_oneway_paired_comparisons_survive_small_units():
    """DEFECT (A), repeated.py ``_paired_raw`` (``sd <= 1e-12 * max(1.0, ...)``): with
    values around 1e-12 (e.g. picoampere currents stored in ampere) some paired post hoc
    comparisons are reported as "could not be tested because the paired differences do not
    vary" although the differences vary as much, relatively, as in any other unit.

    Expected: every comparison has the P value of ``scipy.stats.ttest_rel`` on the same
    columns (a t statistic is scale-free).
    """
    y = _rm_matrix() * 1e-12
    names = list("abcd")
    res = anova_rm_oneway(y, names, posthoc="holm")
    for comp in res.comparisons:
        a, b = y[:, names.index(comp.group_a)], y[:, names.index(comp.group_b)]
        ref = stats.ttest_rel(a, b)
        assert comp.p_value is not None, f"{comp.group_a} vs {comp.group_b} reported as untestable"
        assert comp.p_value == pytest.approx(ref.pvalue, rel=1e-8)


def test_A3_mixed_anova_is_invariant_to_the_unit_of_measurement():
    """DEFECT (A), repeated.py ``anova_mixed`` (same absolute 1e-12 guard): a split-plot
    design measured in small units (values about 5e-13) returns "All values are identical"
    and no test at all.

    Expected: group x time interaction F(2, 16) from the classical split-plot sums of
    squares (``_split_plot_interaction_f``), identical for scaled and unscaled data.
    """
    df = _mixed_long(scale=1e-13)
    f_expected = _split_plot_interaction_f(df)
    assert f_expected == pytest.approx(_split_plot_interaction_f(_mixed_long()), rel=1e-9)
    res = anova_mixed(df, "y", "subject", "group", "time", correction="none")
    assert res.statistic is not None, f"no F test for small-magnitude data: {res.warnings}"
    assert res.statistic == pytest.approx(f_expected, rel=1e-8)


def test_A4_rm_twoway_anova_is_invariant_to_the_unit_of_measurement():
    """DEFECT (A), repeated.py ``anova_rm_twoway`` (same absolute 1e-12 guard).

    Expected: the A x B interaction F equals statsmodels ``AnovaRM`` on the same (scaled)
    data; statsmodels has no such guard and returns the unit-free F.
    """
    rng = np.random.default_rng(8)
    rows = []
    for s in range(6):
        subject_effect = rng.normal()
        for a in "xy":
            for b in "123":
                rows.append((s, a, b, (subject_effect + (a == "y") + rng.normal(0, 0.6)) * 1e-13))
    df = pd.DataFrame(rows, columns=["s", "a", "b", "y"])
    ref = AnovaRM(df, "y", "s", within=["a", "b"]).fit().anova_table
    res = anova_rm_twoway(df, "y", "s", "a", "b", correction="none")
    assert res.statistic is not None, f"no F test for small-magnitude data: {res.warnings}"
    assert res.statistic == pytest.approx(float(ref.loc["a:b", "F Value"]), rel=1e-6)


# ------------------------------------------------------------------ factorial.py (A/B)
def test_A5_additive_model_comparisons_use_the_models_least_squares_means():
    """DEFECT (A-), factorial.py ``_cell_means_cov`` / ``anova_factorial``: when the model
    has NO interaction term (``interactions=False``) and the design is unbalanced, the
    main-effect comparisons (and ``extra['marginal_means']``) are still built from the
    unweighted averages of the OBSERVED cell means, i.e. from the saturated model, while
    the error term comes from the additive model. The estimates therefore differ from the
    least-squares (estimated marginal) means of the model that was actually fitted and
    tested in the ANOVA table, and the SE is not the model-based one.

    Expected (Searle 1987; what R ``emmeans`` / SAS LSMEANS report for ``y ~ A + B``): the
    difference b1 - b2 equals minus the coefficient of B[T.b2] in the additive OLS model
    fitted by statsmodels, with that coefficient's standard error.
    """
    rng = np.random.default_rng(1)
    sizes = {("a1", "b1"): 3, ("a1", "b2"): 5, ("a1", "b3"): 2,
             ("a2", "b1"): 6, ("a2", "b2"): 2, ("a2", "b3"): 4}
    rows = []
    for (a, b), n in sizes.items():
        for _ in range(n):
            mu = 10 + 2 * (a == "a2") + 1.5 * (b == "b2") + 3 * (a == "a2" and b == "b3")
            rows.append((a, b, rng.normal(mu, 1.0)))
    df = pd.DataFrame(rows, columns=["A", "B", "y"])
    ols = smf.ols("y ~ C(A) + C(B)", df).fit()
    expected_diff = -float(ols.params["C(B)[T.b2]"])
    expected_se = float(ols.bse["C(B)[T.b2]"])

    res = anova_factorial(df, "y", ["A", "B"], interactions=False, posthoc="bonferroni",
                          posthoc_compare="main_b")
    comp = res.comparison("b1", "b2")
    sign = 1.0 if comp.group_a == "b1" else -1.0
    assert sign * comp.estimate == pytest.approx(expected_diff, rel=1e-6)
    assert comp.se == pytest.approx(expected_se, rel=1e-6)


def test_B1_factorial_residual_normality_check_is_scale_invariant():
    """DEFECT (B), factorial.py ``_normality_check`` (``scale = max(1.0, max|resid|)`` with a
    1e-10 threshold; shared with repeated.py): for outcomes of small magnitude (about
    1e-12) the Shapiro-Wilk check is skipped with "The residuals are all (nearly)
    identical", so a real violation of normality can go unreported.

    Expected: Shapiro-Wilk W and P are scale-free; the check must return the same P as
    ``scipy.stats.shapiro`` on the cell-mean residuals of the scaled data.
    """
    rng = np.random.default_rng(3)
    rows = [(a, b, rng.normal()) for a in "pq" for b in "xy" for _ in range(5)]
    df = pd.DataFrame(rows, columns=["A", "B", "y"])
    df["y"] = df["y"] * 1e-12
    resid = df["y"] - df.groupby(["A", "B"])["y"].transform("mean")
    ref = stats.shapiro(resid.to_numpy())
    res = anova_factorial(df, "y", ["A", "B"])
    check = next(c for c in res.assumptions if c.name.startswith("Normality"))
    assert check.p_value is not None, check.detail
    assert check.p_value == pytest.approx(ref.pvalue, rel=1e-6)


# ------------------------------------------------------------------ recommend.py (B)
def test_B2_recommend_does_not_mistake_an_exposure_factor_for_a_replicate_column():
    """DEFECT (B, leads to a wrong analysis), recommend.py ``_recommend_mv``: the replicate
    column is detected by SUBSTRING match on ("exp", "run", "day", ...), so a genuine
    experimental factor called "Exposure" (also "Expression", "Experience", ...) is taken to
    be the biological-replicate identifier. For a 2 x 2 between-subjects design
    (Exposure x Sex, 6 independent subjects per cell) the recommendation is the
    replicate-aware analysis, which averages over the 2 exposure levels and runs a paired
    t test on n = 2 "replicates"; its explanation ("'Exposure' identifies independent
    biological replicates") is false.

    Expected: a two-way ANOVA (two crossed fixed factors, independent observations).
    """
    rng = np.random.default_rng(5)
    table = MultipleVariablesTable("m", pd.DataFrame({
        "Exposure": np.tile(["smoker", "non-smoker"], 12),
        "Sex": np.repeat(["m", "f"], 12),
        "Value": rng.normal(size=24),
    }))
    rec = recommend_test(table)
    assert rec.analysis != "nested_analysis", rec.reason
    assert rec.analysis == "anova_twoway"


def test_B3_recommend_long_repeated_measures_table_is_not_a_threeway_anova():
    """DEFECT (B, leads to a wrong / impossible analysis), recommend.py ``_recommend_mv``: a
    long table Subject / Treatment / Time / Value in which every subject is measured at
    every time point (split-plot design) is recommended a THREE-WAY ANOVA with the subject
    identifier as a fixed factor ("One measured value is classified by three grouping
    variables"). That treats repeated measurements as independent, and the recommended
    analysis cannot even run (empty Subject x Treatment cells).

    Expected: subjects nested in Treatment and crossed with Time = mixed-design ANOVA
    (``anova_mixed``), which the registry offers for this table type. At the very least the
    recommended analysis must not use the subject identifier as a fixed factor and must run.
    """
    rng = np.random.default_rng(6)
    table = MultipleVariablesTable("m", pd.DataFrame({
        "Subject": np.repeat([f"S{i}" for i in range(8)], 3),
        "Treatment": np.repeat(["ctl", "drug"], 12),
        "Time": np.tile(["t0", "t1", "t2"], 8),
        "Value": rng.normal(size=24),
    }))
    rec = recommend_test(table)
    factor_values = [v for k, v in rec.params.items() if k.startswith("factor")]
    assert "Subject" not in factor_values, f"{rec.analysis} with params {rec.params}"
    run_analysis(table, rec.analysis, **rec.params)  # must be runnable on the table


def test_B4_recommend_sparse_rxc_table_offers_the_exact_test():
    """DEFECT (B), recommend.py ``_recommend_contingency``: for a 3 x 3 table in which all
    nine expected counts are below 5 the recommendation is still the chi-square test, with
    no alternative, although (i) Cochran's rule (computed in the same function) fails,
    (ii) ``chi_square_test`` itself then says "Use Fisher's exact test", and (iii) the
    registry provides the Fisher-Freeman-Halton exact test for r x c tables.

    Expected: ``fisher_exact`` recommended, or at least listed as an alternative.
    """
    table = ContingencyTable.from_counts("c", [[3, 1, 4], [1, 5, 2], [0, 2, 6]],
                                         ["r1", "r2", "r3"], ["c1", "c2", "c3"])
    rec = recommend_test(table)
    offered = {rec.analysis, *[a.analysis for a in rec.alternatives]}
    assert "fisher_exact" in offered, f"recommended {rec.analysis}, alternatives {sorted(offered)}"


def test_B5_recommend_log_dose_explanation_states_the_real_dose_range():
    """DEFECT (B), recommend.py ``_recommend_xy``: when X is log10(dose) (-9 ... -4, column
    named "log[Agonist]") the explanation says "X values span about 0 decades" (the number
    of decades is computed from the positive X values only, of which there are none),
    directly followed by the reason "X covers several orders of magnitude".

    Expected: X spanning -9 to -4 on a log10 scale is 5 decades; the text must not say 0.
    """
    rng = np.random.default_rng(2)
    logx = np.linspace(-9, -4, 6)
    y = np.column_stack([100 / (1 + 10 ** (-6 - logx)) + rng.normal(0, 3, 6) for _ in range(3)])
    table = XYTable.from_arrays("xy", logx, {"Y": y}, x_name="log[Agonist]")
    rec = recommend_test(table)
    assert rec.params.get("model") == "dose_response_4pl_logx"
    assert "0 decades" not in " ".join(rec.steps)


def test_B6_recommend_flags_that_mann_whitney_cannot_reach_significance_for_3_vs_4():
    """DEFECT (B/D), recommend.py ``_recommend_column``: the caution that a two-sided
    Mann-Whitney test "cannot give P < 0.05" is only issued when BOTH groups have n <= 3
    (``max(ns) <= 3``). For n = 3 vs 4 the smallest attainable two-sided exact P is
    2 / C(7, 3) = 0.0571 > 0.05, so the recommended test is equally powerless but the user
    is not told.

    Expected: the same caution for every pair of sizes whose minimum exact P exceeds 0.05
    (derivation: complete separation gives P = 2 / C(n1 + n2, n1); checked with scipy).
    """
    assert stats.mannwhitneyu([1, 2, 3], [4, 5, 6, 7], method="exact").pvalue > 0.05
    table = ColumnTable.from_dict("c", {"A": [1.0, 1.1, 30.0], "B": [2.0, 2.1, 2.2, 90.0]})
    rec = recommend_test(table)
    assert rec.analysis == "mann_whitney"
    assert any("0.05" in c and "cannot" in c for c in rec.cautions), rec.cautions


# ------------------------------------------------------------------ equation.py (B)
def test_B7_user_equation_parameter_named_e_is_not_silently_replaced_by_2_718():
    """DEFECT (B, silently wrong fit), equation.py ``_convert_name``: a lower-case name ``e``
    is always Euler's number. A user who types the natural polynomial
    ``Y = a + b*x + c*x^2 + d*x^3 + e*x^4`` gets a fit with only four parameters and the
    quartic coefficient frozen at 2.71828, with no warning or note anywhere in the result
    (R squared drops from > 0.99 to about 0.59 for the data below).

    Expected: either ``e`` is fitted as a parameter, or the result tells the user that ``e``
    was read as the constant 2.718...
    """
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 25)
    y = 1 + 2 * x - 0.5 * x**2 + 0.3 * x**3 + 0.7 * x**4 + rng.normal(0, 0.05, 25)
    res = fit_curve(x, y, "user_defined", equation="Y = a + b*x + c*x^2 + d*x^3 + e*x^4")
    names = [p.name for p in res.parameters]
    text = " ".join([*res.warnings, *res.data_notes, res.interpretation]).lower()
    told = "2.718" in text or "euler" in text or "constant e" in text
    assert "e" in names or told, f"parameters {names}; nothing says that e = 2.718 was used"


# ------------------------------------------------------------------ survival.py (B)
def test_B8a_median_survival_ci_of_a_single_subject_is_not_a_zero_width_interval():
    """DEFECT (B/A-), survival.py ``_km`` (``band_hi = np.where(at_zero, 0.0, hi)``): when the
    curve reaches 0 the (undefined) confidence band is replaced by 0, so the upper
    Brookmeyer-Crowley limit of the median is reported as the last event time. For ONE
    subject with an event at t = 5 the "95% CI" of the median is [5, 5].

    Expected: with n = 1 nothing is known about the precision of the median; the upper
    limit is not estimable (R ``survfit`` prints NA: its upper band is NA where S = 0 and the
    limit is the first time the upper band drops to 0.5, Brookmeyer & Crowley 1982).
    """
    km = kaplan_meier([5.0], [1])
    assert km["median"] == 5.0
    assert not (km["median_ci_low"] == km["median_ci_high"] == 5.0), "zero-width 95% CI from n = 1"
    assert km["median_ci_high"] is None


def test_B8b_median_upper_limit_is_open_when_the_upper_band_never_reaches_one_half():
    """DEFECT (B/A-), survival.py ``_km``: same cause as B8a. Three uncensored subjects
    (events at 1, 2, 3): the log-log upper band is 0.945 at t = 1 and 0.774 at t = 2 and is
    undefined at t = 3 (S = 0, Greenwood variance infinite). It never drops to 0.5, so the
    upper limit of the median is "not reached"; the code reports 3 (the largest
    observation). The interval [1, 3] = [min, max] of three observations covers the true
    median with probability 1 - 2 * (1/2)^3 = 0.75, not 0.95.

    Expected: ``median_ci_high is None`` (as for censored data whose band stays above 0.5).
    """
    km = kaplan_meier([1, 2, 3], [1, 1, 1])
    upper_band = [row["ci_high"] for row in km["table"]]
    assert upper_band[0] == pytest.approx(0.945, abs=2e-3) and upper_band[1] == pytest.approx(0.774, abs=2e-3)
    assert upper_band[2] is None  # the band itself is (correctly) undefined at S = 0
    assert km["median_ci_high"] is None


# ------------------------------------------------------------------ nonlinear.py (B/D)
def test_B9_compare_fits_does_not_declare_a_winner_between_identical_fits():
    """DEFECT (B), nonlinear.py ``compare_fits`` wording: 'One site: specific binding' and
    'Michaelis-Menten' are the same function Y = a X / (b + X). Their fits are identical
    (difference in AICc = 0, Akaike weights 50 % / 50 %), yet the interpretation says "the
    simpler model is preferred. By AICc the simpler model is more likely to be correct
    (difference in AICc = 0; probability 50% ...)".

    Expected: with an AICc difference of zero neither model is preferred.
    """
    rng = np.random.default_rng(2)
    x = np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0] * 2)
    y = 100 * x / (5 + x) + rng.normal(0, 2, x.size)
    res = compare_fits(x, y, "one_site_binding_specific", "michaelis_menten")
    assert abs(res.extra["akaike"]["delta_aicc"]) < 1e-6
    assert res.extra["akaike"]["probability_simple"] == pytest.approx(0.5, abs=1e-6)
    assert "is preferred" not in res.interpretation
    assert "more likely to be correct" not in res.interpretation


def test_D1_ic50_labelling_also_covers_the_parameter_description():
    """DEFECT (D), nonlinear.py (``descriptions`` of ``dose_response_4pl`` are not passed
    through ``_display_equation``): with ``kind='IC50'`` the parameter is named LogIC50 but
    its description still reads "log10 of EC50 (the parameter actually fitted)".

    Expected: no mention of EC50 in the note of a parameter labelled LogIC50.
    """
    rng = np.random.default_rng(21)
    conc = np.repeat(np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]), 3)
    y = 5 + 95 / (1 + (conc / 1e-7) ** 1.2) + rng.normal(0, 4, conc.size)
    res = fit_curve(conc, y, "dose_response_4pl", kind="IC50")
    par = res.parameter("LogIC50")
    assert par is not None
    assert "EC50" not in (par.unit_note or "")


# ------------------------------------------------------------------ nested.py (B/D)
def test_D2_nested_paired_summary_counts_only_the_replicates_that_were_used():
    """DEFECT (D/B), nested.py interpretation text: in a paired analysis in which one
    replicate is missing from one group (and is therefore left out), the summary says
    "the 7 biological replicate means were compared between ctl (n = 3), drug (n = 3)":
    seven means are announced but 3 + 3 = 6 entered the test.

    Expected: the sentence reports the 6 replicate means that were compared.
    """
    rng = np.random.default_rng(8)
    rows = []
    for gi, g in enumerate(["ctl", "drug"]):
        for r in range(4):
            if g == "drug" and r == 3:
                continue
            base = rng.normal() + 1.5 * gi
            rows += [(g, f"exp{r + 1}", base + rng.normal(0, 0.5)) for _ in range(5)]
    df = pd.DataFrame(rows, columns=["g", "r", "y"])
    res = nested_analysis(df, "y", "g", "r", paired=True, mixed_model=False)
    assert res.n == {"ctl": 3, "drug": 3}
    assert "the 7 biological replicate" not in res.interpretation
    assert "6 biological replicate" in res.interpretation


# ------------------------------------------------------------------ correlation / regression (C)
@pytest.mark.parametrize("func", ["pearson", "spearman", "linear_regression"])
def test_C1_large_offset_x_values_are_not_called_identical(func):
    """DEFECT (C), correlation.py ``is_constant`` (relative tolerance 1e-13) and
    regression.py ``is_constant``: X = 1e15 + (1, 2, ..., 6) is refused with "All values of
    X are identical", although these doubles are exactly representable (spacing 0.125 at
    1e15), distinct, and correlation / slope are invariant to adding a constant.

    Expected: the same result as for X = 1 ... 6: Pearson r = 0.99894698 (also what
    ``scipy.stats.pearsonr`` returns for the offset data), Spearman rho = 1, slope =
    2.0342857 (exact rational arithmetic on the centred data: S_xy / S_xx = 35.6 / 17.5).
    """
    x = np.array([1, 2, 3, 4, 5, 6.0])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3])
    big = x + 1e15
    assert np.array_equal(big - 1e15, x)  # exactly representable
    if func == "pearson":
        res = pearson(big, y)
        assert res.effect_sizes[0].value == pytest.approx(stats.pearsonr(big, y)[0], rel=1e-9)
    elif func == "spearman":
        res = spearman(big, y)
        assert res.statistic == pytest.approx(1.0)
    else:
        res = linear_regression(big, y)
        assert res.parameter("Slope").value == pytest.approx(35.6 / 17.5, rel=1e-9)


def test_C2_rm_anova_with_very_large_values_does_not_crash():
    """DEFECT (C), repeated.py: values of the order of 1e150 - 1e160 (finite doubles) make
    ``anova_rm_oneway`` raise an unhandled low-level exception (``OverflowError`` at 1e150,
    ``numpy.linalg.LinAlgError: Eigenvalues did not converge`` at 1e160: the sums of squares
    overflow to inf) instead of returning a result or a ``StatsError`` that tells the user
    what to do (as ``linear_regression`` does for such data).

    Expected: either the unit-free F of the data (13.944 for this matrix, see A1) or a
    ``StatsError``; never an unhandled OverflowError.
    """
    y = _rm_matrix() * 1e160
    try:
        res = anova_rm_oneway(y, list("abcd"), correction="none", posthoc="none")
    except StatsError:
        return
    assert res.statistic == pytest.approx(_rm_oneway_f(_rm_matrix())[0], rel=1e-6)


# ======================================================================================
# SECTION 2: GUARD TESTS (PASS ON THE AUDITED CODE; verified behaviour worth protecting)
# ======================================================================================
def test_guard_freireich_logrank_gehan_and_hazard_ratio_direction():
    """GUARD (passes). Freireich et al. (1963) 6-MP vs placebo remission data: log-rank
    chi-square 16.79 and Gehan-Breslow 13.46 (textbook values, e.g. Kleinbaum & Klein),
    reproduced with lifelines in the audit; the hazard ratio of 6-MP relative to placebo
    must be below 1 (6-MP relapses later)."""
    t = [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35,
         1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
    e = [1] * 9 + [0] * 12 + [1] * 21
    g = ["6-MP"] * 21 + ["Placebo"] * 21
    groups = {"6-MP": (t[:21], e[:21]), "Placebo": (t[21:], e[21:])}
    assert logrank_test(groups)["statistic"] == pytest.approx(16.793, abs=2e-3)
    assert logrank_test(groups, weighting="gehan")["statistic"] == pytest.approx(13.458, abs=2e-3)
    res = survival_analysis(t, e, g)
    hr = res.extra["hazard_ratio"]
    assert hr["mantel_haenszel"]["hr"] < 1 and hr["logrank"]["hr"] < 1
    assert hr["logrank"]["hr"] == pytest.approx((9 / 19.2505) / (21 / 10.7495), rel=1e-4)


def test_guard_fisher_freeman_halton_matches_brute_force_enumeration():
    """GUARD (passes). r x c Fisher exact P equals the sum of the probabilities of all
    tables with the observed margins that are no more probable than the observed one,
    enumerated here independently (multivariate hypergeometric probabilities)."""
    table = np.array([[3, 1, 4], [1, 5, 2], [0, 2, 6]])
    rs, cs, n = table.sum(1), table.sum(0), table.sum()

    def logp(m):
        return (sum(math.lgamma(v + 1) for v in rs) + sum(math.lgamma(v + 1) for v in cs)
                - math.lgamma(n + 1) - sum(math.lgamma(v + 1) for v in m.ravel()))

    p_obs, total = logp(table), 0.0
    for a, b, d, e in itertools.product(range(rs[0] + 1), range(rs[0] + 1), range(rs[1] + 1), range(rs[1] + 1)):
        m = np.array([[a, b, rs[0] - a - b], [d, e, rs[1] - d - e], [cs[0] - a - d, cs[1] - b - e, 0]])
        m[2, 2] = rs[2] - m[2, 0] - m[2, 1]
        if (m < 0).any() or m[:, 2].sum() != cs[2]:
            continue
        lp = logp(m)
        if lp <= p_obs + 1e-7:
            total += math.exp(lp)
    assert fisher_exact_test(table.tolist()).p_value == pytest.approx(total, rel=1e-9)


def test_guard_unbalanced_mixed_anova_type3_within_effect_matches_model_comparison():
    """GUARD (passes). Unbalanced split-plot design (group sizes 4 / 7 / 5): the Type III
    sum of squares of the within factor equals RSS(model without the within main-effect
    columns) - RSS(full model) in a sum-coded least-squares fit with subject dummies, and
    the interaction SS equals RSS(no interaction) - RSS(full) (computed with patsy/numpy)."""
    import patsy

    rng = np.random.default_rng(5)
    rows, sid = [], 0
    for g, size in {"g1": 4, "g2": 7, "g3": 5}.items():
        for _ in range(size):
            sid += 1
            s_eff = rng.normal(0, 1.5)
            for w in range(4):
                rows.append((f"s{sid}", g, f"t{w}",
                             10 + s_eff + 0.5 * w * (g == "g2") + (g == "g3") + rng.normal() * (1 + 0.5 * w)))
    df = pd.DataFrame(rows, columns=["s", "g", "w", "y"])

    def rss(x):
        beta, *_ = np.linalg.lstsq(x, df["y"].to_numpy(), rcond=None)
        r = df["y"].to_numpy() - x @ beta
        return float(r @ r)

    full = patsy.dmatrix("C(s) + C(w, Sum) + C(w, Sum):C(g, Sum)", df)
    names = full.design_info.column_names
    x_full = np.asarray(full)
    main_w = [i for i, nm in enumerate(names) if nm.startswith("C(w, Sum)[S.") and ":" not in nm]
    keep = [i for i in range(x_full.shape[1]) if i not in main_w]
    ss_w = rss(x_full[:, keep]) - rss(x_full)
    ss_int = rss(np.asarray(patsy.dmatrix("C(s) + C(w, Sum)", df))) - rss(x_full)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = anova_mixed(df, "y", "s", "g", "w", ss_type=3)
    table = {row["source"]: row for row in res.anova_table}
    assert table["w"]["ss"] == pytest.approx(ss_w, rel=1e-8)
    assert table["g x w"]["ss"] == pytest.approx(ss_int, rel=1e-8)
    assert table["Residual"]["ss"] == pytest.approx(rss(x_full), rel=1e-8)
    assert table["Residual"]["df"] == 3 * (16 - 3)


def test_guard_mauchly_gg_hf_match_independent_formulas():
    """GUARD (passes). Mauchly W and chi-square (Mauchly 1940), Greenhouse-Geisser (1959) and
    Huynh-Feldt (1976) epsilons and the corrected P values, recomputed here from their
    definitions on orthonormal contrasts. (This guard originally used a textbook example; it
    was replaced by constructed data with the same qualitative behaviour, see DECISIONS.md
    D-030. The auditor had confirmed that the textbook's published values were reproduced.)"""
    from scipy import stats

    y = np.array([(1, 12, 12, 12), (4, 5, 6, 8), (4, 7, 6, 10), (5, 12, 11, 5),
                  (1, 4, 5, 7), (12, 10, 11, 10), (6, 9, 8, 2), (3, 12, 11, 2)], dtype=float)
    n, k = y.shape
    q = k - 1
    c = np.zeros((k, q))
    for j in range(1, k):
        c[:j, j - 1] = 1.0
        c[j, j - 1] = -j
        c[:, j - 1] /= np.linalg.norm(c[:, j - 1])
    cov = np.cov(y @ c, rowvar=False)
    w = np.linalg.det(cov) / (np.trace(cov) / q) ** q
    chi2 = -((n - 1) - (2 * q * q + q + 2) / (6 * q)) * np.log(w)
    eps_gg = np.trace(cov) ** 2 / (q * np.trace(cov @ cov))
    eps_hf = min(1.0, (n * q * eps_gg - 2) / (q * (n - 1 - q * eps_gg)))
    res = anova_rm_oneway(y, ["a", "b", "c", "d"], correction="gg", posthoc="none")
    sph = res.extra["sphericity"]
    assert sph["mauchly_w"] == pytest.approx(w, rel=1e-10)
    assert sph["chi2"] == pytest.approx(chi2, rel=1e-10)
    assert sph["eps_gg"] == pytest.approx(eps_gg, rel=1e-10)
    assert sph["eps_hf"] == pytest.approx(eps_hf, rel=1e-10)
    row = res.anova_table[0]
    assert row["p_gg"] == pytest.approx(stats.f.sf(row["f"], q * eps_gg, q * (n - 1) * eps_gg), rel=1e-8)
    assert res.p_value == pytest.approx(row["p_gg"])
