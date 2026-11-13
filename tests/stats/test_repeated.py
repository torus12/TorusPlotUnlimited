"""Repeated-measures ANOVA (one-way, mixed / split-plot, two-way) and sphericity.

Validation layers (see tests/fixtures/sources/factorial_repeated.md):
* a published worked example with a strong sphericity violation;
* cross-implementation checks: statsmodels AnovaRM, OLS and mixedlm; scipy t tests;
* from-first-principles computations written here: classical split-plot sums of
  squares, a full dummy-variable least-squares model for unbalanced mixed designs,
  Box's epsilon from the raw covariance matrix, Mauchly's W from determinants;
* simulations (fixed seeds) and property-based tests with hypothesis.
"""

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.multitest import multipletests

from torusplot.data.tables import ColumnTable, GroupedTable, MultipleVariablesTable
from torusplot.stats import registry
from torusplot.stats import repeated as rp
from torusplot.stats._utils import StatsError


@pytest.fixture
def run(monkeypatch):
    """run_analysis without importing analysis modules owned by other people."""
    monkeypatch.setattr(registry, "_loaded", True)
    return registry.run_analysis


# Field A, Discovering Statistics: "bushtucker trial" example. Time to retch (seconds) for
# 8 celebrities eating 4 foods (stick insect, kangaroo testicle, fish eye, witchetty grub).
FIELD = np.array([
    (8, 7, 1, 6), (9, 5, 2, 5), (6, 2, 3, 8), (5, 3, 1, 9),
    (8, 4, 5, 8), (7, 5, 6, 7), (10, 2, 7, 2), (12, 6, 8, 1),
], dtype=float)
FOODS = ["Stick insect", "Kangaroo testicle", "Fish eye", "Witchetty grub"]


def table_row(result, source):
    for row in result.anova_table:
        if row["source"] == source:
            return row
    raise AssertionError(f"no row {source!r} in {[r['source'] for r in result.anova_table]}")


def long_from_matrix(y, names):
    rows = [(i, name, y[i, j]) for i in range(y.shape[0]) for j, name in enumerate(names)]
    return pd.DataFrame(rows, columns=["subject", "cond", "y"])


def box_epsilon(cov):
    """Greenhouse-Geisser / Box epsilon straight from a k x k covariance matrix
    (classical formula in terms of its elements, no contrasts involved)."""
    k = cov.shape[0]
    grand = cov.mean()
    diag = np.diag(cov).mean()
    rows = cov.mean(axis=1)
    num = (k * (diag - grand)) ** 2
    den = (k - 1) * ((cov**2).sum() - 2 * k * (rows**2).sum() + k**2 * grand**2)
    return num / den


def mixed_frame(seed=1, sizes=(5, 7, 6), levels=("0h", "1h", "4h", "24h"), effect=0.8, rho_noise=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for gi, n in enumerate(sizes):
        for s in range(n):
            base = rng.normal(0, 1.5)
            for ti, lv in enumerate(levels):
                mu = 10 + effect * ti * gi + 0.3 * ti
                rows.append((f"g{gi}s{s}", f"grp{gi}", lv, mu + base + rng.normal(0, rho_noise * (1 + 0.3 * ti))))
    return pd.DataFrame(rows, columns=["id", "group", "time", "y"])


def wide_from_long(frame):
    wide = frame.pivot(index="id", columns="time", values="y")
    wide = wide[list(pd.unique(frame["time"]))]
    group = frame.drop_duplicates("id").set_index("id")["group"].reindex(wide.index)
    order = np.argsort(pd.Categorical(group, categories=list(pd.unique(frame["group"]))).codes, kind="stable")
    return wide.to_numpy()[order], pd.factorize(group.to_numpy()[order])[0]


# =============================================================================
# Published worked example
# =============================================================================
def test_published_field_bushtucker_example():
    r = rp.anova_rm_oneway(FIELD, FOODS, correction="none", posthoc="none")
    row = table_row(r, "Condition")
    # F(3, 21) = 3.79, p = .026 (sphericity assumed)
    assert (row["df"], row["df_error"]) == (3, 21)
    assert row["f"] == pytest.approx(3.79, abs=0.005)
    assert row["p"] == pytest.approx(0.026, abs=0.0005)
    assert r.p_value == row["p"] and (r.df, r.df2) == (3, 21)
    # Mauchly: W = .136, chi-square(5) = 11.41, p = .047
    sph = r.extra["sphericity"]
    assert sph["mauchly_w"] == pytest.approx(0.136, abs=0.0005)
    assert sph["chi2"] == pytest.approx(11.41, abs=0.005) and sph["df"] == 5
    assert sph["p"] == pytest.approx(0.047, abs=0.0005)
    # Greenhouse-Geisser epsilon = .533 (p = .063); Huynh-Feldt epsilon = .666 (p = .048)
    assert row["eps_gg"] == pytest.approx(0.533, abs=0.0005)
    assert row["p_gg"] == pytest.approx(0.063, abs=0.0005)
    assert row["eps_hf"] == pytest.approx(0.666, abs=0.0005)
    assert row["p_hf"] == pytest.approx(0.048, abs=0.0005)
    assert row["eps_lb"] == pytest.approx(1 / 3)
    check = next(a for a in r.assumptions if a.name.startswith("Sphericity"))
    assert check.status == "fail" and check.p_value == sph["p"]
    # the published P value includes the second-order term; the plain chi-square tail is smaller
    assert sph["p_first_order"] == pytest.approx(stats.chi2.sf(sph["chi2"], 5), rel=1e-12)
    assert sph["p_first_order"] == pytest.approx(0.0439, abs=0.0005)


def test_headline_correction_rule():
    auto = rp.anova_rm_oneway(FIELD, FOODS, posthoc="none")
    row = table_row(auto, "Condition")
    assert auto.extra["correction_applied"] == "gg" and row["correction"] == "Greenhouse-Geisser"
    assert auto.p_value == row["p_gg"] == row["p_reported"]
    assert (auto.df, auto.df2) == pytest.approx((3 * row["eps_gg"], 21 * row["eps_gg"]))
    assert auto.p_value == pytest.approx(stats.f.sf(row["f"], 3 * row["eps_gg"], 21 * row["eps_gg"]), rel=1e-12)
    assert "Greenhouse-Geisser" in auto.interpretation and "Greenhouse-Geisser" in auto.methods_text
    assert auto.finalize().significant is False  # .063 with the default correction
    hf = rp.anova_rm_oneway(FIELD, FOODS, correction="hf", posthoc="none")
    assert hf.p_value == row["p_hf"] and hf.finalize().significant is True
    assert hf.df == pytest.approx(3 * row["eps_hf"])
    gg = rp.anova_rm_oneway(FIELD, FOODS, correction="Greenhouse-Geisser", posthoc="none")
    assert gg.p_value == row["p_gg"]
    none = rp.anova_rm_oneway(FIELD, FOODS, correction="none", posthoc="none")
    assert none.p_value == row["p"] and "sphericity assumed" in none.methods_text
    assert row["p_lb"] == pytest.approx(stats.f.sf(row["f"], 1, 7), rel=1e-12)
    assert auto.extra["p_uncorrected"] == row["p"] and auto.extra["p_lower_bound"] == row["p_lb"]


# =============================================================================
# One-way RM: cross-implementation and first-principles checks
# =============================================================================
def test_oneway_matches_statsmodels_anovarm_and_classical_sums_of_squares():
    rng = np.random.default_rng(42)
    for n, k in ((8, 4), (12, 3), (6, 5), (20, 2)):
        y = rng.normal(size=(n, k)) + rng.normal(size=(n, 1)) * 2 + np.linspace(0, 1, k)
        names = [f"c{j}" for j in range(k)]
        r = rp.anova_rm_oneway(y, names, posthoc="none", factor_name="Dose")
        ref = AnovaRM(long_from_matrix(y, names), "y", "subject", within=["cond"]).fit().anova_table
        row = table_row(r, "Dose")
        assert row["f"] == pytest.approx(ref.loc["cond", "F Value"], rel=1e-6)
        assert (row["df"], row["df_error"]) == (ref.loc["cond", "Num DF"], ref.loc["cond", "Den DF"])
        assert row["p"] == pytest.approx(ref.loc["cond", "Pr > F"], rel=1e-6)
        # classical formulas
        grand = y.mean()
        ss_cond = n * ((y.mean(axis=0) - grand) ** 2).sum()
        ss_subj = k * ((y.mean(axis=1) - grand) ** 2).sum()
        ss_total = ((y - grand) ** 2).sum()
        assert row["ss"] == pytest.approx(ss_cond, rel=1e-10)
        assert table_row(r, "Subjects")["ss"] == pytest.approx(ss_subj, rel=1e-10)
        assert table_row(r, "Residual")["ss"] == pytest.approx(ss_total - ss_cond - ss_subj, rel=1e-9)
        assert table_row(r, "Total")["ss"] == pytest.approx(ss_total, rel=1e-12)
        assert [table_row(r, s)["df"] for s in ("Subjects", "Residual", "Total")] == [n - 1, (n - 1) * (k - 1), n * k - 1]
        ms_e = (ss_total - ss_cond - ss_subj) / ((n - 1) * (k - 1))
        assert table_row(r, "Subjects")["f"] == pytest.approx(ss_subj / (n - 1) / ms_e, rel=1e-9)
        # effect sizes
        assert r.effect("partial eta squared").value == pytest.approx(
            ss_cond / (ss_total - ss_subj), rel=1e-9)
        assert r.effect("generalized eta squared").value == pytest.approx(ss_cond / ss_total, rel=1e-9)
        assert row["partial_eta_sq"] == pytest.approx(r.effect("partial eta squared").value)
        assert r.n == dict.fromkeys(names, n) and r.n_total == n and r.extra["n_observations"] == n * k


def test_epsilon_and_mauchly_from_the_covariance_matrix():
    rng = np.random.default_rng(7)
    n, k = 11, 5
    y = rng.normal(size=(n, k)) @ np.diag([1, 1.5, 0.6, 2.0, 1.0]) + rng.normal(size=(n, 1))
    r = rp.anova_rm_oneway(y, list("abcde"), posthoc="none")
    sph = r.extra["sphericity"]
    cov = np.cov(y, rowvar=False)
    q = k - 1
    assert sph["eps_gg"] == pytest.approx(box_epsilon(cov), rel=1e-10)
    # Mauchly's W with an arbitrary orthonormal basis of the contrast space (QR of random contrasts)
    basis, _ = np.linalg.qr((np.eye(k) - 1 / k) @ rng.normal(size=(k, q)))
    s = basis.T @ cov @ basis
    w = np.linalg.det(s) / (np.trace(s) / q) ** q
    assert sph["mauchly_w"] == pytest.approx(w, rel=1e-9)
    chi2 = -((n - 1) - (2 * q * q + q + 2) / (6 * q)) * math.log(w)
    assert sph["chi2"] == pytest.approx(chi2, rel=1e-9) and sph["df"] == q * (q + 1) // 2 - 1
    assert sph["p_first_order"] == pytest.approx(stats.chi2.sf(chi2, sph["df"]), rel=1e-9)
    # second-order term of Box's expansion, written out independently
    rho = 1 - (2 * q * q + q + 2) / (6 * q * (n - 1))
    w2 = (q + 2) * (q - 1) * (q - 2) * (2 * q**3 + 6 * q**2 + 3 * q + 2) / (288 * (q * (n - 1) * rho) ** 2)
    p1, p2 = stats.chi2.sf(chi2, sph["df"]), stats.chi2.sf(chi2, sph["df"] + 4)
    assert sph["p"] == pytest.approx(p1 + w2 * (p2 - p1), rel=1e-9)
    # Huynh-Feldt (one group: the original and Lecoutre formulas coincide)
    eps = sph["eps_gg"]
    hf = (n * q * eps - 2) / (q * (n - 1 - q * eps))
    assert sph["eps_hf_original_uncapped"] == pytest.approx(hf, rel=1e-10)
    assert sph["eps_hf_lecoutre_uncapped"] == pytest.approx(hf, rel=1e-10)
    assert sph["eps_hf"] == pytest.approx(min(hf, 1.0), rel=1e-10)
    assert sph["eps_lower_bound"] == pytest.approx(1 / q)
    # the Helmert contrasts used internally are orthonormal and orthogonal to the constant
    c = rp._orthonormal_contrasts(k)
    np.testing.assert_allclose(c.T @ c, np.eye(q), atol=1e-12)
    np.testing.assert_allclose(c.sum(axis=0), 0, atol=1e-12)


def test_two_levels_equal_the_paired_t_test():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(9, 2)) + rng.normal(size=(9, 1))
    r = rp.anova_rm_oneway(y, ["before", "after"], posthoc="bonferroni")
    ref = stats.ttest_rel(y[:, 0], y[:, 1])
    row = table_row(r, "Condition")
    assert row["f"] == pytest.approx(ref.statistic**2, rel=1e-10)
    assert row["p"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert row["eps_gg"] == row["eps_hf"] == row["eps_lb"] == 1.0
    assert row["p_gg"] == pytest.approx(row["p"]) and r.p_value == row["p"]
    assert r.extra["correction_applied"] == "none" and (r.df, r.df2) == (1, 8)
    check = next(a for a in r.assumptions if a.name.startswith("Sphericity"))
    assert check.status == "pass" and "two" in check.detail
    assert r.extra["sphericity"]["mauchly_w"] is None
    comp = r.comparisons[0]
    assert comp.statistic == pytest.approx(ref.statistic, rel=1e-10) and comp.p_adjusted == pytest.approx(ref.pvalue)


def test_epsilon_reacts_to_the_covariance_structure():
    rng = np.random.default_rng(11)
    n, k = 400, 4
    # compound symmetry (equal variances, equal correlations): sphericity holds, epsilon -> 1
    cs = rng.normal(size=(n, k)) + rng.normal(size=(n, 1)) * 1.3
    sc = rp._analyse_scores(cs @ rp._orthonormal_contrasts(k), np.zeros(n, dtype=int), 1)
    assert sc.eps_gg > 0.97 and sc.eps_hf > 0.97 and sc.mauchly_w > 0.95
    # one condition far noisier than the others: strong violation
    bad = cs.copy()
    bad[:, 0] += rng.normal(0, 8, n)
    sb = rp._analyse_scores(bad @ rp._orthonormal_contrasts(k), np.zeros(n, dtype=int), 1)
    assert sb.eps_gg < 0.5 and sb.mauchly_p < 1e-6 and sb.eps_gg >= 1 / 3


def test_mauchly_p_is_roughly_uniform_under_sphericity():
    """Simulation (fixed seed): with spherical data the test rejects about 5% of the time."""
    rng = np.random.default_rng(2024)
    n, k, reps = 15, 4, 1500
    contrasts = rp._orthonormal_contrasts(k)
    codes = np.zeros(n, dtype=int)
    p_second, p_first = [], []
    for _ in range(reps):
        y = rng.normal(size=(n, k)) + rng.normal(size=(n, 1))
        sc = rp._analyse_scores(y @ contrasts, codes, 1)
        p_second.append(sc.mauchly_p)
        p_first.append(sc.mauchly_p_first_order)
    p_second, p_first = np.array(p_second), np.array(p_first)
    # loose bounds: binomial SE of a 5% rate with 1500 replicates is 0.0056
    assert 0.03 < (p_second < 0.05).mean() < 0.07
    assert 0.44 < p_second.mean() < 0.56
    assert 0.2 < (p_second < 0.25).mean() < 0.3 and 0.7 < (p_second < 0.75).mean() < 0.8
    assert stats.kstest(p_second, "uniform").pvalue > 0.001
    # the second-order term only makes the test slightly less liberal
    assert (p_second >= p_first - 1e-15).all() and np.abs(p_second - p_first).max() < 0.03


def test_oneway_posthoc_paired_t_tests():
    r = rp.anova_rm_oneway(FIELD, FOODS, posthoc="holm")
    assert len(r.comparisons) == 6 and r.correction == "Holm"
    assert "all 6 pairwise comparisons among the 4 conditions" in r.comparison_family
    raw = []
    for a, b in itertools.combinations(range(4), 2):
        comp = r.comparison(FOODS[a], FOODS[b])
        ref = stats.ttest_rel(FIELD[:, a], FIELD[:, b])
        assert comp.estimate == pytest.approx(FIELD[:, a].mean() - FIELD[:, b].mean(), rel=1e-12, abs=1e-12)
        assert comp.statistic == pytest.approx(ref.statistic, rel=1e-10)
        assert comp.df == 7 and comp.p_value == pytest.approx(ref.pvalue, rel=1e-10)
        diff = FIELD[:, a] - FIELD[:, b]
        assert comp.effect_size.name == "Cohen's d_z (paired)"
        assert comp.effect_size.value == pytest.approx(diff.mean() / diff.std(ddof=1), rel=1e-10)
        assert comp.cell_a == {"Condition": FOODS[a]} and comp.label is None
        # Holm P values are paired with Bonferroni simultaneous intervals
        ci = ref.confidence_interval(1 - 0.05 / 6)
        assert (comp.ci_low, comp.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
        raw.append(ref.pvalue)
    np.testing.assert_allclose([c.p_adjusted for c in r.comparisons], multipletests(raw, method="holm")[1], rtol=1e-10)
    for method, sm_name in (("bonferroni", "bonferroni"), ("sidak", "sidak"), ("holm-sidak", "holm-sidak"),
                            ("fdr_bh", "fdr_bh")):
        rm = rp.anova_rm_oneway(FIELD, FOODS, posthoc=method)
        np.testing.assert_allclose([c.p_adjusted for c in rm.comparisons], multipletests(raw, method=sm_name)[1],
                                   rtol=1e-10)
    plain = rp.anova_rm_oneway(FIELD, FOODS, posthoc="fdr_bh")
    ci = stats.ttest_rel(FIELD[:, 0], FIELD[:, 1]).confidence_interval(0.95)
    assert (plain.comparisons[0].ci_low, plain.comparisons[0].ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    sid = rp.anova_rm_oneway(FIELD, FOODS, posthoc="sidak", alpha=0.01)
    ci = stats.ttest_rel(FIELD[:, 0], FIELD[:, 1]).confidence_interval(0.99 ** (1 / 6))
    assert (sid.comparisons[0].ci_low, sid.comparisons[0].ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    # comparisons against a control condition only
    ctrl = rp.anova_rm_oneway(FIELD, FOODS, posthoc="bonferroni", control="Fish eye")
    assert [(c.group_a, c.group_b) for c in ctrl.comparisons] == [
        ("Stick insect", "Fish eye"), ("Kangaroo testicle", "Fish eye"), ("Witchetty grub", "Fish eye")]
    ref = stats.ttest_rel(FIELD[:, 0], FIELD[:, 2])
    assert ctrl.comparisons[0].p_adjusted == pytest.approx(min(1, 3 * ref.pvalue), rel=1e-10)
    assert "control 'Fish eye'" in ctrl.comparison_family
    assert any(a.name.startswith("Normality of paired differences") for a in ctrl.assumptions)
    none = rp.anova_rm_oneway(FIELD, FOODS, posthoc=None)
    assert none.comparisons == [] and none.correction is None
    assert not any(a.name.startswith("Normality of paired differences") for a in none.assumptions)


def test_oneway_missing_data_and_errors():
    y = np.vstack([FIELD, [np.nan] * 4, [1.0, np.nan, 3.0, 4.0]])
    r = rp.anova_rm_oneway(y, FOODS, posthoc="none")
    assert r.n_total == 8 and len(r.data_notes) == 1 and "1 subject(s)" in r.data_notes[0]
    assert table_row(r, "Condition")["f"] == pytest.approx(3.79, abs=0.005)
    with pytest.raises(StatsError, match="at least 2"):
        rp.anova_rm_oneway(FIELD[:1], FOODS)
    with pytest.raises(StatsError, match="at least two conditions"):
        rp.anova_rm_oneway(FIELD[:, :1], FOODS[:1])
    with pytest.raises(StatsError, match="condition name"):
        rp.anova_rm_oneway(FIELD, FOODS[:3])
    with pytest.raises(StatsError, match="unique"):
        rp.anova_rm_oneway(FIELD, ["a", "a", "b", "c"])
    with pytest.raises(StatsError, match="numeric"):
        rp.anova_rm_oneway([["x", "y"], ["z", "w"]], ["a", "b"])
    with pytest.raises(StatsError, match="one row per subject"):
        rp.anova_rm_oneway(FIELD[:, 0], ["a"])
    with pytest.raises(StatsError, match="Control condition"):
        rp.anova_rm_oneway(FIELD, FOODS, control="Tofu")
    with pytest.raises(StatsError, match="correction must be"):
        rp.anova_rm_oneway(FIELD, FOODS, correction="bonferroni")
    with pytest.raises(StatsError, match="hf_variant"):
        rp.anova_rm_oneway(FIELD, FOODS, hf_variant="new")
    with pytest.raises(StatsError, match="multiple-comparison method"):
        rp.anova_rm_oneway(FIELD, FOODS, posthoc="tukey")
    with pytest.raises(StatsError, match="alpha"):
        rp.anova_rm_oneway(FIELD, FOODS, alpha=0)


def test_oneway_degenerate_data_give_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        const = rp.anova_rm_oneway(np.full((5, 3), 2.0), list("abc"))
        # every subject changes by exactly the same amount: zero residual
        additive = rp.anova_rm_oneway(np.arange(5.0)[:, None] + np.array([0.0, 1.0, 3.0]), list("abc"))
        # fewer subjects than conditions: Mauchly and Huynh-Feldt are not available
        few = rp.anova_rm_oneway(FIELD[:2], FOODS, correction="hf")
        three = rp.anova_rm_oneway(FIELD[:3], FOODS)
    assert const.p_value is None and any("identical" in w for w in const.warnings)
    assert all(row["ss"] == 0 for row in const.anova_table)
    assert const.effect_sizes[0].value is None and all(c.p_value is None for c in const.comparisons)
    assert additive.p_value is None and table_row(additive, "Residual")["ss"] == 0.0
    assert table_row(additive, "Condition")["ss"] == pytest.approx(5 * ((np.array([0, 1, 3]) - 4 / 3) ** 2).sum())
    assert any("residual variation is zero" in w for w in additive.warnings)
    assert any("could not be tested" in w for w in additive.warnings)
    assert additive.extra["sphericity"]["eps_gg"] is None
    row = table_row(few, "Condition")
    assert row["eps_hf"] is None and row["p_hf"] is None and row["eps_gg"] is not None
    assert few.extra["correction_applied"] == "gg" and any("Huynh-Feldt" in w for w in few.warnings)
    assert few.extra["sphericity"]["mauchly_w"] is None
    assert next(a for a in few.assumptions if a.name.startswith("Sphericity")).status == "warn"
    assert any("only 2 subjects" in w for w in few.warnings)
    assert three.extra["sphericity"]["p"] is None and three.p_value is not None
    for res in (const, additive, few, three):
        json.loads(res.finalize().model_dump_json())


# =============================================================================
# Mixed (split-plot) ANOVA
# =============================================================================
def split_plot_scratch(y, codes):
    """Classical split-plot sums of squares for a balanced design (Winer et al. 1991)."""
    n_per, p = np.bincount(codes)[0], y.shape[1]
    g = codes.max() + 1
    grand = y.mean()
    group_means = np.array([y[codes == j].mean() for j in range(g)])
    time_means = y.mean(axis=0)
    cell = np.vstack([y[codes == j].mean(axis=0) for j in range(g)])
    subj = y.mean(axis=1)
    ss_a = n_per * p * ((group_means - grand) ** 2).sum()
    ss_s = p * ((subj - group_means[codes]) ** 2).sum()
    ss_b = n_per * g * ((time_means - grand) ** 2).sum()
    ss_ab = n_per * ((cell - group_means[:, None] - time_means[None, :] + grand) ** 2).sum()
    ss_res = ((y - cell[codes] - subj[:, None] + group_means[codes][:, None]) ** 2).sum()
    return ss_a, ss_s, ss_b, ss_ab, ss_res


def dummy_model_ss(y, codes):
    """Type III SS of an (unbalanced) split-plot design from a full least-squares model with
    sum-coded group, time, group x time and subject-within-group columns (nested fits)."""
    n, p = y.shape
    g = codes.max() + 1

    def sum_code(idx, k):
        return np.vstack([np.eye(k - 1), -np.ones((1, k - 1))])[idx]

    subj_idx = np.repeat(np.arange(n), p)
    grp = np.repeat(codes, p)
    time = np.tile(np.arange(p), n)
    x_g = sum_code(grp, g)
    x_t = sum_code(time, p)
    x_gt = np.einsum("ni,nj->nij", x_g, x_t).reshape(n * p, -1)
    blocks = []
    for j in range(g):
        members = np.flatnonzero(codes == j)
        if members.size < 2:
            continue
        local = np.searchsorted(members, subj_idx.clip(members.min(), members.max()))
        cols = sum_code(np.where(grp == j, local, 0), members.size) * (grp == j)[:, None]
        blocks.append(cols)
    x_s = np.column_stack(blocks)
    yy = y.ravel()

    def rss(*parts):
        x = np.column_stack([np.ones(n * p), *parts])
        beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
        return float(((yy - x @ beta) ** 2).sum())

    full = rss(x_g, x_t, x_gt, x_s)
    return {
        "between": rss(x_t, x_gt, x_s) - full,
        "within": rss(x_g, x_gt, x_s) - full,
        "interaction": rss(x_g, x_t, x_s) - full,
        "subjects": rss(x_g, x_t, x_gt) - full,
        "residual": full,
    }


def test_mixed_balanced_matches_classical_split_plot_formulas():
    frame = mixed_frame(seed=5, sizes=(6, 6, 6))
    r = rp.anova_mixed(frame, "y", "id", "group", "time")
    y, codes = wide_from_long(frame)
    ss_a, ss_s, ss_b, ss_ab, ss_res = split_plot_scratch(y, codes)
    assert [row["source"] for row in r.anova_table] == [
        "group", "Subjects within group", "time", "group x time", "Residual", "Total"]
    assert table_row(r, "group")["ss"] == pytest.approx(ss_a, rel=1e-10)
    assert table_row(r, "Subjects within group")["ss"] == pytest.approx(ss_s, rel=1e-10)
    assert table_row(r, "time")["ss"] == pytest.approx(ss_b, rel=1e-10)
    assert table_row(r, "group x time")["ss"] == pytest.approx(ss_ab, rel=1e-10)
    assert table_row(r, "Residual")["ss"] == pytest.approx(ss_res, rel=1e-10)
    assert [row["df"] for row in r.anova_table] == [2, 15, 3, 6, 45, 71]
    assert ss_a + ss_s + ss_b + ss_ab + ss_res == pytest.approx(table_row(r, "Total")["ss"], rel=1e-10)
    f_a = (ss_a / 2) / (ss_s / 15)
    f_b = (ss_b / 3) / (ss_res / 45)
    f_ab = (ss_ab / 6) / (ss_res / 45)
    assert table_row(r, "group")["f"] == pytest.approx(f_a, rel=1e-10)
    assert table_row(r, "group")["p"] == pytest.approx(stats.f.sf(f_a, 2, 15), rel=1e-10)
    assert table_row(r, "group")["error_term"] == "Subjects within group"
    assert table_row(r, "time")["f"] == pytest.approx(f_b, rel=1e-10)
    assert table_row(r, "time")["p"] == pytest.approx(stats.f.sf(f_b, 3, 45), rel=1e-10)
    assert table_row(r, "group x time")["f"] == pytest.approx(f_ab, rel=1e-10)
    assert table_row(r, "group x time")["p"] == pytest.approx(stats.f.sf(f_ab, 6, 45), rel=1e-10)
    # balanced: Type II and III are identical
    r2 = rp.anova_mixed(frame, "y", "id", "group", "time", ss_type=2)
    assert table_row(r2, "time")["ss"] == pytest.approx(ss_b, rel=1e-10)
    # between-subjects test = one-way ANOVA of the subject means
    ref = stats.f_oneway(*[y[codes == j].mean(axis=1) for j in range(3)])
    assert table_row(r, "group")["f"] == pytest.approx(ref.statistic, rel=1e-10)
    # headline = interaction with the Greenhouse-Geisser correction
    row = table_row(r, "group x time")
    assert r.statistic == row["f"] and r.p_value == row["p_gg"]
    assert (r.df, r.df2) == pytest.approx((6 * row["eps_gg"], 45 * row["eps_gg"]))
    assert r.extra["headline_term"] == "group x time" and not r.warnings
    # effect sizes
    eff = {(e.name, e.target): e.value for e in r.effect_sizes}
    assert eff[("partial eta squared", "time")] == pytest.approx(ss_b / (ss_b + ss_res), rel=1e-10)
    assert eff[("partial eta squared", "group")] == pytest.approx(ss_a / (ss_a + ss_s), rel=1e-10)
    assert eff[("generalized eta squared", "group x time")] == pytest.approx(ss_ab / (ss_ab + ss_s + ss_res), rel=1e-10)
    assert eff[("generalized eta squared", "group")] == pytest.approx(ss_a / (ss_a + ss_s + ss_res), rel=1e-10)


def test_mixed_balanced_agrees_with_reml_mixed_model():
    """With balanced data and a positive subject variance, the random-intercept mixed model
    fitted by REML reproduces the split-plot F tests: Wald chi-square / df = F. Tolerance
    1e-3 relative (the mixed model is fitted by numerical optimisation)."""
    frame = mixed_frame(seed=8, sizes=(7, 7, 7), levels=("t0", "t1", "t2"))
    r = rp.anova_mixed(frame, "y", "id", "group", "time", correction="none")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("y ~ C(group, Sum) * C(time, Sum)", frame, groups="id").fit(reml=True)
        wald = fit.wald_test_terms(scalar=True).table
    for ours, theirs in (("group", "C(group, Sum)"), ("time", "C(time, Sum)"),
                         ("group x time", "C(group, Sum):C(time, Sum)")):
        row = table_row(r, ours)
        assert float(wald.loc[theirs, "statistic"]) / row["df"] == pytest.approx(row["f"], rel=1e-3)
    # REML residual variance = MS residual; random-intercept variance = (MS subjects - MS residual) / p
    assert fit.scale == pytest.approx(table_row(r, "Residual")["ms"], rel=1e-3)
    ms_s = table_row(r, "Subjects within group")["ms"]
    assert float(fit.cov_re.iloc[0, 0]) == pytest.approx((ms_s - table_row(r, "Residual")["ms"]) / 3, rel=2e-3)


@pytest.mark.parametrize("sizes", [(5, 7, 6), (3, 9), (2, 4, 8, 3)])
def test_mixed_unbalanced_type3_matches_full_least_squares_model_and_ols(sizes):
    frame = mixed_frame(seed=sum(sizes), sizes=sizes)
    r = rp.anova_mixed(frame, "y", "id", "group", "time", ss_type=3)
    y, codes = wide_from_long(frame)
    ref = dummy_model_ss(y, codes)
    assert table_row(r, "group")["ss"] == pytest.approx(ref["between"], rel=1e-8)
    assert table_row(r, "Subjects within group")["ss"] == pytest.approx(ref["subjects"], rel=1e-8)
    assert table_row(r, "time")["ss"] == pytest.approx(ref["within"], rel=1e-8)
    assert table_row(r, "group x time")["ss"] == pytest.approx(ref["interaction"], rel=1e-8)
    assert table_row(r, "Residual")["ss"] == pytest.approx(ref["residual"], rel=1e-8)
    n, g, p = len(codes), len(sizes), y.shape[1]
    assert [row["df"] for row in r.anova_table] == [g - 1, n - g, p - 1, (g - 1) * (p - 1), (n - g) * (p - 1), n * p - 1]
    # statsmodels OLS on each orthonormal contrast score: Type III intercept = within effect,
    # group term = interaction; Type II within effect = intercept of the intercept-only model
    z = y @ rp._orthonormal_contrasts(p)
    grp = pd.Series(codes).astype(str)
    ss_int3 = ss_grp = ss_int2 = ss_err = 0.0
    for j in range(p - 1):
        d = pd.DataFrame({"z": z[:, j], "g": grp})
        table = anova_lm(smf.ols("z ~ C(g, Sum)", data=d).fit(), typ=3)
        ss_int3 += table.loc["Intercept", "sum_sq"]
        ss_grp += table.loc["C(g, Sum)", "sum_sq"]
        ss_err += table.loc["Residual", "sum_sq"]
        ss_int2 += n * d["z"].mean() ** 2
    assert table_row(r, "time")["ss"] == pytest.approx(ss_int3, rel=1e-8)
    assert table_row(r, "group x time")["ss"] == pytest.approx(ss_grp, rel=1e-8)
    assert table_row(r, "Residual")["ss"] == pytest.approx(ss_err, rel=1e-8)
    r2 = rp.anova_mixed(frame, "y", "id", "group", "time", ss_type=2)
    assert table_row(r2, "time")["ss"] == pytest.approx(ss_int2, rel=1e-8)
    assert table_row(r2, "group x time")["ss"] == pytest.approx(ss_grp, rel=1e-8)
    assert table_row(r2, "group")["ss"] == pytest.approx(ref["between"], rel=1e-8)
    assert abs(table_row(r2, "time")["ss"] - ss_int3) > 1e-6  # the types really differ here
    assert any("unequal" in w for w in r.warnings) and "Type III" in r.methods_text
    assert next(a for a in r.assumptions if a.name == "Balanced groups").status == "warn"


def test_mixed_sphericity_uses_the_pooled_within_group_covariance():
    frame = mixed_frame(seed=21, sizes=(6, 9, 5))
    y, codes = wide_from_long(frame)
    n, p, g = y.shape[0], y.shape[1], 3
    q = p - 1
    pooled = sum((np.cov(y[codes == j], rowvar=False) * ((codes == j).sum() - 1)) for j in range(g)) / (n - g)
    eps = box_epsilon(pooled)
    r = rp.anova_mixed(frame, "y", "id", "group", "time")
    sph = r.extra["sphericity"]
    assert sph["eps_gg"] == pytest.approx(eps, rel=1e-10)
    basis = rp._orthonormal_contrasts(p)
    s = basis.T @ pooled @ basis
    w = np.linalg.det(s) / (np.trace(s) / q) ** q
    assert sph["mauchly_w"] == pytest.approx(w, rel=1e-9)
    assert sph["chi2"] == pytest.approx(-((n - g) - (2 * q * q + q + 2) / (6 * q)) * math.log(w), rel=1e-9)
    # Huynh-Feldt: original formula reported by default, Lecoutre's correction on request
    hf_orig = (n * q * eps - 2) / (q * (n - g - q * eps))
    hf_lec = ((n - g + 1) * q * eps - 2) / (q * (n - g - q * eps))
    assert sph["eps_hf_original_uncapped"] == pytest.approx(hf_orig, rel=1e-10)
    assert sph["eps_hf_lecoutre_uncapped"] == pytest.approx(hf_lec, rel=1e-10)
    assert hf_lec < hf_orig
    assert sph["eps_hf"] == pytest.approx(min(1.0, hf_orig), rel=1e-10) and sph["hf_variant"] == "original"
    r_lec = rp.anova_mixed(frame, "y", "id", "group", "time", hf_variant="lecoutre", correction="hf")
    assert r_lec.extra["sphericity"]["eps_hf"] == pytest.approx(min(1.0, hf_lec), rel=1e-10)
    row = table_row(r_lec, "time")
    assert row["p_hf"] == pytest.approx(stats.f.sf(row["f"], q * row["eps_hf"], q * (n - g) * row["eps_hf"]), rel=1e-10)
    assert r_lec.p_value == table_row(r_lec, "group x time")["p_hf"]
    # the same epsilon is applied to the within factor and to the interaction
    assert table_row(r, "time")["eps_gg"] == table_row(r, "group x time")["eps_gg"] == sph["eps_gg"]
    inter = table_row(r, "group x time")
    assert inter["p_gg"] == pytest.approx(
        stats.f.sf(inter["f"], q * (g - 1) * eps, q * (n - g) * eps), rel=1e-10)


def test_mixed_two_levels_reduces_to_simple_tests():
    frame = mixed_frame(seed=4, sizes=(6, 8), levels=("pre", "post"))
    r = rp.anova_mixed(frame, "y", "id", "group", "time")
    y, codes = wide_from_long(frame)
    diff = y[:, 0] - y[:, 1]
    # interaction = two-sample t test on the change scores; between = t test on the subject sums
    t_int = stats.ttest_ind(diff[codes == 0], diff[codes == 1])
    t_btw = stats.ttest_ind(y[codes == 0].sum(axis=1), y[codes == 1].sum(axis=1))
    assert table_row(r, "group x time")["f"] == pytest.approx(t_int.statistic**2, rel=1e-10)
    assert table_row(r, "group x time")["p"] == pytest.approx(t_int.pvalue, rel=1e-10)
    assert table_row(r, "group")["f"] == pytest.approx(t_btw.statistic**2, rel=1e-10)
    assert table_row(r, "group")["p"] == pytest.approx(t_btw.pvalue, rel=1e-10)
    assert r.extra["correction_applied"] == "none" and r.p_value == table_row(r, "group x time")["p"]
    assert table_row(r, "time")["eps_gg"] == 1.0


def test_mixed_posthoc_families():
    frame = mixed_frame(seed=9, sizes=(6, 8), levels=("t0", "t1", "t2"))
    y, codes = wide_from_long(frame)
    levels = ["t0", "t1", "t2"]
    # groups at each level: with two groups this is the ordinary two-sample t test
    r = rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="holm", posthoc_compare="between_per_level")
    raw = []
    for k, lv in enumerate(levels):
        comp = r.comparison(f"{lv}: grp0", f"{lv}: grp1")
        ref = stats.ttest_ind(y[codes == 0, k], y[codes == 1, k])
        assert comp.label == f"time = {lv}"
        assert comp.cell_a == {"time": lv, "group": "grp0"} and comp.cell_b == {"time": lv, "group": "grp1"}
        assert comp.statistic == pytest.approx(ref.statistic, rel=1e-10) and comp.df == 12
        assert comp.p_value == pytest.approx(ref.pvalue, rel=1e-10)
        assert comp.estimate == pytest.approx(y[codes == 0, k].mean() - y[codes == 1, k].mean(), rel=1e-12)
        assert comp.effect_size.name == "Hedges' g" and comp.effect_size.ci_low < comp.effect_size.ci_high
        raw.append(ref.pvalue)
    np.testing.assert_allclose([c.p_adjusted for c in r.comparisons], multipletests(raw, method="holm")[1], rtol=1e-10)
    assert "one family" in r.comparison_family and r.correction == "Holm"
    # within levels inside each group: paired t tests on that group's subjects
    r = rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="bonferroni", posthoc_compare="within_per_group")
    assert len(r.comparisons) == 6
    comp = r.comparison("grp1: t0", "grp1: t2")
    ref = stats.ttest_rel(y[codes == 1, 0], y[codes == 1, 2])
    assert comp.label == "group = grp1" and comp.df == 7
    assert comp.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert comp.p_adjusted == pytest.approx(min(1.0, 6 * ref.pvalue), rel=1e-10)
    crit = stats.t.ppf(1 - 0.05 / 12, 7)
    assert comp.ci_high - comp.estimate == pytest.approx(crit * comp.se, rel=1e-10)
    assert r.comparison("grp0: t0", "grp0: t1").df == 5
    # between-subjects main effect: t test on the subject averages
    r = rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="sidak", posthoc_compare="between_main")
    ref = stats.ttest_ind(y[codes == 0].mean(axis=1), y[codes == 1].mean(axis=1))
    assert len(r.comparisons) == 1 and r.comparisons[0].statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert r.comparisons[0].p_adjusted == pytest.approx(ref.pvalue, rel=1e-10)
    assert r.comparisons[0].statistic**2 == pytest.approx(table_row(r, "group")["f"], rel=1e-10)
    # within main effect: intercept of an OLS of the difference scores on the (sum-coded) group
    r3 = rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="fdr_bh", posthoc_compare="within_main")
    r2 = rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="fdr_bh", posthoc_compare="within_main", ss_type=2)
    d = pd.DataFrame({"d": y[:, 0] - y[:, 2], "g": pd.Series(codes).astype(str)})
    fit = smf.ols("d ~ C(g, Sum)", data=d).fit()
    comp = r3.comparison("t0", "t2")
    assert comp.estimate == pytest.approx(fit.params["Intercept"], rel=1e-10)
    assert comp.se == pytest.approx(fit.bse["Intercept"], rel=1e-10)
    assert comp.p_value == pytest.approx(fit.pvalues["Intercept"], rel=1e-8)
    assert comp.effect_size.name == "Cohen's d_z (paired)"
    assert comp.effect_size.value == pytest.approx(comp.estimate / math.sqrt(fit.mse_resid), rel=1e-10)
    comp2 = r2.comparison("t0", "t2")
    assert comp2.estimate == pytest.approx(d["d"].mean(), rel=1e-10)
    assert comp2.se == pytest.approx(math.sqrt(fit.mse_resid / len(d)), rel=1e-10)
    assert "unweighted" in r3.comparison_family and "weighted by group size" in r2.comparison_family
    with pytest.raises(StatsError, match="posthoc_compare"):
        rp.anova_mixed(frame, "y", "id", "group", "time", posthoc="holm", posthoc_compare="everything")
    # three groups: error pooled over all groups at that level (df = N - g)
    f3 = mixed_frame(seed=10, sizes=(4, 5, 6), levels=("t0", "t1"))
    r = rp.anova_mixed(f3, "y", "id", "group", "time", posthoc="holm")
    y3, c3 = wide_from_long(f3)
    comp = r.comparison("t1: grp0", "t1: grp2")
    mse = sum(((y3[c3 == j, 1] - y3[c3 == j, 1].mean()) ** 2).sum() for j in range(3)) / 12
    assert comp.df == 12 and comp.se == pytest.approx(math.sqrt(mse * (1 / 4 + 1 / 6)), rel=1e-10)
    assert len(r.comparisons) == 6


def test_mixed_data_handling_and_errors():
    frame = mixed_frame(seed=2, sizes=(4, 4))
    # the same subject labels reused in both groups are treated as different subjects
    reused = frame.assign(id=frame["id"].str.replace(r"g\ds", "s", regex=True))
    r = rp.anova_mixed(reused, "y", "id", "group", "time")
    base = rp.anova_mixed(frame, "y", "id", "group", "time")
    assert r.statistic == pytest.approx(base.statistic) and any("same subject label" in n for n in r.data_notes)
    assert not base.data_notes
    # incomplete subjects are dropped and reported; missing rows are counted
    partial = frame.drop(index=[0]).copy()
    partial.loc[5, "y"] = np.nan
    r = rp.anova_mixed(partial, "y", "id", "group", "time")
    assert r.n_total == 6 and r.extra["n_per_group"] == {"grp0": 2, "grp1": 4}
    assert any("2 subject(s)" in n for n in r.data_notes) and any("1 row(s)" in n for n in r.data_notes)
    assert r.extra["factors"] == ["group", "time"] and len(r.extra["cell_means"]) == 8
    assert set(r.extra["cell_means"][0]) == {"group", "time", "mean", "sd", "sem", "n"}
    # levels of plain text columns are ordered by first appearance (row 0, a "0h" row, was removed)
    assert r.extra["levels"]["time"] == ["1h", "4h", "24h", "0h"]
    assert r.descriptives[0]["group"] == "grp0: 1h" and r.n["grp0: 0h"] == 2
    dup = pd.concat([frame, frame.iloc[[3]]])
    with pytest.raises(StatsError, match="more than one value"):
        rp.anova_mixed(dup, "y", "id", "group", "time")
    with pytest.raises(StatsError, match="only one group"):
        rp.anova_mixed(frame[frame.group == "grp0"], "y", "id", "group", "time")
    with pytest.raises(StatsError, match="at least two levels"):
        rp.anova_mixed(frame[frame.time == "0h"], "y", "id", "group", "time")
    with pytest.raises(StatsError, match="not found"):
        rp.anova_mixed(frame, "y", "id", "group", "week")
    with pytest.raises(StatsError, match="only one role"):
        rp.anova_mixed(frame, "y", "id", "group", "group")
    with pytest.raises(StatsError, match="no complete rows"):
        rp.anova_mixed(frame.assign(y=np.nan), "y", "id", "group", "time")
    with pytest.raises(TypeError):
        rp.anova_mixed(frame.to_dict("list"), "y", "id", "group", "time")
    lost = frame[~((frame.group == "grp1") & (frame.time == "24h"))]
    with pytest.raises(StatsError, match="No subject with complete data"):
        rp.anova_mixed(lost, "y", "id", "group", "time")
    one_each = frame[frame.id.isin(["g0s0", "g1s0"])]
    with pytest.raises(StatsError, match="one subject per group"):
        rp.anova_mixed(one_each, "y", "id", "group", "time")
    # a group with a single subject is allowed but flagged
    lonely = frame[frame.id.isin(["g0s0", "g1s0", "g1s1", "g1s2"])]
    r = rp.anova_mixed(lonely, "y", "id", "group", "time", posthoc="holm", posthoc_compare="within_per_group")
    assert any("single subject" in w for w in r.warnings) and any("could not be tested" in w for w in r.warnings)
    assert sum(c.p_value is None for c in r.comparisons) == 6
    # labels: identical names are disambiguated, custom labels are used in the table
    r = rp.anova_mixed(frame, "y", "id", "group", "time", factor_labels={"group": "Drug", "time": "Hours"})
    assert [row["source"] for row in r.anova_table][:4] == ["Drug", "Subjects within Drug", "Hours", "Drug x Hours"]
    r = rp.anova_mixed(frame, "y", "id", "group", "time", factor_labels={"group": "X", "time": "X"})
    assert r.anova_table[0]["source"] == "X (between)" and r.anova_table[2]["source"] == "X (within)"


def test_mixed_degenerate_data_and_many_levels():
    frame = mixed_frame(seed=3, sizes=(3, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        const = rp.anova_mixed(frame.assign(y=4.0), "y", "id", "group", "time", posthoc="holm")
        # no within-subject noise at all: every subject follows its group's profile exactly
        profile = frame.assign(y=frame["id"].str[-1].astype(float) + frame["time"].map(
            {"0h": 0.0, "1h": 1.0, "4h": 0.5, "24h": 2.0}) * (1 + (frame["group"] == "grp1")))
        exact = rp.anova_mixed(profile, "y", "id", "group", "time", posthoc="holm", posthoc_compare="within_main")
        # subjects within a group have identical averages: the between test is undefined
        flat = frame.assign(y=frame["time"].map({"0h": 1.0, "1h": -1.0, "4h": 1.0, "24h": -1.0})
                            * (frame["id"].str[-1].astype(float) - 1) + (frame["group"] == "grp1") * 2.0)
        nobetween = rp.anova_mixed(flat, "y", "id", "group", "time")
    assert const.p_value is None and any("identical" in w for w in const.warnings)
    assert all(row["ss"] == 0 for row in const.anova_table)
    assert exact.p_value is None and table_row(exact, "Residual")["ss"] == 0.0
    assert table_row(exact, "group")["f"] is not None
    assert any("residual variation is zero" in w for w in exact.warnings)
    assert all(c.p_value is None for c in exact.comparisons)
    assert table_row(nobetween, "group")["f"] is None and table_row(nobetween, "time")["f"] is not None
    assert any("cannot be tested" in w for w in nobetween.warnings)
    for res in (const, exact, nobetween):
        json.loads(res.finalize().model_dump_json())
    # more than six repeated levels: the per-level variance checks are summarised in one entry
    levels = tuple(f"d{i}" for i in range(8))
    many = mixed_frame(seed=6, sizes=(9, 9), levels=levels)
    r = rp.anova_mixed(many, "y", "id", "group", "time")
    checks = [a for a in r.assumptions if a.name.startswith("Equal variances between groups")]
    assert len(checks) == 1 and "each of the 8 levels" in checks[0].detail
    few = rp.anova_mixed(mixed_frame(seed=6, sizes=(9, 9)), "y", "id", "group", "time")
    checks = [a for a in few.assumptions if a.name.startswith("Equal variances between groups")]
    assert [a.target for a in checks] == ["time = 0h", "time = 1h", "time = 4h", "time = 24h"]
    y, codes = wide_from_long(mixed_frame(seed=6, sizes=(9, 9)))
    ref = stats.levene(y[codes == 0, 1], y[codes == 1, 1], center="median")
    assert checks[1].p_value == pytest.approx(ref.pvalue, rel=1e-10)
    # heteroscedastic at many levels -> the summary entry warns
    noisy = many.copy()
    noisy.loc[noisy.group == "grp1", "y"] = 10 + (noisy.loc[noisy.group == "grp1", "y"] - 10) * 12
    r = rp.anova_mixed(noisy, "y", "id", "group", "time")
    summary = next(a for a in r.assumptions if a.name.startswith("Equal variances between groups"))
    assert summary.status == "warn" and "unequal spread" in summary.detail


# =============================================================================
# Two-way repeated measures (both factors within subjects)
# =============================================================================
def rm2_frame(seed=0, n=7, a=3, b=4):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        base = rng.normal(0, 2)
        for i in range(a):
            for j in range(b):
                rows.append((f"s{s}", f"a{i}", f"b{j}", base + 0.5 * i + 0.2 * j * i + rng.normal()))
    return pd.DataFrame(rows, columns=["id", "drug", "time", "y"])


def test_rm_twoway_matches_statsmodels_anovarm():
    frame = rm2_frame()
    r = rp.anova_rm_twoway(frame, "y", "id", "drug", "time", correction="none")
    ref = AnovaRM(frame, "y", "id", within=["drug", "time"]).fit().anova_table
    for ours, theirs in (("drug", "drug"), ("time", "time"), ("drug x time", "drug:time")):
        row = table_row(r, ours)
        assert row["f"] == pytest.approx(ref.loc[theirs, "F Value"], rel=1e-6)
        assert (row["df"], row["df_error"]) == (ref.loc[theirs, "Num DF"], ref.loc[theirs, "Den DF"])
        assert row["p"] == pytest.approx(ref.loc[theirs, "Pr > F"], rel=1e-6)
    # the sums of squares partition the total
    assert sum(row["ss"] for row in r.anova_table if row["kind"] != "total") == pytest.approx(
        table_row(r, "Total")["ss"], rel=1e-10)
    assert r.statistic == table_row(r, "drug x time")["f"] and r.p_value == table_row(r, "drug x time")["p"]
    # each effect has its own sphericity estimate = that of a one-way analysis of the marginal means
    y = frame.pivot(index="id", columns=["drug", "time"], values="y")
    drug_means = np.column_stack([y[d].mean(axis=1) for d in ("a0", "a1", "a2")])
    one = rp.anova_rm_oneway(drug_means, ["a0", "a1", "a2"], posthoc="none")
    assert table_row(r, "drug")["eps_gg"] == pytest.approx(one.extra["sphericity"]["eps_gg"], rel=1e-10)
    assert table_row(r, "drug")["f"] == pytest.approx(table_row(one, "Condition")["f"], rel=1e-10)
    assert r.extra["sphericity"][0]["mauchly_w"] == pytest.approx(one.extra["sphericity"]["mauchly_w"], rel=1e-9)
    auto = rp.anova_rm_twoway(frame, "y", "id", "drug", "time")
    row = table_row(auto, "drug x time")
    assert auto.p_value == row["p_gg"] and auto.df == pytest.approx(6 * row["eps_gg"])
    assert "Greenhouse-Geisser" in auto.interpretation
    gen = next(e for e in auto.effect_sizes if e.name == "generalized eta squared" and e.target == "drug")
    errors = sum(row["ss"] for row in auto.anova_table if row["kind"] in {"error", "subjects"})
    assert gen.value == pytest.approx(table_row(auto, "drug")["ss"] / (table_row(auto, "drug")["ss"] + errors))


def test_rm_twoway_errors_and_degenerate_cases():
    frame = rm2_frame(n=4, a=2, b=2)
    r = rp.anova_rm_twoway(frame, "y", "id", "drug", "time", correction="hf")
    assert all(row["eps_gg"] == 1.0 for row in r.anova_table if row["kind"] in {"within", "interaction"})
    incomplete = frame.drop(index=[0])
    r = rp.anova_rm_twoway(incomplete, "y", "id", "drug", "time")
    assert r.n_total == 3 and any("1 subject(s)" in n for n in r.data_notes)
    with pytest.raises(StatsError, match="no data at all"):
        rp.anova_rm_twoway(frame[~((frame.drug == "a0") & (frame.time == "b1"))], "y", "id", "drug", "time")
    with pytest.raises(StatsError, match="more than one value"):
        rp.anova_rm_twoway(pd.concat([frame, frame.iloc[[2]]]), "y", "id", "drug", "time")
    with pytest.raises(StatsError, match="at least two levels"):
        rp.anova_rm_twoway(frame[frame.drug == "a0"], "y", "id", "drug", "time")
    with pytest.raises(StatsError, match="was not found"):
        rp.anova_rm_twoway(frame, "y", "id", "drug", "nope")
    with pytest.raises(StatsError, match="only one role"):
        rp.anova_rm_twoway(frame, "y", "id", "drug", "drug")
    with pytest.raises(StatsError, match="Only 1 subject"):
        rp.anova_rm_twoway(frame[frame.id == "s0"], "y", "id", "drug", "time")
    with pytest.raises(StatsError, match="no complete rows"):
        rp.anova_rm_twoway(frame.assign(y=np.nan), "y", "id", "drug", "time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        const = rp.anova_rm_twoway(frame.assign(y=1.0), "y", "id", "drug", "time")
        nonoise = rp.anova_rm_twoway(
            frame.assign(y=frame["id"].str[1:].astype(float) + (frame["drug"] == "a1") * 2.0), "y", "id", "drug", "time")
    assert const.p_value is None and any("identical" in w for w in const.warnings)
    assert table_row(nonoise, "drug")["f"] is None and any("cannot be tested" in w for w in nonoise.warnings)
    json.loads(const.finalize().model_dump_json())


# =============================================================================
# Registry (table-level) analyses
# =============================================================================
def check_registry_result(result, key, table_name):
    assert result.analysis == key and result.table == table_name
    assert result.interpretation and result.why_chosen and result.methods_text and result.citations
    assert len(result.assumptions) >= 1 and result.significant is not None
    payload = json.loads(result.model_dump_json())
    assert payload["anova_table"] and payload["extra"]["factors"]


def test_registry_rm_oneway_on_column_table(run):
    table = ColumnTable.from_dict("Bushtucker", {name: FIELD[:, j] for j, name in enumerate(FOODS)})
    r = run(table, "anova_rm_oneway")
    check_registry_result(r, "anova_rm_oneway", "Bushtucker")
    assert r.params == {"groups": None, "correction": "auto", "posthoc": "holm", "control": None, "alpha": 0.05}
    assert r.p_value == pytest.approx(0.063, abs=0.0005) and len(r.comparisons) == 6
    assert all(c.id.startswith(r.id) and c.significant is not None for c in r.comparisons)
    r = run(table, "anova_rm_oneway", groups=FOODS[:3], control=FOODS[0], correction="none", posthoc="bonferroni")
    assert len(r.comparisons) == 2 and r.df == 2 and list(r.n) == FOODS[:3]
    direct = rp.anova_rm_oneway(FIELD[:, :3], FOODS[:3], correction="none", posthoc="bonferroni", control=FOODS[0])
    assert r.statistic == pytest.approx(direct.statistic)
    # an excluded cell removes that subject (row) and is reported; unequal column lengths are fine
    table.exclude([(2, FOODS[1])])
    r = run(table, "anova_rm_oneway", posthoc="none")
    assert r.n_total == 7 and any("left out" in n for n in r.data_notes) and any("excluded" in n for n in r.data_notes)
    ragged = ColumnTable.from_dict("r", {"a": [1, 2, 4, 3, 6], "b": [2, 4, 5, 5], "c": [3, 3, 7, 4]})
    r = run(ragged, "anova_rm_oneway")
    assert r.n_total == 4 and len(r.data_notes) == 1
    with pytest.raises(StatsError, match="Unknown column"):
        run(table, "anova_rm_oneway", groups=["nope", FOODS[0]])
    with pytest.raises(StatsError, match="only once"):
        run(table, "anova_rm_oneway", groups=[FOODS[0], FOODS[0]])
    with pytest.raises(StatsError, match="at least two columns"):
        run(table, "anova_rm_oneway", groups=FOODS[0])


def test_registry_mixed_on_grouped_table(run):
    frame = mixed_frame(seed=14, sizes=(5, 6), levels=("0 h", "4 h", "24 h"))
    y, codes = wide_from_long(frame)
    cells = {lv: {grp: list(y[codes == j, k]) for j, grp in enumerate(("Control", "NP"))}
             for k, lv in enumerate(("0 h", "4 h", "24 h"))}
    table = GroupedTable.from_cells("Uptake", cells, row_label="Time", column_label="Treatment")
    r = run(table, "anova_mixed")
    check_registry_result(r, "anova_mixed", "Uptake")
    direct = rp.anova_mixed(frame, "y", "id", "group", "time")
    assert r.statistic == pytest.approx(direct.statistic, rel=1e-12)
    assert r.p_value == pytest.approx(direct.p_value, rel=1e-12)
    assert [row["source"] for row in r.anova_table][:4] == [
        "Treatment", "Subjects within Treatment", "Time", "Treatment x Time"]
    assert r.extra["factors"] == ["column", "row"]
    assert r.extra["factor_labels"] == {"column": "Treatment", "row": "Time"}
    assert r.extra["n_per_group"] == {"Control": 5, "NP": 6} and r.n_total == 11
    assert r.data_notes[0].startswith("Layout: rows = repeated levels (Time)")
    assert not any("same subject label" in n for n in r.data_notes)
    r = run(table, "anova_mixed", posthoc="holm")
    comp = r.comparison("4 h: Control", "4 h: NP")
    assert comp.label == "Time = 4 h" and comp.cell_a == {"row": "4 h", "column": "Control"}
    assert comp.cell_b == {"row": "4 h", "column": "NP"} and len(r.comparisons) == 3
    r = run(table, "anova_mixed", posthoc="bonferroni", posthoc_compare="within_per_group", ss_type=2,
            correction="hf")
    assert len(r.comparisons) == 6 and r.comparisons[0].label == "Treatment = Control"
    # excluding one cell drops that subject
    table.exclude([(1, "NP:2")])
    r = run(table, "anova_mixed")
    assert r.n_total == 10 and any("1 subject(s)" in n for n in r.data_notes)


def test_registry_mixed_and_rm_twoway_on_multiple_variables(run):
    frame = mixed_frame(seed=15, sizes=(4, 5))
    table = MultipleVariablesTable("long", frame)
    r = run(table, "anova_mixed")
    check_registry_result(r, "anova_mixed", "long")
    direct = rp.anova_mixed(frame, "y", "id", "group", "time")
    assert r.statistic == pytest.approx(direct.statistic)
    notes = [n for n in r.data_notes if "automatically" in n]
    assert len(notes) == 4 and "'id'" in notes[1] and "'group'" in notes[2] and "'time'" in notes[3]
    r = run(table, "anova_mixed", value="y", subject="id", between="group", within="time", posthoc="holm",
            posthoc_compare="between_main")
    assert not any("automatically" in n for n in r.data_notes) and len(r.comparisons) == 1
    # numeric subject / time columns work when named, and are guessed when unnamed
    numeric = frame.assign(id=pd.factorize(frame["id"])[0] + 1.0, time=frame["time"].map(
        {"0h": 0.0, "1h": 1.0, "4h": 4.0, "24h": 24.0}))
    rn = run(MultipleVariablesTable("num", numeric[["group", "id", "time", "y"]]), "anova_mixed")
    assert rn.statistic == pytest.approx(direct.statistic)
    assert rn.data_notes[0].startswith("Outcome column chosen automatically: 'y'")
    assert rn.extra["levels"]["time"] == ["0", "1", "4", "24"]
    with pytest.raises(StatsError, match="not in the table"):
        run(table, "anova_mixed", within="week")
    with pytest.raises(StatsError, match="between-subjects column"):
        run(MultipleVariablesTable("nb", frame[["id", "time", "y"]]), "anova_mixed")
    with pytest.raises(StatsError, match="within-subjects column"):
        run(MultipleVariablesTable("nw", frame[["id", "group", "y"]].drop_duplicates("id")), "anova_mixed")
    with pytest.raises(StatsError, match="no numeric column"):
        run(MultipleVariablesTable("nn", frame[["id", "group", "time"]]), "anova_mixed")
    with pytest.raises(StatsError, match="subject column"):
        run(MultipleVariablesTable("ns", frame[["y"]]), "anova_mixed")

    two = rm2_frame(n=5, a=2, b=3)
    t2 = MultipleVariablesTable("rm2", two)
    r = run(t2, "anova_rm_twoway")
    check_registry_result(r, "anova_rm_twoway", "rm2")
    assert r.extra["factors"] == ["drug", "time"] and len([n for n in r.data_notes if "automatically" in n]) == 4
    assert r.statistic == pytest.approx(rp.anova_rm_twoway(two, "y", "id", "drug", "time").statistic)
    r = run(t2, "anova_rm_twoway", value="y", subject="id", within_a="time", within_b="drug", correction="none")
    assert r.extra["factors"] == ["time", "drug"]
    with pytest.raises(StatsError, match="not in the table"):
        run(t2, "anova_rm_twoway", subject="who")
    with pytest.raises(StatsError, match="second repeated factor"):
        run(MultipleVariablesTable("short", two[["id", "drug", "y"]]), "anova_rm_twoway")
    with pytest.raises(StatsError, match="no numeric column"):
        run(MultipleVariablesTable("nonum", two[["id", "drug", "time"]]), "anova_rm_twoway")
    coded = two.assign(id=pd.factorize(two["id"])[0] + 100.0)  # numeric subject codes are found too
    rc = run(MultipleVariablesTable("coded", coded[["drug", "time", "id", "y"]]), "anova_rm_twoway")
    assert rc.statistic == pytest.approx(r.statistic) and rc.extra["factors"] == ["drug", "time"]
    assert any("Subject column chosen automatically: 'id'" in n for n in rc.data_notes)
    with pytest.raises(StatsError, match="subject column"):
        run(MultipleVariablesTable("nosubj", two[["y"]]), "anova_rm_twoway")
    for key in ("anova_rm_oneway", "anova_mixed", "anova_rm_twoway"):
        info = registry.REGISTRY[key]
        assert info.category == "Repeated measures" and all(p.help for p in info.params) and info.description


# =============================================================================
# Property-based tests
# =============================================================================
@settings(max_examples=50, deadline=None)
@given(n=st.integers(2, 9), k=st.integers(2, 5), seed=st.integers(0, 2**31 - 1),
       shift=st.floats(-1e3, 1e3), scale=st.floats(0.01, 100.0))
def test_property_oneway_bounds_and_invariances(n, k, seed, shift, scale):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, k)) * rng.uniform(0.5, 3, k) + rng.normal(size=(n, 1))
    names = [f"c{j}" for j in range(k)]
    r = rp.anova_rm_oneway(y, names, posthoc="none")
    row = table_row(r, "Condition")
    q = k - 1
    sph = r.extra["sphericity"]
    assert 1 / q - 1e-12 <= row["eps_gg"] <= 1 + 1e-12
    if sph["eps_hf_original_uncapped"] is not None:
        assert sph["eps_hf_original_uncapped"] >= row["eps_gg"] - 1e-12  # before capping
        assert row["eps_gg"] - 1e-12 <= row["eps_hf"] <= 1.0
    for key in ("p", "p_gg", "p_hf", "p_lb", "p_reported"):
        assert row[key] is None or 0.0 <= row[key] <= 1.0
    if sph["p"] is not None:
        assert 0.0 <= sph["p"] <= 1.0 and 0.0 < sph["mauchly_w"] <= 1.0
    assert row["eps_gg"] == pytest.approx(box_epsilon(np.cov(y, rowvar=False)), rel=1e-7)
    # invariances: subject order, condition order, units, and adding a constant to each subject
    perm_rows, perm_cols = rng.permutation(n), rng.permutation(k)
    y2 = (y[perm_rows][:, perm_cols] + rng.normal(size=(n, 1)) * 5) * scale + shift
    r2 = rp.anova_rm_oneway(y2, [names[j] for j in perm_cols], posthoc="none")
    row2 = table_row(r2, "Condition")
    assert row2["f"] == pytest.approx(row["f"], rel=1e-6, abs=1e-9)
    assert row2["eps_gg"] == pytest.approx(row["eps_gg"], rel=1e-6)
    if sph["mauchly_w"] is not None and r2.extra["sphericity"]["mauchly_w"] is not None:
        assert r2.extra["sphericity"]["mauchly_w"] == pytest.approx(sph["mauchly_w"], rel=1e-5, abs=1e-12)
    assert r2.effect("partial eta squared").value == pytest.approx(r.effect("partial eta squared").value, rel=1e-6)


@settings(max_examples=50, deadline=None)
@given(data=st.data(), n=st.integers(1, 7), k=st.integers(2, 4),
       posthoc=st.sampled_from(["none", "holm", "bonferroni", "fdr_bh"]),
       correction=st.sampled_from(["auto", "none", "gg", "hf"]))
def test_property_oneway_ties_zeros_constants_and_missing_never_crash(data, n, k, posthoc, correction):
    cells = data.draw(st.lists(st.one_of(st.integers(-2, 2).map(float), st.just(float("nan"))),
                               min_size=n * k, max_size=n * k))
    y = np.array(cells).reshape(n, k)
    names = [f"c{j}" for j in range(k)]
    complete = int(np.isfinite(y).all(axis=1).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numerical warnings must not leak
        if complete < 2:
            with pytest.raises(StatsError):
                rp.anova_rm_oneway(y, names, posthoc=posthoc, correction=correction)
            return
        r = rp.anova_rm_oneway(y, names, posthoc=posthoc, correction=correction)
    assert r.n_total == complete
    assert r.p_value is None or 0.0 <= r.p_value <= 1.0
    if r.p_value is None:
        assert r.warnings
    for comp in r.comparisons:
        for p in (comp.p_value, comp.p_adjusted):
            assert p is None or 0.0 <= p <= 1.0
        if comp.p_value is not None:
            assert comp.p_adjusted >= comp.p_value - 1e-12 and comp.ci_low <= comp.estimate <= comp.ci_high
    for e in r.effect_sizes:
        assert e.value is None or 0.0 <= e.value <= 1.0
    json.loads(r.finalize().model_dump_json())


@settings(max_examples=30, deadline=None)
@given(sizes=st.lists(st.integers(1, 5), min_size=2, max_size=3), p=st.integers(2, 4),
       seed=st.integers(0, 2**31 - 1), ss_type=st.sampled_from([2, 3]),
       compare=st.sampled_from(list(rp.MIXED_COMPARE)), integer=st.booleans())
def test_property_mixed_matches_least_squares_and_is_order_invariant(sizes, p, seed, ss_type, compare, integer):
    rng = np.random.default_rng(seed)
    levels = tuple(f"t{j}" for j in range(p))
    frame = mixed_frame(seed=seed, sizes=tuple(sizes), levels=levels)
    if integer:
        frame["y"] = np.round(frame["y"] - 10).clip(-2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if sum(sizes) - len(sizes) < 1:
            with pytest.raises(StatsError):
                rp.anova_mixed(frame, "y", "id", "group", "time", ss_type=ss_type)
            return
        r = rp.anova_mixed(frame, "y", "id", "group", "time", ss_type=ss_type, posthoc="holm",
                           posthoc_compare=compare)
    y, codes = wide_from_long(frame)
    ref = dummy_model_ss(y, codes)
    total = max(table_row(r, "Total")["ss"], 1e-12)
    assert table_row(r, "group")["ss"] == pytest.approx(ref["between"], rel=1e-6, abs=1e-9 * total)
    assert table_row(r, "group x time")["ss"] == pytest.approx(ref["interaction"], rel=1e-6, abs=1e-9 * total)
    assert table_row(r, "Residual")["ss"] == pytest.approx(ref["residual"], rel=1e-6, abs=1e-9 * total)
    if ss_type == 3:
        assert table_row(r, "time")["ss"] == pytest.approx(ref["within"], rel=1e-6, abs=1e-9 * total)
    for row in r.anova_table:
        assert row["ss"] >= 0
        for key in ("p", "p_gg", "p_hf", "p_lb"):
            assert row.get(key) is None or 0.0 <= row[key] <= 1.0
    row = table_row(r, "time")
    if row["eps_gg"] is not None:
        assert 1 / (p - 1) - 1e-12 <= row["eps_gg"] <= 1 + 1e-12
        sph = r.extra["sphericity"]
        if sph["eps_hf_original_uncapped"] is not None:
            assert sph["eps_hf_original_uncapped"] >= sph["eps_hf_lecoutre_uncapped"] >= row["eps_gg"] - 1e-12
    for comp in r.comparisons:
        assert comp.p_value is None or (0.0 <= comp.p_value <= comp.p_adjusted + 1e-12 <= 1.0 + 1e-12)
    # row order of the long table does not matter
    shuffled = frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    shuffled["group"] = pd.Categorical(shuffled["group"], categories=list(pd.unique(frame["group"])))
    shuffled["time"] = pd.Categorical(shuffled["time"], categories=list(levels))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r2 = rp.anova_mixed(shuffled, "y", "id", "group", "time", ss_type=ss_type)
    for name in ("group", "time", "group x time"):
        a, b = table_row(r, name), table_row(r2, name)
        assert b["ss"] == pytest.approx(a["ss"], rel=1e-8, abs=1e-9 * total)
        assert (a["f"] is None) == (b["f"] is None)
        if a["f"] is not None:
            assert b["f"] == pytest.approx(a["f"], rel=1e-6, abs=1e-9)
    json.loads(r.finalize().model_dump_json())


def test_engine_and_fallback_edge_cases():
    # a single subject per group leaves no error df: the engine reports it instead of dividing by zero
    sc = rp._analyse_scores(np.array([[1.0, 2.0], [0.5, 1.0]]), np.array([0, 1]), 2)
    assert sc.df_error == 0 and sc.eps_gg is None and "Too few subjects" in sc.note
    # 3 subjects in 2 groups, 4 levels: one error df, epsilon sits on its lower bound and the
    # Huynh-Feldt estimate is undefined -> Greenhouse-Geisser is used and the user is told
    frame = mixed_frame(seed=1, sizes=(1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = rp.anova_mixed(frame, "y", "id", "group", "time", correction="hf")
    row = table_row(r, "time")
    assert row["eps_gg"] == pytest.approx(1 / 3) and row["eps_hf"] is None
    assert r.extra["correction_applied"] == "gg" and any("Huynh-Feldt" in w for w in r.warnings)
    assert r.p_value == table_row(r, "group x time")["p_gg"]
    # eight levels but one group has a single subject: the variance check cannot run anywhere
    many = mixed_frame(seed=1, sizes=(1, 2), levels=tuple(f"d{i}" for i in range(8)))
    r = rp.anova_mixed(many, "y", "id", "group", "time")
    summary = next(a for a in r.assumptions if a.name.startswith("Equal variances between groups"))
    assert summary.status == "warn" and "could not be checked" in summary.detail
    # two-way RM with too few subjects for Huynh-Feldt on the interaction
    two = rm2_frame(n=2, a=3, b=3)
    r = rp.anova_rm_twoway(two, "y", "id", "drug", "time", correction="hf")
    assert any("Huynh-Feldt" in w for w in r.warnings)
    assert table_row(r, "drug x time")["correction"] == "Greenhouse-Geisser"


# =============================================================================
# Unit of measurement: every statistic is scale-free (audit part 2: A1-A4, C2)
# =============================================================================
SCALES = [1e-13, 1.0, 1e9, 1e150]


def _scale_matrix():
    rng = np.random.default_rng(7)
    return rng.normal(5.0, 1.0, (8, 4)) + np.array([0.0, 0.5, 1.0, 2.0])


def _scale_mixed(scale=1.0):
    rng = np.random.default_rng(11)
    rows = []
    for s in range(11):
        group = "g1" if s < 5 else "g2"
        effect = rng.normal(0.0, 1.0)
        for w in range(3):
            rows.append((f"s{s}", group, f"t{w}",
                         (5.0 + effect + (1.0 * w if group == "g1" else 0.0) + rng.normal(0.0, 0.7)) * scale))
    return pd.DataFrame(rows, columns=["subject", "group", "time", "y"])


def _scale_twoway(scale=1.0):
    rng = np.random.default_rng(8)
    rows = []
    for s in range(6):
        effect = rng.normal()
        for a in "xy":
            for b in "123":
                rows.append((s, a, b, (effect + (a == "y") + 0.3 * int(b) + rng.normal(0, 0.6)) * scale))
    return pd.DataFrame(rows, columns=["s", "a", "b", "y"])


def _unit_free(res):
    """Everything in a result that must not depend on the unit of the data."""
    return {
        "f": res.statistic, "p": res.p_value, "df": res.df, "df2": res.df2,
        "table_f": [r["f"] for r in res.anova_table], "table_p": [r.get("p_reported") for r in res.anova_table],
        "pct": [r["percent_of_variation"] for r in res.anova_table],
        "effects": [(e.value, e.ci_low, e.ci_high) for e in res.effect_sizes],
        "assumption_p": [a.p_value for a in res.assumptions],
        "comp": [(c.statistic, c.p_value, c.p_adjusted, c.effect_size.value if c.effect_size else None)
                 for c in res.comparisons],
    }


def _in_units(res, scale):
    """Quantities that carry the unit of the data, divided by the unit (power 1 or 2)."""
    return {
        "ss": [None if r["ss"] is None else r["ss"] / scale / scale for r in res.anova_table],
        "ms": [None if r["ms"] is None else r["ms"] / scale / scale for r in res.anova_table],
        "means": [d["mean"] / scale for d in res.descriptives],
        "sds": [d["sd"] / scale for d in res.descriptives],
        "comp": [(c.estimate / scale, c.se / scale, c.ci_low / scale, c.ci_high / scale) for c in res.comparisons],
    }


def _assert_same(a, b, rel=1e-9, path=""):
    """Recursive comparison of nested dicts / lists of numbers (None must match None)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            _assert_same(a[key], b[key], rel, f"{path}/{key}")
    elif isinstance(a, list | tuple):
        assert isinstance(b, list | tuple) and len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b)):
            _assert_same(u, v, rel, f"{path}[{i}]")
    elif a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} vs {b!r}"
    else:
        assert a == pytest.approx(b, rel=rel, abs=1e-12), f"{path}: {a!r} vs {b!r}"


@pytest.mark.parametrize("scale", SCALES)
def test_rm_oneway_does_not_depend_on_the_unit_of_measurement(scale):
    base = rp.anova_rm_oneway(_scale_matrix(), list("abcd"), posthoc="holm")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow / underflow warnings may leak
        res = rp.anova_rm_oneway(_scale_matrix() * scale, list("abcd"), posthoc="holm")
    assert res.statistic is not None and len(res.comparisons) == 6
    _assert_same(_unit_free(res), _unit_free(base))
    _assert_same(_in_units(res, scale), _in_units(base, 1.0))
    assert res.extra["subject_means"] == pytest.approx([v * scale for v in base.extra["subject_means"]], rel=1e-12)
    json.loads(res.model_dump_json())


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("compare", rp.MIXED_COMPARE)
def test_mixed_anova_does_not_depend_on_the_unit_of_measurement(scale, compare):
    kwargs = {"posthoc": "holm", "posthoc_compare": compare}
    base = rp.anova_mixed(_scale_mixed(), "y", "subject", "group", "time", **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = rp.anova_mixed(_scale_mixed(scale), "y", "subject", "group", "time", **kwargs)
    assert res.statistic is not None and res.comparisons
    _assert_same(_unit_free(res), _unit_free(base))
    _assert_same(_in_units(res, scale), _in_units(base, 1.0))


@pytest.mark.parametrize("scale", SCALES)
def test_rm_twoway_does_not_depend_on_the_unit_of_measurement(scale):
    base = rp.anova_rm_twoway(_scale_twoway(), "y", "s", "a", "b")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = rp.anova_rm_twoway(_scale_twoway(scale), "y", "s", "a", "b")
    assert res.statistic is not None
    _assert_same(_unit_free(res), _unit_free(base))
    _assert_same(_in_units(res, scale), _in_units(base, 1.0))


@pytest.mark.parametrize("scale", [1e-170, 1e160, 1e300])
def test_rm_anova_with_extreme_magnitudes_gives_the_unit_free_result_and_says_what_is_missing(scale):
    """Squares of such values cannot be represented: the F test is still right (computed on
    rescaled data), sums of squares are left empty and a warning explains why."""
    base = rp.anova_rm_oneway(_scale_matrix(), list("abcd"), posthoc="holm")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = rp.anova_rm_oneway(_scale_matrix() * scale, list("abcd"), posthoc="holm")
        mixed = rp.anova_mixed(_scale_mixed(scale), "y", "subject", "group", "time")
        twoway = rp.anova_rm_twoway(_scale_twoway(scale), "y", "s", "a", "b")
    _assert_same(_unit_free(res), _unit_free(base))
    assert mixed.statistic == pytest.approx(
        rp.anova_mixed(_scale_mixed(), "y", "subject", "group", "time").statistic, rel=1e-9)
    assert twoway.statistic == pytest.approx(rp.anova_rm_twoway(_scale_twoway(), "y", "s", "a", "b").statistic,
                                             rel=1e-9)
    for r in (res, mixed, twoway):
        assert all(row["ss"] is None for row in r.anova_table if row["kind"] != "total" or True)
        assert any("cannot be written as ordinary numbers" in w for w in r.warnings)
        assert all(math.isfinite(d["mean"]) for d in r.descriptives)
        json.loads(r.model_dump_json())
    # means and paired differences are still reported in the units of the data
    assert res.comparisons[0].estimate == pytest.approx(base.comparisons[0].estimate * scale, rel=1e-9)


def test_paired_comparisons_survive_small_units_and_agree_with_scipy():
    y = _scale_matrix() * 1e-12
    names = list("abcd")
    res = rp.anova_rm_oneway(y, names, posthoc="holm")
    assert not any("could not be tested" in w for w in res.warnings)
    for comp in res.comparisons:
        ref = stats.ttest_rel(y[:, names.index(comp.group_a)], y[:, names.index(comp.group_b)])
        assert comp.statistic == pytest.approx(ref.statistic, rel=1e-9)
        assert comp.p_value == pytest.approx(ref.pvalue, rel=1e-9)


@pytest.mark.parametrize("scale", [1e-13, 1.0, 1e9])
def test_constant_differences_are_untestable_at_every_scale(scale):
    """b = a + constant: the paired differences do not vary (only rounding noise), whatever
    the unit; c varies and stays testable."""
    rng = np.random.default_rng(3)
    a = rng.normal(10.0, 2.0, 7)
    y = np.column_stack([a, a + 0.1, a + rng.normal(1.0, 0.5, 7)]) * scale
    res = rp.anova_rm_oneway(y, ["a", "b", "c"], posthoc="holm")
    ab = res.comparison("a", "b")
    assert ab.p_value is None and ab.statistic is None
    assert res.comparison("a", "c").p_value is not None
    assert any("1 comparison(s) could not be tested" in w for w in res.warnings)


def test_large_offset_with_small_spread_is_not_called_identical():
    """ulp-based 'identical' test: 1e9 + noise of order 1 is real variation."""
    y = _scale_matrix() + 1e9
    res = rp.anova_rm_oneway(y, list("abcd"), posthoc="none")
    base = rp.anova_rm_oneway(_scale_matrix(), list("abcd"), posthoc="none")
    assert res.statistic == pytest.approx(base.statistic, rel=1e-6)
    same = rp.anova_rm_oneway(np.full((5, 3), 1e9), list("abc"), posthoc="none")
    assert same.statistic is None and any("identical" in w for w in same.warnings)


def test_grouped_table_text_does_not_leak_the_internal_column_name(run):
    table = GroupedTable.from_cells(
        "uptake", {t: {g: list(np.random.default_rng(i + 3 * j).normal(5 + i + j, 1, 4))
                       for j, g in enumerate(["Ctl", "Drug"])} for i, t in enumerate(["t0", "t1", "t2"])},
        row_label="Time", column_label="Treatment")
    res = run(table, "anova_mixed")
    assert res.interpretation.startswith("Mixed-design ANOVA of the measured values with Treatment")
    assert " of value " not in res.interpretation
