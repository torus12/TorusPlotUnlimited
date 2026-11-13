"""Factorial ANOVA and ANCOVA.

Validation layers (see tests/fixtures/sources/factorial_repeated.md):
* published worked examples (typed in by hand, values reproduced to published precision);
* cross-implementation checks against statsmodels and scipy;
* from-first-principles sums of squares written in this file (classical balanced
  formulas, nested-model comparison with a hand-built sum-coded design matrix, and the
  unweighted-means hypothesis for Type III main effects);
* property-based robustness tests with hypothesis.
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
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from torusplot.data.tables import GroupedTable, MultipleVariablesTable
from torusplot.stats import effect_sizes as es
from torusplot.stats import factorial as fx
from torusplot.stats import registry
from torusplot.stats._utils import StatsError


@pytest.fixture
def run(monkeypatch):
    """run_analysis without importing analysis modules owned by other people."""
    monkeypatch.setattr(registry, "_loaded", True)
    return registry.run_analysis


# ----------------------------------------------------------------------------- data
# Montgomery DC, Design and Analysis of Experiments: battery life (hours), 3 material
# types x 3 temperatures (deg F), n = 4 per cell.
BATTERY = {
    ("1", "15"): (130, 155, 74, 180), ("1", "70"): (34, 40, 80, 75), ("1", "125"): (20, 70, 82, 58),
    ("2", "15"): (150, 188, 159, 126), ("2", "70"): (136, 122, 106, 115), ("2", "125"): (25, 70, 58, 45),
    ("3", "15"): (138, 110, 168, 160), ("3", "70"): (174, 120, 150, 139), ("3", "125"): (96, 104, 82, 60),
}

# Crampton EW (1947) J Nutr 33:491-504 (tooth growth in guinea pigs; also Bliss 1952):
# odontoblast length by supplement (VC = ascorbic acid, OJ = orange juice) and dose (mg/day).
TOOTH = {
    ("VC", "0.5"): [4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7],
    ("VC", "1"): [16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5],
    ("VC", "2"): [23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5],
    ("OJ", "0.5"): [15.2, 21.5, 17.6, 9.7, 14.5, 10, 8.2, 9.4, 16.5, 9.7],
    ("OJ", "1"): [19.7, 23.3, 23.6, 26.4, 20, 25.2, 25.8, 21.2, 14.5, 27.3],
    ("OJ", "2"): [25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23],
}


def long_from_cells(cells, names=("A", "B"), value="y"):
    rows = [(*key, float(v)) for key, values in cells.items() for v in values]
    return pd.DataFrame(rows, columns=[*names, value])


def battery_frame():
    return long_from_cells(BATTERY, ("Material", "Temperature"), "Life")


def unbalanced_two_way(seed=3, counts=((3, 5, 2), (4, 2, 6)), effect=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in enumerate(counts):
        for j, n in enumerate(row):
            mu = 10 + effect * (1.5 * i + 0.8 * j + 0.9 * i * j)
            rows += [(f"a{i}", f"b{j}", mu + rng.normal(0, 1.2)) for _ in range(n)]
    return pd.DataFrame(rows, columns=["A", "B", "y"])


def unbalanced_three_way(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i, j, k in itertools.product(range(2), range(3), range(2)):
        n = int(rng.integers(2, 5))
        mu = 5 + i + 0.5 * j - 0.7 * k + 0.6 * i * k + 0.4 * i * j * k
        rows += [(f"a{i}", f"b{j}", f"c{k}", mu + rng.normal()) for _ in range(n)]
    return pd.DataFrame(rows, columns=["A", "B", "C", "y"])


def table_row(result, source):
    for row in result.anova_table:
        if row["source"] == source:
            return row
    raise AssertionError(f"no row {source!r} in {[r['source'] for r in result.anova_table]}")


# ------------------------------------------------- from-first-principles helpers
def sum_coded(codes, k):
    """N x (k - 1) deviation-coded columns (last level = -1 on every column)."""
    basis = np.vstack([np.eye(k - 1), -np.ones((1, k - 1))])
    return basis[codes]


def scratch_blocks(frame, factors):
    """{term (tuple of factor indices): design columns} with sum-to-zero coding."""
    mains = []
    for f in factors:
        codes, levels = pd.factorize(frame[f])
        mains.append(sum_coded(codes, len(levels)))
    blocks = {}
    for order in range(1, len(factors) + 1):
        for combo in itertools.combinations(range(len(factors)), order):
            cols = mains[combo[0]]
            for i in combo[1:]:
                cols = np.einsum("ni,nj->nij", cols, mains[i]).reshape(len(frame), -1)
            blocks[combo] = cols
    return blocks


def rss(y, *blocks):
    x = np.column_stack([np.ones(len(y)), *blocks])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(((y - x @ beta) ** 2).sum())


def scratch_ss(y, blocks, ss_type):
    """Type II / III SS by comparing nested least-squares fits."""
    out = {}
    for term in blocks:
        if ss_type == 3:
            others = [b for t, b in blocks.items() if t != term]
            out[term] = rss(y, *others) - rss(y, *blocks.values())
        else:
            base = [b for t, b in blocks.items() if t != term and not set(term) < set(t)]
            out[term] = rss(y, *base) - rss(y, *base, blocks[term])
    return out


# =============================================================================
# Published worked examples
# =============================================================================
def test_published_montgomery_battery_life():
    r = fx.anova_twoway(battery_frame(), "Life", "Material", "Temperature")
    assert r.test_name == "Two-way ANOVA"
    # sums of squares as printed in the textbook ANOVA table
    assert table_row(r, "Material")["ss"] == pytest.approx(10683.72, abs=0.005)
    assert table_row(r, "Temperature")["ss"] == pytest.approx(39118.72, abs=0.005)
    assert table_row(r, "Material x Temperature")["ss"] == pytest.approx(9613.78, abs=0.005)
    assert table_row(r, "Residual")["ss"] == pytest.approx(18230.75, abs=0.005)
    assert table_row(r, "Total")["ss"] == pytest.approx(77646.97, abs=0.005)
    assert [table_row(r, s)["df"] for s in ("Material", "Temperature", "Material x Temperature", "Residual", "Total")] \
        == [2, 2, 4, 27, 35]
    assert table_row(r, "Residual")["ms"] == pytest.approx(675.21, abs=0.005)
    assert table_row(r, "Material")["f"] == pytest.approx(7.91, abs=0.005)
    assert table_row(r, "Temperature")["f"] == pytest.approx(28.97, abs=0.005)
    assert table_row(r, "Material x Temperature")["f"] == pytest.approx(3.56, abs=0.005)
    # headline = the interaction
    assert r.statistic == pytest.approx(3.56, abs=0.005) and (r.df, r.df2) == (4, 27)
    assert r.extra["headline_term"] == "Material x Temperature"


def test_published_tooth_growth():
    frame = long_from_cells(TOOTH, ("supp", "dose"), "len")
    # typing check against the well-known cell means and grand mean
    means = frame.groupby(["supp", "dose"])["len"].mean()
    assert means[("VC", "0.5")] == pytest.approx(7.98) and means[("OJ", "2")] == pytest.approx(26.06)
    assert frame["len"].mean() == pytest.approx(18.8133, abs=5e-5)
    r = fx.anova_twoway(frame, "len", "supp", "dose")
    assert table_row(r, "supp")["f"] == pytest.approx(15.572, abs=5e-4)
    assert table_row(r, "dose")["f"] == pytest.approx(92.000, abs=5e-4)
    assert table_row(r, "supp x dose")["f"] == pytest.approx(4.107, abs=5e-4)
    assert [table_row(r, s)["df"] for s in ("supp", "dose", "supp x dose", "Residual")] == [1, 2, 2, 54]
    assert table_row(r, "supp")["ss"] == pytest.approx(205.35, abs=0.005)
    assert table_row(r, "dose")["ss"] == pytest.approx(2426.43, abs=0.005)
    assert table_row(r, "supp x dose")["ss"] == pytest.approx(108.32, abs=0.005)
    assert table_row(r, "Residual")["ss"] == pytest.approx(712.11, abs=0.005)
    assert table_row(r, "supp x dose")["p"] == pytest.approx(0.02186, abs=5e-6)


# =============================================================================
# Cross-implementation and from-scratch checks: ANOVA table
# =============================================================================
@pytest.mark.parametrize("ss_type", [2, 3])
def test_unbalanced_two_way_matches_statsmodels(ss_type):
    frame = unbalanced_two_way()
    r = fx.anova_twoway(frame, "y", "A", "B", ss_type=ss_type)
    ref = anova_lm(smf.ols("y ~ C(A, Sum) * C(B, Sum)", data=frame).fit(), typ=ss_type)
    for ours, theirs in (("A", "C(A, Sum)"), ("B", "C(B, Sum)"), ("A x B", "C(A, Sum):C(B, Sum)")):
        row = table_row(r, ours)
        assert row["ss"] == pytest.approx(ref.loc[theirs, "sum_sq"], rel=1e-8)
        assert row["df"] == ref.loc[theirs, "df"]
        assert row["f"] == pytest.approx(ref.loc[theirs, "F"], rel=1e-8)
        assert row["p"] == pytest.approx(ref.loc[theirs, "PR(>F)"], rel=1e-6, abs=1e-12)
    assert table_row(r, "Residual")["ss"] == pytest.approx(ref.loc["Residual", "sum_sq"], rel=1e-10)
    assert any("unbalanced" in w for w in r.warnings)
    assert next(a for a in r.assumptions if a.name == "Balanced design").status == "warn"


@pytest.mark.parametrize("ss_type", [2, 3])
def test_unbalanced_two_way_matches_from_scratch_model_comparison(ss_type):
    frame = unbalanced_two_way(seed=8, counts=((2, 6, 3), (5, 3, 4), (4, 2, 7)))
    r = fx.anova_twoway(frame, "y", "A", "B", ss_type=ss_type)
    ss = scratch_ss(frame["y"].to_numpy(), scratch_blocks(frame, ["A", "B"]), ss_type)
    assert table_row(r, "A")["ss"] == pytest.approx(ss[(0,)], rel=1e-8)
    assert table_row(r, "B")["ss"] == pytest.approx(ss[(1,)], rel=1e-8)
    assert table_row(r, "A x B")["ss"] == pytest.approx(ss[(0, 1)], rel=1e-8)
    # F and P from the definition
    mse = table_row(r, "Residual")["ss"] / table_row(r, "Residual")["df"]
    row = table_row(r, "A")
    assert row["f"] == pytest.approx(ss[(0,)] / 2 / mse, rel=1e-8)
    assert row["p"] == pytest.approx(stats.f.sf(row["f"], 2, len(frame) - 9), rel=1e-8)
    assert row["partial_eta_sq"] == pytest.approx(ss[(0,)] / (ss[(0,)] + mse * (len(frame) - 9)), rel=1e-8)


@pytest.mark.parametrize("ss_type", [2, 3])
def test_unbalanced_three_way_matches_statsmodels_and_scratch(ss_type):
    frame = unbalanced_three_way()
    r = fx.anova_threeway(frame, "y", "A", "B", "C", ss_type=ss_type)
    assert r.test_name == "Three-way ANOVA"
    # a user column called "C" would shadow the formula function C(): our code renames columns
    # internally; for the reference fit the column is renamed by hand
    ref = anova_lm(smf.ols("y ~ C(A, Sum) * C(B, Sum) * C(K, Sum)", data=frame.rename(columns={"C": "K"})).fit(),
                   typ=ss_type)
    ss = scratch_ss(frame["y"].to_numpy(), scratch_blocks(frame, ["A", "B", "C"]), ss_type)
    names = {(0,): "A", (1,): "B", (2,): "C", (0, 1): "A x B", (0, 2): "A x C", (1, 2): "B x C",
             (0, 1, 2): "A x B x C"}
    for combo, name in names.items():
        theirs = ":".join(f"C({'ABK'[i]}, Sum)" for i in combo)
        row = table_row(r, name)
        assert row["ss"] == pytest.approx(ref.loc[theirs, "sum_sq"], rel=1e-8)
        assert row["ss"] == pytest.approx(ss[combo], rel=1e-7)
        assert row["f"] == pytest.approx(ref.loc[theirs, "F"], rel=1e-8)
        assert row["p"] == pytest.approx(ref.loc[theirs, "PR(>F)"], rel=1e-6, abs=1e-12)
    assert r.extra["headline_term"] == "A x B x C"
    assert r.statistic == pytest.approx(table_row(r, "A x B x C")["f"])


def test_balanced_two_way_all_types_agree_with_classical_formulas():
    frame = battery_frame()
    y = frame["Life"].to_numpy()
    grand = y.mean()
    a_means = frame.groupby("Material", sort=False)["Life"].mean()
    b_means = frame.groupby("Temperature", sort=False)["Life"].mean()
    cell = frame.groupby(["Material", "Temperature"], sort=False)["Life"].mean()
    n = 4
    ss_a = 3 * n * ((a_means - grand) ** 2).sum()
    ss_b = 3 * n * ((b_means - grand) ** 2).sum()
    ss_cells = n * ((cell - grand) ** 2).sum()
    ss_ab = ss_cells - ss_a - ss_b
    ss_e = ((y - grand) ** 2).sum() - ss_cells
    type1 = anova_lm(smf.ols("Life ~ C(Material) * C(Temperature)", data=frame).fit(), typ=1)
    for ss_type in (2, 3):
        r = fx.anova_twoway(frame, "Life", "Material", "Temperature", ss_type=ss_type)
        assert table_row(r, "Material")["ss"] == pytest.approx(ss_a, rel=1e-10)
        assert table_row(r, "Temperature")["ss"] == pytest.approx(ss_b, rel=1e-10)
        assert table_row(r, "Material x Temperature")["ss"] == pytest.approx(ss_ab, rel=1e-10)
        assert table_row(r, "Residual")["ss"] == pytest.approx(ss_e, rel=1e-10)
        assert table_row(r, "Material")["ss"] == pytest.approx(type1.loc["C(Material)", "sum_sq"], rel=1e-10)
        assert table_row(r, "Material x Temperature")["ss"] == pytest.approx(
            type1.loc["C(Material):C(Temperature)", "sum_sq"], rel=1e-10)
        # balanced: the percentages add up to 100
        parts = [row["percent_of_variation"] for row in r.anova_table if row["kind"] != "total"]
        assert sum(parts) == pytest.approx(100.0, rel=1e-10)
        if ss_type == 3:
            assert not r.warnings
        else:  # the interaction is significant in these data, which Type II main effects ignore
            assert len(r.warnings) == 1 and "Type II" in r.warnings[0]
        assert next(a for a in r.assumptions if a.name == "Balanced design").status == "pass"


def test_balanced_three_way_classical_formulas_and_type_identity():
    rng = np.random.default_rng(5)
    a, b, c, n = 2, 3, 2, 3
    rows = [(f"a{i}", f"b{j}", f"c{k}", rng.normal(i + 0.5 * j * k, 1.0))
            for i, j, k in itertools.product(range(a), range(b), range(c)) for _ in range(n)]
    frame = pd.DataFrame(rows, columns=["A", "B", "C", "y"])
    y = frame["y"].to_numpy()
    grand = y.mean()

    def ss_of(cols):
        m = frame.groupby(cols, sort=False)["y"].mean()
        reps = len(frame) / len(m)
        return float(reps * ((m - grand) ** 2).sum())

    ss = {"A": ss_of(["A"]), "B": ss_of(["B"]), "C": ss_of(["C"])}
    ss["A x B"] = ss_of(["A", "B"]) - ss["A"] - ss["B"]
    ss["A x C"] = ss_of(["A", "C"]) - ss["A"] - ss["C"]
    ss["B x C"] = ss_of(["B", "C"]) - ss["B"] - ss["C"]
    ss["A x B x C"] = ss_of(["A", "B", "C"]) - sum(ss.values())
    ss_e = float(((y - grand) ** 2).sum()) - ss_of(["A", "B", "C"])
    type1 = anova_lm(smf.ols("y ~ C(A) * C(B) * C(K)", data=frame.rename(columns={"C": "K"})).fit(), typ=1)
    for ss_type in (2, 3):
        r = fx.anova_factorial(frame, "y", ["A", "B", "C"], ss_type=ss_type)
        for name, value in ss.items():
            assert table_row(r, name)["ss"] == pytest.approx(value, rel=1e-9, abs=1e-10)
        assert table_row(r, "Residual")["ss"] == pytest.approx(ss_e, rel=1e-10)
        assert table_row(r, "Residual")["df"] == a * b * c * (n - 1)
        assert table_row(r, "B x C")["ss"] == pytest.approx(type1.loc["C(B):C(K)", "sum_sq"], rel=1e-9)
        assert table_row(r, "A x B x C")["ss"] == pytest.approx(type1.loc["C(A):C(B):C(K)", "sum_sq"], rel=1e-9)


def test_type3_main_effect_tests_unweighted_marginal_means():
    """Type III SS(A) equals the quadratic form for 'all unweighted row means are equal'."""
    frame = unbalanced_two_way(seed=21, counts=((2, 5, 3), (6, 2, 4)))
    r = fx.anova_twoway(frame, "y", "A", "B")
    cell = frame.groupby(["A", "B"], sort=False)["y"].agg(["mean", "count"])
    m = cell["mean"].to_numpy()  # order a0b0 a0b1 a0b2 a1b0 ...
    d = np.diag(1.0 / cell["count"].to_numpy())
    contrast = np.array([[1 / 3, 1 / 3, 1 / 3, -1 / 3, -1 / 3, -1 / 3]])
    ss_a = float((contrast @ m) @ np.linalg.inv(contrast @ d @ contrast.T) @ (contrast @ m))
    assert table_row(r, "A")["ss"] == pytest.approx(ss_a, rel=1e-9)
    # ...and the estimated marginal means in `extra` are those unweighted means
    marg = {(e["factor"], e["level"]): e for e in r.extra["marginal_means"]}
    assert marg[("A", "a0")]["mean"] == pytest.approx(m[:3].mean(), rel=1e-12)
    assert marg[("A", "a0")]["observed_mean"] == pytest.approx(frame.loc[frame.A == "a0", "y"].mean(), rel=1e-12)
    mse = table_row(r, "Residual")["ms"]
    assert marg[("A", "a0")]["se"] == pytest.approx(math.sqrt(mse * (1 / 2 + 1 / 5 + 1 / 3) / 9), rel=1e-9)


def test_additive_model_and_single_replicate_design():
    frame = unbalanced_two_way(seed=4)
    r = fx.anova_twoway(frame, "y", "A", "B", interactions=False, ss_type=2)
    ref = anova_lm(smf.ols("y ~ C(A) + C(B)", data=frame).fit(), typ=2)
    assert [row["source"] for row in r.anova_table] == ["A", "B", "Residual", "Total"]
    assert table_row(r, "A")["f"] == pytest.approx(ref.loc["C(A)", "F"], rel=1e-8)
    assert table_row(r, "B")["p"] == pytest.approx(ref.loc["C(B)", "PR(>F)"], rel=1e-6)
    assert r.extra["interaction_in_model"] is False and r.extra["headline_term"] == "A"

    # one value per cell: the interaction is dropped automatically (randomized-block formulas)
    rng = np.random.default_rng(2)
    grid = rng.normal(size=(4, 3)) + np.arange(4)[:, None]
    single = pd.DataFrame([(f"r{i}", f"c{j}", grid[i, j]) for i in range(4) for j in range(3)],
                          columns=["R", "C", "y"])
    r1 = fx.anova_twoway(single, "y", "R", "C")
    grand = grid.mean()
    ss_r = 3 * ((grid.mean(axis=1) - grand) ** 2).sum()
    ss_c = 4 * ((grid.mean(axis=0) - grand) ** 2).sum()
    ss_e = ((grid - grand) ** 2).sum() - ss_r - ss_c
    assert [row["source"] for row in r1.anova_table] == ["R", "C", "Residual", "Total"]
    assert table_row(r1, "R")["ss"] == pytest.approx(ss_r, rel=1e-10)
    assert table_row(r1, "C")["ss"] == pytest.approx(ss_c, rel=1e-10)
    assert table_row(r1, "Residual")["ss"] == pytest.approx(ss_e, rel=1e-10)
    assert table_row(r1, "Residual")["df"] == 6
    assert table_row(r1, "R")["f"] == pytest.approx((ss_r / 3) / (ss_e / 6), rel=1e-10)
    assert r1.extra["dropped_term"] == "R x C"
    assert any("single value" in w for w in r1.warnings)
    assert next(a for a in r1.assumptions if a.name == "Replication").status == "warn"
    # scipy cross-check: with one value per cell this is the randomized-block F test
    assert r1.anova_table[0]["p"] == pytest.approx(stats.f.sf(table_row(r1, "R")["f"], 3, 6), rel=1e-10)


def test_three_way_single_replicate_drops_only_the_three_way_interaction():
    rng = np.random.default_rng(9)
    rows = [(f"a{i}", f"b{j}", f"c{k}", rng.normal()) for i, j, k in itertools.product(range(2), range(3), range(2))]
    frame = pd.DataFrame(rows, columns=["A", "B", "C", "y"])
    r = fx.anova_threeway(frame, "y", "A", "B", "C")
    sources = [row["source"] for row in r.anova_table]
    assert "A x B x C" not in sources and "B x C" in sources
    assert table_row(r, "Residual")["df"] == 2  # (2-1)(3-1)(2-1)
    assert r.extra["dropped_term"] == "A x B x C"


def test_model_comparison_fallback_matches_anova_lm():
    """The degenerate-case SS routine gives the same numbers as anova_lm on regular data."""
    frame = unbalanced_two_way(seed=13)
    for ss_type in (2, 3):
        design = fx._prepare(frame, "y", ["A", "B"], None, None)
        fit = fx._fit(design, ss_type, 2)
        exog = np.asarray(fit.result.model.exog)
        keys = [fx._term_key(n) for n in fit.result.model.exog_names]
        ss = fx._ss_model_comparison(design.y, exog, keys, fit.terms, ss_type)
        for term in fit.terms:
            assert ss[term] == pytest.approx(fit.ss[term], rel=1e-8)


def test_perfect_fit_and_constant_data_give_warnings_not_numbers():
    # cell means differ but there is no variation within cells: MS residual = 0
    perfect = long_from_cells({("a", "x"): [1, 1], ("a", "y"): [2, 2], ("b", "x"): [4, 4], ("b", "y"): [9, 9]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = fx.anova_twoway(perfect, "y", "A", "B", posthoc="tukey")
    assert r.statistic is None and r.p_value is None
    assert table_row(r, "Residual")["ss"] == 0.0
    # the sums of squares are still the classical ones: SS_A = 4 * (1.5 - 4)^2 + 4 * (6.5 - 4)^2 ...
    assert table_row(r, "A")["ss"] == pytest.approx(2 * 2 * ((1.5 - 4) ** 2 + (6.5 - 4) ** 2))
    assert table_row(r, "A x B")["ss"] == pytest.approx(8.0)
    assert table_row(r, "A")["f"] is None and table_row(r, "A")["partial_eta_sq"] == 1.0
    assert any("perfectly" in w for w in r.warnings) and any("skipped" in w for w in r.warnings)
    assert r.comparisons == []

    constant = long_from_cells({("a", "x"): [3, 3], ("a", "y"): [3, 3], ("b", "x"): [3, 3], ("b", "y"): [3, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = fx.anova_twoway(constant, "y", "A", "B")
    assert r.p_value is None and all(row["ss"] == 0 for row in r.anova_table)
    assert any("identical" in w for w in r.warnings)
    assert r.effect_sizes[0].value is None
    json.loads(r.finalize().model_dump_json())


# =============================================================================
# Multiple comparisons
# =============================================================================
def test_all_cells_tukey_matches_statsmodels_tukeyhsd():
    for frame in (battery_frame().rename(columns={"Material": "A", "Temperature": "B", "Life": "y"}),
                  unbalanced_two_way(seed=6)):
        r = fx.anova_twoway(frame, "y", "A", "B", posthoc="tukey", posthoc_compare="all_cells")
        cells = frame["A"] + ": " + frame["B"]
        ref = pairwise_tukeyhsd(frame["y"].to_numpy(), cells.to_numpy())
        names = list(ref.groupsunique)
        pairs = list(zip(*np.triu_indices(len(names), 1)))
        assert len(r.comparisons) == len(pairs)
        for (i, j), diff, p, (lo, hi) in zip(pairs, ref.meandiffs, ref.pvalues, ref.confint):
            comp = r.comparison(names[i], names[j])
            sign = 1.0 if comp.group_a == names[j] else -1.0  # statsmodels reports group2 - group1
            assert comp.estimate == pytest.approx(sign * diff, rel=1e-9, abs=1e-12)
            assert comp.p_adjusted == pytest.approx(p, rel=1e-4, abs=1e-7)
            got = (comp.ci_low, comp.ci_high) if sign > 0 else (-comp.ci_high, -comp.ci_low)
            assert got == pytest.approx((lo, hi), rel=1e-4, abs=1e-6)
            assert comp.statistic_name == "q" and comp.df == len(frame) - len(names)
        assert "k = " in r.comparison_family
    assert r.correction == "Tukey-Kramer"  # the last frame is unbalanced


def test_simple_effects_use_pooled_error_and_documented_families():
    frame = battery_frame()
    r = fx.anova_twoway(frame, "Life", "Material", "Temperature", posthoc="tukey",
                        posthoc_compare="simple_a_within_b")
    assert len(r.comparisons) == 9 and r.correction == "Tukey HSD"
    mse, df = 18230.75 / 27, 27
    comp = r.comparison("70: 1", "70: 2")
    assert comp.label == "Temperature = 70"
    assert comp.cell_a == {"Temperature": "70", "Material": "1"}
    assert comp.cell_b == {"Temperature": "70", "Material": "2"}
    est = np.mean(BATTERY[("1", "70")]) - np.mean(BATTERY[("2", "70")])
    se = math.sqrt(mse * 2 / 4)
    assert comp.estimate == pytest.approx(est) and comp.se == pytest.approx(se, rel=1e-10)
    t = est / se
    assert comp.statistic == pytest.approx(abs(t) * math.sqrt(2), rel=1e-10)  # q
    assert comp.p_value == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-9)
    # Tukey with k = 3 (the three materials compared within one temperature)
    assert comp.p_adjusted == pytest.approx(stats.studentized_range.sf(abs(t) * math.sqrt(2), 3, df), rel=1e-6)
    half = stats.studentized_range.ppf(0.95, 3, df) / math.sqrt(2) * se
    assert (comp.ci_low, comp.ci_high) == pytest.approx((est - half, est + half), rel=1e-6)
    assert r.extra["tukey_k"] == 3
    # Hedges' g uses the pooled residual SD and J(df residual)
    assert comp.effect_size.name == "Hedges' g"
    assert comp.effect_size.value == pytest.approx(est / math.sqrt(mse) * es.hedges_correction(df), rel=1e-10)
    assert comp.effect_size.ci_low < comp.effect_size.value < comp.effect_size.ci_high
    scale = se / math.sqrt(mse) * es.hedges_correction(df)
    assert stats.nct.cdf(t, df, comp.effect_size.ci_low / scale) == pytest.approx(0.975, abs=1e-6)

    # the swapped mode compares temperatures within each material
    r2 = fx.anova_twoway(frame, "Life", "Material", "Temperature", posthoc="bonferroni",
                         posthoc_compare="simple_b_within_a")
    comp2 = r2.comparison("1: 15", "1: 70")
    assert comp2.label == "Material = 1" and comp2.statistic_name == "t"
    raw = np.array([c.p_value for c in r2.comparisons])
    np.testing.assert_allclose([c.p_adjusted for c in r2.comparisons], np.minimum(raw * 9, 1.0), rtol=1e-12)
    crit = stats.t.ppf(1 - 0.05 / (2 * 9), df)
    assert comp2.ci_high - comp2.estimate == pytest.approx(crit * comp2.se, rel=1e-10)
    assert "one family" in r2.comparison_family
    # significance flags follow the adjusted P values after finalize()
    r2.finalize()
    assert all(c.significant == (c.p_adjusted < 0.05) for c in r2.comparisons)
    assert all(c.id.startswith(r2.id + ":c") for c in r2.comparisons)


@pytest.mark.parametrize("method,sm_name", [("holm", "holm"), ("sidak", "sidak"), ("holm-sidak", "holm-sidak"),
                                            ("fdr_bh", "fdr_bh"), ("bonferroni", "bonferroni")])
def test_adjustment_methods_match_statsmodels_multipletests(method, sm_name):
    frame = unbalanced_two_way(seed=17)
    r = fx.anova_twoway(frame, "y", "A", "B", posthoc=method, posthoc_compare="simple_b_within_a")
    raw = [c.p_value for c in r.comparisons]
    np.testing.assert_allclose([c.p_adjusted for c in r.comparisons], multipletests(raw, method=sm_name)[1],
                               rtol=1e-10)
    # every raw P value is the pooled-error t test of the two cell means
    mse = table_row(r, "Residual")["ms"]
    df = table_row(r, "Residual")["df"]
    comp = r.comparisons[0]
    ya = frame.loc[(frame.A == comp.cell_a["A"]) & (frame.B == comp.cell_a["B"]), "y"]
    yb = frame.loc[(frame.A == comp.cell_b["A"]) & (frame.B == comp.cell_b["B"]), "y"]
    t = (ya.mean() - yb.mean()) / math.sqrt(mse * (1 / len(ya) + 1 / len(yb)))
    assert comp.statistic == pytest.approx(t, rel=1e-9)
    assert comp.p_value == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-9)
    m = len(raw)
    width = (comp.ci_high - comp.ci_low) / (2 * comp.se)
    expected = {
        "bonferroni": stats.t.ppf(1 - 0.05 / (2 * m), df), "holm": stats.t.ppf(1 - 0.05 / (2 * m), df),
        "sidak": stats.t.ppf(1 - (1 - 0.95 ** (1 / m)) / 2, df),
        "holm-sidak": stats.t.ppf(1 - (1 - 0.95 ** (1 / m)) / 2, df),
        "fdr_bh": stats.t.ppf(0.975, df),
    }[method]
    assert width == pytest.approx(expected, rel=1e-9)
    assert "comparisons_note" in r.extra


def test_main_effect_comparisons_match_statsmodels_contrasts():
    frame = unbalanced_two_way(seed=30, counts=((3, 4, 2), (5, 2, 4), (2, 6, 3)))
    r = fx.anova_twoway(frame, "y", "A", "B", posthoc="none")
    assert r.comparisons == [] and r.correction is None
    r = fx.anova_twoway(frame, "y", "A", "B", posthoc="tukey", posthoc_compare="main_a")
    assert len(r.comparisons) == 3
    # In the sum-coded model the difference between unweighted marginal means of a0 and a1 is
    # alpha_0 - alpha_1: a linear contrast of the coefficients.
    fit = smf.ols("y ~ C(A, Sum) * C(B, Sum)", data=frame).fit()
    ref = fit.t_test("C(A, Sum)[S.a0] - C(A, Sum)[S.a1] = 0")
    comp = r.comparison("a0", "a1")
    assert comp.label == "Main effect of A" and comp.cell_a == {"A": "a0"}
    assert comp.estimate == pytest.approx(float(ref.effect[0]), rel=1e-9)
    assert comp.se == pytest.approx(float(ref.sd[0, 0]), rel=1e-9)
    assert comp.p_value == pytest.approx(float(ref.pvalue), rel=1e-7)
    t = comp.estimate / comp.se
    assert comp.p_adjusted == pytest.approx(stats.studentized_range.sf(abs(t) * math.sqrt(2), 3, fit.df_resid), rel=1e-6)
    assert "marginal means" in r.comparison_family
    # the other factor, and a three-factor conditional comparison averaged over the third factor
    rb = fx.anova_twoway(frame, "y", "A", "B", posthoc="holm", posthoc_compare="main_b")
    assert {c.label for c in rb.comparisons} == {"Main effect of B"}
    f3 = unbalanced_three_way()
    r3 = fx.anova_factorial(f3, "y", ["A", "B", "C"], posthoc="sidak", posthoc_compare="simple_c_within_a")
    assert len(r3.comparisons) == 2 and "averaged (unweighted) over B" in r3.comparison_family
    comp = r3.comparisons[0]
    cellmeans = f3.groupby(["A", "B", "C"])["y"].mean()
    est = np.mean([cellmeans[("a0", b, "c0")] - cellmeans[("a0", b, "c1")] for b in ("b0", "b1", "b2")])
    assert comp.estimate == pytest.approx(est, rel=1e-10)
    r3b = fx.anova_threeway(f3, "y", "A", "B", "C", posthoc="tukey")  # default simple_a_within_bc
    assert len(r3b.comparisons) == 6 and r3b.comparisons[0].label == "B = b0, C = c0"
    r3c = fx.anova_factorial(f3, "y", ["A", "B", "C"], posthoc="holm", posthoc_compare="simple_b")
    assert len(r3c.comparisons) == 12


# =============================================================================
# ANCOVA
# =============================================================================
def ancova_frame(seed=1, slopes=(0.8, 0.8, 0.8), n=(9, 7, 11)):
    rng = np.random.default_rng(seed)
    rows = []
    for gi, (slope, count) in enumerate(zip(slopes, n)):
        x = rng.normal(10 + gi, 2, count)
        y = 3 + 1.2 * gi + slope * x + rng.normal(0, 1, count)
        rows += [(f"g{gi}", xv, yv) for xv, yv in zip(x, y)]
    return pd.DataFrame(rows, columns=["group", "baseline", "response"])


def test_ancova_matches_statsmodels_and_closed_forms():
    frame = ancova_frame()
    r = fx.ancova(frame, "response", "group", "baseline")
    fit = smf.ols("response ~ C(group) + baseline", data=frame).fit()
    ref = anova_lm(fit, typ=2)
    assert r.test_name.startswith("Analysis of covariance")
    assert r.statistic == pytest.approx(ref.loc["C(group)", "F"], rel=1e-8)
    assert r.p_value == pytest.approx(ref.loc["C(group)", "PR(>F)"], rel=1e-6)
    assert (r.df, r.df2) == (2, len(frame) - 4)
    assert table_row(r, "baseline")["f"] == pytest.approx(ref.loc["baseline", "F"], rel=1e-8)
    # from scratch: SS(group | covariate) = RSS(covariate only) - RSS(full)
    y, x = frame["response"].to_numpy(), frame["baseline"].to_numpy()
    codes = pd.factorize(frame["group"])[0]
    dummies = np.eye(3)[codes][:, 1:]
    assert table_row(r, "group")["ss"] == pytest.approx(rss(y, x) - rss(y, x, dummies), rel=1e-8)
    assert table_row(r, "baseline")["ss"] == pytest.approx(rss(y, dummies) - rss(y, x, dummies), rel=1e-8)
    # slope
    slope = r.parameter("Slope: baseline")
    assert slope.value == pytest.approx(fit.params["baseline"], rel=1e-9)
    assert slope.se == pytest.approx(fit.bse["baseline"], rel=1e-9)
    assert (slope.ci_low, slope.ci_high) == pytest.approx(tuple(fit.conf_int().loc["baseline"]), rel=1e-8)
    assert slope.p_value == pytest.approx(fit.pvalues["baseline"], rel=1e-7)
    # adjusted means: ybar_i - b (xbar_i - xbar), SE^2 = MSE (1/n_i + (xbar_i - xbar)^2 / Exx)
    mse = fit.mse_resid
    exx = sum(((g["baseline"] - g["baseline"].mean()) ** 2).sum() for _, g in frame.groupby("group"))
    for entry in r.extra["adjusted_means"]:
        g = frame[frame.group == entry["group"]]
        dx = g["baseline"].mean() - x.mean()
        assert entry["adjusted_mean"] == pytest.approx(g["response"].mean() - slope.value * dx, rel=1e-9)
        assert entry["se"] == pytest.approx(math.sqrt(mse * (1 / len(g) + dx**2 / exx)), rel=1e-8)
        assert entry["mean"] == pytest.approx(g["response"].mean())
        assert entry["ci_low"] < entry["adjusted_mean"] < entry["ci_high"]
    assert r.extra["covariate_means"]["baseline"] == pytest.approx(x.mean())
    line = r.extra["group_lines"]["g1"]
    assert line["intercept"] == pytest.approx(fit.params["Intercept"] + fit.params["C(group)[T.g1]"], rel=1e-8)
    assert r.effect("partial eta squared").value == pytest.approx(
        table_row(r, "group")["ss"] / (table_row(r, "group")["ss"] + fit.ssr), rel=1e-9)
    # same analysis through the general factorial function
    r2 = fx.anova_factorial(frame, "response", ["group"], covariates=["baseline"])
    assert r2.test_name == "One-way ANCOVA"
    assert table_row(r2, "group")["f"] == pytest.approx(r.statistic, rel=1e-10)
    assert r2.extra["cell_means"][0]["adjusted_mean"] == pytest.approx(r.extra["adjusted_means"][0]["adjusted_mean"])
    assert r2.parameter("Slope: baseline").value == pytest.approx(slope.value)


def test_ancova_homogeneity_of_slopes_check():
    frame = ancova_frame(seed=2)
    r = fx.ancova(frame, "response", "group", "baseline")
    check = next(a for a in r.assumptions if a.name.startswith("Equal slopes"))
    full = smf.ols("response ~ C(group) * baseline", data=frame).fit()
    ref = anova_lm(full, typ=2).loc["C(group):baseline"]
    assert check.statistic == pytest.approx(ref["F"], rel=1e-8)
    assert check.p_value == pytest.approx(ref["PR(>F)"], rel=1e-6)
    assert check.status == "pass" and check.target == "baseline"
    # clearly different slopes -> fail + warning
    bad = ancova_frame(seed=2, slopes=(0.1, 1.0, 2.2), n=(12, 12, 12))
    rb = fx.ancova(bad, "response", "group", "baseline")
    check = next(a for a in rb.assumptions if a.name.startswith("Equal slopes"))
    assert check.status == "fail" and "not appropriate" in check.detail
    assert any("slopes differ" in w for w in rb.warnings)
    assert rb.worst_assumption_status() == "fail"
    # too few values to fit separate slopes -> warn, not crash
    tiny = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [1.0, 2, 3, 5, 4], "y": [1.0, 2.5, 2, 4, 3]})
    rt = fx.ancova(tiny, "y", "g", "x")
    assert next(a for a in rt.assumptions if a.name.startswith("Equal slopes")).status == "warn"


def test_ancova_pairwise_comparisons_of_adjusted_means():
    frame = ancova_frame(seed=5)
    fit = smf.ols("response ~ C(group) + baseline", data=frame).fit()
    r = fx.ancova(frame, "response", "group", "baseline", posthoc="tukey")
    assert "approximate" in r.correction
    comp = r.comparison("g0", "g1")
    assert comp.estimate == pytest.approx(-fit.params["C(group)[T.g1]"], rel=1e-9)
    assert comp.se == pytest.approx(fit.bse["C(group)[T.g1]"], rel=1e-9)
    assert comp.p_value == pytest.approx(fit.pvalues["C(group)[T.g1]"], rel=1e-7)
    t = comp.estimate / comp.se
    assert comp.p_adjusted == pytest.approx(stats.studentized_range.sf(abs(t) * math.sqrt(2), 3, fit.df_resid), rel=1e-6)
    ref = fit.t_test("C(group)[T.g1] - C(group)[T.g2] = 0")
    comp12 = r.comparison("g1", "g2")
    assert comp12.estimate == pytest.approx(float(ref.effect[0]), rel=1e-9)
    assert comp12.se == pytest.approx(float(ref.sd[0, 0]), rel=1e-9)
    rh = fx.ancova(frame, "response", "group", "baseline", posthoc="holm")
    np.testing.assert_allclose([c.p_adjusted for c in rh.comparisons],
                               multipletests([c.p_value for c in rh.comparisons], method="holm")[1], rtol=1e-10)
    assert comp.effect_size.value == pytest.approx(
        comp.estimate / math.sqrt(fit.mse_resid) * es.hedges_correction(fit.df_resid), rel=1e-9)


def test_ancova_two_covariates_and_errors():
    rng = np.random.default_rng(12)
    n = 40
    frame = pd.DataFrame({"g": rng.choice(["a", "b", "c"], n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    frame["y"] = frame["x1"] * 0.5 - frame["x2"] * 0.3 + (frame["g"] == "c") * 1.0 + rng.normal(size=n)
    r = fx.ancova(frame, "y", "g", ["x1", "x2"], ss_type=2)
    ref = anova_lm(smf.ols("y ~ C(g) + x1 + x2", data=frame).fit(), typ=2)
    assert r.statistic == pytest.approx(ref.loc["C(g)", "F"], rel=1e-8)
    assert table_row(r, "x2")["p"] == pytest.approx(ref.loc["x2", "PR(>F)"], rel=1e-6)
    assert [p.name for p in r.parameters] == ["Slope: x1", "Slope: x2"]
    assert "group_lines" not in r.extra
    assert len([a for a in r.assumptions if a.name.startswith("Equal slopes")]) == 2
    with pytest.raises(StatsError, match="at least one covariate"):
        fx.ancova(frame, "y", "g", [])
    with pytest.raises(StatsError, match="same value in every row"):
        fx.ancova(frame.assign(k=1.0), "y", "g", "k")
    with pytest.raises(StatsError, match="completely determined"):
        fx.ancova(frame.assign(k=(frame.g == "a").astype(float)), "y", "g", "k")
    with pytest.raises(StatsError, match="exactly one"):
        fx.ancova(frame, "y", ["g"], "x1")
    # the covariate differs between groups -> flagged
    shifted = ancova_frame(seed=3)
    shifted.loc[shifted.group == "g2", "baseline"] += 15
    rs = fx.ancova(shifted, "response", "group", "baseline")
    assert next(a for a in rs.assumptions if a.name == "Covariate independent of group").status == "warn"


def test_ancova_degenerate_data_give_warnings():
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 2.0, 6.0, 9.0])
    g = np.repeat(["a", "b", "c"], 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exact = fx.ancova(pd.DataFrame({"g": g, "x": x, "y": 2 * x + np.repeat([0.0, 1.0, 5.0], 3)}), "y", "g", "x",
                          posthoc="tukey")
        const = fx.ancova(pd.DataFrame({"g": g, "x": x, "y": 3.0}), "y", "g", "x", posthoc="holm")
    assert exact.p_value is None and any("perfectly" in w for w in exact.warnings)
    assert any("skipped" in w for w in exact.warnings) and exact.comparisons == []
    assert exact.parameter("Slope: x").value == pytest.approx(2.0) and exact.parameter("Slope: x").p_value is None
    assert [e["adjusted_mean"] for e in exact.extra["adjusted_means"]] == pytest.approx(
        [2 * x.mean() + s for s in (0.0, 1.0, 5.0)])
    assert "could not be tested" in exact.interpretation
    assert const.p_value is None and any("identical" in w for w in const.warnings)
    assert next(a for a in exact.assumptions if a.name.startswith("Equal slopes")).status == "warn"
    for res in (exact, const):
        json.loads(res.finalize().model_dump_json())
    assert fx._rss(np.array([1.0, 2.0]), np.empty((2, 0))) == pytest.approx(5.0)


def test_factorial_with_covariate_uses_adjusted_means():
    rng = np.random.default_rng(41)
    frame = unbalanced_two_way(seed=40, counts=((4, 5), (3, 6)))
    frame["x"] = rng.normal(size=len(frame))
    frame["y"] += 0.9 * frame["x"]
    r = fx.anova_twoway(frame, "y", "A", "B", covariates=["x"], posthoc="bonferroni", posthoc_compare="all_cells")
    assert r.test_name == "Two-way ANCOVA"
    fit = smf.ols("y ~ C(A, Sum) * C(B, Sum) + x", data=frame).fit()
    ref = anova_lm(fit, typ=3)
    assert table_row(r, "A x B")["f"] == pytest.approx(ref.loc["C(A, Sum):C(B, Sum)", "F"], rel=1e-8)
    assert table_row(r, "x")["kind"] == "covariate"
    assert table_row(r, "x")["f"] == pytest.approx(ref.loc["x", "F"], rel=1e-8)
    slope = fit.params["x"]
    for entry in r.extra["cell_means"]:
        cell = frame[(frame.A == entry["A"]) & (frame.B == entry["B"])]
        expected = cell["y"].mean() - slope * (cell["x"].mean() - frame["x"].mean())
        assert entry["adjusted_mean"] == pytest.approx(expected, rel=1e-9)
    comp = r.comparison("a0: b0", "a1: b1")
    means = {(e["A"], e["B"]): e["adjusted_mean"] for e in r.extra["cell_means"]}
    assert comp.estimate == pytest.approx(means[("a0", "b0")] - means[("a1", "b1")], rel=1e-10)
    assert "adjusted" in r.extra["comparisons_note"]
    assert any(a.name == "Covariate model" for a in r.assumptions)


# =============================================================================
# Assumptions, input handling, errors
# =============================================================================
def test_assumption_checks_react_to_violations():
    rng = np.random.default_rng(0)
    rows = []
    for a, b in itertools.product("pq", "xy"):
        sd = 12.0 if (a, b) == ("q", "y") else 0.5
        rows += [(a, b, v) for v in rng.normal(10, sd, 14)]
    hetero = pd.DataFrame(rows, columns=["A", "B", "y"])
    r = fx.anova_twoway(hetero, "y", "A", "B")
    var = next(a for a in r.assumptions if a.name.startswith("Equal variances"))
    cells = [g["y"].to_numpy() for _, g in hetero.groupby(["A", "B"])]
    ref = stats.levene(*cells, center="median")
    assert var.status == "fail" and var.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert var.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    skewed = pd.DataFrame([(a, b, v) for a, b in itertools.product("pq", "xy") for v in rng.lognormal(0, 1.5, 15)],
                          columns=["A", "B", "y"])
    r = fx.anova_twoway(skewed, "y", "A", "B")
    norm = next(a for a in r.assumptions if a.name.startswith("Normality"))
    resid = skewed["y"] - skewed.groupby(["A", "B"])["y"].transform("mean")
    assert norm.status == "fail" and norm.p_value == pytest.approx(stats.shapiro(resid).pvalue, rel=1e-8)
    assert r.extra["residuals"]["residual"] == pytest.approx(resid.to_list())
    assert [a.status for a in r.assumptions if a.name == "Independent observations"] == ["info"]
    # cells with a single value are left out of the variance test
    partial = long_from_cells({("a", "x"): [1.0, 2, 4], ("a", "y"): [3.0], ("b", "x"): [2.0, 5, 3], ("b", "y"): [4.0, 7]})
    r = fx.anova_twoway(partial, "y", "A", "B")
    assert "single value were left out" in next(a for a in r.assumptions if a.name.startswith("Equal var")).detail
    assert next(a for a in r.assumptions if a.name == "Replication").status == "warn"


def test_helper_checks_handle_degenerate_input():
    assert fx._normality_check(np.array([1.0, 2.0])).status == "warn"
    assert fx._normality_check(np.zeros(8)).status == "warn"
    assert fx._brown_forsythe({"a": np.array([1.0]), "b": np.array([1.0, 2.0])}, "x").status == "warn"
    assert fx._brown_forsythe({"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0])}, "x").status == "warn"
    assert fx._hedges_g_pooled(1.0, 0.5, 0.0, 10) is None
    assert fx._smd_pooled(1.0, 0.0, 1.0, 10) is None
    assert math.isnan(fx._tukey_p(float("nan"), 3, 10))
    assert fx._tukey_p(0.0, 3, 10) == pytest.approx(1.0)
    assert all(math.isnan(v) for v in fx._nct_limits(float("inf"), 5, 0.95))
    assert fx._none_if_nan(None) is None and fx._none_if_nan(float("nan")) is None
    assert fx._level_label(4.0) == "4" and fx._level_label(np.int64(3)) == "3" and fx._level_label(True) == "True"
    assert fx._describe(np.array([2.0]))["sd"] is None


def test_missing_values_level_order_and_labels():
    frame = battery_frame()
    frame.loc[0, "Life"] = np.nan
    frame.loc[5, "Material"] = None
    frame["Temperature"] = pd.Categorical(frame["Temperature"], categories=["125", "70", "15"], ordered=True)
    r = fx.anova_twoway(frame, "Life", "Material", "Temperature",
                        factor_labels={"Material": "Plate material", "Temperature": "Temp (F)"})
    assert r.data_notes == ["2 row(s) with a missing or non-numeric value were ignored."]
    assert r.n_total == 34 and r.n["1: 15"] == 3
    assert r.extra["levels"]["Temperature"] == ["125", "70", "15"]  # categorical order is respected
    assert r.extra["factor_labels"] == {"Material": "Plate material", "Temperature": "Temp (F)"}
    assert [row["source"] for row in r.anova_table][:3] == ["Plate material", "Temp (F)", "Plate material x Temp (F)"]
    assert set(r.extra["cell_means"][0]) == {"Material", "Temperature", "mean", "sd", "sem", "n"}
    assert r.descriptives[0]["group"] == "1: 125" and r.descriptives[0]["median"] is not None
    # numeric factor columns are ordered numerically and labelled without a trailing ".0"
    numeric = long_from_cells(TOOTH, ("supp", "dose"), "len")
    numeric["dose"] = numeric["dose"].astype(float)
    numeric = numeric.sample(frac=1.0, random_state=1)
    r = fx.anova_twoway(numeric, "len", "supp", "dose")
    assert r.extra["levels"]["dose"] == ["0.5", "1", "2"]
    assert table_row(r, "dose")["f"] == pytest.approx(92.000, abs=5e-4)
    # identical labels for two factors are disambiguated
    r = fx.anova_twoway(numeric, "len", "supp", "dose", factor_labels={"supp": "X", "dose": "X"})
    assert r.extra["factor_labels"] == {"supp": "X (supp)", "dose": "X (dose)"}


def test_impossible_input_raises_stats_error():
    frame = unbalanced_two_way()
    hole = frame[~((frame.A == "a1") & (frame.B == "b2"))]
    with pytest.raises(StatsError, match="no data"):
        fx.anova_twoway(hole, "y", "A", "B")
    with pytest.raises(StatsError, match="only one level"):
        fx.anova_twoway(frame[frame.A == "a0"], "y", "A", "B")
    with pytest.raises(StatsError, match="not found"):
        fx.anova_twoway(frame, "y", "A", "nope")
    with pytest.raises(StatsError, match="only one role"):
        fx.anova_twoway(frame, "y", "A", "A")
    with pytest.raises(StatsError, match="fewer than 3"):
        fx.anova_twoway(frame.assign(y=np.nan), "y", "A", "B")
    with pytest.raises(StatsError, match="ss_type"):
        fx.anova_twoway(frame, "y", "A", "B", ss_type=1)
    with pytest.raises(StatsError, match="multiple-comparison method"):
        fx.anova_twoway(frame, "y", "A", "B", posthoc="dunnett")
    with pytest.raises(StatsError, match="comparison set"):
        fx.anova_twoway(frame, "y", "A", "B", posthoc="tukey", posthoc_compare="simple_c_within_a")
    with pytest.raises(StatsError, match="comparison set"):
        fx.anova_twoway(frame, "y", "A", "B", posthoc="tukey", posthoc_compare="simple_a_within_a")
    with pytest.raises(StatsError, match="alpha"):
        fx.anova_twoway(frame, "y", "A", "B", alpha=1.5)
    with pytest.raises(StatsError, match="at least one grouping factor"):
        fx.anova_factorial(frame, "y", [])
    with pytest.raises(TypeError):
        fx.anova_twoway(frame.to_dict("list"), "y", "A", "B")
    # 2 x 2 with one value per cell and no interaction still has 1 residual df; a saturated
    # model with a covariate has none
    one = long_from_cells({("a", "x"): [1.0], ("a", "y"): [2.0], ("b", "x"): [4.0], ("b", "y"): [3.0]})
    assert table_row(fx.anova_twoway(one, "y", "A", "B"), "Residual")["df"] == 1
    one["x"] = [0.1, 0.7, 0.3, 0.2]
    with pytest.raises(StatsError, match="no residual degrees of freedom"):
        fx.anova_twoway(one, "y", "A", "B", covariates=["x"])


def test_option_spellings_and_type2_warning():
    frame = battery_frame()
    r = fx.anova_twoway(frame, "Life", "Material", "Temperature", ss_type="II", posthoc="FDR-BH",
                        posthoc_compare="MAIN_B")
    assert r.extra["ss_type"] == 2 and r.correction == "Benjamini-Hochberg FDR"
    assert any("Type II main-effect tests assume" in w for w in r.warnings)
    assert "interpret the main effects with caution" in r.interpretation
    assert "Benjamini" in " ".join(r.citations)
    assert "two-way ANOVA" in r.methods_text and "Type II" in r.methods_text
    for term in ("Material", "Temperature", "Material x Temperature"):
        eta = next(e for e in r.effect_sizes if e.name == "partial eta squared" and e.target == term)
        row = table_row(r, term)
        assert eta.value == pytest.approx(row["partial_eta_sq"])
        lo, hi = es.eta_squared_ci(row["f"], row["df"], 27)
        assert (eta.ci_low, eta.ci_high) == pytest.approx((lo, hi))
        omega = next(e for e in r.effect_sizes if e.name == "partial omega squared" and e.target == term)
        assert omega.value == pytest.approx(row["df"] * (row["f"] - 1) / (row["df"] * (row["f"] - 1) + 36))


# =============================================================================
# Registry (table-level) analyses
# =============================================================================
def grouped_battery():
    cells = {temp: {mat: list(BATTERY[(mat, temp)]) for mat in ("1", "2", "3")} for temp in ("15", "70", "125")}
    return GroupedTable.from_cells("Battery", cells, row_label="Temperature", column_label="Material")


def check_registry_result(result, key, table_name):
    assert result.analysis == key and result.table == table_name
    assert result.interpretation and result.why_chosen and result.methods_text and result.citations
    assert len(result.assumptions) >= 1 and result.significant is not None
    assert result.n_total == sum(result.n.values())
    payload = json.loads(result.model_dump_json())
    assert payload["anova_table"] and payload["extra"]["factors"]


def test_registry_two_way_on_grouped_table(run):
    table = grouped_battery()
    r = run(table, "anova_twoway")
    check_registry_result(r, "anova_twoway", "Battery")
    assert r.params["ss_type"] == 3 and r.params["posthoc"] == "none" and r.comparisons == []
    assert table_row(r, "Temperature")["ss"] == pytest.approx(39118.72, abs=0.005)
    assert table_row(r, "Temperature x Material")["f"] == pytest.approx(3.56, abs=0.005)
    assert r.extra["factors"] == ["row", "column"]
    assert r.extra["factor_labels"] == {"row": "Temperature", "column": "Material"}
    assert set(r.extra["cell_means"][0]) == {"row", "column", "mean", "sd", "sem", "n"}
    assert "Factor A = rows" in r.data_notes[0]
    r = run(table, "anova_twoway", posthoc="tukey")  # default: columns compared within each row
    assert len(r.comparisons) == 9
    comp = r.comparison("70: 1", "70: 2")
    assert comp.label == "Temperature = 70"
    assert comp.cell_a == {"row": "70", "column": "1"} and comp.cell_b == {"row": "70", "column": "2"}
    assert comp.significant is True and comp.id.endswith(tuple("0123456789"))
    # user exclusions are honoured and reported
    table.exclude([(0, "1:1")])
    r = run(table, "anova_twoway", interaction=False, posthoc="holm", posthoc_compare="main_b")
    assert r.n_total == 35 and any("excluded" in n for n in r.data_notes)
    assert [row["source"] for row in r.anova_table] == ["Temperature", "Material", "Residual", "Total"]
    same = GroupedTable.from_cells("T", {"r1": {"c1": [1, 2, 3], "c2": [2, 4, 5]}, "r2": {"c1": [2, 2, 4], "c2": [7, 8, 9]}},
                                   row_label="X", column_label="X")
    assert run(same, "anova_twoway").extra["factor_labels"] == {"row": "X (rows)", "column": "X (columns)"}
    with pytest.raises(StatsError, match="Unknown option"):
        run(table, "anova_twoway", nonsense=1)


def test_registry_two_way_and_three_way_on_multiple_variables(run):
    frame = unbalanced_three_way()
    frame.insert(0, "dose_mg", np.arange(len(frame), dtype=float))  # a numeric column that is not the outcome
    frame = frame[["A", "B", "C", "y", "dose_mg"]]
    table = MultipleVariablesTable("mv", frame)
    r = run(table, "anova_twoway")
    check_registry_result(r, "anova_twoway", "mv")
    assert r.extra["factors"] == ["A", "B"]
    assert [n for n in r.data_notes if "automatically" in n] == [
        "Outcome column chosen automatically: 'y' (the first numeric column).",
        "Factor A chosen automatically: 'A' (the next text column).",
        "Factor B chosen automatically: 'B' (the next text column).",
    ]
    direct = fx.anova_twoway(frame, "y", "A", "B")
    assert r.statistic == pytest.approx(direct.statistic)
    r = run(table, "anova_twoway", value="y", factor_a="C", factor_b="A", ss_type=2, posthoc="sidak")
    assert r.extra["factors"] == ["C", "A"] and len(r.comparisons) == 2
    r3 = run(table, "anova_threeway")
    check_registry_result(r3, "anova_threeway", "mv")
    assert r3.extra["factors"] == ["A", "B", "C"] and r3.test_name == "Three-way ANOVA"
    r3 = run(table, "anova_threeway", posthoc="holm", posthoc_compare="main_b", interaction=False)
    assert len(r3.comparisons) == 3 and len(r3.anova_table) == 5
    with pytest.raises(StatsError, match="not in the table"):
        run(table, "anova_twoway", factor_a="missing")
    with pytest.raises(StatsError, match="automatically"):
        run(MultipleVariablesTable("few", frame[["A", "y"]]), "anova_twoway")
    with pytest.raises(StatsError, match="no numeric column"):
        run(MultipleVariablesTable("txt", frame[["A", "B"]]), "anova_twoway")
    with pytest.raises(StatsError, match="not available"):
        run(grouped_battery(), "anova_threeway")
    # a numeric column may be used as a factor when named explicitly
    tooth = long_from_cells(TOOTH, ("supp", "dose"), "len")
    tooth["dose"] = tooth["dose"].astype(float)
    r = run(MultipleVariablesTable("tooth", tooth), "anova_twoway", value="len", factor_a="supp", factor_b="dose")
    assert table_row(r, "supp x dose")["f"] == pytest.approx(4.107, abs=5e-4)


def test_registry_ancova(run):
    frame = ancova_frame()
    frame["other"] = np.linspace(0, 1, len(frame))
    table = MultipleVariablesTable("anc", frame[["response", "group", "baseline", "other"]])
    r = run(table, "ancova")
    check_registry_result(r, "ancova", "anc")
    assert r.extra["covariates"] == ["baseline"] and r.extra["factors"] == ["group"]
    assert len([n for n in r.data_notes if "automatically" in n]) == 3
    direct = fx.ancova(frame, "response", "group", "baseline")
    assert r.statistic == pytest.approx(direct.statistic)
    r = run(table, "ancova", value="response", factor="group", covariates=["baseline", "other"], posthoc="tukey")
    assert len(r.parameters) == 2 and len(r.comparisons) == 3 and not any("automatically" in n for n in r.data_notes)
    r = run(table, "ancova", covariates="other")
    assert r.extra["covariates"] == ["other"]
    with pytest.raises(StatsError, match="not in the table"):
        run(table, "ancova", covariates=["nope"])
    with pytest.raises(StatsError, match="cannot also be a covariate"):
        run(table, "ancova", value="baseline", covariates=["baseline"])
    with pytest.raises(StatsError, match="covariate column"):
        run(MultipleVariablesTable("nocov", frame[["response", "group"]]), "ancova")
    with pytest.raises(StatsError, match="no numeric column"):
        run(MultipleVariablesTable("nonum", frame[["group"]].assign(g2=frame["group"])), "ancova")
    info = registry.REGISTRY
    for key in ("anova_twoway", "anova_threeway", "ancova"):
        assert info[key].category == "Factorial designs"
        assert all(p.help for p in info[key].params) and info[key].description


# =============================================================================
# Property-based tests
# =============================================================================
@st.composite
def random_design(draw, integer_values=False):
    a = draw(st.integers(2, 3))
    b = draw(st.integers(2, 3))
    counts = draw(st.lists(st.integers(1, 4), min_size=a * b, max_size=a * b))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    rows = []
    for (i, j), n in zip(itertools.product(range(a), range(b)), counts):
        if integer_values:
            values = rng.integers(-2, 3, n).astype(float)
        else:
            values = rng.normal(i - 0.5 * j + 0.3 * i * j, 1.0, n)
        rows += [(f"a{i}", f"b{j}", v) for v in values]
    return pd.DataFrame(rows, columns=["A", "B", "y"])


@settings(max_examples=40, deadline=None)
@given(frame=random_design(), ss_type=st.sampled_from([2, 3]), shift=st.floats(-1e3, 1e3),
       scale=st.floats(0.01, 100.0), seed=st.integers(0, 10**6))
def test_property_table_matches_scratch_and_is_invariant(frame, ss_type, shift, scale, seed):
    counts = frame.groupby(["A", "B"]).size()
    replicated = bool((counts > 1).any())
    r = fx.anova_twoway(frame, "y", "A", "B", ss_type=ss_type)
    blocks = scratch_blocks(frame, ["A", "B"])
    if not replicated:
        blocks.pop((0, 1))
        assert r.extra["dropped_term"] == "A x B"
    ss = scratch_ss(frame["y"].to_numpy(), blocks, ss_type)
    total = table_row(r, "Total")["ss"]
    for combo, name in (((0,), "A"), ((1,), "B"), ((0, 1), "A x B")):
        if combo in ss:
            assert table_row(r, name)["ss"] == pytest.approx(ss[combo], rel=1e-6, abs=1e-9 * total)
    for row in r.anova_table:
        assert row["ss"] >= 0
        assert row["p"] is None or 0.0 <= row["p"] <= 1.0
    # invariance: row order, level names, and a linear change of units leave F unchanged
    rng = np.random.default_rng(seed)
    other = frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    other["A"] = other["A"].map({"a0": "zeta", "a1": "alpha", "a2": "mid"})
    other["y"] = other["y"] * scale + shift
    r2 = fx.anova_twoway(other, "y", "A", "B", ss_type=ss_type)
    for name in ("A", "B", "A x B"):
        if any(row["source"] == name for row in r.anova_table):
            f1, f2 = table_row(r, name)["f"], table_row(r2, name)["f"]
            assert f2 == pytest.approx(f1, rel=1e-5, abs=1e-7)
    assert r2.n_total == r.n_total


@settings(max_examples=40, deadline=None)
@given(frame=random_design(integer_values=True), posthoc=st.sampled_from(["none", "tukey", "holm", "fdr_bh"]),
       compare=st.sampled_from(["all_cells", "main_a", "simple_a_within_b", "simple_b_within_a"]),
       n_missing=st.integers(0, 2))
def test_property_ties_zeros_constants_and_missing_never_crash(frame, posthoc, compare, n_missing):
    frame = frame.copy()
    for k in range(n_missing):
        frame.loc[len(frame)] = ["a0", "b0", np.nan] if k == 0 else [None, "b1", 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numerical warnings must not leak
        r = fx.anova_twoway(frame, "y", "A", "B", posthoc=posthoc, posthoc_compare=compare)
    assert r.p_value is None or 0.0 <= r.p_value <= 1.0
    if n_missing:
        assert any("ignored" in n for n in r.data_notes)
    resid = table_row(r, "Residual")
    if resid["ss"] == 0:
        assert r.p_value is None and r.warnings
    for comp in r.comparisons:
        for p in (comp.p_value, comp.p_adjusted):
            assert p is None or 0.0 <= p <= 1.0
        if comp.p_value is not None and comp.p_adjusted is not None:
            assert comp.p_adjusted >= comp.p_value - 1e-12
            assert comp.ci_low <= comp.estimate <= comp.ci_high
    for e in r.effect_sizes:
        assert e.value is None or 0.0 <= e.value <= 1.0
    json.loads(r.finalize().model_dump_json())


@settings(max_examples=25, deadline=None)
@given(frame=random_design(), data=st.data())
def test_property_empty_cell_is_a_stats_error(frame, data):
    cells = sorted(set(zip(frame["A"], frame["B"])))
    a, b = data.draw(st.sampled_from(cells))
    hole = frame[~((frame.A == a) & (frame.B == b))]
    with pytest.raises(StatsError):
        fx.anova_twoway(hole, "y", "A", "B")


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), k=st.integers(2, 4), posthoc=st.sampled_from(["none", "tukey", "bonferroni"]))
def test_property_ancova_is_invariant_to_covariate_units_and_group_order(seed, k, posthoc):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 6, k)
    rows = []
    for g, count in enumerate(n):
        x = rng.normal(0, 1, count)
        rows += [(f"g{g}", xv, 0.5 * xv + g + rng.normal()) for xv in x]
    frame = pd.DataFrame(rows, columns=["g", "x", "y"])
    try:
        r = fx.ancova(frame, "y", "g", "x", posthoc=posthoc)
    except StatsError:
        return  # e.g. no residual df with the smallest designs
    other = frame.iloc[::-1].assign(x=lambda d: d["x"] * 1000.0 + 7.0)
    r2 = fx.ancova(other, "y", "g", "x", posthoc=posthoc)
    assert r2.statistic == pytest.approx(r.statistic, rel=1e-6, abs=1e-9)
    assert r.p_value is None or 0 <= r.p_value <= 1
    adj = {e["group"]: e["adjusted_mean"] for e in r.extra["adjusted_means"]}
    adj2 = {e["group"]: e["adjusted_mean"] for e in r2.extra["adjusted_means"]}
    for g in adj:
        assert adj2[g] == pytest.approx(adj[g], rel=1e-6, abs=1e-8)
    assert r2.parameter("Slope: x").value == pytest.approx(r.parameter("Slope: x").value / 1000.0, rel=1e-6)
    json.loads(r.finalize().model_dump_json())
