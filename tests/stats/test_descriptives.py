"""Descriptive statistics.

Validation sources (see tests/fixtures/sources/basics.md):
* NIST StRD univariate summary statistics, data sets NumAcc1-NumAcc4 (certified mean/SD).
* Cross-checks against numpy, scipy, pandas and statsmodels (independent implementations).
* First-principles formulas written out in the tests; hypothesis property tests.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

from torusplot.data.tables import ColumnTable, GroupedTable, MultipleVariablesTable, XYTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import descriptives as ds

X = np.array([5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.1, 9.4, 4.4, 5.0])


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    """Do not require sibling analysis modules (owned by other people) to be importable."""
    monkeypatch.setattr(registry, "_loaded", True)


# ----------------------------------------------------------------- reference values


def test_nist_strd_numacc_certified_values():
    """NIST StRD univariate data sets NumAcc1-4 (numerical accuracy of mean and SD).

    Certified values: NumAcc1 mean 10000002, SD 1; NumAcc2 mean 1.2, SD 0.1;
    NumAcc3 mean 1000000.2, SD 0.1; NumAcc4 mean 10000000.2, SD 0.1.
    """
    numacc1 = [10000001.0, 10000003.0, 10000002.0]
    numacc2 = [1.2] + [1.1, 1.3] * 500
    numacc3 = [1000000.2] + [1000000.1, 1000000.3] * 500
    numacc4 = [10000000.2] + [10000000.1, 10000000.3] * 500
    d1 = ds.describe(numacc1)
    assert d1["mean"] == pytest.approx(10000002.0, rel=1e-15) and d1["sd"] == pytest.approx(1.0, rel=1e-12)
    d2 = ds.describe(numacc2)
    assert d2["n"] == 1001
    assert d2["mean"] == pytest.approx(1.2, rel=1e-14) and d2["sd"] == pytest.approx(0.1, rel=1e-12)
    d3 = ds.describe(numacc3)
    assert d3["mean"] == pytest.approx(1000000.2, rel=1e-14) and d3["sd"] == pytest.approx(0.1, rel=1e-8)
    d4 = ds.describe(numacc4)
    # the data themselves are only representable to ~2e-9 in double precision at 1e7
    assert d4["mean"] == pytest.approx(10000000.2, rel=1e-14) and d4["sd"] == pytest.approx(0.1, rel=1e-7)


def test_quartiles_are_hyndman_fan_type_7():
    # h = (n - 1) p + 1; Q(p) = x[floor(h)] + (h - floor(h)) (x[floor(h) + 1] - x[floor(h)])
    d = ds.describe([1, 2, 3, 4])
    assert (d["q1"], d["median"], d["q3"], d["iqr"]) == (1.75, 2.5, 3.25, 1.5)
    d = ds.describe([7, 1, 3, 9, 5, 11])  # sorted 1 3 5 7 9 11; h(0.25) = 2.25 -> 3 + 0.25 * 2
    assert d["q1"] == pytest.approx(3.5) and d["q3"] == pytest.approx(8.5)
    xs = np.sort(X)
    for p, key in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
        h = (xs.size - 1) * p
        lo = int(math.floor(h))
        expected = xs[lo] + (h - lo) * (xs[min(lo + 1, xs.size - 1)] - xs[lo])
        assert ds.describe(X)[key] == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------- cross-implementation


def test_describe_matches_numpy_scipy_pandas_statsmodels():
    d = ds.describe(X)
    s = pd.Series(X)
    assert d["n"] == X.size and d["n_missing"] == 0
    assert d["mean"] == pytest.approx(np.mean(X), rel=1e-12)
    assert d["sd"] == pytest.approx(np.std(X, ddof=1), rel=1e-12)
    assert d["sd"] == pytest.approx(s.std(), rel=1e-12)
    assert d["sem"] == pytest.approx(stats.sem(X), rel=1e-12)
    assert d["median"] == pytest.approx(np.median(X), rel=1e-12)
    assert d["q1"] == pytest.approx(s.quantile(0.25), rel=1e-12)
    assert d["q3"] == pytest.approx(s.quantile(0.75), rel=1e-12)
    assert d["iqr"] == pytest.approx(stats.iqr(X), rel=1e-12)
    assert (d["min"], d["max"]) == (X.min(), X.max())
    assert d["sum"] == pytest.approx(X.sum(), rel=1e-12)
    lo, hi = stats.t.interval(0.95, X.size - 1, loc=X.mean(), scale=stats.sem(X))
    assert (d["ci_low"], d["ci_high"]) == pytest.approx((lo, hi), rel=1e-10)
    sm_lo, sm_hi = DescrStatsW(X).tconfint_mean(alpha=0.05)
    assert (d["ci_low"], d["ci_high"]) == pytest.approx((sm_lo, sm_hi), rel=1e-10)
    assert d["cv_percent"] == pytest.approx(100 * stats.variation(X, ddof=1), rel=1e-12)
    assert d["skewness"] == pytest.approx(stats.skew(X, bias=False), rel=1e-10)
    assert d["skewness"] == pytest.approx(s.skew(), rel=1e-10)
    assert d["kurtosis"] == pytest.approx(stats.kurtosis(X, bias=False), rel=1e-10)
    assert d["kurtosis"] == pytest.approx(s.kurt(), rel=1e-10)
    assert d["geometric_mean"] == pytest.approx(stats.gmean(X), rel=1e-12)


def test_ci_level_changes_interval_only():
    d90, d99 = ds.describe(X, ci_level=0.90), ds.describe(X, ci_level=0.99)
    lo, hi = DescrStatsW(X).tconfint_mean(alpha=0.10)
    assert (d90["ci_low"], d90["ci_high"]) == pytest.approx((lo, hi), rel=1e-10)
    assert d99["ci_low"] < d90["ci_low"] < d90["mean"] < d90["ci_high"] < d99["ci_high"]
    assert d90["sd"] == d99["sd"]
    for bad in (0, 1, 1.5, -0.2, "x", None):
        with pytest.raises(StatsError):
            ds.describe(X, ci_level=bad)


# ------------------------------------------------------------------- edge cases


def test_small_and_degenerate_samples():
    one = ds.describe([3.5])
    assert one["n"] == 1 and one["mean"] == 3.5 and one["median"] == 3.5
    for key in ("sd", "sem", "ci_low", "ci_high", "cv_percent", "skewness", "kurtosis"):
        assert one[key] is None
    two = ds.describe([1.0, 3.0])
    assert two["sd"] == pytest.approx(math.sqrt(2)) and two["skewness"] is None and two["kurtosis"] is None
    three = ds.describe([1.0, 2.0, 6.0])
    assert three["skewness"] is not None and three["kurtosis"] is None
    const = ds.describe([2.0, 2.0, 2.0, 2.0])
    assert const["sd"] == 0.0 and const["ci_low"] == const["ci_high"] == 2.0
    assert const["skewness"] is None and const["kurtosis"] is None and const["cv_percent"] == 0.0
    zero_mean = ds.describe([-1.0, 1.0])
    assert zero_mean["cv_percent"] is None and zero_mean["geometric_mean"] is None
    assert ds.describe([0.0, 1.0, 2.0])["geometric_mean"] is None
    with pytest.raises(StatsError):
        ds.describe([])
    with pytest.raises(StatsError):
        ds.describe([np.nan, np.inf])
    with pytest.raises(StatsError):
        ds.describe(None)


def test_missing_values_are_dropped_and_counted():
    d = ds.describe([1.0, np.nan, 2.0, np.inf, 3.0])
    assert d["n"] == 3 and d["n_missing"] == 2 and d["mean"] == 2.0
    assert ds.describe(v for v in (1.0, 2.0, 3.0))["n"] == 3  # generators are fine


def test_descriptive_statistics_result():
    res = ds.descriptive_statistics({"A": X, "B": [1.0, 2.0, np.nan, 4.0], "C": [5.0], "D": [np.nan]})
    assert res.test_name == "Descriptive statistics" and res.p_value is None
    assert [r["group"] for r in res.descriptives] == ["A", "B", "C", "D"]
    assert res.n == {"A": 11, "B": 3, "C": 1, "D": 0}
    assert res.descriptives[0]["mean"] == pytest.approx(X.mean())
    assert res.descriptives[3]["mean"] is None and res.descriptives[3]["n"] == 0
    assert any("D" in w and "empty" in w for w in res.warnings)
    assert any("C" in w and "Only one value" in w for w in res.warnings)
    assert any("B: 1 missing" in n for n in res.data_notes)
    assert res.assumptions and res.interpretation and res.why_chosen and res.methods_text
    # A is strongly right-skewed (one high value) -> flagged
    assert res.assumptions[0].status == "warn" and "A" in res.assumptions[0].detail
    assert "no P value" in res.interpretation
    json.loads(res.finalize().model_dump_json())

    single = ds.descriptive_statistics({"Only": [1.0, 2.0, 3.0, 4.0]}, ci_level=0.9)
    assert single.assumptions[0].status == "info"
    assert "Only: n = 4" in single.interpretation and "90% CI" in single.interpretation
    assert single.extra["ci_level"] == 0.9
    many = ds.descriptive_statistics({f"g{i}": [float(i)] for i in range(12)})
    assert "and 4 more" in many.warnings[0]

    with pytest.raises(StatsError):
        ds.descriptive_statistics({})
    with pytest.raises(StatsError):
        ds.descriptive_statistics({"A": [np.nan], "B": []})
    with pytest.raises(StatsError):
        ds.descriptive_statistics({"A": [1, 2]}, ci_level=2)


# ----------------------------------------------------------------------- registry


def _check_registry_result(res):
    assert res.analysis == "descriptives" and res.interpretation and res.why_chosen
    assert len(res.assumptions) >= 1
    json.loads(res.model_dump_json())


def test_registry_column_table():
    table = ColumnTable.from_dict("t", {"Control": [1.0, 2.0, 3.0, 4.0], "Drug": [2.0, 4.0, 6.0]})
    res = run_analysis(table, "descriptives")
    _check_registry_result(res)
    assert res.table == "t" and res.n == {"Control": 4, "Drug": 3} and res.n_total == 7
    assert res.descriptives[1]["mean"] == pytest.approx(4.0)
    assert res.params["ci_level"] == 0.95
    table.exclude([(3, "Control")])
    res = run_analysis(table, "descriptives")
    assert res.n["Control"] == 3 and any("excluded" in n for n in res.data_notes)


def test_registry_grouped_table():
    table = GroupedTable.from_cells(
        "g", {"Day 1": {"Ctrl": [1.0, 2.0, 3.0], "Drug": [2.0, 3.0, 4.0]},
              "Day 2": {"Ctrl": [1.0, 2.0, 6.0], "Drug": [5.0, 7.0]}},
        row_label="Day", column_label="Treatment")
    res = run_analysis(table, "descriptives")
    _check_registry_result(res)
    assert [r["group"] for r in res.descriptives] == [
        "Day 1 | Ctrl", "Day 1 | Drug", "Day 2 | Ctrl", "Day 2 | Drug"]
    last = res.descriptives[-1]
    assert (last["row"], last["column"], last["n"], last["mean"]) == ("Day 2", "Drug", 2, 6.0)


def test_registry_xy_table():
    table = XYTable.from_arrays(
        "xy", [0.0, 1.0, 1.0, np.nan],
        {"A": [[1.0, 3.0], [2.0, 4.0], [5.0, np.nan], [9.0, 9.0]], "B": [1.0, np.nan, 3.0, 4.0]},
        x_name="Dose")
    res = run_analysis(table, "descriptives")
    _check_registry_result(res)
    rows = {r["group"]: r for r in res.descriptives}
    assert rows["A | Dose = 0"]["mean"] == 2.0 and rows["A | Dose = 0"]["x"] == 0.0
    assert rows["A | Dose = 0"]["series"] == "A" and rows["A | Dose = 0"]["n"] == 2
    # the X value 1 occurs in two rows: kept apart, nothing pooled silently
    assert rows["A | Dose = 1"]["n"] == 2 and rows["A | Dose = 1 (row 3)"]["n"] == 1
    assert rows["B | Dose = 1"]["mean"] == 3.0 and "B | Dose = 1 (row 3)" not in rows  # B is empty in row 2
    assert all(r["x"] is not None for r in res.descriptives)  # the row with a missing X is skipped
    empty = XYTable.from_arrays("e", [1.0, 2.0], {"A": [np.nan, np.nan]})
    with pytest.raises(StatsError):
        run_analysis(empty, "descriptives")


def test_registry_multiple_variables_table():
    frame = pd.DataFrame({
        "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "height": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "treatment": ["a", "a", "b", "b", None, "b"],
        "dose": [1, 1, 2, 2, 2, 2],
    })
    table = MultipleVariablesTable("m", frame)
    res = run_analysis(table, "descriptives")
    _check_registry_result(res)
    assert [r["group"] for r in res.descriptives] == ["weight", "height", "dose"]
    assert res.descriptives[0]["variable"] == "weight"

    res = run_analysis(table, "descriptives", by="treatment", columns=["weight"])
    assert [(r["group"], r["n"], r["by"]) for r in res.descriptives] == [
        ("weight | a", 2, "a"), ("weight | b", 3, "b")]
    assert any("no value in 'treatment'" in n for n in res.data_notes)
    res = run_analysis(table, "descriptives", by="dose", columns="height")  # numeric grouping column
    assert [r["group"] for r in res.descriptives] == ["height | 1.0", "height | 2.0"]
    res = run_analysis(table, "descriptives", by="dose")  # default columns leave the grouping column out
    assert all(not r["group"].startswith("dose") for r in res.descriptives)

    for bad in ({"columns": ["nope"]}, {"columns": ["treatment"]}, {"by": "nope"}):
        with pytest.raises(StatsError):
            run_analysis(table, "descriptives", **bad)
    text_only = MultipleVariablesTable("t", pd.DataFrame({"a": ["x", "y"]}))
    with pytest.raises(StatsError):
        run_analysis(text_only, "descriptives")
    no_levels = MultipleVariablesTable("n", pd.DataFrame({"y": [1.0, 2.0], "g": ["", ""]}))
    with pytest.raises(StatsError):
        run_analysis(no_levels, "descriptives", by="g")


# ------------------------------------------------------------------ property tests

_values = st.lists(
    st.one_of(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), st.just(float("nan"))),
    min_size=0, max_size=40,
)


@settings(max_examples=60, deadline=None)
@given(_values)
def test_describe_properties(values):
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        with pytest.raises(StatsError):
            ds.describe(values)
        return
    d = ds.describe(values)
    assert d["n"] == len(finite) and d["n_missing"] == len(values) - len(finite)
    tol = 1e-9 * max(1.0, max(abs(v) for v in finite))
    assert d["min"] <= d["q1"] + tol and d["q1"] <= d["median"] + tol
    assert d["median"] <= d["q3"] + tol and d["q3"] <= d["max"] + tol
    assert d["min"] - tol <= d["mean"] <= d["max"] + tol
    assert d["iqr"] >= -tol
    if d["n"] >= 2:
        assert d["sd"] >= 0 and d["sem"] <= d["sd"] + tol
        assert d["ci_low"] <= d["mean"] <= d["ci_high"]
    else:
        assert d["sd"] is None and d["ci_low"] is None
    for key in ("skewness", "kurtosis", "geometric_mean", "cv_percent"):
        assert d[key] is None or math.isfinite(d[key])
    if d["geometric_mean"] is not None:
        assert d["geometric_mean"] <= d["mean"] * (1 + 1e-9) + tol  # AM-GM inequality


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=4, max_size=30),
    st.floats(min_value=-100, max_value=100),
    st.floats(min_value=0.1, max_value=50),
)
def test_describe_location_scale_behaviour(values, shift, scale):
    x = np.array(values, dtype=float)
    a, b = ds.describe(x), ds.describe(x * scale + shift)
    assert b["mean"] == pytest.approx(a["mean"] * scale + shift, rel=1e-9, abs=1e-7)
    assert b["median"] == pytest.approx(a["median"] * scale + shift, rel=1e-9, abs=1e-7)
    assert b["sd"] == pytest.approx(a["sd"] * scale, rel=1e-7, abs=1e-9)
    assert b["iqr"] == pytest.approx(a["iqr"] * scale, rel=1e-7, abs=1e-7)
    if a["skewness"] is not None and b["skewness"] is not None and a["sd"] > 1e-6:
        assert b["skewness"] == pytest.approx(a["skewness"], rel=1e-5, abs=1e-6)
        assert b["kurtosis"] == pytest.approx(a["kurtosis"], rel=1e-5, abs=1e-6)


@settings(max_examples=40, deadline=None)
@given(st.dictionaries(st.text(min_size=1, max_size=5), _values, min_size=0, max_size=4))
def test_descriptive_statistics_never_crashes(groups):
    total = sum(1 for v in groups.values() for x in v if not math.isnan(x))
    if total == 0:
        with pytest.raises(StatsError):
            ds.descriptive_statistics(groups)
        return
    res = ds.descriptive_statistics(groups).finalize()
    assert len(res.descriptives) == len(groups) and res.p_value is None
    assert sum(res.n.values()) == total
    json.loads(res.model_dump_json())
