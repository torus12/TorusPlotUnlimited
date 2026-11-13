import numpy as np
import pandas as pd
import pytest

from torusplot.data import (
    ColumnTable,
    ContingencyTable,
    GroupedTable,
    MultipleVariablesTable,
    TableError,
    XYTable,
    long_to_column,
    long_to_contingency,
    long_to_grouped,
    long_to_xy,
    table_from_meta,
    to_multiple_variables,
)


def test_column_table_groups_and_long():
    t = ColumnTable.from_dict("sizes", {"A": [1, 2, 3], "B": [4, 5], "C": [6, 7, 8, 9]})
    g = t.groups()
    assert list(g) == ["A", "B", "C"]
    assert g["B"].tolist() == [4.0, 5.0]
    long = t.to_long()
    assert list(long.columns) == ["group", "subject", "value"]
    assert len(long) == 9
    assert t.paired_frame().shape == (2, 3)
    assert t.paired_frame(["A", "C"]).shape == (3, 2)
    with pytest.raises(TableError):
        t.paired_frame(["nope"])


def test_column_table_coerces_text_and_dedupes_names():
    df = pd.DataFrame([[1, "x"], [2, 3]], columns=["A", "A"])
    t = ColumnTable("t", df)
    assert t.group_names == ["A", "A (2)"]
    assert np.isnan(t.data["A (2)"][0])


def test_exclusions_are_explicit_and_reported():
    t = ColumnTable.from_dict("t", {"A": [1, 2, 300], "B": [4, 5, 6]})
    assert t.exclusion_notes() == []
    t.exclude([(2, "A")])
    assert t.groups()["A"].tolist() == [1.0, 2.0]
    assert t.data["A"].tolist() == [1.0, 2.0, 300.0]  # raw data untouched
    assert "explicitly excluded" in t.exclusion_notes()[0]
    t2 = table_from_meta(t.data, t.meta())
    assert t2.excluded == [(2, "A")]
    t.clear_exclusions()
    assert len(t.groups()["A"]) == 3


def test_xy_table():
    t = XYTable.from_arrays(
        "dose", [1, 2, 3], {"Drug": [[10, 12], [20, 22], [30, np.nan]], "Vehicle": [1, 2, 3]}
    )
    assert t.series_names == ["Drug", "Vehicle"]
    x, y = t.xy_points("Drug")
    assert len(x) == 5 and x.tolist() == [1, 1, 2, 2, 3]
    means = t.xy_means("Drug")
    assert means["mean"].tolist() == [11.0, 21.0, 30.0]
    assert means["n"].tolist() == [2, 2, 1]
    assert np.isnan(means["sd"].iloc[2])
    long = t.to_long()
    assert set(long.columns) == {"x", "series", "replicate", "y"}
    assert len(long) == 8
    with pytest.raises(TableError):
        t.series("nope")
    with pytest.raises(TableError):
        XYTable("bad", pd.DataFrame({"x": [1, 2]}))


def test_grouped_table():
    t = GroupedTable.from_cells(
        "uptake",
        {"1 h": {"Ctrl": [1, 2, 3], "NP": [4, 5, 6]}, "4 h": {"Ctrl": [2, 3, 4], "NP": [8, 9]}},
        row_label="Time",
        column_label="Treatment",
    )
    assert t.row_levels == ["1 h", "4 h"]
    assert t.column_levels == ["Ctrl", "NP"]
    assert t.cell("4 h", "NP").tolist() == [8.0, 9.0]
    long = t.to_long()
    assert list(long.columns) == ["row", "column", "replicate", "value"]
    assert len(long) == 11
    rebuilt = table_from_meta(t.data, t.meta())
    assert rebuilt.equals(t)
    with pytest.raises(TableError):
        t.cell("nope", "NP")


def test_multiple_variables_table():
    df = pd.DataFrame({"dose": ["1", "2", "3"], "sex": ["F", "M", ""], "y": [1.5, 2.5, 3.5]})
    t = MultipleVariablesTable("mv", df)
    assert t.numeric_columns == ["dose", "y"]
    assert t.categorical_columns == ["sex"]
    assert len(t.frame()) == 2
    assert len(t.frame(["dose", "y"])) == 3
    t.exclude([(0, "*")])
    assert len(t.frame(["dose", "y"])) == 2
    assert t.exclusion_notes()


def test_contingency_table():
    t = ContingencyTable.from_counts("c", [[10, 20], [30, 5]], ["Drug", "Placebo"], ["Yes", "No"])
    assert t.counts.loc["Placebo", "Yes"] == 30
    assert t.row_levels == ["Drug", "Placebo"] and t.column_levels == ["Yes", "No"]
    assert not t.has_non_integer_counts()
    assert len(t.to_long()) == 4
    with pytest.raises(TableError):
        ContingencyTable.from_counts("c", [[-1, 2], [3, 4]], ["a", "b"], ["c", "d"])


def test_reshape_round_trips():
    long = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b"],
            "t": ["x", "y", "x", "y", "y"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
            "d": [1, 2, 1, 2, 2],
        }
    )
    col = long_to_column(long, "g", "v")
    assert col.groups()["b"].tolist() == [3.0, 4.0, 5.0]
    grp = long_to_grouped(long, "t", "g", "v")
    assert grp.cell("y", "b").tolist() == [4.0, 5.0]
    xy = long_to_xy(long, "d", "v", series="g")
    assert xy.series_names == ["a", "b"]
    assert xy.xy_means("b")["mean"].tolist() == [3.0, 4.5]
    cont = long_to_contingency(long, "g", "t")
    assert cont.counts.loc["b", "y"] == 2
    mv = to_multiple_variables(grp)
    assert set(mv.data.columns) == {"row", "column", "replicate", "value"}
    with pytest.raises(TableError):
        long_to_column(long, "nope", "v")


def test_paired_long_to_column():
    long = pd.DataFrame({"s": [1, 2, 3, 1, 2, 3], "g": ["pre"] * 3 + ["post"] * 3, "v": [1, 2, 3, 2, 4, 6]})
    t = long_to_column(long, "g", "v", subject="s")
    assert t.paired_frame().shape == (3, 2)
    assert t.group_names == ["pre", "post"]


def test_approximate_flag_survives():
    t = ColumnTable.from_dict("dig", {"A": [1, 2]}, approximate=True)
    assert "APPROXIMATE" in t.summary()
    assert table_from_meta(t.data, t.meta()).approximate is True
    assert t.copy().approximate is True
