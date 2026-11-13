import numpy as np
import pandas as pd
import pytest

from torusplot.data import (
    TableError,
    build_table,
    export_table_csv,
    guess_table_kind,
    import_table,
    list_sheets,
    parse_number,
    parse_pasted_text,
    table_from_text,
)
from torusplot.data.parse import detect_delimiter

EXCEL_PASTE = "Control\tDrug A\tDrug B\n1.2\t2.4\t3.1\n1.4\t2.2\t\n1.1\t\t3.3\n"


def test_excel_tab_paste():
    df = parse_pasted_text(EXCEL_PASTE)
    assert list(df.columns) == ["Control", "Drug A", "Drug B"]
    assert df.attrs["header_detected"] is True
    t = build_table("column", df, "t")
    assert t.groups()["Drug A"].tolist() == [2.4, 2.2]


def test_no_header_detected():
    df = parse_pasted_text("1\t2\n3\t4\n")
    assert list(df.columns) == ["Column 1", "Column 2"]
    assert df.shape == (2, 2)


def test_decimal_comma_semicolon():
    df = parse_pasted_text("a;b\n1,5;2,25\n3,0;4\n")
    assert df["a"].tolist() == [1.5, 3.0]
    assert df["b"].tolist() == [2.25, 4.0]
    assert df.attrs["decimal_comma"] is True


def test_csv_with_thousands_and_missing_markers():
    df = parse_pasted_text('x,y\n"1,200",NA\n3.5e2,-\n7,#DIV/0!\n')
    assert df["x"].tolist() == [1200.0, 350.0, 7.0]
    assert df["y"].isna().all()


@pytest.mark.parametrize(
    "cell,expected",
    [("1.5", 1.5), (" 2 ", 2.0), ("−3", -3.0), ("50%", 50.0), ("1e-3", 0.001), ("1,000.5", 1000.5)],
)
def test_parse_number(cell, expected):
    assert parse_number(cell) == pytest.approx(expected)


@pytest.mark.parametrize("cell", ["", "abc", "N/A", "-", "1.2.3", None, "+"])
def test_parse_number_nan(cell):
    assert np.isnan(parse_number(cell))


def test_delimiter_detection():
    assert detect_delimiter("a\tb\n1\t2") == "\t"
    assert detect_delimiter("a;b\n1;2") == ";"
    assert detect_delimiter("a,b\n1,2") == ","
    assert detect_delimiter("1 2\n3 4") == " "


def test_empty_paste_raises():
    with pytest.raises(TableError):
        parse_pasted_text("   \n ")


def test_xy_replicates_by_name_and_count():
    text = "Dose\tDrug\tDrug\tDrug\tVeh\tVeh\tVeh\n1\t1\t2\t3\t4\t5\t6\n2\t2\t3\t4\t5\t6\t7\n"
    t = table_from_text("xy", text, "dr")
    assert t.series_names == ["Drug", "Veh"]
    assert len(t.series_map["Drug"]) == 3
    t2 = table_from_text("xy", "x\ta1\ta2\tb1\tb2\n1\t1\t2\t3\t4\n", "t2", replicates=2)
    assert [len(v) for v in t2.series_map.values()] == [2, 2]
    with pytest.raises(TableError):
        table_from_text("xy", "x\ta\tb\tc\n1\t1\t2\t3\n", "t3", replicates=2)


def test_grouped_and_contingency_from_text():
    g = table_from_text("grouped", "Time\tCtrl:1\tCtrl:2\tNP:1\tNP:2\n1h\t1\t2\t3\t4\n4h\t2\t3\t6\t7\n", "g")
    assert g.column_levels == ["Ctrl", "NP"]
    assert g.cell("4h", "NP").tolist() == [6.0, 7.0]
    c = table_from_text("contingency", "\tYes\tNo\nDrug\t12\t30\nPlacebo\t25\t18\n", "c")
    assert c.counts.to_numpy().tolist() == [[12, 30], [25, 18]]


def test_guess_table_kind():
    assert guess_table_kind(parse_pasted_text(EXCEL_PASTE)) == "column"
    assert guess_table_kind(parse_pasted_text("x\ty\n1\t5\n2\t3\n3\t9\n")) == "xy"
    assert guess_table_kind(parse_pasted_text("\tYes\tNo\nA\t1\t2\nB\t3\t4\n")) == "contingency"
    assert guess_table_kind(parse_pasted_text("t\ta\tb\n1h\t1.5\t2.5\n2h\t3.5\t4.5\n")) == "grouped"
    assert guess_table_kind(parse_pasted_text("g\ts\ty\na\tm\t1\na\tf\t2\n")) == "multiple_variables"


def test_csv_and_xlsx_import(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Control,Treated\n1,2\n3,4\n", encoding="utf-8")
    t = import_table(csv_path, "column")
    assert t.name == "data" and t.groups()["Treated"].tolist() == [2.0, 4.0]

    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame({"Dose": [1, 10, 100], "Resp": [99.5, 50.25, 2.0]}).to_excel(
            writer, sheet_name="dr", index=False
        )
        pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}).to_excel(writer, sheet_name="mv", index=False)
    assert list_sheets(xlsx) == ["dr", "mv"]
    xy = import_table(xlsx, "xy", sheet="dr")
    assert xy.x.tolist() == [1.0, 10.0, 100.0]
    assert xy.series("Resp").iloc[:, 0].tolist() == [99.5, 50.25, 2.0]
    mv = import_table(xlsx, "multiple_variables", sheet=1)
    assert mv.categorical_columns == ["g"]
    with pytest.raises(TableError):
        import_table(xlsx, "xy", sheet="missing")
    with pytest.raises(TableError):
        import_table(tmp_path / "x.json", "column")
    out = export_table_csv(t, tmp_path / "out.csv")
    assert out.read_text(encoding="utf-8").startswith("Control,Treated")


def test_utf16_file(tmp_path):
    p = tmp_path / "u16.txt"
    p.write_bytes("A\tB\n1\t2\n".encode("utf-16"))
    assert import_table(p, "column").group_names == ["A", "B"]
