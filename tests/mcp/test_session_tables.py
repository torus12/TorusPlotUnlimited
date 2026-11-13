"""TorusSession: loading tables, state listing, file-system safety for inputs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.mcp import strict_json
from torusplot.data.samples import SAMPLES
from torusplot.mcp_server.session import (
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIR_ENV,
    SessionError,
    TorusSession,
    jsonable,
    looks_like_statistic,
    safe_stem,
)
from torusplot.spec import plot_types_for

COLUMN_TEXT = "Control\tDrug A\tDrug B\n1.1\t2.2\t3.1\n1.3\t2.9\t3.6\n0.9\t2.4\t\n1.2\t2.6\t3.3\n"


@pytest.fixture
def session(tmp_path: Path) -> TorusSession:
    return TorusSession(tmp_path / "out")


# ------------------------------------------------------------------ samples
@pytest.mark.parametrize("sample", list(SAMPLES))
def test_load_every_bundled_sample(session: TorusSession, sample: str):
    out = session.load_table(name=f"t_{sample}", sample=sample)
    info = SAMPLES[sample]
    assert out["name"] == f"t_{sample}" and out["kind"] == info["kind"]
    assert out["shape"]["rows"] > 0 and out["shape"]["columns"] > 0
    assert out["plot_types"] == plot_types_for(info["kind"])
    assert isinstance(out["analyses"], list)
    for entry in out["analyses"]:
        assert set(entry) == {"key", "label", "category"}
    assert "SYNTHETIC" in out["notes"]
    assert out["replaced"] is False and "recommend_test" in out["next"]
    assert len(out["preview"]) <= 5
    strict_json(out)
    assert [t["name"] for t in session.list_state()["tables"]] == [f"t_{sample}"]


def test_sample_summaries_describe_the_design(session: TorusSession):
    col = session.load_table("np", "column", sample="nanoparticle_size")
    assert col["groups"] == ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]
    assert col["n_per_group"] == dict.fromkeys(col["groups"], 8)

    xy = session.load_table("dr", sample="dose_response")
    assert xy["series"] == ["Free drug", "Nanoformulation"] and xy["x"] == "Dose (uM)"
    assert xy["replicates_per_series"] == {"Free drug": 3, "Nanoformulation": 3}
    assert xy["n_per_group"]["Free drug"] == 27 and xy["x_values"] == 9

    grouped = session.load_table("up", sample="cell_uptake")
    assert grouped["row_levels"] == ["1 h", "4 h", "24 h"] and grouped["row_label"] == "Time"
    assert grouped["column_label"] == "Formulation"
    assert grouped["n_per_cell"]["4 h"]["Nanoparticle"] == 4

    mv = session.load_table("cells", sample="cell_uptake_cells")
    assert mv["numeric_columns"] == ["Uptake"]
    assert mv["levels"]["Treatment"] == {"Control": 60, "Targeted NP": 60}

    ct = session.load_table("resp", sample="contingency")
    assert ct["row_totals"] == {"Treated": 24.0, "Control": 24.0} and ct["n_total"] == 48.0


def test_sample_errors(session: TorusSession):
    with pytest.raises(SessionError, match="nanoparticle_size"):
        session.load_table("x", sample="no_such_sample")
    with pytest.raises(SessionError, match="is a column table"):
        session.load_table("x", "xy", sample="nanoparticle_size")
    assert session.list_state()["tables"] == []


# ------------------------------------------------------------------ pasted text
def test_load_column_table_from_text(session: TorusSession):
    out = session.load_table("assay", "column", text=COLUMN_TEXT)
    assert out["groups"] == ["Control", "Drug A", "Drug B"]
    assert out["n_per_group"] == {"Control": 4, "Drug A": 4, "Drug B": 3}
    assert out["preview"][2]["Drug B"] is None  # blank cell -> null, never NaN
    assert out["source"] == "pasted text"
    strict_json(out)


def test_load_xy_grouped_contingency_and_tidy_text(session: TorusSession):
    xy = session.load_table("xy", "xy", text="x,a1,a2,b1,b2\n1,2,3,4,5\n2,3,4,5,6\n3,4,5,6,7", replicates=2)
    assert xy["series"] == ["a1", "b1"] and xy["replicates_per_series"] == {"a1": 2, "b1": 2}

    grouped = session.load_table(
        "g", "grouped", text="Time;A:1;A:2;B:1;B:2\n1h;1;2;3;4\n2h;2;3;4;5", row_label="Time", column_label="Drug"
    )
    assert grouped["groups"] == ["A", "B"] and grouped["column_label"] == "Drug"
    assert grouped["replicates_per_group"] == {"A": 2, "B": 2}

    ct = session.load_table("c", "contingency", text="Group,Yes,No\nT,10,5\nC,3,12")
    assert ct["column_levels"] == ["Yes", "No"] and ct["n_total"] == 30.0

    mv = session.load_table("m", "multiple_variables", text="grp,val\na,1\na,2\nb,3\nb,4")
    assert mv["categorical_columns"] == ["grp"] and mv["n_per_column"] == {"grp": 4, "val": 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "column"}, "exactly one of text, path or sample"),
        ({"kind": "column", "text": COLUMN_TEXT, "path": "data.csv"}, "exactly one of text, path or sample"),
        ({"kind": "column", "text": COLUMN_TEXT, "sample": "paired"}, "exactly one of text, path or sample"),
        ({"text": COLUMN_TEXT}, "kind must be one of"),
        ({"kind": "pie", "text": COLUMN_TEXT}, "kind must be one of"),
        ({"kind": "column", "text": "   \n  "}, "empty"),
        ({"kind": "column", "text": "hello world"}, "No numbers were found"),
        ({"kind": "xy", "text": "x,a,b,c\n1,2,3,4\n2,3,4,5", "replicates": 2}, "cannot be split"),
        ({"kind": "xy", "text": "x,a\n1,2\n2,3", "replicates": 0}, "replicates must be"),
        ({"kind": "xy", "text": "x,a\n1,2\n2,3", "replicates": True}, "replicates must be"),
        ({"kind": "contingency", "text": "Group,Yes,No\nT,-10,5\nC,3,12"}, "negative"),
        ({"kind": "grouped", "text": "Time;A;B\n1h;1;2", "row_label": "p < 0.05"}, "statistical result"),
    ],
)
def test_load_table_rejects_bad_input(session: TorusSession, kwargs: dict, message: str):
    with pytest.raises(SessionError, match=message):
        session.load_table("t", **kwargs)
    assert session.project.tables == {}


@pytest.mark.parametrize("name", ["", "   ", None, 5, "x" * 200, "bad\x07name", "p < 0.05", "Size ***"])
def test_table_name_must_be_sensible(session: TorusSession, name):
    with pytest.raises(SessionError, match="Table name"):  # it would become the default axis title
        session.load_table(name, "column", text=COLUMN_TEXT)
    assert session.project.tables == {}


@pytest.mark.parametrize(
    "text",
    [
        "Control\tDrug **\n1\t2\n2\t3",
        "Control\tp < 0.05\n1\t2\n2\t3",
        "Control\t***\n1\t2\n2\t3",
        "Control\tP = 0.003\n1\t2\n2\t3",
    ],
)
def test_statistics_cannot_be_smuggled_through_column_names(session: TorusSession, text: str):
    with pytest.raises(SessionError, match="look like statistical results"):
        session.load_table("t", "column", text=text)
    assert session.project.tables == {}


def test_statistics_cannot_be_smuggled_through_category_levels(session: TorusSession):
    with pytest.raises(SessionError, match="look like statistical results"):
        session.load_table("t", "multiple_variables", text="grp,val\na,1\na,2\n***,3\n***,4")
    with pytest.raises(SessionError, match="look like statistical results"):
        session.load_table("t", "grouped", text="Time;A;B\n1h;1;2\np=0.01;2;3")


def test_ordinary_names_are_not_mistaken_for_statistics(session: TorusSession):
    out = session.load_table("t", "column", text="pH = 7.4\tp53 level\tns\n1\t2\t3\n2\t3\t4")
    assert out["groups"] == ["pH = 7.4", "p53 level", "ns"]
    for text in ["Response (%)", "Temp > 0.5", "EC50 = 0.3 uM", "n = 3", "Top 5", "P = 1 bar", None, "", 7]:
        assert not looks_like_statistic(text)
    for text in ["p < 0.05", "P=0.003", "p-value = 0.01", "p adj < 0.05", "***", "ns", "n.s.", "Drug **",
                 "Dose *", "q < 0.05", "P<.001", "p = 1e-5", "P = 2.3 x 10^-4", "Padj", "FDR < 5%"]:
        assert looks_like_statistic(text), text


def test_replacing_a_table_is_reported(session: TorusSession):
    session.load_table("t", "column", text=COLUMN_TEXT)
    session.make_figure("f", "t", "bar")
    out = session.load_table("t", "column", text="A\tB\n1\t2\n3\t4\n5\t7")
    assert out["replaced"] is True and out["groups"] == ["A", "B"]
    assert "replaced" in out["warnings"][0] and "1 figure(s)" in out["warnings"][0]
    assert len(session.list_state()["tables"]) == 1


# ------------------------------------------------------------------ files
def test_load_table_from_csv_and_tsv(session: TorusSession, tmp_path: Path):
    csv = tmp_path / "my data.csv"
    csv.write_text("Control,Drug\n1.5,2.5\n1.7,2.9\n1.2,3.1\n", encoding="utf-8")
    out = session.load_table("from_csv", "column", path=str(csv))
    assert out["groups"] == ["Control", "Drug"] and out["source"] == "file 'my data.csv'"
    tsv = tmp_path / "data.TSV"
    tsv.write_text("x\ty\n1\t2\n2\t4\n3\t6\n", encoding="utf-8")
    assert session.load_table("from_tsv", "xy", path=str(tsv))["series"] == ["y"]


def test_load_table_from_xlsx(session: TorusSession, tmp_path: Path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "First"
    for row in (["Control", "Drug"], [1.0, 2.0], [1.5, 2.5], [0.9, 2.2]):
        ws.append(row)
    other = wb.create_sheet("Second")
    for row in (["A", "B", "C"], [1, 2, 3], [4, 5, 6]):
        other.append(row)
    path = tmp_path / "book.xlsx"
    wb.save(path)
    assert session.load_table("first", "column", path=str(path))["groups"] == ["Control", "Drug"]
    assert session.load_table("second", "column", path=str(path), sheet="Second")["groups"] == ["A", "B", "C"]
    with pytest.raises(SessionError, match="Sheet 'Nope' not found"):
        session.load_table("third", "column", path=str(path), sheet="Nope")


def test_load_table_only_reads_whitelisted_existing_files(session: TorusSession, tmp_path: Path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(SessionError, match=r"Only \.csv, \.tsv, \.txt, \.xlsx files"):
        session.load_table("t", "column", path=str(secret))
    with pytest.raises(SessionError, match="no extension"):
        session.load_table("t", "column", path=str(tmp_path / "passwd"))
    with pytest.raises(SessionError, match="File not found"):
        session.load_table("t", "column", path=str(tmp_path / "missing.csv"))
    folder = tmp_path / "folder.csv"
    folder.mkdir()
    with pytest.raises(SessionError, match="Not a file"):
        session.load_table("t", "column", path=str(folder))
    with pytest.raises(SessionError, match="path must be"):
        session.load_table("t", "column", path="   ")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"this is not a workbook")
    with pytest.raises(SessionError, match="could not be read as a table"):
        session.load_table("t", "column", path=str(broken))
    assert session.project.tables == {}


def test_load_table_does_not_follow_links_to_other_file_types(session: TorusSession, tmp_path: Path):
    target = tmp_path / "secret.conf"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    link = tmp_path / "innocent.csv"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")
    with pytest.raises(SessionError, match="Only .csv"):
        session.load_table("t", "column", path=str(link))


def test_oversized_files_are_refused(session: TorusSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from torusplot.mcp_server import session as session_module

    csv = tmp_path / "big.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    monkeypatch.setattr(session_module, "MAX_TABLE_FILE_BYTES", 4)
    with pytest.raises(SessionError, match="limit for table files"):
        session.load_table("t", "column", path=str(csv))


# ------------------------------------------------------------------ state, output dir, helpers
def test_list_state_starts_empty_and_lists_samples(session: TorusSession):
    state = session.list_state()
    assert state["tables"] == state["results"] == state["figures"] == state["compositions"] == []
    assert set(state["samples"]) == set(SAMPLES)
    assert state["output_dir"] == str(session.output_dir)
    strict_json(state)


def test_unknown_table_error_lists_what_exists(session: TorusSession):
    with pytest.raises(SessionError, match="no tables are loaded yet"):
        session.make_figure("f", "ghost", "bar")
    session.load_table("real", "column", text=COLUMN_TEXT)
    with pytest.raises(SessionError, match=r"Loaded tables: \['real'\]"):
        session.make_figure("f", "ghost", "bar")


def test_output_directory_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)
    assert TorusSession().output_dir == (tmp_path / DEFAULT_OUTPUT_DIR).resolve()
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path / "from_env"))
    assert TorusSession().output_dir == (tmp_path / "from_env").resolve()
    assert TorusSession(tmp_path / "explicit").output_dir == (tmp_path / "explicit").resolve()
    assert not (tmp_path / "explicit").exists()  # created lazily, on the first write


def test_jsonable_is_strict_json_safe():
    import numpy as np
    import pandas as pd

    from torusplot.spec import LegendSpec

    raw = {
        1: np.float64("nan"), "inf": float("inf"), "ninf": -np.inf, "i": np.int64(3), "b": np.bool_(True),
        "arr": np.array([1.0, np.nan]), "tuple": (1, 2), "set": {5}, "path": Path("a") / "b", "na": pd.NA,
        "model": LegendSpec(), "obj": object, "nat": pd.NaT,
    }
    out = jsonable(raw)
    assert out["1"] is None and out["inf"] == "Infinity" and out["ninf"] == "-Infinity"
    assert out["i"] == 3 and out["b"] is True and out["arr"] == [1.0, None]
    assert out["tuple"] == [1, 2] and out["set"] == [5] and out["na"] is None and out["nat"] is None
    assert out["path"] == str(Path("a") / "b") and out["model"]["position"] == "best"
    assert isinstance(out["obj"], str)
    strict_json(out)


@pytest.mark.parametrize(
    ("name", "stem"),
    [("Figure 1", "Figure_1"), ("../../etc/passwd", "etc_passwd"), ("a/b\\c:d*e?", "a_b_c_d_e"), ("...", "figure"),
     ("CON", "_CON"), ("fig.png", "fig_png"), ("Größe (nm)", "Größe_nm"), ("x" * 300, "x" * 100)],
)
def test_safe_stem(name: str, stem: str):
    assert safe_stem(name) == stem
