import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from torusplot.data import (
    ColumnTable,
    ContingencyTable,
    GroupedTable,
    MultipleVariablesTable,
    Project,
    ProjectError,
    XYTable,
)
from torusplot.stats.result import StatResult


def make_project() -> Project:
    p = Project("Demo")
    p.add_table(ColumnTable.from_dict("Sizes (nm)", {"A": [100.123456789012, 2, np.nan], "B": [3, 4, 5]}))
    p.add_table(XYTable.from_arrays("Dose/Response", [0.1, 1, 10], {"Drug": [[1, 2], [3, 4], [5, 6]]}))
    p.add_table(
        GroupedTable.from_cells(
            "Uptake",
            {"1": {"Ctrl": [1, 2], "NP": [3, 4]}, "4": {"Ctrl": [5, 6], "NP": [7, 8]}},
            row_label="Time (h)",
            column_label="Treatment",
        )
    )
    p.add_table(
        MultipleVariablesTable(
            "MV", pd.DataFrame({"id": ["s01", "s02", "s03"], "sex": ["F", None, "M"], "y": [1.0, 2.0, 3.5]})
        )
    )
    p.add_table(ContingencyTable.from_counts("Cont", [[1, 2], [3, 4]], ["1", "2"], ["Yes", "No"]))
    p.tables["Sizes (nm)"].exclude([(1, "A")])
    p.figures["fig1"] = {"plot_type": "bar", "data": {"table": "Sizes (nm)"}}
    p.add_result(StatResult(id="sr_1", analysis="demo", test_name="Demo test", p_value=0.04))
    p.log.append("ran demo")
    return p


def test_save_load_round_trip(tmp_path):
    p = make_project()
    path = p.save(tmp_path / "demo")
    assert path.suffix == ".tplot"
    q = Project.load(path)
    assert q.name == "Demo"
    assert list(q.tables) == list(p.tables)
    for name in p.tables:
        assert q.tables[name].equals(p.tables[name]), name
    assert q.tables["Sizes (nm)"].data["A"][0] == 100.123456789012  # exact float round trip
    assert q.tables["MV"].data["id"].tolist() == ["s01", "s02", "s03"]
    assert q.tables["Uptake"].row_levels == ["1", "4"]
    assert q.tables["Cont"].row_levels == ["1", "2"]
    assert q.log == ["ran demo"]
    fig = q.figures["fig1"]
    fig = fig if isinstance(fig, dict) else fig.model_dump()
    assert fig["plot_type"] == "bar"
    assert q.results["sr_1"].p_value == 0.04
    assert q.results["sr_1"].test_name == "Demo test"


def test_save_is_deterministic(tmp_path):
    a = make_project().save(tmp_path / "a.tplot").read_bytes()
    b = make_project().save(tmp_path / "b.tplot").read_bytes()
    assert a == b


def test_archive_layout(tmp_path):
    path = make_project().save(tmp_path / "x.tplot")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert "project.yaml" in names and "results.json" in names
    assert sum(n.startswith("data/") and n.endswith(".csv") for n in names) == 5


def test_bad_files(tmp_path):
    bad = tmp_path / "bad.tplot"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ProjectError):
        Project.load(bad)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other.txt", "x")
    buf.seek(0)
    with pytest.raises(ProjectError):
        Project.load(buf)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("project.yaml", "format: torusplot-project\nformat_version: 999\n")
    buf.seek(0)
    with pytest.raises(ProjectError, match="newer"):
        Project.load(buf)


def test_unique_names_and_no_replace():
    p = Project()
    p.add_table(ColumnTable.from_dict("T", {"A": [1]}))
    assert p.unique_table_name("T") == "T 2"
    with pytest.raises(ProjectError):
        p.add_table(ColumnTable.from_dict("T", {"A": [1]}), replace=False)
