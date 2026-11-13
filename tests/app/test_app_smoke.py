"""Streamlit AppTest smoke tests: every page loads, and the main workflow works end to end."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
PAGES = ["1 Data", "2 Analyze", "3 Graph", "4 Style", "5 Compose", "6 Export", "AI Assistant", "Settings"]


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
    return at


def click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    assert not at.exception, (key, [e.value for e in at.exception])
    return at


def load_sample(at: AppTest, sample: str) -> AppTest:
    goto(at, "1 Data")
    at.selectbox(key="sample_choice").set_value(sample).run()
    return click(at, "load_sample")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_on_an_empty_project(page):
    at = fresh()
    goto(at, page)
    assert at.header[0].value


def test_sidebar_shows_workflow_in_order():
    at = fresh()
    assert list(at.sidebar.radio(key="page").options) == PAGES
    assert at.sidebar.radio(key="page").value == "1 Data"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_with_data(page):
    at = load_sample(fresh(), "nanoparticle_size")
    goto(at, page)
    assert at.header[0].value


def test_paste_from_excel_creates_a_table():
    at = fresh()
    at.text_area(key="paste_text").set_value("Control\tDrug A\tDrug B\n1.2\t2.4\t3.1\n1.4\t2.2\t2.9\n1.1\t2.6\t3.3\n").run()
    at.text_input(key="paste_name").set_value("Pasted").run()
    click(at, "paste_create")
    project = at.session_state["project"]
    assert "Pasted" in project.tables
    assert project.tables["Pasted"].group_names == ["Control", "Drug A", "Drug B"]
    assert project.tables["Pasted"].kind == "column"


def test_analysis_menu_only_lists_valid_analyses_for_the_table_type():
    at = load_sample(fresh(), "contingency")
    goto(at, "2 Analyze")
    categories = list(at.selectbox(key="an_category").options)
    assert categories == ["Contingency tables"]
    labels = list(at.selectbox(key="an_choice").options)
    assert any("Fisher" in label for label in labels)
    assert not any("ANOVA" in label or label.endswith(" t test") or "regression" in label.lower() for label in labels)


def test_full_workflow_recommend_run_graph_style_compose_export():
    at = load_sample(fresh(), "nanoparticle_size")
    goto(at, "2 Analyze")
    click(at, "recommend")
    rec = at.session_state["recommendation"]
    assert rec.analysis in {"anova_oneway", "welch_anova", "kruskal_wallis"}
    click(at, "run_recommended")
    project = at.session_state["project"]
    assert len(project.results) == 1
    result = next(iter(project.results.values()))
    assert result.comparisons and result.interpretation and result.assumptions
    assert any("Why this test?" in e.label for e in at.expander)

    goto(at, "3 Graph")
    click(at, "create_figure")
    project = at.session_state["project"]
    assert "Figure 1" in project.figures
    at.selectbox(key="ann_Figure 1").set_value(result.id).run()
    assert not at.exception, [e.value for e in at.exception]
    spec = at.session_state["project"].figures["Figure 1"]
    assert spec.annotations and spec.annotations[0].stat_result_id == result.id
    at.selectbox(key="err_Figure 1").set_value("SD").run()
    assert any(lay.type == "sd_errorbar" for lay in at.session_state["project"].figures["Figure 1"].layers)

    goto(at, "4 Style")
    at.selectbox(key="preset_Figure 1").set_value("Classic Lab").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["project"].figures["Figure 1"].style.preset == "Classic Lab"

    goto(at, "5 Compose")
    click(at, "comp_save")
    assert len(at.session_state["project"].compositions) == 1

    goto(at, "6 Export")
    assert not at.exception
    methods = at.text_area(key="methods_text").value
    assert "TorusPlot" in methods and "SD" in methods
    assert len(at.get("download_button")) >= 4


def test_grouped_and_xy_samples_can_be_graphed():
    at = fresh()
    for sample in ("cell_uptake", "dose_response", "cell_uptake_cells", "survival", "paired"):
        load_sample(at, sample)
        goto(at, "3 Graph")
        click(at, "create_figure")
    assert len(at.session_state["project"].figures) == 5
    for page in ("4 Style", "5 Compose", "6 Export"):
        goto(at, page)


def test_outliers_are_only_flagged_until_the_user_explicitly_excludes():
    at = fresh()
    at.text_area(key="paste_text").set_value("A\tB\n10\t11\n11\t12\n9\t10\n10\t11\n12\t13\n11\t10\n10\t12\n95\t11\n").run()
    at.text_input(key="paste_name").set_value("Out").run()
    click(at, "paste_create")
    goto(at, "2 Analyze")
    at.selectbox(key="an_category").set_value("Outliers").run()
    click(at, "run_manual")
    project = at.session_state["project"]
    table = project.tables["Out"]
    result = next(iter(project.results.values()))
    assert result.extra["flagged"], "the planted outlier should be flagged"
    assert table.excluded == [] and len(table.groups()["A"]) == 8  # nothing removed automatically
    button = at.button(key=f"excl_btn_{result.id}")
    assert button.disabled  # needs a selection and an explicit confirmation


def test_approximate_table_blocks_statistics_until_override():
    at = load_sample(fresh(), "nanoparticle_size")
    project = at.session_state["project"]
    next(iter(project.tables.values())).approximate = True
    goto(at, "2 Analyze")
    assert any("APPROXIMATE" in e.value for e in at.error)
    click(at, "run_manual")
    assert len(at.session_state["project"].results) == 0
    assert any("APPROXIMATE" in e.value for e in at.error)
    at.checkbox(key="allow_approx").check().run()
    click(at, "run_manual")
    results = at.session_state["project"].results
    assert len(results) == 1 and next(iter(results.values())).approximate_data


def test_power_calculator_solves_for_sample_size():
    at = goto(fresh(), "2 Analyze")
    at.text_input(key="pw_es").set_value("0.5").run()
    click(at, "pw_go")
    assert any("64" in s.value for s in at.success), [s.value for s in at.success]
