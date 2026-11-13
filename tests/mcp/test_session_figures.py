"""TorusSession: figures, restyling, panels, export, reports, project files.

These tests do not need the analysis modules: where a figure needs a statistics result
they attach a hand-built ``StatResult`` test double (see ``tests.mcp.synthetic_result``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.mcp import NANO_GROUPS, PNG_MAGIC, strict_json, synthetic_result
from torusplot.export import validate_svg
from torusplot.mcp_server.session import SessionError, TorusSession
from torusplot.plot import comparison_label, preset_names, summary_text
from torusplot.spec import PLOT_TYPES, CompositionSpec, FigureSpec, StyleSpec


@pytest.fixture
def session(tmp_path: Path) -> TorusSession:
    s = TorusSession(tmp_path / "out")
    s.load_table("np", sample="nanoparticle_size")
    return s


@pytest.fixture
def result(session: TorusSession):
    return session.project.add_result(synthetic_result("np"))


def _files(session: TorusSession) -> set[str]:
    return {p.name for p in session.output_dir.iterdir()} if session.output_dir.exists() else set()


# ====================================================================== make_figure
def test_make_figure_returns_spec_warnings_and_preview(session: TorusSession):
    out = session.make_figure("Fig 1", "np", "bar", title="Particle size", x_label="Method", width_mm=100,
                              height_mm=80, style_preset="classic lab", palette="Okabe-Ito")
    assert out["name"] == "Fig 1" and out["kind"] == "figure" and out["replaced"] is False
    assert out["warnings"] == [] and out["brackets_drawn"] == 0 and out["stat_result_ids"] == []
    assert out["size_mm"] == [100.0, 80.0]
    spec = FigureSpec.model_validate(out["spec"])  # the returned spec is a valid, re-usable spec
    assert spec.title == "Particle size" and spec.axes.x.label == "Method"
    assert spec.style.preset == "Classic Lab" and spec.style.palette == "Okabe-Ito"
    assert spec.axes.y.label == "Nanoparticle diameter (nm)"  # the sample's own axis title
    preview = Path(out["preview_path"])
    assert preview == session.output_dir / "Fig_1_preview.png" and preview.read_bytes().startswith(PNG_MAGIC)
    assert session.preview_png("Fig 1") == preview.read_bytes() and session.preview_png("nope") is None
    assert out["preview_size_px"][0] > out["preview_size_px"][1] > 100
    assert session.list_state()["figures"][0]["name"] == "Fig 1"
    strict_json(out)
    again = session.make_figure("Fig 1", "np", "box")
    assert again["replaced"] is True and len(session.project.figures) == 1


@pytest.mark.parametrize("plot_type", ["bar", "scatter", "box", "violin", "box_violin", "beeswarm", "histogram", "qq", "heatmap"])
def test_plot_types_of_a_column_table_render(session: TorusSession, plot_type: str):
    out = session.make_figure(f"f_{plot_type}", "np", plot_type)
    assert out["plot_type"] == plot_type and out["warnings"] == []


def test_make_figure_from_spec_and_simple_arguments_override(session: TorusSession):
    spec = {
        "plot_type": "box",
        "data": {"table": "np", "groups": ["Extrusion", "Microfluidic"]},
        "layers": [{"type": "boxplot"}, {"type": "data_points_beeswarm"}],
        "title": "From the spec",
        "axes": {"y": {"label": "Diameter (nm)", "min": 0}},
        "texts": [{"text": "pilot batch", "x": 0.05, "y": 0.95}],
        "style": {"preset": "Minimal"},
    }
    out = session.make_figure("f", spec=spec, title="From the argument", groups=["Microfluidic", "Extrusion"])
    got = FigureSpec.model_validate(out["spec"])
    assert got.plot_type == "box" and got.title == "From the argument"
    assert got.data.groups == ["Microfluidic", "Extrusion"] and got.axes.y.label == "Diameter (nm)"
    assert [lay.type for lay in got.layers] == ["boxplot", "data_points_beeswarm"]
    assert got.texts[0].text == "pilot batch" and got.style.preset == "Minimal"
    # a JSON object typed as text (some clients send that) is accepted too
    out = session.make_figure("g", spec='{"plot_type": "bar", "data": {"table": "np"}}')
    assert out["plot_type"] == "bar"


def test_multiple_variables_tables_need_column_roles(session: TorusSession):
    session.load_table("cells", sample="cell_uptake_cells")
    out = session.make_figure("sp", "cells", "superplot", x="Treatment", y="Uptake", replicate="Experiment")
    assert out["warnings"] == [] and out["spec"]["data"]["replicate"] == "Experiment"
    out = session.make_figure("bars", "cells", "bar", x="Treatment", y="Uptake", hue="Experiment")
    assert out["spec"]["data"]["hue"] == "Experiment"
    with pytest.raises(SessionError, match="'Nope'"):
        session.make_figure("bad", "cells", "bar", x="Nope", y="Uptake")
    assert "bad" not in session.project.figures


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"table": "ghost", "plot_type": "bar"}, "Table 'ghost' not found"),
        ({"table": "np", "plot_type": "xy"}, "cannot draw a Column table"),
        ({"table": "np", "plot_type": "sparkles"}, "is not a plot type"),
        ({"plot_type": "bar"}, "Give the table"),
        ({"table": "np"}, "Give the plot type"),
        ({"table": "np", "plot_type": "bar", "spec": ["not", "a", "dict"]}, "spec must be a JSON object"),
        ({"table": "np", "plot_type": "bar", "spec": "plain words"}, "spec must be a JSON object"),
        ({"table": "np", "plot_type": "bar", "spec": {"data": "np"}}, "spec.data must be a JSON object"),
        ({"table": "np", "plot_type": "bar", "spec": {"sparkle": True}}, "Invalid figure spec"),
        ({"table": "np", "plot_type": "bar", "spec": {"layers": [{"type": "pie_slice"}]}}, "layers.0.type"),
        ({"table": "np", "plot_type": "bar", "style_preset": "Disco"}, "Unknown style preset"),
        ({"table": "np", "plot_type": "bar", "palette": "Disco"}, "Unknown palette"),
        ({"table": "np", "plot_type": "bar", "width_mm": 2}, "layout.width_mm"),
        ({"table": "np", "plot_type": "bar", "groups": ["Nope"]}, "Unknown group"),
        ({"table": "np", "plot_type": "bar", "groups": [["nested"]]}, "groups must be a list of text values"),
        ({"table": "np", "plot_type": "bar", "spec": {"style": {"palette": ["not-a-colour"]}}}, "not a valid colour"),
        ({"table": "np", "plot_type": "bar", "spec": {"layers": [{"type": "mean_bar", "marker": 5}]}}, "layers.0.marker"),
        ({"table": "np", "plot_type": "residuals"}, "needs a fit result"),
    ],
)
def test_make_figure_errors_are_clear_and_store_nothing(session: TorusSession, kwargs: dict, message: str):
    with pytest.raises(SessionError, match=message) as err:
        session.make_figure("f", **kwargs)
    assert "Traceback" not in str(err.value)
    assert session.project.figures == {} and _files(session) == set()


def test_figure_name_rules(session: TorusSession):
    for bad in ["", None, "x" * 121]:
        with pytest.raises(SessionError, match="Figure name"):
            session.make_figure(bad, "np", "bar")
    session.make_figure("a", "np", "bar")
    session.compose_panels("panel", ["a"])
    with pytest.raises(SessionError, match="already the name of a composition"):
        session.make_figure("panel", "np", "bar")
    with pytest.raises(SessionError, match="already the name of a figure"):
        session.compose_panels("a", ["a"])
    # path-like names are allowed as names but never become paths
    out = session.make_figure("../../escape", "np", "bar")
    assert Path(out["preview_path"]).parent == session.output_dir
    assert Path(out["preview_path"]).name == "escape_preview.png"


# ===================================================== the hard rule: no typed statistics
@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "p < 0.05"},
        {"title": "Treatment effect (P = 0.003)"},
        {"title": "***"},
        {"x_label": "ns"},
        {"y_label": "Diameter ***"},
        {"y_label": "p-value = 0.01"},
        {"spec": {"title": "p<0.01"}},
        {"spec": {"texts": [{"text": "***", "x": 0.5, "y": 0.9}]}},
        {"spec": {"texts": [{"text": "p = 0.04", "x": 0.5, "y": 0.9}]}},
        {"spec": {"texts": [{"text": "Bulk vs Micro **", "x": 0.5, "y": 0.9}]}},
        {"spec": {"axes": {"y": {"label": "q < 0.05"}}}},
        {"spec": {"axes": {"x": {"ticks": [0, 1], "tick_labels": ["*", "n.s."]}}}},
        {"spec": {"legend": {"title": "P adj"}}},
        {"spec": {"layers": [{"type": "reference_line", "value": 100, "label": "p = 0.01"}]}},
    ],
)
def test_typed_statistics_are_rejected_everywhere_text_is_drawn(session: TorusSession, kwargs: dict):
    with pytest.raises(SessionError, match="looks like a statistical result"):
        session.make_figure("f", "np", "bar", **kwargs)
    assert session.project.figures == {} and _files(session) == set()


@pytest.mark.parametrize(
    "annotation",
    [
        {"stat_result_id": "sr_real", "label": "***"},
        {"stat_result_id": "sr_real", "text": "p < 0.001"},
        {"stat_result_id": "sr_real", "p_value": 0.001},
        {"stat_result_id": "sr_real", "stars": "**"},
        {"label": "***", "pairs": [["Bulk mixing", "Sonication"]]},
    ],
)
def test_annotations_have_no_field_for_a_number_or_stars(session: TorusSession, result, annotation: dict):
    if "stat_result_id" in annotation:
        annotation = {**annotation, "stat_result_id": result.id}
    with pytest.raises(SessionError, match="Invalid figure spec"):
        session.make_figure("f", "np", "bar", spec={"annotations": [annotation]})
    assert session.project.figures == {}


def test_a_result_id_that_does_not_exist_is_an_error_never_a_drawn_fake(session: TorusSession, result):
    with pytest.raises(SessionError, match="No statistics result with ID 'sr_deadbeef'") as err:
        session.make_figure("f", "np", "bar", annotate_result_id="sr_deadbeef")
    assert result.id in str(err.value) and "cannot be invented" in str(err.value)
    with pytest.raises(SessionError, match="No statistics result with ID 'sr_fake'"):
        session.make_figure("f", "np", "bar", spec={"annotations": [{"stat_result_id": "sr_fake"}]})
    with pytest.raises(SessionError, match="No statistics result with ID 'sr_fake'"):
        session.make_figure("f", "np", "bar", spec={"layers": [{"type": "curve_fit", "stat_result_id": "sr_fake"}]})
    with pytest.raises(SessionError, match="No statistics result with ID 'sr_fake'"):
        session.make_figure("f", "np", "qq", data_result_id="sr_fake")
    with pytest.raises(SessionError, match="Unknown comparison ID"):
        session.make_figure("f", "np", "bar", annotate_result_id=result.id, comparisons=[f"{result.id}:c99"])
    with pytest.raises(SessionError, match="comparisons must be"):
        session.make_figure("f", "np", "bar", annotate_result_id=result.id, comparisons=[1, 2])
    assert session.project.figures == {} and _files(session) == set()


def test_no_results_at_all_gives_a_helpful_message(session: TorusSession):
    with pytest.raises(SessionError, match="No analyses have been run yet"):
        session.make_figure("f", "np", "bar", annotate_result_id="sr_00000000")


# ============================================================ annotations by result ID
def test_brackets_are_drawn_from_the_referenced_result(session: TorusSession, result):
    out = session.make_figure("f", "np", "bar", annotate_result_id=result.id)
    assert out["brackets_drawn"] == len(result.comparisons) == 6
    assert out["stat_result_ids"] == [result.id] and out["warnings"] == []
    by_id = {c.id: c for c in result.comparisons}
    for bracket in out["brackets"]:
        comp = by_id[bracket["comparison_id"]]
        assert bracket["label"] == comparison_label(comp, "stars") and bracket["significant"] == comp.significant
    annotation = out["spec"]["annotations"][0]
    assert annotation["stat_result_id"] == result.id and annotation["kind"] == "brackets"
    assert not {"label", "text", "p_value"} & set(annotation)


def test_comparison_selection_and_label_format(session: TorusSession, result):
    significant = [c for c in result.comparisons if c.significant]
    out = session.make_figure("f", "np", "bar", annotate_result_id=result.id, comparisons="significant")
    assert out["brackets_drawn"] == len(significant) == 4
    chosen = [result.comparisons[0].id, result.comparisons[2].id]
    out = session.make_figure("f", "np", "bar", annotate_result_id=result.id, comparisons=chosen, annotation_format="p")
    assert [b["comparison_id"] for b in out["brackets"]] == chosen
    assert [b["label"] for b in out["brackets"]] == [comparison_label(result.comparisons[i], "p") for i in (0, 2)]
    out = session.make_figure("f", "np", "bar", annotate_result_id=result.id, comparisons=chosen[0])
    assert out["brackets_drawn"] == 1
    with pytest.raises(SessionError, match="annotation_format must be one of"):
        session.make_figure("f", "np", "bar", annotate_result_id=result.id, annotation_format="p = 0.01")
    with pytest.raises(SessionError, match="annotation_kind must be one of"):
        session.make_figure("f", "np", "bar", annotate_result_id=result.id, annotation_kind="text")


def test_re_annotating_replaces_the_annotation_of_the_same_result(session: TorusSession, result):
    spec = {"annotations": [{"stat_result_id": result.id, "comparisons": "significant"}]}
    out = session.make_figure("f", "np", "bar", spec=spec, annotate_result_id=result.id)
    assert len(out["spec"]["annotations"]) == 1 and out["brackets_drawn"] == 6
    with pytest.raises(SessionError, match="spec.annotations must be a list"):
        session.make_figure("g", "np", "bar", spec={"annotations": {"x": 1}}, annotate_result_id=result.id)


def test_summary_text_annotation(session: TorusSession, result):
    out = session.make_figure("f", "np", "bar", annotate_result_id=result.id, annotation_kind="summary_text")
    assert out["brackets_drawn"] == 0
    assert out["summary_texts"] == [{"stat_result_id": result.id, "text": summary_text(result)}]
    # a result without pairwise comparisons falls back to the summary line automatically
    plain = session.project.add_result(synthetic_result("np", with_comparisons=False))
    out = session.make_figure("g", "np", "bar", annotate_result_id=plain.id)
    assert out["spec"]["annotations"][0]["kind"] == "summary_text" and out["brackets_drawn"] == 0
    with pytest.raises(SessionError, match="no pairwise comparisons"):
        session.make_figure("h", "np", "bar", annotate_result_id=plain.id, annotation_kind="brackets")
    # brackets need a categorical plot
    out = session.make_figure("h", "np", "histogram", annotate_result_id=result.id)
    assert out["spec"]["annotations"][0]["kind"] == "summary_text"
    with pytest.raises(SessionError, match="only drawn on categorical plots"):
        session.make_figure("i", "np", "histogram", annotate_result_id=result.id, annotation_kind="brackets")


def test_questionable_result_links_are_reported(session: TorusSession, result):
    session.load_table("other", "column", text="\t".join(NANO_GROUPS) + "\n1\t2\t3\t4\n2\t3\t4\t5\n3\t4\t5\t7")
    out = session.make_figure("f", "other", "bar", annotate_result_id=result.id)
    assert any("was computed on table 'np'" in w for w in out["warnings"])
    # replacing the analysed table makes its results stale
    session.load_table("np", sample="nanoparticle_size")
    out = session.make_figure("g", "np", "bar", annotate_result_id=result.id)
    assert any("re-run the analysis" in w for w in out["warnings"])
    assert session.list_state()["results"][0]["stale"] is True
    assert "replaced after this analysis" in session.get_result(result.id)["warnings"][0]


def test_fit_curves_need_a_fit_result_and_an_xy_plot(session: TorusSession, result):
    session.load_table("dr", sample="dose_response")
    with pytest.raises(SessionError, match="needs plot_type 'xy' or 'dose_response'"):
        session.make_figure("f", "np", "bar", fit_result_ids=[result.id])
    with pytest.raises(SessionError, match="has no fitted curve"):
        session.make_figure("f", "dr", "dose_response", fit_result_ids=[result.id])
    with pytest.raises(SessionError, match="No statistics result with ID"):
        session.make_figure("f", "dr", "xy", fit_result_ids="sr_missing")
    out = session.make_figure("f", "dr", "dose_response")  # no fit attached: drawn without a curve, with a warning
    assert any("No fit is attached" in w for w in out["warnings"])
    assert session.make_figure("g", "dr", "xy")["spec"]["axes"]["y"]["label"] == "Viability (%)"


# ==================================================================== restyle_figure
def test_restyle_changes_only_style_and_layout(session: TorusSession, result):
    before = session.make_figure("f", "np", "bar", annotate_result_id=result.id)
    out = session.restyle_figure(
        "f", style_preset="minimal", palette=["#112233", "#445566", "#778899", "#AABBCC"],
        colors={"Extrusion": "red"}, font_family="Source Sans 3", font_size=9, width_mm=120, height_mm=90,
        title="New title", x_label="Method", y_label="Diameter (nm)", y_min=0, y_max=260, y_scale="linear",
        x_scale="linear", error_bars="sd", points="beeswarm", legend_position="outside right",
        style_overrides={"bar_width": 0.5, "spines": ["left", "bottom"]},
    )
    spec = FigureSpec.model_validate(out["spec"])
    assert spec.style.preset == "Minimal" and spec.style.palette[0] == "#112233"
    assert spec.style.colors == {"Extrusion": "#FF0000"} and spec.style.font_family == "Source Sans 3"
    assert spec.style.font_size == 9 and spec.style.bar_width == 0.5 and spec.style.spines == ["left", "bottom"]
    assert (spec.layout.width_mm, spec.layout.height_mm) == (120, 90) and out["size_mm"] == [120.0, 90.0]
    assert spec.title == "New title" and spec.axes.x.label == "Method" and spec.axes.y.label == "Diameter (nm)"
    assert (spec.axes.y.min, spec.axes.y.max) == (0, 260) and spec.legend.position == "outside right"
    assert [lay.type for lay in spec.layers] == ["mean_bar", "sd_errorbar", "data_points_beeswarm"]
    # data and statistics are untouched; the brackets are still the result's own
    assert out["spec"]["data"] == before["spec"]["data"]
    assert out["spec"]["annotations"] == before["spec"]["annotations"]
    assert out["brackets_drawn"] == 6 and out["brackets"] == before["brackets"]
    assert out["replaced"] is True and len(session.project.figures) == 1

    out = session.restyle_figure("f", title="", x_label="", error_bars="none", points="none", font_family="")
    spec = FigureSpec.model_validate(out["spec"])
    assert spec.title is None and spec.axes.x.label is None and spec.style.font_family is None
    assert [lay.type for lay in spec.layers] == ["mean_bar"]
    assert spec.style.preset == "Minimal"  # unchanged options are kept
    assert session.restyle_figure("f")["spec"] == out["spec"]  # nothing to change: just re-rendered


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"style_preset": "Disco"}, "Unknown style preset"),
        ({"palette": "Disco"}, "Unknown palette"),
        ({"colors": {"Extrusion": "not-a-colour"}}, "not a valid colour"),
        ({"colors": "red"}, "colors must be a JSON object"),
        ({"error_bars": "stderr"}, "error_bars must be one of"),
        ({"points": "dots"}, "points must be one of"),
        ({"y_scale": "logarithmic"}, "y_scale must be one of"),
        ({"x_scale": 10}, "x_scale must be one of"),
        ({"legend_position": "middle"}, "legend_position must be one of"),
        ({"y_min": 10, "y_max": 1}, "Axis min must be smaller than max"),
        ({"font_size": -3}, "style.font_size"),
        ({"width_mm": 5}, "layout.width_mm"),
        ({"style_overrides": {"sparkle": 1}}, "Unknown style option"),
        ({"style_overrides": {"bar_width": 7}}, "style.bar_width"),
        ({"style_overrides": "bar_width=7"}, "style_overrides must be a JSON object"),
        ({"title": "p < 0.05"}, "looks like a statistical result"),
        ({"y_label": "**"}, "looks like a statistical result"),
        ({"x_label": "P = 0.01"}, "looks like a statistical result"),
    ],
)
def test_restyle_errors_leave_the_figure_unchanged(session: TorusSession, kwargs: dict, message: str):
    before = session.make_figure("f", "np", "bar")["spec"]
    with pytest.raises(SessionError, match=message):
        session.restyle_figure("f", **kwargs)
    assert session.project.figures["f"].model_dump(mode="json", exclude_none=True) == before


def test_restyle_unknown_figure_and_inapplicable_options(session: TorusSession):
    with pytest.raises(SessionError, match=r"Figure 'ghost' not found\. Figures: \[\]"):
        session.restyle_figure("ghost", title="x")
    session.make_figure("h", "np", "histogram")
    with pytest.raises(SessionError, match="error_bars does not apply"):
        session.restyle_figure("h", error_bars="sd")
    with pytest.raises(SessionError, match="points does not apply"):
        session.restyle_figure("h", points="jitter")


def test_restyle_cannot_reach_data_or_statistics(session: TorusSession):
    session.make_figure("f", "np", "bar")
    with pytest.raises(TypeError):  # there simply is no such parameter
        session.restyle_figure("f", annotations=[{"stat_result_id": "sr_x"}])  # type: ignore[call-arg]
    for key in ("annotations", "data", "texts", "layers", "title"):
        assert key not in StyleSpec.model_fields
        with pytest.raises(SessionError, match="Unknown style option"):
            session.restyle_figure("f", style_overrides={key: "p < 0.05"})


# ==================================================================== compose_panels
@pytest.mark.parametrize(("n", "grid"), [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)), (5, (2, 3))])
def test_compose_places_figures_on_an_automatic_grid(session: TorusSession, n: int, grid: tuple[int, int]):
    names = [f"f{i}" for i in range(n)]
    for name in names:
        session.make_figure(name, "np", "box")
    out = session.compose_panels("panel", names)
    assert (out["grid"]["rows"], out["grid"]["cols"]) == grid
    assert [p["figure"] for p in out["panels"]] == names
    assert [p["label"] for p in out["panels"]] == [chr(ord("A") + i) for i in range(n)]
    assert [(p["row"], p["col"]) for p in out["panels"]] == [(i // grid[1], i % grid[1]) for i in range(n)]
    assert out["size_mm"][0] == 183.0
    CompositionSpec.model_validate(out["spec"])
    assert Path(out["preview_path"]).read_bytes().startswith(PNG_MAGIC)
    strict_json(out)


def test_compose_options_and_explicit_layout(session: TorusSession, result):
    session.make_figure("a", "np", "bar", annotate_result_id=result.id)
    session.make_figure("b", "np", "box")
    session.make_figure("c", "np", "violin")
    out = session.compose_panels("p1", ["a", "b", "c"], rows=3, width_mm=89, height_mm=200, panel_labels="(a)",
                                 share_y="col", shared_legend=True, style_preset="Classic Lab")
    assert (out["grid"]["rows"], out["grid"]["cols"]) == (3, 1) and out["size_mm"] == [89.0, 200.0]
    assert [p["label"] for p in out["panels"]] == ["(a)", "(b)", "(c)"]
    assert out["spec"]["style_preset"] == "Classic Lab" and out["spec"]["share_y"] == "col"
    assert out["stat_result_ids"] == [result.id]
    assert session.compose_panels("p1", ["a", "b"], cols=1)["replaced"] is True

    layout = [
        {"figure": "a", "row": 0, "col": 0, "colspan": 2},
        {"figure": "b", "row": 1, "col": 0},
        {"figure": "c", "row": 1, "col": 1, "label": "Z"},
    ]
    out = session.compose_panels("p2", layout=layout, height_mm=150)
    assert (out["grid"]["rows"], out["grid"]["cols"]) == (2, 2)
    assert [p["label"] for p in out["panels"]] == ["A", "B", "Z"] and out["panels"][0]["colspan"] == 2
    assert {c["name"] for c in session.list_state()["compositions"]} == {"p1", "p2"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Give the figures to combine"),
        ({"figures": []}, "Give the figures to combine"),
        ({"figures": ["a", "ghost"]}, r"Figure\(s\) not found: \['ghost'\]"),
        ({"figures": ["a", "b"], "rows": 1, "cols": 1}, "do not fit in a 1 x 1 grid"),
        ({"figures": ["a"], "rows": 0}, "rows must be a whole number"),
        ({"figures": ["a"], "cols": 99}, "cols must be a whole number"),
        ({"figures": ["a"], "width_mm": 5}, "width_mm must be between"),
        ({"figures": ["a"], "height_mm": "tall"}, "height_mm must be a number"),
        ({"figures": ["a"], "panel_labels": "i"}, "panel_labels must be one of"),
        ({"figures": ["a"], "share_x": "yes"}, "share_x must be one of"),
        ({"figures": ["a"], "style_preset": "Disco"}, "Unknown style preset"),
        ({"layout": [{"figure": "a"}]}, "needs 'figure', 'row' and 'col'"),
        ({"layout": ["a"]}, "layout must be a list of"),
        ({"layout": [{"figure": "a", "row": "top", "col": 0}]}, "must be whole numbers"),
        ({"layout": [{"figure": "a", "row": 0, "col": 0}, {"figure": "b", "row": 0, "col": 0}]}, "overlaps another panel"),
        ({"layout": [{"figure": "a", "row": 0, "col": 0, "label": "***"}]}, "looks like a statistical result"),
        ({"layout": [{"figure": "a", "row": 0, "col": 0, "label": "p = 0.01"}]}, "looks like a statistical result"),
        ({"layout": [{"figure": "a", "row": 0, "col": 0, "label": "a very long panel label"}]}, "short text value"),
    ],
)
def test_compose_errors(session: TorusSession, kwargs: dict, message: str):
    session.make_figure("a", "np", "bar")
    session.make_figure("b", "np", "box")
    with pytest.raises(SessionError, match=message):
        session.compose_panels("panel", **kwargs)
    assert session.project.compositions == {}


def test_a_layout_that_does_not_fit_is_reported_as_a_warning(session: TorusSession):
    names = [f"f{i}" for i in range(6)]
    for name in names:
        session.make_figure(name, "np", "bar")
    out = session.compose_panels("tiny", names, cols=6, width_mm=40, height_mm=20)
    assert any("does not fit at this size" in w for w in out["warnings"])


# ===================================================================== export_figure
def test_export_writes_every_format_inside_the_output_directory(session: TorusSession, result):
    from PIL import Image

    session.make_figure("Fig 1", "np", "bar", annotate_result_id=result.id)
    out = session.export_figure("Fig 1", formats=["svg", "pdf", "tiff", "png"], dpi=150)
    paths = {f["format"]: Path(f["path"]) for f in out["files"]}
    assert set(paths) == {"svg", "pdf", "tiff", "png"}
    for f in out["files"]:
        assert Path(f["path"]).parent == session.output_dir and Path(f["path"]).stat().st_size == f["size_bytes"] > 0
    assert paths["svg"].name == "Fig_1.svg" and paths["tiff"].name == "Fig_1.tif"
    assert paths["pdf"].read_bytes().startswith(b"%PDF") and paths["png"].read_bytes().startswith(PNG_MAGIC)
    with Image.open(paths["tiff"]) as tiff:
        assert tiff.info["compression"] == "tiff_lzw"
    check = validate_svg(paths["svg"])
    assert check["ok"], check["problems"]
    assert check["text_elements"] > 5 and out["svg_check"] == {
        "ok": True, "editable_text_elements": check["text_elements"], "problems": [],
    }
    assert sum(1 for gid in check["group_ids"] if re.fullmatch(r"bracket-\d+", gid)) == len(result.comparisons)
    report = {r["format"]: Path(r["path"]) for r in out["stats_report"]}
    assert report["markdown"].name == "Fig_1_stats.md" and report["csv"].name == "Fig_1_stats.csv"
    assert result.id in report["markdown"].read_text(encoding="utf-8")
    assert result.id in report["csv"].read_text(encoding="utf-8")
    assert out["stat_result_ids"] == [result.id] and out["size_mm"] == [89.0, 70.0]
    strict_json(out)


def test_export_defaults_and_report_switch(session: TorusSession, result):
    session.make_figure("plain", "np", "box")
    out = session.export_figure("plain")
    assert [f["format"] for f in out["files"]] == ["svg", "pdf", "png"]
    assert out["stats_report"] == [] and "no report was written" in out["stats_report_note"]
    assert "plain_stats.md" not in _files(session)
    session.make_figure("annotated", "np", "bar", annotate_result_id=result.id)
    out = session.export_figure("annotated", ["png", "PNG", ".png", "tif"], with_stats_report=False, transparent=True)
    assert [f["format"] for f in out["files"]] == ["png", "tiff"] and "stats_report" not in out
    assert "annotated_stats.md" not in _files(session)


def test_export_a_composition_with_the_results_of_all_panels(session: TorusSession, result):
    other = session.project.add_result(synthetic_result("np", with_comparisons=False))
    session.make_figure("a", "np", "bar", annotate_result_id=result.id)
    session.make_figure("b", "np", "box", annotate_result_id=other.id)
    session.compose_panels("panel", ["a", "b"])
    out = session.export_figure("panel", ["svg", "png"], dpi=100)
    assert out["kind"] == "composition" and out["stat_result_ids"] == [result.id, other.id]
    assert validate_svg(Path(out["files"][0]["path"]))["ok"]
    text = Path(out["stats_report"][0]["path"]).read_text(encoding="utf-8")
    assert result.id in text and other.id in text


@pytest.mark.parametrize(
    "filename",
    ["../evil", "..", "sub/../../evil", "..\\evil", "/tmp/evil", "/etc/passwd", "C:\\Windows\\evil", "c:/evil",
     "~/evil", "~root/evil", "a/../../../evil", "", "   ", "bad\x00name"],
)
def test_export_rejects_path_traversal_and_outside_paths(session: TorusSession, tmp_path: Path, filename: str):
    session.make_figure("f", "np", "bar")
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SessionError, match="outside the output directory|must be a non-empty text value"):
        session.export_figure("f", ["png"], filename=filename)
    assert {p for p in tmp_path.rglob("*")} == before  # nothing was written anywhere


def test_export_file_names_inside_the_output_directory(session: TorusSession):
    session.make_figure("f", "np", "bar")
    out = session.export_figure("f", ["png"], filename="Figure 2: final?.png", dpi=72)
    assert Path(out["files"][0]["path"]) == session.output_dir / "Figure_2_final.png"
    out = session.export_figure("f", ["svg"], filename="paper/fig 3/panel")
    assert Path(out["files"][0]["path"]) == session.output_dir / "paper" / "fig_3" / "panel.svg"
    inside = session.output_dir / "abs" / "ok"
    out = session.export_figure("f", ["png"], filename=str(inside), dpi=72)
    assert Path(out["files"][0]["path"]) == inside.with_suffix(".png")
    with pytest.raises(SessionError, match="output directory itself"):
        session.export_figure("f", ["png"], filename=str(session.output_dir))


def test_export_never_follows_a_link_out_of_the_output_directory(session: TorusSession, tmp_path: Path):
    import os

    session.make_figure("f", "np", "bar")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        os.symlink(outside, session.output_dir / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")
    with pytest.raises(SessionError, match="outside the output directory"):
        session.export_figure("f", ["png"], filename="link/fig")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "ghost"}, "No figure or composition named 'ghost'"),
        ({"name": "f", "formats": ["bmp"]}, "formats must be a non-empty list"),
        ({"name": "f", "formats": []}, "formats must be a non-empty list"),
        ({"name": "f", "dpi": 10}, "dpi must be a whole number from 72 to 1200"),
        ({"name": "f", "dpi": 300.5}, "dpi must be a whole number"),
    ],
)
def test_export_errors(session: TorusSession, kwargs: dict, message: str):
    session.make_figure("f", "np", "bar")
    before = _files(session)
    with pytest.raises(SessionError, match=message):
        session.export_figure(**kwargs)
    assert _files(session) == before


def test_an_unwritable_output_directory_is_a_clean_error(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    session = TorusSession(blocker / "out")
    session.load_table("np", sample="nanoparticle_size")
    with pytest.raises(SessionError, match="cannot be created"):
        session.make_figure("f", "np", "bar")
    assert session.project.figures == {}
    with pytest.raises(SessionError, match="could not be written"):
        session.save_project("p.tplot")


# ================================================================== get_stats_report
def test_stats_report_for_figures_ids_and_everything(session: TorusSession, result):
    other = session.project.add_result(synthetic_result("np", with_comparisons=False))
    session.make_figure("f", "np", "bar", annotate_result_id=result.id)
    session.make_figure("g", "np", "box")
    session.compose_panels("panel", ["f", "g"])

    out = session.get_stats_report(figure="f")
    assert out["format"] == "markdown" and out["result_ids"] == [result.id]
    assert out["report"].startswith("# Statistics report: f") and result.id in out["report"]
    assert "Data are presented as" in out["methods_paragraph"] and "ns, not significant" in out["significance_legend"]
    assert session.get_stats_report(figure="panel")["result_ids"] == [result.id]
    assert session.get_stats_report(figure="f", result_ids=[other.id, result.id])["result_ids"] == [result.id, other.id]
    assert session.get_stats_report(result_ids=[other.id])["result_ids"] == [other.id]
    assert session.get_stats_report()["result_ids"] == [result.id, other.id]  # neither given: every result

    csv = session.get_stats_report(result_ids=[result.id], format="csv")
    assert csv["format"] == "csv" and csv["report"].splitlines()[0].startswith("result_id,test,table")
    assert sum(line.startswith(result.id) for line in csv["report"].splitlines()) > 6

    empty = session.get_stats_report(figure="g")
    assert empty["result_ids"] == [] and empty["methods_paragraph"] == "" and "run_analysis" in empty["note"]
    with pytest.raises(SessionError, match="No statistics result with ID 'sr_nope'"):
        session.get_stats_report(result_ids=["sr_nope"])
    with pytest.raises(SessionError, match="format must be one of"):
        session.get_stats_report(format="docx")
    with pytest.raises(SessionError, match="No figure or composition named"):
        session.get_stats_report(figure="ghost")
    strict_json(out)


# ====================================================================== project files
def test_save_and_load_round_trip(session: TorusSession, result, tmp_path: Path):
    session.load_table("dr", sample="dose_response")
    session.make_figure("f", "np", "bar", annotate_result_id=result.id, title="Sizes")
    session.make_figure("g", "dr", "xy")
    session.compose_panels("panel", ["f", "g"])
    saved = session.save_project("My study")
    path = Path(saved["path"])
    assert path == session.output_dir / "My_study.tplot" and path.stat().st_size == saved["size_bytes"]
    assert (saved["tables"], saved["results"], saved["figures"], saved["compositions"]) == (2, 1, 2, 1)
    assert Path(session.save_project("other.zip")["path"]).name == "other.tplot"
    assert Path(session.save_project()["path"]).name == "project.tplot"

    fresh = TorusSession(tmp_path / "second")
    state = fresh.load_project(str(path))
    assert state["loaded_from"] == str(path)
    before = session.list_state()
    for key in ("tables", "results", "figures", "compositions"):
        assert state[key] == before[key]
    # the loaded figure re-renders with the stored result: same brackets, nothing recalculated
    assert fresh.get_result(result.id)["comparisons"] == session.get_result(result.id)["comparisons"]
    assert fresh.restyle_figure("f")["brackets"] == session.restyle_figure("f")["brackets"]
    exported = fresh.export_figure("panel", ["png"], dpi=72)
    assert Path(exported["files"][0]["path"]).parent == fresh.output_dir
    # a bare file name is looked up in the output directory
    assert session.load_project("My_study.tplot")["figures"] == before["figures"]


@pytest.mark.parametrize("filename", ["../x.tplot", "/tmp/x.tplot", "..\\..\\x.tplot", "C:\\x.tplot", ""])
def test_save_project_stays_inside_the_output_directory(session: TorusSession, tmp_path: Path, filename: str):
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SessionError, match="outside the output directory|must be a non-empty text value"):
        session.save_project(filename)
    assert {p for p in tmp_path.rglob("*")} == before


def test_load_project_errors_keep_the_session(session: TorusSession, tmp_path: Path):
    session.make_figure("f", "np", "bar")
    with pytest.raises(SessionError, match="Only .tplot project files"):
        session.load_project(str(tmp_path / "data.csv"))
    with pytest.raises(SessionError, match="Project file not found"):
        session.load_project(str(tmp_path / "missing.tplot"))
    with pytest.raises(SessionError, match="path must be"):
        session.load_project("  ")
    garbage = tmp_path / "garbage.tplot"
    garbage.write_bytes(b"not a zip archive")
    with pytest.raises(SessionError, match="Not a readable .tplot file"):
        session.load_project(str(garbage))
    assert list(session.project.figures) == ["f"] and list(session.project.tables) == ["np"]


# ======================================================================== catalogues
def test_list_styles(session: TorusSession):
    out = session.list_styles()
    assert [p["name"] for p in out["presets"]] == preset_names()
    assert all(p["description"] and p["width_mm"] > 0 for p in out["presets"])
    palettes = {p["name"]: p["colors"] for p in out["palettes"]}
    assert palettes["Okabe-Ito"][0] == "#0072B2" and len(palettes["viridis"]) == 5
    assert [p["key"] for p in out["plot_types"]] == list(PLOT_TYPES)
    bar = out["plot_types"][0]
    assert bar["default_layers"] == ["mean_bar", "sem_errorbar", "data_points_jitter"] and "column" in bar["tables"]
    assert out["error_bars"] == ["sem", "sd", "ci95", "range", "iqr", "none"]
    assert out["points"] == ["jitter", "beeswarm", "plain", "none"]
    assert "stars" in out["annotation_formats"] and "log10" in out["axis_scales"]
    assert "bar_width" in out["style_options"]
    strict_json(out)


def test_get_figure_schema(session: TorusSession):
    out = session.get_figure_schema()
    assert out["schema"]["title"] == "FigureSpec" and "plot_type" in out["schema"]["required"]
    annotation = out["schema"]["$defs"]["StatAnnotation"]
    assert annotation["additionalProperties"] is False and "stat_result_id" in annotation["required"]
    assert not {"label", "text", "p_value", "stars"} & set(annotation["properties"])
    example = dict(out["example"], data={"table": "np"})
    assert session.make_figure("from_example", spec=example)["plot_type"] == "bar"
    strict_json(out)


# ================================================================= unexpected failures
def test_values_matplotlib_rejects_are_clean_errors(session: TorusSession):
    spec = {"layers": [{"type": "mean_bar"}, {"type": "data_points", "marker": "no-such-marker"}]}
    with pytest.raises(SessionError, match="could not be rendered") as err:
        session.make_figure("f", "np", "bar", spec=spec)
    assert "Traceback" not in str(err.value) and session.project.figures == {}


def test_a_fit_test_double_draws_a_curve_and_layers_must_be_a_list(session: TorusSession):
    session.load_table("dr", sample="dose_response")
    fit = synthetic_result("dr", with_comparisons=False)
    fit.extra = {"curve": {"x": [0.01, 1.0, 100.0], "y": [95.0, 50.0, 5.0]}, "series": "Free drug"}
    session.project.add_result(fit)
    out = session.make_figure("g", "dr", "dose_response", fit_result_ids=[fit.id])
    assert out["spec"]["layers"][-1] == {"type": "curve_fit", "stat_result_id": fit.id} and out["warnings"] == []
    with pytest.raises(SessionError, match="spec.layers must be a list"):
        session.make_figure("h", "dr", "xy", spec={"layers": {"type": "curve_fit"}}, fit_result_ids=[fit.id])


def test_a_project_file_with_an_invalid_figure_is_a_clean_error(session: TorusSession, tmp_path: Path):
    from torusplot.data import Project

    broken = Project("broken")
    broken.figures["bad"] = {"plot_type": "sparkles", "data": {"table": "np"}, "p_value": 0.01}
    path = broken.save(tmp_path / "broken.tplot")
    with pytest.raises(SessionError, match="Invalid input") as err:
        session.load_project(str(path))
    assert "plot_type" in str(err.value) and list(session.project.tables) == ["np"]


def test_many_validation_problems_are_summarised(session: TorusSession):
    layers = [{"type": f"nonsense_{i}"} for i in range(20)]
    with pytest.raises(SessionError, match=r"Invalid figure spec \(20 problem\(s\)\)") as err:
        session.make_figure("f", "np", "bar", spec={"layers": layers})
    assert "and 8 more problem(s)" in str(err.value)


def test_only_user_warnings_become_tool_warnings():
    import warnings

    from torusplot.mcp_server.session import _captured_warnings

    notes: list[str] = []
    with pytest.warns(RuntimeWarning, match="numeric detail"):
        with _captured_warnings(notes):
            warnings.warn("Glyph 9731 missing\n from font", UserWarning, stacklevel=1)
            warnings.warn("Glyph 9731 missing from font", UserWarning, stacklevel=1)
            warnings.warn("numeric detail", RuntimeWarning, stacklevel=1)
    assert notes == ["Glyph 9731 missing from font"]
