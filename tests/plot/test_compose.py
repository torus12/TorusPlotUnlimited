import pytest

from tests.golden import cases
from torusplot.data import Project
from torusplot.export import figure_bytes, figure_size_mm, validate_svg
from torusplot.plot import PlotError
from torusplot.plot.compose import panel_letter, render_composition
from torusplot.spec import CompositionSpec

TABLES, RESULTS, SPECS = cases.tables(), cases.results(), cases.specs()


def test_panel_letters():
    assert [panel_letter(i, "A") for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]
    assert panel_letter(2, "a") == "c" and panel_letter(2, "(A)") == "(C)" and panel_letter(2, "(a)") == "(c)"
    assert panel_letter(2, "1") == "3" and panel_letter(2, "none") == ""


def test_composition_exact_size_letters_and_unique_ids():
    fig = render_composition(cases.composition(), SPECS, TABLES, RESULTS)
    assert figure_size_mm(fig) == pytest.approx((183.0, 140.0))
    assert fig._torusplot_warnings == []
    letters = [c._letter for c in fig._torusplot_contexts]
    assert letters == ["A", "B", "C", "D"]
    info = validate_svg(figure_bytes(fig, "svg"))
    assert info["ok"], info["problems"]
    for letter in "ABCD":
        assert f"panel-letter-{letter}" in info["group_ids"]
        assert f"panel-{letter}-axes" in info["group_ids"]
    # panels do not overlap
    boxes = [c.ax.get_position() for c in fig._torusplot_contexts]
    assert boxes[0].x1 < boxes[1].x0 and boxes[0].y0 > boxes[2].y1


def test_shared_axes_and_shared_legend_and_preset_override():
    comp = CompositionSpec(
        rows=1, cols=2, width_mm=183, height_mm=80, share_y="row", shared_legend=True, style_preset="Classic Lab",
        panel_labels="(a)", panels=[{"figure": "grouped_bar", "row": 0, "col": 0}, {"figure": "heatmap", "row": 0, "col": 1, "label": "Z"}],
    )
    specs = {"grouped_bar": SPECS["grouped_bar"], "heatmap": SPECS["stacked_bar"].model_copy(update={"data": SPECS["grouped_bar"].data, "plot_type": "bar"})}
    fig = render_composition(comp, specs, TABLES, RESULTS)
    a, b = fig._torusplot_contexts
    assert a.ax.get_ylim() == b.ax.get_ylim()
    assert a.ax.get_legend() is None and b.ax.get_legend() is None and len(fig.legends) == 1
    assert [c._letter for c in fig._torusplot_contexts] == ["(a)", "Z"]
    assert a.style["font_family"] == "Liberation Sans"


def test_project_input_and_errors():
    p = Project("demo")
    for t in TABLES.values():
        p.add_table(t)
    for r in RESULTS.values():
        p.add_result(r)
    p.figures.update({"bar": SPECS["bar"], "qq": SPECS["qq"].model_dump()})
    comp = CompositionSpec(rows=1, cols=2, panels=[{"figure": "bar", "row": 0, "col": 0}, {"figure": "qq", "row": 0, "col": 1}], panel_labels="none")
    fig = render_composition(comp, p)
    assert len(fig._torusplot_contexts) == 2
    with pytest.raises(PlotError, match="at least one panel"):
        render_composition(CompositionSpec(), SPECS, TABLES)
    with pytest.raises(PlotError, match="not found"):
        render_composition(CompositionSpec(panels=[{"figure": "zzz", "row": 0, "col": 0}]), SPECS, TABLES)
