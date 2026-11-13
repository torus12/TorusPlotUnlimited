import io
import shutil
import subprocess
from xml.etree import ElementTree

import pytest
from PIL import Image

from tests.golden import cases
from torusplot.export import (
    ExportError,
    data_presentation_sentence,
    export_all,
    export_figure,
    figure_bytes,
    methods_paragraph,
    significance_legend,
    stats_report_csv,
    stats_report_markdown,
    stats_report_rows,
    validate_svg,
    write_stats_report,
)
from torusplot.plot import render_figure
from torusplot.spec import FigureSpec
from torusplot.stats.result import AssumptionCheck, EffectSize, StatResult

TABLES, RESULTS, SPECS = cases.tables(), cases.results(), cases.specs()


@pytest.fixture(scope="module")
def fig():
    return render_figure(SPECS["bar"], TABLES, RESULTS)


def test_svg_has_editable_text_and_named_groups(fig):
    svg = figure_bytes(fig, "svg").decode("utf-8")
    info = validate_svg(svg)
    assert info["ok"], info["problems"]
    assert info["text_elements"] >= 10
    root = ElementTree.fromstring(svg)
    texts = ["".join(el.itertext()).strip() for el in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "Size by synthesis method" in texts and "Bulk mixing" in texts and "****" in texts
    for gid in ("torusplot-figure", "axes", "bar-Bulk_mixing", "points-Sonication", "errorbar-Extrusion", "bracket-1",
                "bracket-label-1", "title", "y-label"):
        assert gid in info["group_ids"], gid
    assert "font-family" in svg and "Inter" in svg
    assert info["width"].endswith("pt") and float(info["width"][:-2]) == pytest.approx(89 / 25.4 * 72, abs=0.01)


def test_svg_and_pdf_are_deterministic(fig):
    assert figure_bytes(fig, "svg") == figure_bytes(fig, "svg")
    assert figure_bytes(fig, "pdf") == figure_bytes(fig, "pdf")


def test_svg_validator_catches_problems(tmp_path):
    assert not validate_svg("<svg><g></svg>")["ok"]
    no_text = '<svg xmlns="http://www.w3.org/2000/svg"><g id="a"/><g id="a"/><use href="#missing"/></svg>'
    info = validate_svg(no_text)
    assert not info["ok"] and len(info["problems"]) == 3
    path = tmp_path / "x.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>', encoding="utf-8")
    assert validate_svg(path)["ok"] and validate_svg(str(path))["ok"]


@pytest.mark.skipif(shutil.which("xmllint") is None, reason="xmllint not installed")
def test_svg_passes_xmllint(fig, tmp_path):
    path = export_figure(fig, tmp_path / "f.svg")
    assert subprocess.run(["xmllint", "--noout", str(path)], capture_output=True).returncode == 0


def test_pdf_tiff_png_exports(fig, tmp_path):
    paths = export_all(fig, tmp_path / "figure", ["svg", "pdf", "tiff", "png"], dpi=300)
    assert sorted(p.name for p in paths.values()) == ["figure.pdf", "figure.png", "figure.svg", "figure.tif"]
    pdf = paths["pdf"].read_bytes()
    assert pdf.startswith(b"%PDF") and b"TorusPlot" in pdf
    assert b"/Type /Font" in pdf and b"/Subtype /Type3" not in pdf  # text embedded as TrueType, not outlines
    tif = Image.open(paths["tiff"])
    assert tif.format == "TIFF" and tif.mode == "RGB"
    assert tif.info["compression"] == "tiff_lzw"
    assert tuple(round(float(d)) for d in tif.info["dpi"]) == (300, 300)
    assert tif.size[0] == round(89 / 25.4 * 300) and abs(tif.size[1] - 70 / 25.4 * 300) <= 1
    png = Image.open(paths["png"])
    assert png.size == tif.size
    tif600 = Image.open(io.BytesIO(figure_bytes(fig, "tiff", dpi=600)))
    assert tif600.size[0] == pytest.approx(2 * tif.size[0], abs=1)
    assert Image.open(io.BytesIO(figure_bytes(fig, "tif", dpi=100, transparent=True))).mode == "RGBA"
    assert export_figure(fig, tmp_path / "noext", fmt="tiff").suffix == ".tif"
    assert export_all(fig, tmp_path / "again.png", ["png"])["png"].name == "again.png"
    with pytest.raises(ExportError):
        figure_bytes(fig, "bmp")


def full_result() -> StatResult:
    res = RESULTS["sr_anova"].model_copy(deep=True)
    res.table = cases.T_SIZE
    res.why_chosen = "Three or more independent groups with roughly normal data."
    res.interpretation = "The groups differ."
    res.methods_text = "Groups were compared by one-way ANOVA followed by Tukey's multiple comparisons test"
    res.effect_sizes = [EffectSize(name="eta squared", value=0.85, ci_low=0.7, ci_high=0.9, magnitude="large")]
    res.assumptions = [AssumptionCheck(name="Normality (Shapiro-Wilk)", target="residuals", statistic=0.97, p_value=0.5, status="pass", detail="ok")]
    res.anova_table = [{"source": "Between", "ss": 100.0, "df": 3, "ms": 33.3, "f": 52.3, "p": 1.2e-11},
                       {"source": "Within", "ss": 17.8, "df": 28, "ms": 0.64, "f": None, "p": None}]
    res.n = {"Bulk mixing": 8, "Sonication": 8, "Extrusion": 8, "Microfluidic": 8}
    res.descriptives = [{"group": "Bulk mixing", "n": 8, "mean": 142.1}]
    res.warnings = ["Example warning"]
    res.data_notes = ["1 value(s) were explicitly excluded by the user before analysis: Sonication row 3."]
    res.citations = ["Tukey JW (1949) Biometrics 5:99-114"]
    res.comparison_family = "all 6 pairwise comparisons"
    return res


def test_markdown_report_contains_everything():
    res = full_result()
    md = stats_report_markdown([res], "Figure 1 statistics", SPECS["bar"])
    for needle in ("# Figure 1 statistics", "## One-way ANOVA", "`sr_anova`", "F(3, 28)", "<0.0001", "Why this test?",
                   "eta squared", "Multiple comparisons** (Tukey HSD)", "`sr_anova:c1`", "Bulk mixing vs Sonication",
                   "Assumption checks", "PASS", "Example warning", "explicitly excluded", "Tukey JW (1949)",
                   "Methods paragraph", "mean ± SEM", "ANOVA table", "Descriptive statistics"):
        assert needle in md, needle
    assert "No statistical analyses" in stats_report_markdown([], spec=SPECS["qq"])
    approx = res.model_copy(update={"approximate_data": True})
    assert "APPROXIMATE DATA" in stats_report_markdown([approx])


def test_csv_report_is_tidy_and_matches_results():
    import csv

    res = full_result()
    rows = list(csv.DictReader(io.StringIO(stats_report_csv([res]))))
    assert rows[0]["section"] == "test" and float(rows[0]["statistic"]) == 52.3
    comps = [r for r in rows if r["section"] == "comparison"]
    assert len(comps) == 6
    assert {r["name"] for r in comps} == {c.id for c in res.comparisons}
    assert float(comps[0]["p_adjusted"]) == res.comparisons[0].p_adjusted
    assert {r["section"] for r in rows} >= {"effect_size", "anova", "n", "assumption", "warning", "data_note"}
    assert len(stats_report_rows([res, None])) == len(rows)


def test_write_stats_report(tmp_path):
    out = write_stats_report([full_result()], tmp_path / "fig1.svg", spec=SPECS["bar"])
    assert out["markdown"].name == "fig1_stats.md" and out["csv"].name == "fig1_stats.csv"
    assert out["markdown"].read_text(encoding="utf-8").startswith("# Statistics report")


def test_methods_paragraph():
    res = full_result()
    text = methods_paragraph([res], SPECS["bar"])
    assert text.startswith("Data are presented as mean ± SEM, with individual data points shown.")
    assert "one-way ANOVA followed by Tukey's multiple comparisons test." in text
    assert "P < 0.05 was considered statistically significant." in text
    assert "explicitly excluded" in text and "n = 8" in text and "TorusPlot" in text
    assert text.count("one-way ANOVA followed") == 1 == methods_paragraph([res, res], SPECS["bar"]).count("one-way ANOVA followed")
    bare = StatResult(analysis="x", test_name="One-way ANOVA", p_value=0.01, correction="Tukey HSD", comparisons=res.comparisons)
    assert "Data were analyzed using one-way ANOVA followed by Tukey HSD multiple-comparisons testing." in methods_paragraph([bare])
    welch = StatResult(analysis="x", test_name="Welch's t test", p_value=0.01, alpha=0.01)
    both = methods_paragraph([bare, welch], include_software=False)
    assert "using Welch's t test." in both and "thresholds (alpha) are given" in both and "TorusPlot" not in both
    assert "digitized" in methods_paragraph([welch.model_copy(update={"approximate_data": True})])
    assert methods_paragraph([]) == ""


def test_data_presentation_sentences():
    def sentence(plot_type, **kw):
        return data_presentation_sentence(FigureSpec(plot_type=plot_type, data={"table": "t"}, **kw))

    assert "mean ± SD" in sentence("scatter")
    assert "interquartile range (box)" in sentence("box")
    assert "SuperPlots" in sentence("superplot") and "replicate means" in sentence("superplot")
    assert "Kaplan-Meier" in sentence("survival")
    assert sentence("bar", layers=[{"type": "mean_bar"}]) == "Data are presented as mean."
    assert sentence("paired") == "Individual data points are shown."
    assert sentence("heatmap") == "" and data_presentation_sentence(None) == ""
    assert "95% confidence interval" in sentence("bar", layers=[{"type": "mean_bar"}, {"type": "ci95_errorbar"}])
    assert "****" in significance_legend() and "****" not in significance_legend("three")
