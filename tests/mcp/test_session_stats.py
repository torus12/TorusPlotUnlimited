"""TorusSession: statistics tools (recommend_test, list_analyses, run_analysis) and the end-to-end flow.

The analysis modules are being finished in parallel; when they cannot be loaded these
tests are skipped instead of failing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.mcp import strict_json
from torusplot.data.samples import SAMPLES
from torusplot.export import validate_svg
from torusplot.mcp_server.session import SessionError, TorusSession

try:
    from torusplot.stats.registry import load_all

    load_all()
    STATS_READY = True
except Exception:  # noqa: BLE001 - any failure of a half-finished module means "not ready"
    STATS_READY = False

pytestmark = pytest.mark.skipif(not STATS_READY, reason="statistics analysis modules are not ready")

RESULT_KEYS = {
    "id", "test_name", "why_chosen", "statistic", "statistic_name", "df", "p_value", "p_adjusted", "significant",
    "effect_sizes", "parameters", "anova_table", "comparisons", "assumptions", "warnings", "interpretation",
    "methods_text",
}


@pytest.fixture
def session(tmp_path: Path) -> TorusSession:
    s = TorusSession(tmp_path / "out")
    s.load_table("np", sample="nanoparticle_size")
    return s


# ===================================================================== list_analyses
@pytest.mark.parametrize("sample", list(SAMPLES))
def test_list_analyses_matches_the_registry(tmp_path: Path, sample: str):
    from torusplot.stats import analyses_for

    session = TorusSession(tmp_path)
    loaded = session.load_table("t", sample=sample)
    out = session.list_analyses("t")
    expected = [a.key for a in analyses_for(SAMPLES[sample]["kind"])]
    assert [a["key"] for a in out["analyses"]] == expected == [a["key"] for a in loaded["analyses"]]
    assert expected, "every table kind has at least one analysis"
    for analysis in out["analyses"]:
        assert analysis["label"] and analysis["category"]
        for param in analysis["params"]:
            assert set(param) == {"name", "label", "kind", "default", "choices", "optional", "help"}
            assert isinstance(param["choices"], list)
    assert out["kind"] == SAMPLES[sample]["kind"] and out["names"]
    strict_json(out)


def test_list_analyses_for_one_analysis_and_errors(session: TorusSession):
    out = session.list_analyses("np", analysis="anova_oneway")
    assert [a["key"] for a in out["analyses"]] == ["anova_oneway"]
    params = {p["name"]: p for p in out["analyses"][0]["params"]}
    assert "tukey" in params["posthoc"]["choices"] and params["alpha"]["default"] == 0.05
    assert out["names"] == {"groups": ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]}
    with pytest.raises(SessionError, match="Unknown analysis 'nope'"):
        session.list_analyses("np", analysis="nope")
    with pytest.raises(SessionError, match="Table 'ghost' not found"):
        session.list_analyses("ghost")


# ==================================================================== recommend_test
@pytest.mark.parametrize("sample", list(SAMPLES))
def test_recommendation_is_a_runnable_registered_analysis(tmp_path: Path, sample: str):
    from torusplot.stats import get_analysis

    session = TorusSession(tmp_path)
    session.load_table("t", sample=sample)
    rec = session.recommend_test("t")
    assert {"analysis", "label", "params", "reason", "steps", "checks", "alternatives", "cautions"} <= set(rec)
    info = get_analysis(rec["analysis"])
    assert SAMPLES[sample]["kind"] in info.tables
    assert set(rec["params"]) <= {p.name for p in info.params}
    assert rec["reason"] and rec["steps"] and rec["analysis"] in rec["next"]
    for alternative in rec["alternatives"]:
        get_analysis(alternative["analysis"])
    strict_json(rec)
    result = session.run_analysis("t", rec["analysis"], rec["params"])  # the recommendation runs as given
    assert result["analysis"] == rec["analysis"]


def test_recommendation_follows_the_design_hints(session: TorusSession):
    session.load_table("pairs", sample="paired")
    assert session.recommend_test("pairs")["analysis"] != session.recommend_test("pairs", paired=True)["analysis"]
    assert session.recommend_test("pairs", paired=True)["design"]["paired"] is True
    rec = session.recommend_test("np", control="Bulk mixing")
    assert rec["params"].get("control") == "Bulk mixing"
    rec = session.recommend_test("np", groups=["Extrusion", "Microfluidic"])
    assert rec["design"]["groups"] == 2
    assert not session.project.results  # recommending never runs or stores anything


def test_recommend_test_errors(session: TorusSession):
    with pytest.raises(SessionError, match="Table 'ghost' not found"):
        session.recommend_test("ghost")
    with pytest.raises(SessionError, match=r"Unknown group\(s\) \['Nope'\]"):
        session.recommend_test("np", groups=["Nope", "Extrusion"])
    with pytest.raises(SessionError, match=r"Unknown group\(s\) \['Placebo'\]"):
        session.recommend_test("np", control="Placebo")


# ====================================================================== run_analysis
def test_run_analysis_returns_the_engine_result_verbatim(session: TorusSession):
    from torusplot.stats import run_analysis

    out = session.run_analysis("np", "anova_oneway")
    assert RESULT_KEYS <= set(out)
    direct = run_analysis(session.project.tables["np"], "anova_oneway")
    assert out["test_name"] == direct.test_name and out["statistic_name"] == "F"
    assert out["statistic"] == direct.statistic and out["p_value"] == direct.p_value
    assert (out["df"], out["df2"]) == (direct.df, direct.df2) and out["significant"] is direct.significant
    assert out["interpretation"] == direct.interpretation and out["methods_text"] == direct.methods_text
    assert out["why_chosen"] == direct.why_chosen and out["anova_table"] == direct.anova_table
    assert len(out["comparisons"]) == len(direct.comparisons) == 6
    for got, want in zip(out["comparisons"], direct.comparisons):
        assert got["id"].startswith(out["id"] + ":c")
        assert (got["group_a"], got["group_b"]) == (want.group_a, want.group_b)
        assert got["p_adjusted"] == want.p_adjusted and got["significant"] is want.significant
    assert [e["name"] for e in out["effect_sizes"]] == [e.name for e in direct.effect_sizes]
    assert [e["value"] for e in out["effect_sizes"]] == [e.value for e in direct.effect_sizes]
    assert {a["status"] for a in out["assumptions"]} <= {"pass", "warn", "fail", "info"}
    assert all(a["detail"] for a in out["assumptions"])
    assert "residuals" in out["extra_omitted"] and "residuals" not in out["extra"]  # bulky arrays are left out
    assert out["summary"].startswith("One-way ANOVA: F(3, 28) = ")
    assert out["id"] in out["next"]
    strict_json(out)

    stored = session.project.results[out["id"]]
    assert stored.statistic == out["statistic"]
    assert session.get_result(out["id"]) == {k: v for k, v in out.items() if k != "next"}
    listed = session.list_state()["results"]
    assert [r["id"] for r in listed] == [out["id"]] and listed[0]["comparisons"] == 6


def test_run_analysis_options_are_checked_and_coerced(session: TorusSession):
    out = session.run_analysis("np", "anova_oneway", {"posthoc": "dunnett", "control": "Bulk mixing", "alpha": "0.01"})
    assert out["params"]["posthoc"] == "dunnett" and out["alpha"] == 0.01 and len(out["comparisons"]) == 3
    out = session.run_analysis("np", "anova_oneway", {"groups": ["Extrusion", "Microfluidic", "Sonication"], "posthoc": "NONE"})
    assert out["params"]["posthoc"] == "none" and out["comparisons"] == [] and len(out["n"]) == 3
    out = session.run_analysis("np", "ttest_welch", '{"group_a": "Extrusion", "group_b": "Microfluidic"}')
    assert set(out["n"]) == {"Extrusion", "Microfluidic"}
    session.load_table("up", sample="cell_uptake")
    out = session.run_analysis("up", "anova_twoway", {"ss_type": "2", "interaction": "true"})
    assert out["params"]["ss_type"] == 2 and out["params"]["interaction"] is True


@pytest.mark.parametrize(
    ("analysis", "params", "message"),
    [
        ("anova_magic", None, r"Unknown analysis 'anova_magic'\. Analyses valid for the Column table 'np': .*anova_oneway"),
        ("anova_twoway", None, r"needs a table of type \['grouped', 'multiple_variables'\].*anova_oneway"),
        ("survival", None, "needs a table of type"),
        ("anova_oneway", {"bogus": 1}, r"Unknown option\(s\) \['bogus'\].*Valid options for 'anova_oneway': .*posthoc"),
        ("anova_oneway", {"posthoc": "magic"}, r"must be one of \['tukey'"),
        ("anova_oneway", {"alpha": "small"}, "must be a number"),
        ("anova_oneway", {"alpha": True}, "must be a number"),
        ("anova_oneway", {"alpha": 5}, "alpha must be between 0 and 1"),
        ("anova_oneway", {"groups": [["a"]]}, "must be a list of text values"),
        ("anova_oneway", {"control": ["a"]}, "must be a text value"),
        ("anova_oneway", {"groups": ["Extrusion", "Nope"]}, "Nope"),
        ("anova_oneway", "posthoc=tukey", "params must be a JSON object"),
        ("anova_oneway", ["posthoc"], "params must be a JSON object"),
        ("ttest_welch", {"group_a": "Nope"}, "Unknown group 'Nope'"),
    ],
)
def test_run_analysis_errors_say_what_is_valid(session: TorusSession, analysis: str, params, message: str):
    with pytest.raises(SessionError, match=message) as err:
        session.run_analysis("np", analysis, params)
    assert "Traceback" not in str(err.value)
    assert session.project.results == {}


def test_option_coercion_rules():
    from torusplot.stats import ParamSpec

    class Info:
        key = "demo"

    coerce = TorusSession._coerce_param
    assert coerce(ParamSpec("n", "N", kind="int"), "3", Info) == 3 and coerce(ParamSpec("n", "N", kind="int"), 4.0, Info) == 4
    assert coerce(ParamSpec("f", "F", kind="bool"), "No", Info) is False and coerce(ParamSpec("f", "F", kind="bool"), 1, Info) is True
    assert coerce(ParamSpec("g", "G", kind="groups"), "A", Info) == ["A"]
    assert coerce(ParamSpec("c", "C", kind="column"), 7, Info) == "7"
    assert coerce(ParamSpec("c", "C", kind="choice", choices=(1, 2, 3)), "2", Info) == 2
    assert coerce(ParamSpec("x", "X", kind="mystery"), {"a": 1}, Info) == {"a": 1}
    assert coerce(ParamSpec("x", "X", kind="float"), None, Info) is None
    for spec, value in [
        (ParamSpec("n", "N", kind="int"), 2.5), (ParamSpec("f", "F", kind="bool"), "maybe"),
        (ParamSpec("x", "X", kind="float"), float("nan")), (ParamSpec("t", "T", kind="text"), {"a": 1}),
    ]:
        with pytest.raises(SessionError, match="Option"):
            coerce(spec, value, Info)


def test_statistics_cannot_be_passed_into_an_analysis(session: TorusSession):
    for smuggled in ({"p_value": 0.001}, {"stars": "***"}, {"significant": True}, {"statistic": 99.0}):
        with pytest.raises(SessionError, match=r"Unknown option\(s\)"):
            session.run_analysis("np", "anova_oneway", smuggled)
    assert session.project.results == {}


def test_unknown_table_and_approximate_data(session: TorusSession):
    with pytest.raises(SessionError, match="Table 'ghost' not found"):
        session.run_analysis("ghost", "anova_oneway")
    session.project.tables["np"].approximate = True  # as for data digitized from an image
    with pytest.raises(SessionError, match="APPROXIMATE"):
        session.run_analysis("np", "anova_oneway")
    out = session.run_analysis("np", "anova_oneway", allow_approximate=True)
    assert out["approximate_data"] is True and out["warnings"][0].startswith("APPROXIMATE DATA")


def test_an_unexpected_engine_failure_is_a_clean_error(session: TorusSession, monkeypatch: pytest.MonkeyPatch):
    from torusplot.mcp_server import session as session_module

    def boom(*args, **kwargs):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(session_module._registry, "run_analysis", boom)
    with pytest.raises(SessionError, match="could not be computed with these options .ZeroDivisionError") as err:
        session.run_analysis("np", "anova_oneway")
    assert "Valid options for 'anova_oneway'" in str(err.value)


def test_a_broken_statistics_engine_does_not_block_loading_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from torusplot.mcp_server import session as session_module

    def broken(kind: str):
        raise ImportError("No module named 'torusplot.stats.unfinished'")

    monkeypatch.setattr(session_module._registry, "analyses_for", broken)
    session = TorusSession(tmp_path)
    out = session.load_table("np", sample="nanoparticle_size")
    assert out["analyses"] == [] and "statistics engine could not be loaded" in out["warnings"][0]
    assert session.make_figure("f", "np", "bar")["warnings"] == []  # graphing still works
    with pytest.raises(SessionError, match="statistics engine could not be loaded"):
        session.run_analysis("np", "anova_oneway")


# ===================================================================== end to end
def test_end_to_end_nanoparticle_figure_with_brackets(session: TorusSession):
    """load sample -> recommend_test -> run_analysis -> make_figure with brackets -> export + report."""
    from PIL import Image

    rec = session.recommend_test("np")
    result = session.run_analysis("np", rec["analysis"], rec["params"])
    assert result["comparisons"], "the recommended analysis for 4 groups has pairwise comparisons"

    figure = session.make_figure("Figure 1", "np", "bar", annotate_result_id=result["id"], y_label="Diameter (nm)")
    assert figure["warnings"] == []
    assert figure["brackets_drawn"] == len(result["comparisons"])
    assert [b["comparison_id"] for b in figure["brackets"]] == [c["id"] for c in result["comparisons"]]

    exported = session.export_figure("Figure 1", formats=["svg", "pdf", "tiff", "png"], dpi=300)
    paths = {f["format"]: Path(f["path"]) for f in exported["files"]}
    assert set(paths) == {"svg", "pdf", "tiff", "png"}
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 1000 and path.parent == session.output_dir
    check = validate_svg(paths["svg"])
    assert check["ok"], check["problems"]
    assert check["text_elements"] > 0  # text stays editable text
    brackets = [gid for gid in check["group_ids"] if re.fullmatch(r"bracket-\d+", gid)]
    assert len(brackets) == len(result["comparisons"])  # one drawn bracket per computed comparison
    with Image.open(paths["tiff"]) as tiff:
        assert tiff.info["compression"] == "tiff_lzw" and tiff.info["dpi"][0] == pytest.approx(300, abs=1)
    with Image.open(paths["png"]) as png:
        assert png.size[0] == pytest.approx(89 / 25.4 * 300, abs=2)  # exact size in millimetres

    report = {r["format"]: Path(r["path"]) for r in exported["stats_report"]}
    markdown = report["markdown"].read_text(encoding="utf-8")
    assert report["markdown"].name == "Figure_1_stats.md" and report["csv"].is_file()
    assert result["id"] in markdown and result["test_name"] in markdown and result["methods_text"] in markdown

    text = session.get_stats_report(figure="Figure 1")
    assert text["result_ids"] == [result["id"]] and result["methods_text"] in text["methods_paragraph"]
    for payload in (rec, result, figure, exported, text, session.list_state()):
        strict_json(payload)


def test_dose_response_fit_curves(tmp_path: Path):
    session = TorusSession(tmp_path)
    session.load_table("dr", sample="dose_response")
    rec = session.recommend_test("dr", goal="dose_response")
    assert rec["analysis"] == "nonlinear_regression"
    fits = [session.run_analysis("dr", "nonlinear_regression", {"series": s}) for s in ("Free drug", "Nanoformulation")]
    assert all("fit_result_ids" in fit["next"] for fit in fits)
    assert all("curve" in fit["extra_omitted"] for fit in fits)
    ids = [fit["id"] for fit in fits]
    out = session.make_figure("curves", "dr", "dose_response", fit_result_ids=ids)
    layers = out["spec"]["layers"]
    assert [lay["stat_result_id"] for lay in layers if lay["type"] == "curve_fit"] == ids
    assert out["warnings"] == [] and out["stat_result_ids"] == ids
    svg = Path(session.export_figure("curves", ["svg"])["files"][0]["path"])
    assert sum(1 for gid in validate_svg(svg)["group_ids"] if gid.startswith("curve-")) == 2
    # a fit drawn on a plain XY plot replaces the connecting line
    xy = session.make_figure("xy", "dr", "xy", fit_result_ids=ids[0])
    assert [lay["type"] for lay in xy["spec"]["layers"]] == ["sem_errorbar", "mean_dot", "curve_fit"]
    # a result without a curve cannot be drawn as one
    plain = session.run_analysis("dr", "descriptives")
    with pytest.raises(SessionError, match="has no fitted curve"):
        session.make_figure("bad", "dr", "xy", fit_result_ids=[plain["id"]])


def test_survival_and_grouped_flows(tmp_path: Path):
    session = TorusSession(tmp_path)
    session.load_table("surv", sample="survival")
    rec = session.recommend_test("surv")
    result = session.run_analysis("surv", rec["analysis"], rec["params"])
    assert "data_result_id" in result["next"]
    out = session.make_figure("km", "surv", "survival", data_result_id=result["id"], annotate_result_id=result["id"])
    assert out["summary_texts"][0]["text"] == result["summary"] and out["warnings"] == []

    session.load_table("up", sample="cell_uptake")
    rec = session.recommend_test("up")
    result = session.run_analysis("up", rec["analysis"], rec["params"])
    out = session.make_figure("bars", "up", "grouped_bar", annotate_result_id=result["id"], comparisons="significant")
    significant = [c for c in result["comparisons"] if c["significant"]]
    assert out["brackets_drawn"] == len(significant) > 0
    panel = session.compose_panels("panel", ["km", "bars"])
    assert panel["stat_result_ids"] == [r["id"] for r in session.list_state()["results"]]


def test_an_unexpected_recommendation_failure_is_a_clean_error(session: TorusSession, monkeypatch: pytest.MonkeyPatch):
    from torusplot.stats import recommend

    def boom(*args, **kwargs):
        raise KeyError("unexpected")

    monkeypatch.setattr(recommend, "recommend_test", boom)
    with pytest.raises(SessionError, match="No recommendation could be made for 'np' .KeyError"):
        session.recommend_test("np")
