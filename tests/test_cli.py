import json

import pytest

from torusplot.cli import _parse_value, main


def test_samples_and_analyses(capsys):
    assert main(["samples"]) == 0
    out = capsys.readouterr().out
    assert "nanoparticle_size" in out and "SYNTHETIC" in out
    assert main(["analyses", "--kind", "contingency"]) == 0
    out = capsys.readouterr().out
    assert "fisher_exact" in out and "anova_oneway" not in out


def test_recommend_and_analyze(capsys, tmp_path):
    assert main(["recommend", "--sample", "paired", "--paired"]) == 0
    out = capsys.readouterr().out
    assert "Recommended:" in out and "--analysis" in out
    assert main(["analyze", "--sample", "nanoparticle_size", "--analysis", "anova_oneway", "--param", "posthoc=tukey",
                 "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "One-way ANOVA" in out and "Multiple comparisons" in out
    assert (tmp_path / "anova_oneway_stats.md").exists() and (tmp_path / "anova_oneway_stats.csv").exists()


def test_analyze_file_input(capsys, tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("A,B\n1,2\n2,4\n3,5\n4,7\n5,8\n", encoding="utf-8")
    assert main(["analyze", "--file", str(csv), "--kind", "column", "--analysis", "ttest_welch"]) == 0
    assert "Welch" in capsys.readouterr().out
    assert main(["analyze", "--file", str(csv), "--kind", "column", "--analysis", "nope"]) == 2
    assert "Unknown analysis" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["analyze", "--analysis", "ttest_welch"])
    with pytest.raises(SystemExit):
        main(["analyze", "--sample", "paired", "--analysis", "ttest_paired", "--param", "broken"])


@pytest.mark.slow
def test_demo_then_render_from_project(capsys, tmp_path):
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    capsys.readouterr()
    for name in ("uptake_over_time.svg", "uptake_over_time.pdf", "uptake_over_time.tif", "uptake_over_time.png",
                 "uptake_over_time_stats.md", "uptake_over_time_stats.csv", "demo.tplot"):
        assert (tmp_path / "demo" / name).exists(), name
    assert main(["render", "--project", str(tmp_path / "demo" / "demo.tplot"), "--out", str(tmp_path / "again"),
                 "--formats", "svg"]) == 0
    first = (tmp_path / "demo" / "uptake_over_time.svg").read_bytes()
    second = (tmp_path / "again" / "Uptake_over_time.svg").read_bytes()
    assert first == second  # a project file re-renders identically
    with pytest.raises(SystemExit, match="not a figure"):
        main(["render", "--project", str(tmp_path / "demo" / "demo.tplot"), "--figure", "missing"])


def test_schema_and_value_parsing(capsys):
    assert main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "FigureSpec"
    assert _parse_value("true") is True and _parse_value("no") is False and _parse_value("none") is None
    assert _parse_value("0.01") == 0.01 and _parse_value("a,b") == ["a", "b"] and _parse_value("tukey") == "tukey"
    assert _parse_value('["A", "B"]') == ["A", "B"]
