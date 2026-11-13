"""Package-level guarantees of the AI layer: cheap import, public API, clean sources."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import torusplot.ai as ai

ROOT = Path(__file__).resolve().parents[2]
AI_DIR = ROOT / "torusplot" / "ai"
OWNED = [
    *sorted(AI_DIR.rglob("*.py")),
    *sorted(AI_DIR.glob("recorded/*.json")),
    *sorted((ROOT / "tests" / "ai").glob("*.py")),
    ROOT / "tools" / "build_prompt_template.py",
    ROOT / "docs" / "PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md",
]


def test_import_is_cheap_and_pulls_in_no_network_or_sdk_modules():
    code = (
        "import sys, json\n"
        "import torusplot.ai\n"
        "import torusplot.ai.providers\n"
        "import torusplot.ai.settings\n"
        "watched = ['anthropic', 'urllib.request', 'http.client', 'ssl', 'skimage', 'matplotlib', 'PIL',\n"
        "           'pandas', 'torusplot.ai.providers.ollama', 'torusplot.ai.providers.openai',\n"
        "           'torusplot.ai.providers.gemini', 'torusplot.ai.providers.anthropic', 'torusplot.ai.match',\n"
        "           'torusplot.ai.digitize']\n"
        "print(json.dumps([m for m in watched if m in sys.modules]))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == []


def test_backends_do_not_import_network_code_at_import_time():
    code = (
        "import sys, json\n"
        "import torusplot.ai.providers.ollama, torusplot.ai.providers.openai\n"
        "import torusplot.ai.providers.gemini, torusplot.ai.providers.anthropic\n"
        "print(json.dumps([m for m in ('anthropic', 'urllib.request', 'http.client') if m in sys.modules]))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == []


def test_public_api_is_re_exported():
    for name in ai.__all__:
        assert getattr(ai, name) is not None, name
    assert set(ai.__all__) <= set(dir(ai))
    from torusplot.ai.describe import describe_to_spec
    from torusplot.ai.digitize import digitize, to_table
    from torusplot.ai.match import match_figure
    from torusplot.ai.outside import import_outside_instructions

    assert ai.describe_to_spec is describe_to_spec
    assert ai.import_outside_instructions is import_outside_instructions
    assert ai.match_figure is match_figure
    assert ai.digitize_image is digitize and ai.to_table is to_table
    for expected in ("AISettings", "load_settings", "get_provider", "MockProvider", "ProviderError", "extract_json",
                     "SpecOutcome", "outside_prompt_template", "extract_style", "style_to_spec", "dominant_colors",
                     "analyze_image", "ssim_score", "apply_style_patch", "critique_patch", "AxisCalibration",
                     "detect_plot_area", "candidate_colors", "pixel_to_data", "RECOMMENDED_OLLAMA_MODELS"):
        assert expected in ai.__all__, expected
    with pytest.raises(AttributeError):
        _ = ai.no_such_thing


def test_default_settings_keep_ai_off():
    assert ai.AISettings().enabled is False
    assert ai.active_provider() is None
    with pytest.raises(ai.ProviderError):
        ai.get_provider(ai.AISettings())


def test_sources_contain_no_forbidden_trademarks_or_secrets():
    forbidden = ["graph" + "pad", "pri" + "sm"]
    for path in OWNED:
        text = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in text, f"{path} mentions a forbidden trademark"
    key_like = ("sk-" + "ant-", "AIza" + "Sy", "sk-" + "proj-")
    for path in OWNED:
        text = path.read_text(encoding="utf-8")
        assert not [k for k in key_like if k in text], path


def test_recorded_responses_are_well_formed():
    files = sorted(AI_DIR.glob("recorded/*.json"))
    assert len(files) >= 5
    tasks = set()
    for path in files:
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert {"task", "response"} <= set(entry) and isinstance(entry["response"], dict), path.name
        assert isinstance(entry.get("keywords", []), list)
        tasks.add(entry["task"])
    assert tasks == {"describe_figure", "extract_style", "critique_figure"}
