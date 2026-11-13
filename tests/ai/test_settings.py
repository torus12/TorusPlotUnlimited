import json

import pytest

from torusplot.ai.settings import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_TEXT_MODEL,
    DEFAULT_OLLAMA_VISION_MODEL,
    RECOMMENDED_OLLAMA_MODELS,
    AISettings,
    load_settings,
    parse_dotenv,
    read_dotenv,
    redact,
)

KEY = "sk-test-SECRET-1234567890abcdef"


def test_defaults_are_off_and_offline():
    s = AISettings()
    assert s.enabled is False
    assert s.provider == "mock"
    assert s.max_retries == 3
    assert s.max_feedback_loops == 3
    assert s.ollama_text_model == DEFAULT_OLLAMA_TEXT_MODEL == "qwen2.5:7b"
    assert s.ollama_vision_model == DEFAULT_OLLAMA_VISION_MODEL == "granite3.2-vision"
    assert s.anthropic_model == DEFAULT_ANTHROPIC_MODEL == "claude-opus-5-5"
    # no built-in default for services whose model names change
    assert s.gemini_model == "" and s.openai_model == ""
    assert not s.has_key("openai") and not s.has_key("gemini") and not s.has_key("anthropic")


def test_load_settings_with_empty_env_is_off():
    s = load_settings(env={})
    assert s.enabled is False and s.provider == "mock"
    assert s.ollama_host == "http://localhost:11434"


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("on", True),
                                            ("0", False), ("false", False), ("", False), ("maybe", False)])
def test_enabled_flag_parsing(value, expected):
    assert load_settings(env={"TORUSPLOT_AI_ENABLED": value}).enabled is expected


def test_load_settings_reads_every_variable():
    env = {
        "TORUSPLOT_AI_ENABLED": "1",
        "TORUSPLOT_AI_PROVIDER": "Ollama",
        "OLLAMA_HOST": "127.0.0.1:11434/",
        "TORUSPLOT_OLLAMA_TEXT_MODEL": "qwen3",
        "TORUSPLOT_OLLAMA_VISION_MODEL": "minicpm-v",
        "GEMINI_API_KEY": "g-" + KEY,
        "ANTHROPIC_API_KEY": "a-" + KEY,
        "OPENAI_API_KEY": "o-" + KEY,
        "TORUSPLOT_GEMINI_MODEL": "gemini-x",
        "TORUSPLOT_ANTHROPIC_MODEL": "claude-sonnet-5-5",
        "TORUSPLOT_OPENAI_MODEL": "gpt-x",
        "OPENAI_BASE_URL": "http://localhost:8080/v1/",
        "TORUSPLOT_AI_TIMEOUT": "30",
        "TORUSPLOT_AI_MAX_RETRIES": "1",
        "TORUSPLOT_AI_MAX_FEEDBACK_LOOPS": "9",
    }
    s = load_settings(env=env)
    assert s.enabled and s.provider == "ollama"
    assert s.ollama_host == "http://127.0.0.1:11434"
    assert (s.ollama_text_model, s.ollama_vision_model) == ("qwen3", "minicpm-v")
    assert (s.gemini_model, s.anthropic_model, s.openai_model) == ("gemini-x", "claude-sonnet-5-5", "gpt-x")
    assert s.openai_base_url == "http://localhost:8080/v1"
    assert s.api_key("gemini") == "g-" + KEY
    assert s.api_key("anthropic") == "a-" + KEY
    assert s.api_key("openai") == "o-" + KEY
    assert s.timeout == 30.0 and s.max_retries == 1
    assert s.max_feedback_loops == 3  # hard cap
    assert s.key_env_name("openai") == "OPENAI_API_KEY"
    assert s.api_key("ollama") is None and s.key_env_name("ollama") is None


def test_empty_values_fall_back_to_defaults():
    # .env.example ships with empty model lines: they must not blank the defaults
    s = load_settings(env={"TORUSPLOT_OLLAMA_TEXT_MODEL": "", "TORUSPLOT_ANTHROPIC_MODEL": "  ", "OPENAI_API_KEY": ""})
    assert s.ollama_text_model == "qwen2.5:7b"
    assert s.anthropic_model == "claude-opus-5-5"
    assert not s.has_key("openai")


def test_parse_dotenv():
    text = """
    # a comment
    TORUSPLOT_AI_ENABLED=1
    export TORUSPLOT_AI_PROVIDER = "openai"
    OPENAI_API_KEY='sk-abc # not a comment'
    TORUSPLOT_OPENAI_MODEL=gpt-x   # trailing comment
    EMPTY=
    not a valid line
    BAD KEY=1
    =nokey
    """
    env = parse_dotenv(text)
    assert env == {
        "TORUSPLOT_AI_ENABLED": "1",
        "TORUSPLOT_AI_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-abc # not a comment",
        "TORUSPLOT_OPENAI_MODEL": "gpt-x",
        "EMPTY": "",
    }


def test_dotenv_file_and_precedence(tmp_path):
    path = tmp_path / ".env"
    path.write_text("TORUSPLOT_AI_ENABLED=1\nTORUSPLOT_AI_PROVIDER=gemini\nGEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert read_dotenv(tmp_path / "missing.env") == {}
    s = load_settings(env={}, dotenv_path=path)
    assert s.enabled and s.provider == "gemini" and s.api_key() == "from-file"
    # real environment variables win over the file
    s = load_settings(env={"TORUSPLOT_AI_PROVIDER": "openai", "GEMINI_API_KEY": "from-env"}, dotenv_path=path)
    assert s.provider == "openai" and s.api_key("gemini") == "from-env"
    # explicit env without dotenv_path never reads a file
    assert load_settings(env={}).enabled is False


def test_load_settings_from_real_environment(tmp_path, monkeypatch):
    for name in ("TORUSPLOT_AI_ENABLED", "TORUSPLOT_AI_PROVIDER", "OPENAI_API_KEY", "TORUSPLOT_OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_settings().enabled is False  # nothing set, no .env
    (tmp_path / ".env").write_text("TORUSPLOT_AI_ENABLED=true\nTORUSPLOT_OPENAI_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("TORUSPLOT_AI_PROVIDER", "openai")
    s = load_settings()
    assert s.enabled and s.provider == "openai" and s.openai_model == "from-dotenv"


def test_api_keys_are_masked_everywhere():
    s = load_settings(env={"OPENAI_API_KEY": KEY, "GEMINI_API_KEY": KEY + "-g", "ANTHROPIC_API_KEY": KEY + "-a"})
    for text in (repr(s), str(s), f"{s}", s.model_dump_json(), json.dumps(s.public_dict()), repr(s.model_dump())):
        assert KEY not in text
    assert s.public_dict()["openai_api_key"] == "set"
    assert AISettings().public_dict()["openai_api_key"] == "not set"
    assert "enabled=False" in repr(s)
    # the real value is only available on explicit request
    assert s.api_key("openai") == KEY
    assert set(s.secrets()) == {KEY, KEY + "-g", KEY + "-a"}


def test_redact():
    s = load_settings(env={"OPENAI_API_KEY": KEY})
    assert s.redact(f"Bearer {KEY} failed") == "Bearer *** failed"
    assert redact("abc abcdef", ["abc", "abcdef"]) == "*** ***"
    assert redact(ValueError("x"), []) == "x"


def test_invalid_settings_errors_do_not_leak_keys():
    with pytest.raises(ValueError) as err:
        load_settings(env={"TORUSPLOT_AI_PROVIDER": "skynet", "OPENAI_API_KEY": KEY})
    assert "skynet" in str(err.value) and KEY not in str(err.value)
    with pytest.raises(ValueError, match="TORUSPLOT_AI_TIMEOUT must be a number"):
        load_settings(env={"TORUSPLOT_AI_TIMEOUT": "soon"})
    with pytest.raises(ValueError) as err:
        load_settings(env={"TORUSPLOT_AI_MAX_RETRIES": "99", "ANTHROPIC_API_KEY": KEY})
    assert "max_retries" in str(err.value) and KEY not in str(err.value)
    with pytest.raises(ValueError):
        AISettings(provider="nope")


def test_recommended_ollama_models_are_permissively_licensed():
    tags = {m["tag"] for m in RECOMMENDED_OLLAMA_MODELS}
    assert DEFAULT_OLLAMA_TEXT_MODEL in tags and DEFAULT_OLLAMA_VISION_MODEL in tags
    for model in RECOMMENDED_OLLAMA_MODELS:
        assert model["license"] in {"Apache-2.0", "MIT"}
        assert model["source"].startswith("https://")
        assert model["role"] in {"text", "vision"}
        assert model["pull"] == f"ollama pull {model['tag']}"
    # sizes the vendor does not release under Apache-2.0 must never be suggested
    assert not tags & {"qwen2.5:3b", "qwen2.5:72b", "llava"}
