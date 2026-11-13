"""Provider tests. No network: ``urllib.request.urlopen`` and the ``anthropic`` SDK are faked."""

import base64
import importlib.machinery
import io
import json
import sys
import types
import urllib.error
import urllib.request

import pytest

from tests.ai import T_DOSE, T_SIZE, flat_figure, jpeg_of, png_of, sample_tables
from torusplot.ai.prompts import describe_user_prompt
from torusplot.ai.providers import (
    MockProvider,
    Provider,
    ProviderError,
    active_provider,
    get_provider,
)
from torusplot.ai.providers.anthropic import INSTALL_HINT, AnthropicProvider
from torusplot.ai.providers.base import dig, http_json, image_media_type
from torusplot.ai.providers.gemini import GeminiProvider
from torusplot.ai.providers.ollama import OllamaProvider
from torusplot.ai.providers.openai import OpenAIProvider
from torusplot.ai.settings import AISettings, load_settings

KEY = "sk-test-SECRET-abcdef0123456789"
PNG = png_of(flat_figure())
JPEG = jpeg_of(flat_figure())


class FakeResponse:
    def __init__(self, payload):
        self._data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHTTP:
    """Replacement for ``urllib.request.urlopen`` that records requests and replays answers."""

    def __init__(self, monkeypatch, *answers):
        self.answers = list(answers)
        self.requests = []
        monkeypatch.setattr(urllib.request, "urlopen", self)

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append(
            {"url": request.full_url, "method": request.get_method(), "body": body, "timeout": timeout,
             "headers": {k.lower(): v for k, v in request.header_items()}}
        )
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)


def http_error(code, body=""):
    return urllib.error.HTTPError("http://x", code, "error", {}, io.BytesIO(body.encode("utf-8")))


# ====================================================================== base
def test_image_media_type():
    assert image_media_type(PNG) == "image/png"
    assert image_media_type(JPEG) == "image/jpeg"
    with pytest.raises(ProviderError, match="PNG or JPEG"):
        image_media_type(b"GIF89a....")


def test_dig():
    data = {"a": [{"b": 1}]}
    assert dig(data, "a", 0, "b") == 1
    assert dig(data, "a", 5, "b") is None and dig(data, "x") is None and dig(data, "a", "b") is None


def test_http_json_get_post_and_errors(monkeypatch):
    http = FakeHTTP(monkeypatch, {"ok": 1}, {"ok": 2})
    assert http_json("http://h/x", timeout=3) == {"ok": 1}
    assert http.requests[0]["method"] == "GET" and http.requests[0]["timeout"] == 3
    assert http_json("http://h/x", {"q": 1}, headers={"X-K": "v"}) == {"ok": 2}
    assert http.requests[1]["method"] == "POST" and http.requests[1]["body"] == {"q": 1}
    assert http.requests[1]["headers"]["content-type"] == "application/json"
    assert http.requests[1]["headers"]["x-k"] == "v"

    with pytest.raises(ProviderError, match="http:// or https://"):
        http_json("file:///etc/passwd")
    FakeHTTP(monkeypatch, b"<html>not json</html>")
    with pytest.raises(ProviderError, match="not valid JSON"):
        http_json("http://h/x")
    FakeHTTP(monkeypatch, TimeoutError())
    with pytest.raises(ProviderError, match="did not answer within 5 seconds"):
        http_json("http://h/x", timeout=5)
    FakeHTTP(monkeypatch, urllib.error.URLError("timed out"))
    with pytest.raises(ProviderError, match="did not answer"):
        http_json("http://h/x")
    FakeHTTP(monkeypatch, urllib.error.URLError(f"refused for {KEY}"))
    with pytest.raises(ProviderError) as err:
        http_json("http://h/x", secrets=[KEY], service="Svc")
    assert "Svc could not be reached" in str(err.value) and KEY not in str(err.value)
    FakeHTTP(monkeypatch, OSError(f"socket {KEY}"))
    with pytest.raises(ProviderError) as err:
        http_json("http://h/x", secrets=[KEY])
    assert KEY not in str(err.value)
    FakeHTTP(monkeypatch, http_error(429, f'{{"error": "quota for {KEY}"}}'))
    with pytest.raises(ProviderError) as err:
        http_json("http://h/x", secrets=[KEY])
    assert "HTTP 429" in str(err.value) and "rate limit" in str(err.value) and KEY not in str(err.value)
    FakeHTTP(monkeypatch, http_error(500))
    with pytest.raises(ProviderError, match="HTTP 500"):
        http_json("http://h/x")


def test_get_provider_is_off_by_default():
    with pytest.raises(ProviderError, match="turned off"):
        get_provider(AISettings())
    with pytest.raises(ProviderError, match="turned off"):
        get_provider()
    assert active_provider() is None
    assert active_provider(AISettings(enabled=False, provider="openai")) is None
    assert isinstance(get_provider(AISettings(), allow_disabled=True), MockProvider)


@pytest.mark.parametrize("name,cls", [("mock", MockProvider), ("ollama", OllamaProvider), ("gemini", GeminiProvider),
                                      ("anthropic", AnthropicProvider), ("openai", OpenAIProvider)])
def test_get_provider_builds_each_backend(name, cls):
    provider = get_provider(AISettings(enabled=True, provider=name))
    assert isinstance(provider, cls) and isinstance(provider, Provider)
    assert provider.name == name and provider.supports_vision
    assert name in repr(provider)
    assert isinstance(active_provider(AISettings(enabled=True, provider=name)), cls)


def test_get_provider_unknown_name():
    settings = AISettings(enabled=True).model_copy(update={"provider": "hal9000"})
    with pytest.raises(ProviderError, match="Unknown AI provider"):
        get_provider(settings)


# ====================================================================== mock
def test_mock_queued_mode_records_calls():
    mock = MockProvider(responses=["one", "two"])
    assert mock.available()[0] is True
    assert mock.complete("sys", "first") == "one"
    assert mock.complete("sys2", "second", images=[PNG, PNG], json_mode=False, max_tokens=10) == "two"
    assert [c["prompt"] for c in mock.calls] == ["first", "second"]
    assert mock.calls[0]["system"] == "sys" and mock.calls[0]["n_images"] == 0
    assert mock.calls[1] == {"system": "sys2", "prompt": "second", "n_images": 2, "json_mode": False,
                             "max_tokens": 10}
    with pytest.raises(ProviderError, match="called 3 times"):
        mock.complete("s", "third")


def test_mock_recorded_mode_matches_task_keywords_and_table_kind():
    tables = sample_tables()
    mock = MockProvider()
    answer = json.loads(mock.complete("", describe_user_prompt("a box plot please", tables, current_table=T_SIZE)))
    assert answer["plot_type"] == "box" and answer["data"]["table"] == T_SIZE
    # more keywords win: "bar" + "sd" beats "bar"
    answer = json.loads(mock.complete("", describe_user_prompt("bar graph with SD", tables, current_table=T_SIZE)))
    assert [lay["type"] for lay in answer["layers"]][1] == "sd_errorbar"
    # the "bar" recording is only for column/grouped tables: an XY table gets the generic XY answer
    answer = json.loads(mock.complete("", describe_user_prompt("bar graph", tables, current_table=T_DOSE)))
    assert answer == {"plot_type": "xy", "data": {"table": T_DOSE}}
    answer = json.loads(mock.complete("", describe_user_prompt("log axis", tables, current_table=T_DOSE)))
    assert answer["axes"]["x"]["scale"] == "log10"
    # deterministic
    again = json.loads(mock.complete("", describe_user_prompt("log axis", tables, current_table=T_DOSE)))
    assert again == answer
    assert len(mock.calls) == 5


def test_mock_generic_fallbacks():
    mock = MockProvider()
    assert json.loads(mock.complete("", "TASK: describe_figure\nno tables here")) == {
        "plot_type": "bar", "data": {"table": "My table"}}
    assert json.loads(mock.complete("", "TASK: critique_figure\n"))["done"] is True
    assert json.loads(mock.complete("", "TASK: extract_style\n"))["font_category"] == "sans"
    assert json.loads(mock.complete("", "TASK: something_else\n"))["ok"] is True
    assert json.loads(mock.complete("", "no task line at all"))["ok"] is True


def test_mock_custom_recorded_dir(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(
        {"task": "describe_figure", "keywords": ["zebra"],
         "response": {"plot_type": "violin", "data": {"table": "{{CURRENT_TABLE}}"}}}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"task": "extract_style", "response": {"grid": True}}),
                                     encoding="utf-8")
    mock = MockProvider(recorded_dir=tmp_path)
    tables = sample_tables()
    answer = json.loads(mock.complete("", describe_user_prompt("zebra", tables, current_table=T_SIZE)))
    assert answer == {"plot_type": "violin", "data": {"table": T_SIZE}}
    # placeholder but no table in the prompt -> generic answer
    assert json.loads(mock.complete("", "TASK: describe_figure\nREQUEST:\n<<<\nzebra\n>>>"))["plot_type"] == "bar"
    assert json.loads(mock.complete("", "TASK: extract_style\n")) == {"grid": True}
    assert json.loads(mock.complete("", "TASK: describe_figure\nREQUEST:\n<<<\nlion\n>>>"))["plot_type"] == "bar"

    (tmp_path / "c.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ProviderError, match="not valid JSON"):
        MockProvider(recorded_dir=tmp_path).complete("", "TASK: x")
    (tmp_path / "c.json").write_text('{"task": "x"}', encoding="utf-8")
    with pytest.raises(ProviderError, match="needs 'task' and 'response'"):
        MockProvider(recorded_dir=tmp_path).complete("", "TASK: x")


def test_recorded_answers_are_valid_figurespecs():
    from torusplot.spec import FigureSpec

    for entry in MockProvider().recorded():
        if entry["task"] == "describe_figure":
            FigureSpec.model_validate(json.loads(json.dumps(entry["response"]).replace("{{CURRENT_TABLE}}", "T")))


# ====================================================================== ollama
def test_ollama_text_request_shape(monkeypatch):
    http = FakeHTTP(monkeypatch, {"message": {"role": "assistant", "content": '{"a": 1}'}})
    provider = OllamaProvider(AISettings(ollama_host="http://localhost:11434/", timeout=42))
    assert provider.complete("SYSTEM", "PROMPT", max_tokens=123) == '{"a": 1}'
    request = http.requests[0]
    assert request["url"] == "http://localhost:11434/api/chat" and request["method"] == "POST"
    assert request["timeout"] == 42
    assert request["body"] == {
        "model": "qwen2.5:7b",
        "messages": [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "PROMPT"}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 123},
    }
    assert "authorization" not in request["headers"]


def test_ollama_vision_request_uses_vision_model(monkeypatch):
    http = FakeHTTP(monkeypatch, {"message": {"content": "text"}})
    provider = OllamaProvider(AISettings())
    assert provider.complete("S", "P", images=[PNG, JPEG], json_mode=False) == "text"
    body = http.requests[0]["body"]
    assert body["model"] == "granite3.2-vision"
    assert "format" not in body
    assert body["messages"][1]["images"] == [base64.b64encode(PNG).decode(), base64.b64encode(JPEG).decode()]
    with pytest.raises(ProviderError, match="PNG or JPEG"):
        provider.complete("S", "P", images=[b"not an image"])


def test_ollama_errors(monkeypatch):
    provider = OllamaProvider(AISettings())
    FakeHTTP(monkeypatch, http_error(404, '{"error":"model not found"}'))
    with pytest.raises(ProviderError, match=r"model qwen2.5:7b is not pulled: run `ollama pull qwen2.5:7b`"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, urllib.error.URLError("Connection refused"))
    with pytest.raises(ProviderError, match="Ollama is not running at http://localhost:11434"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"message": {"content": ""}, "error": "boom"})
    with pytest.raises(ProviderError, match=r"no text \(boom\)"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, http_error(500, "oops"))
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.complete("S", "P")
    no_vision = OllamaProvider(AISettings(ollama_vision_model=""))
    assert no_vision.supports_vision is False
    with pytest.raises(ProviderError, match="No Ollama vision model"):
        no_vision.complete("S", "P", images=[PNG])


def test_ollama_available(monkeypatch):
    provider = OllamaProvider(AISettings())
    http = FakeHTTP(monkeypatch, {"models": [{"name": "qwen2.5:7b"}, {"name": "granite3.2-vision:latest"}]})
    ok, why = provider.available()
    assert ok and "ready" in why
    assert http.requests[0]["url"] == "http://localhost:11434/api/tags" and http.requests[0]["method"] == "GET"
    assert http.requests[0]["timeout"] <= 3

    FakeHTTP(monkeypatch, urllib.error.URLError("refused"))
    ok, why = provider.available()
    assert not ok and "Ollama is not running at http://localhost:11434" in why

    FakeHTTP(monkeypatch, {"models": [{"name": "llama3:latest"}]})
    ok, why = provider.available()
    assert not ok and why == "model qwen2.5:7b is not pulled: run `ollama pull qwen2.5:7b`"

    FakeHTTP(monkeypatch, {"models": [{"name": "qwen2.5:7b"}]})
    ok, why = provider.available()
    assert ok and "granite3.2-vision is not pulled" in why


# ====================================================================== openai
def openai_settings(**extra):
    return load_settings(env={"TORUSPLOT_AI_ENABLED": "1", "TORUSPLOT_AI_PROVIDER": "openai", "OPENAI_API_KEY": KEY,
                              "TORUSPLOT_OPENAI_MODEL": "gpt-test", **extra})


def test_openai_request_shape(monkeypatch):
    http = FakeHTTP(monkeypatch, {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})
    provider = OpenAIProvider(openai_settings())
    assert provider.available() == (True, "OpenAI is configured (model: gpt-test).")
    assert provider.complete("SYSTEM", "PROMPT") == '{"ok": true}'
    request = http.requests[0]
    assert request["url"] == "https://api.openai.com/v1/chat/completions"
    assert request["headers"]["authorization"] == f"Bearer {KEY}"
    assert request["body"] == {
        "model": "gpt-test",
        "messages": [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "PROMPT"}],
        "response_format": {"type": "json_object"},
    }
    assert request["timeout"] == 120.0


def test_openai_images_and_base_url(monkeypatch):
    http = FakeHTTP(monkeypatch, {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]})
    provider = OpenAIProvider(openai_settings(OPENAI_BASE_URL="http://localhost:9999/v1/"))
    assert provider.complete("S", "P", images=[PNG, JPEG], json_mode=False) == "hi"
    request = http.requests[0]
    assert request["url"] == "http://localhost:9999/v1/chat/completions"
    assert "response_format" not in request["body"]
    content = request["body"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "P"}
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_errors_never_contain_the_key(monkeypatch):
    provider = OpenAIProvider(openai_settings())
    FakeHTTP(monkeypatch, http_error(401, f'{{"error": {{"message": "Incorrect API key provided: {KEY}"}}}}'))
    with pytest.raises(ProviderError) as err:
        provider.complete("S", "P")
    assert "HTTP 401" in str(err.value) and "API key was rejected" in str(err.value)
    assert KEY not in str(err.value) and "***" in str(err.value)

    FakeHTTP(monkeypatch, {"choices": [{"message": {"content": None, "refusal": "I cannot help"}}]})
    with pytest.raises(ProviderError, match="declined this request: I cannot help"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"choices": [{"message": {"content": '{"a":'}, "finish_reason": "length"}]})
    with pytest.raises(ProviderError, match="cut off"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    with pytest.raises(ProviderError, match="cut off"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"choices": []})
    with pytest.raises(ProviderError, match="no text"):
        provider.complete("S", "P")


def test_openai_availability_explains_what_is_missing(monkeypatch):
    http = FakeHTTP(monkeypatch)
    no_key = OpenAIProvider(AISettings(openai_model="gpt-test"))
    assert no_key.available() == (False, "Set OPENAI_API_KEY (in your environment or .env file) to use OpenAI.")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        no_key.complete("S", "P")
    no_model = OpenAIProvider(load_settings(env={"OPENAI_API_KEY": KEY}))
    ok, why = no_model.available()
    assert not ok and "TORUSPLOT_OPENAI_MODEL" in why and "no built-in default" in why
    with pytest.raises(ProviderError, match="TORUSPLOT_OPENAI_MODEL"):
        no_model.complete("S", "P")
    assert http.requests == []  # nothing was sent


# ====================================================================== gemini
def gemini_settings(model="gemini-test"):
    return load_settings(env={"GEMINI_API_KEY": KEY, "TORUSPLOT_GEMINI_MODEL": model})


def test_gemini_request_shape(monkeypatch):
    http = FakeHTTP(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                                         {"text": '{"a"'}, {"text": ": 1}"}]},
                                                  "finishReason": "STOP"}]})
    provider = GeminiProvider(gemini_settings("models/gemini-test"))
    assert provider.available()[0] is True
    assert provider.complete("SYSTEM", "PROMPT", images=[PNG, JPEG]) == '{"a": 1}'
    request = http.requests[0]
    assert request["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    assert KEY not in request["url"]
    assert request["headers"]["x-goog-api-key"] == KEY
    assert request["body"] == {
        "system_instruction": {"parts": [{"text": "SYSTEM"}]},
        "contents": [{"role": "user", "parts": [
            {"text": "PROMPT"},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(PNG).decode()}},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(JPEG).decode()}},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }


def test_gemini_text_mode_and_errors(monkeypatch):
    provider = GeminiProvider(gemini_settings())
    http = FakeHTTP(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "plain"}]}}]})
    assert provider.complete("S", "P", json_mode=False) == "plain"
    assert http.requests[0]["body"]["generationConfig"] == {"temperature": 0}

    FakeHTTP(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError, match=r"declined this request \(reason: SAFETY\)"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"candidates": [{"finishReason": "RECITATION"}]})
    with pytest.raises(ProviderError, match="RECITATION"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(ProviderError, match="cut off"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, {"candidates": [{"content": {"parts": []}}]})
    with pytest.raises(ProviderError, match="no text"):
        provider.complete("S", "P")
    FakeHTTP(monkeypatch, http_error(403, f"key {KEY} not allowed"))
    with pytest.raises(ProviderError) as err:
        provider.complete("S", "P")
    assert "HTTP 403" in str(err.value) and KEY not in str(err.value)


def test_gemini_availability_explains_what_is_missing():
    assert GeminiProvider(AISettings()).available() == (
        False, "Set GEMINI_API_KEY (in your environment or .env file) to use Gemini.")
    ok, why = GeminiProvider(load_settings(env={"GEMINI_API_KEY": KEY})).available()
    assert not ok and "TORUSPLOT_GEMINI_MODEL" in why and "no built-in default" in why
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider(AISettings()).complete("S", "P")


# ====================================================================== anthropic (fake SDK)
class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Message:
    def __init__(self, text='{"ok": true}', stop_reason="end_turn", stop_details=None):
        self.content = [_Block("thinking"), _Block("text", text[:5]), _Block("text", text[5:])]
        self.stop_reason = stop_reason
        self.stop_details = stop_details


def make_fake_anthropic(*, beta=None, plain=None):
    """A stand-in for the ``anthropic`` package recording how it is called."""
    fake = types.ModuleType("anthropic")
    fake.__spec__ = importlib.machinery.ModuleSpec("anthropic", None)
    fake.calls = []

    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        def __init__(self, message="status error", status_code=500):
            super().__init__(message)
            self.message, self.status_code = message, status_code

    class BadRequestError(APIStatusError):
        def __init__(self, message="bad request"):
            super().__init__(message, 400)

    class AuthenticationError(APIStatusError):
        def __init__(self, message="auth"):
            super().__init__(message, 401)

    class PermissionDeniedError(APIStatusError):
        def __init__(self, message="denied"):
            super().__init__(message, 403)

    class NotFoundError(APIStatusError):
        def __init__(self, message="not found"):
            super().__init__(message, 404)

    class RateLimitError(APIStatusError):
        def __init__(self, message="rate"):
            super().__init__(message, 429)

    for cls in (APIError, APIConnectionError, APIStatusError, BadRequestError, AuthenticationError,
                PermissionDeniedError, NotFoundError, RateLimitError):
        setattr(fake, cls.__name__, cls)

    def run(kind, behaviour, kwargs):
        fake.calls.append((kind, kwargs))
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour if behaviour is not None else _Message()

    class _Messages:
        def __init__(self, kind, behaviour):
            self.kind, self.behaviour = kind, behaviour

        def create(self, **kwargs):
            return run(self.kind, self.behaviour, kwargs)

    class Anthropic:
        def __init__(self, **kwargs):
            fake.client_kwargs = kwargs
            self.messages = _Messages("plain", plain)
            self.beta = types.SimpleNamespace(messages=_Messages("beta", beta))

    fake.Anthropic = Anthropic
    return fake


def anthropic_settings(**env):
    return load_settings(env={"ANTHROPIC_API_KEY": KEY, **env})


def test_anthropic_missing_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    provider = AnthropicProvider(anthropic_settings())
    assert provider.available() == (False, "Install the optional package: pip install anthropic")
    assert INSTALL_HINT == "Install the optional package: pip install anthropic"
    with pytest.raises(ProviderError, match="pip install anthropic"):
        provider.complete("S", "P")


def test_anthropic_available_needs_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", make_fake_anthropic())
    ok, why = AnthropicProvider(AISettings()).available()
    assert not ok and "ANTHROPIC_API_KEY" in why
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(AISettings()).complete("S", "P")
    assert AnthropicProvider(anthropic_settings()).available() == (True, "Claude is configured (model: claude-opus-5-5).")


def test_anthropic_request_shape(monkeypatch):
    fake = make_fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    provider = AnthropicProvider(anthropic_settings())
    assert provider.complete("SYSTEM json", "PROMPT", images=[PNG, JPEG]) == '{"ok": true}'

    assert fake.client_kwargs == {"api_key": KEY, "timeout": 120.0}
    assert len(fake.calls) == 1
    kind, kwargs = fake.calls[0]
    assert kind == "beta"  # refusal fallbacks are opted in to first
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"] and kwargs["fallbacks"] == "default"
    assert kwargs["model"] == "claude-opus-5-5"  # exact ID, no date suffix
    assert kwargs["max_tokens"] == 16000
    assert kwargs["system"] == "SYSTEM json"
    assert kwargs["messages"] == [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.b64encode(PNG).decode()}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.b64encode(JPEG).decode()}},
        {"type": "text", "text": "PROMPT"},
    ]}]
    # nothing current models reject: no sampling parameters, no thinking config, no prefill
    assert set(kwargs) == {"betas", "fallbacks", "model", "max_tokens", "system", "messages"}
    assert all(m["role"] == "user" for m in kwargs["messages"])


def test_anthropic_json_instruction_and_model_override(monkeypatch):
    fake = make_fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    provider = AnthropicProvider(anthropic_settings(TORUSPLOT_ANTHROPIC_MODEL="claude-sonnet-5-5"))
    provider.complete("Be brief.", "Hello", max_tokens=32000)
    kwargs = fake.calls[0][1]
    assert kwargs["model"] == "claude-sonnet-5-5" and kwargs["max_tokens"] == 32000
    assert kwargs["system"].endswith("Answer with one JSON object and nothing else.")
    assert kwargs["messages"][0]["content"] == [{"type": "text", "text": "Hello"}]
    provider.complete("Be brief.", "Hello", json_mode=False)
    assert fake.calls[1][1]["system"] == "Be brief."


@pytest.mark.parametrize("first_error", ["type", "bad_request"])
def test_anthropic_falls_back_to_plain_create(monkeypatch, first_error):
    fake = make_fake_anthropic()
    if first_error == "type":  # an older SDK that does not know the `fallbacks` argument
        error = TypeError("create() got an unexpected keyword argument 'fallbacks'")
    else:  # a model without server-side fallbacks
        error = fake.BadRequestError("fallbacks are not supported for this model")
    fake = _with_beta(fake, error)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert AnthropicProvider(anthropic_settings()).complete("S json", "P") == '{"ok": true}'
    assert [kind for kind, _ in fake.calls] == ["beta", "plain"]
    plain_kwargs = fake.calls[1][1]
    assert set(plain_kwargs) == {"model", "max_tokens", "system", "messages"}


def _with_beta(fake, behaviour):
    """Rebuild the fake client so its beta endpoint raises an error of this fake module."""
    original = fake.Anthropic

    class Anthropic(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.beta.messages.behaviour = behaviour

    fake.Anthropic = Anthropic
    return fake


def test_anthropic_refusal_and_truncation(monkeypatch):
    details = types.SimpleNamespace(category="cyber", explanation="This request was declined by a safety filter.")
    fake = make_fake_anthropic(beta=_Message(stop_reason="refusal", stop_details=details))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError, match="declined this request. This request was declined by a safety filter."):
        AnthropicProvider(anthropic_settings()).complete("S json", "P")

    fake = make_fake_anthropic(beta=_Message(stop_reason="refusal"))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError, match=r"declined this request\.$"):
        AnthropicProvider(anthropic_settings()).complete("S json", "P")

    fake = make_fake_anthropic(beta=_Message(stop_reason="max_tokens"))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError, match="cut off"):
        AnthropicProvider(anthropic_settings()).complete("S json", "P")

    fake = make_fake_anthropic(beta=_Message(text=""))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError, match="no text"):
        AnthropicProvider(anthropic_settings()).complete("S json", "P")


@pytest.mark.parametrize("error_name,args,expected", [
    ("AuthenticationError", (), "rejected the API key"),
    ("PermissionDeniedError", (), "may not use the model 'claude-opus-5-5'"),
    ("NotFoundError", (), "does not know the model 'claude-opus-5-5'"),
    ("RateLimitError", (), "rate limit"),
    ("APIStatusError", (f"overloaded while using {KEY}", 529), "HTTP 529"),
    ("APIConnectionError", (), "Could not reach Anthropic"),
])
def test_anthropic_error_mapping(monkeypatch, error_name, args, expected):
    fake = make_fake_anthropic()
    fake = _with_beta(fake, getattr(fake, error_name)(*args))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError) as err:
        AnthropicProvider(anthropic_settings()).complete("S json", "P")
    assert expected in str(err.value)
    assert KEY not in str(err.value)
    assert [kind for kind, _ in fake.calls] == ["beta"]  # not retried on the plain endpoint


def test_anthropic_bad_request_on_both_endpoints_is_reported(monkeypatch):
    fake = make_fake_anthropic()
    error = fake.BadRequestError("image too large")
    fake = _with_beta(fake, error)
    original = fake.Anthropic

    class Anthropic(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.messages.behaviour = error

    fake.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(ProviderError, match=r"HTTP 400\). image too large"):
        AnthropicProvider(anthropic_settings()).complete("S json", "P")
    assert [kind for kind, _ in fake.calls] == ["beta", "plain"]
