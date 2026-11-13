"""Settings of the optional AI layer.

AI features are **off by default** and TorusPlot works fully offline without them.
Everything here is read from environment variables and an optional ``.env`` file; API
keys are never written anywhere and never appear in ``repr``, logs, exceptions or saved
settings (they are held in :class:`pydantic.SecretStr` and masked by :func:`redact`).

Environment variables (see ``.env.example``)::

    TORUSPLOT_AI_ENABLED            1/true/yes/on turns AI features on (default: off)
    TORUSPLOT_AI_PROVIDER           mock | ollama | gemini | anthropic | openai (default: mock)
    OLLAMA_HOST                     default http://localhost:11434
    TORUSPLOT_OLLAMA_TEXT_MODEL     default qwen2.5:7b          (Apache-2.0)
    TORUSPLOT_OLLAMA_VISION_MODEL   default granite3.2-vision   (Apache-2.0)
    GEMINI_API_KEY / TORUSPLOT_GEMINI_MODEL        no built-in default model
    ANTHROPIC_API_KEY / TORUSPLOT_ANTHROPIC_MODEL  default claude-opus-5-5
    OPENAI_API_KEY / TORUSPLOT_OPENAI_MODEL        no built-in default model
    OPENAI_BASE_URL                 default https://api.openai.com/v1
    TORUSPLOT_AI_TIMEOUT            seconds per request (default 120)
    TORUSPLOT_AI_MAX_RETRIES        validation retries of "Describe it" (default 3)
    TORUSPLOT_AI_MAX_FEEDBACK_LOOPS visual feedback loops (default 3, hard cap 3)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROVIDERS = ("mock", "ollama", "gemini", "anthropic", "openai")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_VISION_MODEL = "granite3.2-vision"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Hard upper bound of the visual feedback loop ("Match this figure").
MAX_FEEDBACK_LOOPS_CAP = 3

#: Local models TorusPlot suggests in its UI. Only Apache-2.0 / MIT weights are listed
#: (see docs/MODEL_LICENSE_RESEARCH.md). TorusPlot never bundles or downloads weights:
#: the user pulls a model with Ollama (``ollama pull <tag>``).
RECOMMENDED_OLLAMA_MODELS: tuple[dict[str, str], ...] = (
    {
        "tag": "qwen2.5:7b",
        "role": "text",
        "license": "Apache-2.0",
        "source": "https://github.com/QwenLM/Qwen2.5",
        "card": "https://ollama.com/library/qwen2.5",
        "pull": "ollama pull qwen2.5:7b",
        "note": "Default text model. Pin the 7b size: the 3b and 72b sizes are not Apache-2.0.",
    },
    {
        "tag": "granite3.2-vision",
        "role": "vision",
        "license": "Apache-2.0",
        "source": "https://github.com/ibm-granite/granite-vision-models",
        "card": "https://ollama.com/library/granite3.2-vision",
        "pull": "ollama pull granite3.2-vision",
        "note": "Default vision model (figure style extraction and critique).",
    },
    {
        "tag": "qwen3",
        "role": "text",
        "license": "Apache-2.0",
        "source": "https://github.com/QwenLM/Qwen3",
        "card": "https://ollama.com/library/qwen3",
        "pull": "ollama pull qwen3",
        "note": "Alternative text model.",
    },
    {
        "tag": "granite3.3",
        "role": "text",
        "license": "Apache-2.0",
        "source": "https://github.com/ibm-granite/granite-3.3-language-models",
        "card": "https://ollama.com/library/granite3.3",
        "pull": "ollama pull granite3.3",
        "note": "Alternative text model.",
    },
)

_TRUE = {"1", "true", "yes", "on", "y", "t", "enabled"}
_KEY_FIELDS = {"gemini": "gemini_api_key", "anthropic": "anthropic_api_key", "openai": "openai_api_key"}
_KEY_ENV = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class AISettings(BaseModel):
    """Configuration of the AI layer. Safe to print: API keys are always masked."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = Field(default=False, description="Master switch. Off by default.")
    provider: str = Field(default="mock", description="mock | ollama | gemini | anthropic | openai")
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_text_model: str = DEFAULT_OLLAMA_TEXT_MODEL
    ollama_vision_model: str = DEFAULT_OLLAMA_VISION_MODEL
    gemini_model: str = Field(default="", description="No built-in default: the user must choose one")
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_model: str = Field(default="", description="No built-in default: the user must choose one")
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    timeout: float = Field(default=120.0, gt=0, description="Seconds per provider request")
    max_retries: int = Field(default=3, ge=0, le=10)
    max_feedback_loops: int = Field(default=3, ge=1, le=MAX_FEEDBACK_LOOPS_CAP)

    @field_validator("provider", mode="before")
    @classmethod
    def _provider(cls, v: Any) -> str:
        name = str(v or "mock").strip().lower()
        if name not in PROVIDERS:
            raise ValueError(f"Unknown AI provider '{name}'. Choose one of: {', '.join(PROVIDERS)}.")
        return name

    @field_validator("ollama_host", "openai_base_url", mode="before")
    @classmethod
    def _url(cls, v: Any) -> str:
        return str(v or "").strip().rstrip("/")

    # ------------------------------------------------------------------ keys
    def api_key(self, provider: str | None = None) -> str | None:
        """The real key of a cloud provider (None if not set). Never log the return value."""
        field = _KEY_FIELDS.get(provider or self.provider)
        secret = getattr(self, field) if field else None
        value = secret.get_secret_value().strip() if secret is not None else ""
        return value or None

    def has_key(self, provider: str | None = None) -> bool:
        return self.api_key(provider) is not None

    def key_env_name(self, provider: str | None = None) -> str | None:
        return _KEY_ENV.get(provider or self.provider)

    def secrets(self) -> list[str]:
        """Every configured secret value (used to scrub messages)."""
        return [k for k in (self.api_key(p) for p in _KEY_FIELDS) if k]

    def redact(self, text: Any) -> str:
        """``text`` with every configured API key replaced by ``***``."""
        return redact(text, self.secrets())

    def public_dict(self) -> dict[str, Any]:
        """Settings that are safe to show or save: keys are reduced to 'set' / 'not set'."""
        out = self.model_dump(exclude=set(_KEY_FIELDS.values()))
        for provider in _KEY_FIELDS:
            out[f"{provider}_api_key"] = "set" if self.has_key(provider) else "not set"
        return out

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.public_dict().items())
        return f"AISettings({body})"

    __str__ = __repr__


def redact(text: Any, secrets: list[str] | tuple[str, ...]) -> str:
    """Replace every secret in ``text`` with ``***`` (used for errors and logs)."""
    out = str(text)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        out = out.replace(secret, "***")
    return out


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. ``#`` starts a comment; values may be quoted; ``export`` is allowed."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            hash_at = value.find(" #")
            if hash_at >= 0:
                value = value[:hash_at].rstrip()
        out[key] = value
    return out


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Read a ``.env`` file (missing file = empty dict)."""
    path = Path(path)
    if not path.is_file():
        return {}
    return parse_dotenv(path.read_text(encoding="utf-8"))


def load_settings(
    env: Mapping[str, str] | None = None, dotenv_path: str | Path | None = None
) -> AISettings:
    """Build :class:`AISettings` from environment variables and an optional ``.env`` file.

    * ``env=None`` uses ``os.environ``; then, if ``dotenv_path`` is also None, a ``.env``
      file in the current directory is read when it exists.
    * When ``env`` is given explicitly, a ``.env`` file is read only if ``dotenv_path`` is
      given (so tests and callers stay hermetic).
    * Real environment variables win over the ``.env`` file. Empty values count as unset.
    """
    if env is None:
        env = os.environ
        if dotenv_path is None:
            dotenv_path = Path.cwd() / ".env"
    merged: dict[str, str] = dict(read_dotenv(dotenv_path)) if dotenv_path is not None else {}
    merged.update({k: v for k, v in env.items() if isinstance(v, str) and v.strip() != ""})

    def get(name: str) -> str | None:
        value = merged.get(name)
        value = value.strip() if isinstance(value, str) else None
        return value or None

    values: dict[str, Any] = {
        "enabled": (get("TORUSPLOT_AI_ENABLED") or "").lower() in _TRUE,
        "provider": get("TORUSPLOT_AI_PROVIDER") or "mock",
    }
    mapping = {
        "ollama_host": "OLLAMA_HOST",
        "ollama_text_model": "TORUSPLOT_OLLAMA_TEXT_MODEL",
        "ollama_vision_model": "TORUSPLOT_OLLAMA_VISION_MODEL",
        "gemini_model": "TORUSPLOT_GEMINI_MODEL",
        "anthropic_model": "TORUSPLOT_ANTHROPIC_MODEL",
        "openai_model": "TORUSPLOT_OPENAI_MODEL",
        "openai_base_url": "OPENAI_BASE_URL",
        "gemini_api_key": "GEMINI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
    }
    for field, name in mapping.items():
        value = get(name)
        if value is not None:
            values[field] = value
    host = values.get("ollama_host")
    if host and "://" not in host:
        values["ollama_host"] = "http://" + host
    for field, name, cast in (
        ("timeout", "TORUSPLOT_AI_TIMEOUT", float),
        ("max_retries", "TORUSPLOT_AI_MAX_RETRIES", int),
        ("max_feedback_loops", "TORUSPLOT_AI_MAX_FEEDBACK_LOOPS", int),
    ):
        value = get(name)
        if value is None:
            continue
        try:
            number = cast(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got '{value}'.") from exc
        if field == "max_feedback_loops":
            number = max(1, min(int(number), MAX_FEEDBACK_LOOPS_CAP))
        values[field] = number
    try:
        return AISettings(**values)
    except ValueError as exc:
        # pydantic would echo input values; keep the message key-free.
        secrets = [values[k] for k in ("gemini_api_key", "anthropic_api_key", "openai_api_key") if values.get(k)]
        raise ValueError(redact(_short_error(exc), secrets)) from None


def _short_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts = []
        for err in errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            parts.append(f"{loc}: {err.get('msg', '')}".strip(": "))
        if parts:
            return "Invalid AI settings: " + "; ".join(parts)
    return f"Invalid AI settings: {exc}"
