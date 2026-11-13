"""Provider interface shared by every AI backend.

A provider turns ``(system prompt, user prompt, optional images)`` into raw text. It knows
nothing about figures or statistics: all validation happens in the callers
(:mod:`torusplot.ai.describe`, :mod:`torusplot.ai.match` ...), which never trust the text.

This module imports no network or SDK code at import time.
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..settings import PROVIDERS, AISettings, redact


class ProviderError(RuntimeError):
    """A provider could not answer. The message is user-facing and never contains an API key."""


class Provider(ABC):
    """Base class of all providers."""

    name: str = "provider"
    supports_vision: bool = False

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """``(usable, explanation)``. Cheap: no network unless the service is trivially local."""

    @abstractmethod
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        images: list[bytes] | None = None,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        """Return the model's raw text. ``images`` are PNG or JPEG bytes."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} vision={self.supports_vision}>"


# ---------------------------------------------------------------------- factory
def get_provider(settings: AISettings | None = None, *, allow_disabled: bool = False) -> Provider:
    """Build the provider selected in ``settings`` (the backend module is imported lazily).

    AI features are off by default: unless ``settings.enabled`` is true (or
    ``allow_disabled=True`` is passed, e.g. for a "test connection" button) this raises
    :class:`ProviderError` explaining how to turn them on.
    """
    settings = settings or AISettings()
    if not settings.enabled and not allow_disabled:
        raise ProviderError(
            "AI features are turned off. Turn them on in Settings (or set TORUSPLOT_AI_ENABLED=1). "
            "TorusPlot works fully without them."
        )
    name = settings.provider
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(settings)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(settings)
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(settings)
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(settings)
    raise ProviderError(f"Unknown AI provider '{name}'. Choose one of: {', '.join(PROVIDERS)}.")


def active_provider(settings: AISettings | None = None) -> Provider | None:
    """The configured provider, or None when AI features are off (the default)."""
    settings = settings or AISettings()
    if not settings.enabled:
        return None
    return get_provider(settings)


# ---------------------------------------------------------------------- helpers
def image_media_type(data: bytes) -> str:
    """``image/png`` or ``image/jpeg`` from the magic bytes (anything else is an error)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    raise ProviderError("Images sent to an AI provider must be PNG or JPEG.")


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def data_url(data: bytes) -> str:
    return f"data:{image_media_type(data)};base64,{b64(data)}"


def http_json(
    url: str,
    payload: Mapping[str, Any] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 120.0,
    secrets: Sequence[str] = (),
    service: str = "The AI service",
) -> Any:
    """POST ``payload`` as JSON (or GET when it is None) with the standard library.

    Every failure becomes a :class:`ProviderError` whose message has all ``secrets``
    (API keys) removed. ``urllib`` is imported here, not at module import time.
    """
    import urllib.error
    import urllib.request

    if not url.lower().startswith(("http://", "https://")):
        raise ProviderError(f"{service}: the address must start with http:// or https://.")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method="GET" if body is None else "POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # pragma: no cover - best effort only
            detail = ""
        hint = {
            400: "the request was rejected (check the model name)",
            401: "the API key was rejected",
            403: "access was denied for this key or model",
            404: "the model or address was not found",
            429: "the rate limit or quota was reached; wait and try again",
        }.get(exc.code, "the service reported an error")
        message = f"{service} returned HTTP {exc.code}: {hint}."
        if detail.strip():
            message += f" Details: {' '.join(detail.split())}"
        raise ProviderError(redact(message, list(secrets))) from None
    except TimeoutError:
        raise ProviderError(f"{service} did not answer within {timeout:g} seconds.") from None
    except urllib.error.URLError as exc:
        reason = redact(getattr(exc, "reason", exc), list(secrets))
        if "timed out" in reason.lower():
            raise ProviderError(f"{service} did not answer within {timeout:g} seconds.") from None
        raise ProviderError(f"{service} could not be reached ({reason}).") from None
    except OSError as exc:
        raise ProviderError(f"{service} could not be reached ({redact(exc, list(secrets))}).") from None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderError(f"{service} sent a reply that is not valid JSON.") from None


def dig(obj: Any, *path: Any) -> Any:
    """Safe nested lookup in dicts/lists (None when any step is missing)."""
    for key in path:
        if isinstance(obj, Mapping) and key in obj:
            obj = obj[key]
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    return obj
