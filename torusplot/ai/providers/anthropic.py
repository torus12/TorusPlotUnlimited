"""Anthropic (Claude) provider (bring your own key).

Uses the official ``anthropic`` Python SDK as an **optional** extra that is imported
lazily inside the methods: TorusPlot does not depend on it. If the package is missing,
:meth:`AnthropicProvider.available` says how to install it.

Request shape (deliberate, do not "tidy"):

* no ``temperature`` / ``top_p`` / ``top_k`` and no ``thinking`` parameter: current models
  use adaptive thinking by default and reject sampling parameters and thinking budgets;
* no assistant prefill (rejected by current models): JSON-only output is requested in the
  system prompt and the answer is always validated by the caller;
* model IDs are used exactly as configured (no date suffix is appended);
* the request first opts in to server-side refusal fallbacks (beta); SDKs or models
  without that feature fall back to a plain ``messages.create`` call.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from ..settings import DEFAULT_ANTHROPIC_MODEL, AISettings
from .base import Provider, ProviderError, b64, image_media_type

INSTALL_HINT = "Install the optional package: pip install anthropic"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Floor for ``max_tokens``: thinking and the answer share this budget.
MIN_MAX_TOKENS = 16000


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_vision = True

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or AISettings()
        self.model = self.settings.anthropic_model or DEFAULT_ANTHROPIC_MODEL

    def available(self) -> tuple[bool, str]:
        try:
            found = importlib.util.find_spec("anthropic") is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return False, INSTALL_HINT
        if not self.settings.has_key("anthropic"):
            return False, "Set ANTHROPIC_API_KEY (in your environment or .env file) to use Claude."
        return True, f"Claude is configured (model: {self.model})."

    def build_request(self, system: str, prompt: str, images: list[bytes] | None, max_tokens: int) -> dict[str, Any]:
        """Keyword arguments for ``messages.create``: images first, then the text."""
        content: list[dict[str, Any]] = []
        for image in images or []:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": image_media_type(image), "data": b64(image)},
                }
            )
        content.append({"type": "text", "text": prompt})
        return {
            "model": self.model,
            "max_tokens": max(int(max_tokens), MIN_MAX_TOKENS),
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        images: list[bytes] | None = None,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError:
            raise ProviderError(INSTALL_HINT) from None
        key = self.settings.api_key("anthropic")
        if not key:
            raise ProviderError("Set ANTHROPIC_API_KEY (in your environment or .env file) to use Claude.")
        if json_mode and "json" not in (system + prompt).lower():
            system = f"{system}\n\nAnswer with one JSON object and nothing else."
        kwargs = self.build_request(system, prompt, images, max_tokens)
        redact = self.settings.redact
        try:
            client = anthropic.Anthropic(api_key=key, timeout=self.settings.timeout)
            try:
                # Refusal fallback (opt-in): the API re-runs a declined request on a fallback model.
                response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            except (TypeError, anthropic.BadRequestError):
                # Older SDK, or a model without server-side fallbacks.
                response = client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            raise ProviderError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from None
        except anthropic.PermissionDeniedError:
            raise ProviderError(f"This Anthropic API key may not use the model '{self.model}'.") from None
        except anthropic.NotFoundError:
            raise ProviderError(
                f"Anthropic does not know the model '{self.model}'. Check TORUSPLOT_ANTHROPIC_MODEL."
            ) from None
        except anthropic.RateLimitError:
            raise ProviderError("Anthropic rate limit reached. Wait a moment and try again.") from None
        except anthropic.APIStatusError as exc:
            status = getattr(exc, "status_code", "?")
            detail = redact(getattr(exc, "message", "") or "")
            raise ProviderError(f"Anthropic returned an error (HTTP {status}). {detail}".strip()) from None
        except anthropic.APIConnectionError:
            raise ProviderError("Could not reach Anthropic. Check your internet connection.") from None

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            explanation = getattr(details, "explanation", None) if details is not None else None
            message = "The model declined this request."
            if explanation:
                message += f" {explanation}"
            raise ProviderError(redact(message))
        if response.stop_reason == "max_tokens":
            raise ProviderError("The model's answer was cut off (token limit reached). Try again.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise ProviderError("Claude returned no text.")
        return text
