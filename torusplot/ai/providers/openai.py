"""OpenAI provider (bring your own key). Plain HTTPS with the standard library.

Also works with any service that speaks the same chat-completions protocol when
``OPENAI_BASE_URL`` is set. There is no built-in default model: model names change, so
the user chooses one with ``TORUSPLOT_OPENAI_MODEL``.
"""

from __future__ import annotations

from typing import Any

from ..settings import AISettings
from .base import Provider, ProviderError, data_url, dig, http_json


class OpenAIProvider(Provider):
    name = "openai"
    supports_vision = True

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or AISettings()
        self.model = self.settings.openai_model
        self.base_url = self.settings.openai_base_url.rstrip("/")

    def available(self) -> tuple[bool, str]:
        if not self.settings.has_key("openai"):
            return False, "Set OPENAI_API_KEY (in your environment or .env file) to use OpenAI."
        if not self.model:
            return False, (
                "Choose a model: set TORUSPLOT_OPENAI_MODEL to a current OpenAI model name. TorusPlot "
                "has no built-in default because model names change."
            )
        return True, f"OpenAI is configured (model: {self.model})."

    def build_request(
        self, system: str, prompt: str, images: list[bytes] | None, json_mode: bool
    ) -> dict[str, Any]:
        """Request body. No token limit is sent: models that reason before answering count
        hidden reasoning against it, so the service's own default is used."""
        if images:
            content: Any = [{"type": "text", "text": prompt}]
            content += [{"type": "image_url", "image_url": {"url": data_url(image)}} for image in images]
        else:
            content = prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        images: list[bytes] | None = None,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        key = self.settings.api_key("openai") or ""
        reply = http_json(
            f"{self.base_url}/chat/completions",
            self.build_request(system, prompt, images, json_mode),
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.settings.timeout,
            secrets=[key],
            service="OpenAI",
        )
        choice = dig(reply, "choices", 0) or {}
        refusal = dig(choice, "message", "refusal")
        if refusal:
            raise ProviderError(f"The model declined this request: {refusal}")
        text = dig(choice, "message", "content")
        if isinstance(text, list):  # some compatible servers return content parts
            text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
        if not isinstance(text, str) or not text.strip():
            if dig(choice, "finish_reason") == "length":
                raise ProviderError("The model's answer was cut off (token limit reached). Try again.")
            raise ProviderError("OpenAI returned no text.")
        if dig(choice, "finish_reason") == "length":
            raise ProviderError("The model's answer was cut off (token limit reached). Try again.")
        return text
