"""Google Gemini provider (bring your own key). Plain HTTPS with the standard library.

There is no built-in default model: model names change, so the user chooses one with
``TORUSPLOT_GEMINI_MODEL``. The key travels in the ``x-goog-api-key`` header, never in the URL.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from ..settings import AISettings
from .base import Provider, ProviderError, b64, dig, http_json, image_media_type

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(Provider):
    name = "gemini"
    supports_vision = True

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or AISettings()
        self.model = self.settings.gemini_model

    def available(self) -> tuple[bool, str]:
        if not self.settings.has_key("gemini"):
            return False, "Set GEMINI_API_KEY (in your environment or .env file) to use Gemini."
        if not self.model:
            return False, (
                "Choose a model: set TORUSPLOT_GEMINI_MODEL to a current Gemini model name. TorusPlot "
                "has no built-in default because model names change."
            )
        return True, f"Gemini is configured (model: {self.model})."

    def url(self) -> str:
        model = self.model.removeprefix("models/")
        return f"{BASE_URL}/models/{quote(model, safe='')}:generateContent"

    def build_request(
        self, system: str, prompt: str, images: list[bytes] | None, json_mode: bool
    ) -> dict[str, Any]:
        """Request body. No token limit is sent: models that think before answering count
        hidden thinking against it, so the service's own default is used."""
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image in images or []:
            parts.append({"inline_data": {"mime_type": image_media_type(image), "data": b64(image)}})
        config: dict[str, Any] = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        return {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
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
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        key = self.settings.api_key("gemini") or ""
        reply = http_json(
            self.url(),
            self.build_request(system, prompt, images, json_mode),
            headers={"x-goog-api-key": key},
            timeout=self.settings.timeout,
            secrets=[key],
            service="Gemini",
        )
        block = dig(reply, "promptFeedback", "blockReason")
        if block:
            raise ProviderError(f"The model declined this request (reason: {block}).")
        candidate = dig(reply, "candidates", 0) or {}
        finish = dig(candidate, "finishReason")
        parts = dig(candidate, "content", "parts") or []
        text = "".join(
            str(p["text"]) for p in parts if isinstance(p, dict) and "text" in p and not p.get("thought")
        )
        if finish in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
            raise ProviderError(f"The model declined this request (reason: {finish}).")
        if finish == "MAX_TOKENS":
            raise ProviderError("The model's answer was cut off (token limit reached). Try again.")
        if not text.strip():
            raise ProviderError("Gemini returned no text.")
        return text
