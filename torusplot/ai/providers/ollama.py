"""Ollama provider: free, local models. Nothing leaves the computer.

TorusPlot never bundles or downloads model weights. The user installs Ollama
(https://ollama.com) and pulls a model; the defaults are Apache-2.0 models only
(see :data:`torusplot.ai.settings.RECOMMENDED_OLLAMA_MODELS`).
"""

from __future__ import annotations

from typing import Any

from ..settings import AISettings
from .base import Provider, ProviderError, b64, dig, http_json, image_media_type


class OllamaProvider(Provider):
    name = "ollama"
    supports_vision = True

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or AISettings()
        self.host = self.settings.ollama_host.rstrip("/")
        self.text_model = self.settings.ollama_text_model
        self.vision_model = self.settings.ollama_vision_model
        self.supports_vision = bool(self.vision_model)

    def available(self) -> tuple[bool, str]:
        """Ask the local Ollama server which models are pulled (short timeout)."""
        try:
            tags = http_json(f"{self.host}/api/tags", timeout=2.0, service="Ollama")
        except ProviderError:
            return False, (
                f"Ollama is not running at {self.host}. Install it from https://ollama.com, start it, "
                f"then run `ollama pull {self.text_model}`."
            )
        pulled = {str(m.get("name", "")) for m in (dig(tags, "models") or []) if isinstance(m, dict)}
        pulled |= {str(m.get("model", "")) for m in (dig(tags, "models") or []) if isinstance(m, dict)}
        if not _has_model(pulled, self.text_model):
            return False, f"model {self.text_model} is not pulled: run `ollama pull {self.text_model}`"
        if self.vision_model and not _has_model(pulled, self.vision_model):
            return True, (
                f"Ollama is ready for text ({self.text_model}). For image features, model "
                f"{self.vision_model} is not pulled: run `ollama pull {self.vision_model}`"
            )
        return True, f"Ollama is ready at {self.host} (text: {self.text_model}, vision: {self.vision_model})."

    def build_request(
        self, system: str, prompt: str, images: list[bytes] | None, json_mode: bool, max_tokens: int
    ) -> dict[str, Any]:
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            for image in images:
                image_media_type(image)  # PNG/JPEG only
            user["images"] = [b64(image) for image in images]
        payload: dict[str, Any] = {
            "model": self.vision_model if images else self.text_model,
            "messages": [{"role": "system", "content": system}, user],
            "stream": False,
            "options": {"temperature": 0, "num_predict": int(max_tokens)},
        }
        if json_mode:
            payload["format"] = "json"
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
        if images and not self.vision_model:
            raise ProviderError("No Ollama vision model is configured (TORUSPLOT_OLLAMA_VISION_MODEL).")
        payload = self.build_request(system, prompt, images, json_mode, max_tokens)
        try:
            reply = http_json(
                f"{self.host}/api/chat", payload, timeout=self.settings.timeout, service="Ollama"
            )
        except ProviderError as exc:
            if "HTTP 404" in str(exc):
                model = payload["model"]
                raise ProviderError(f"model {model} is not pulled: run `ollama pull {model}`") from None
            if "could not be reached" in str(exc):
                raise ProviderError(
                    f"Ollama is not running at {self.host}. Start Ollama and try again."
                ) from None
            raise
        text = dig(reply, "message", "content")
        if not isinstance(text, str) or not text.strip():
            error = dig(reply, "error")
            raise ProviderError(f"Ollama returned no text{f' ({error})' if error else ''}.")
        return text


def _has_model(pulled: set[str], wanted: str) -> bool:
    """``qwen2.5:7b`` must match exactly; an untagged name matches ``name:latest``."""
    if wanted in pulled:
        return True
    if ":" not in wanted:
        return f"{wanted}:latest" in pulled
    return False
