"""Offline mock provider (the default). No network, no model, fully deterministic.

Two modes:

* **Queued**: ``MockProvider(responses=[...])`` returns the given texts in order (tests).
* **Recorded**: ``MockProvider()`` looks up a recorded answer in ``torusplot/ai/recorded/*.json``
  from the ``TASK: <name>`` line of the prompt and keywords of the request, and otherwise
  falls back to a generic valid answer built from the table names in the prompt.

Every call is appended to ``.calls`` (system, prompt, number of images) for assertions.

Recorded file format::

    {"task": "describe_figure",            # required; matches the TASK line
     "keywords": ["box"],                  # all must occur in the request (case-insensitive)
     "table_kinds": ["column", "grouped"], # optional: only for these kinds of current table
     "response": {...}}                    # JSON answer; "{{CURRENT_TABLE}}" is filled in
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import Provider, ProviderError

RECORDED_DIR = Path(__file__).resolve().parent.parent / "recorded"
_PLACEHOLDER = "{{CURRENT_TABLE}}"
_TASK = re.compile(r"^TASK:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
_CURRENT = re.compile(r"^CURRENT_TABLE:\s*(.+?)\s*$", re.MULTILINE)
_REQUEST = re.compile(r"^REQUEST:\s*\n<<<\n(.*?)\n>>>", re.MULTILINE | re.DOTALL)


class MockProvider(Provider):
    """Deterministic stand-in for a language/vision model."""

    name = "mock"
    supports_vision = True

    def __init__(self, responses: list[str] | None = None, recorded_dir: str | Path | None = None) -> None:
        self._queue: list[str] | None = None if responses is None else [str(r) for r in responses]
        self.recorded_dir = Path(recorded_dir) if recorded_dir is not None else RECORDED_DIR
        self.calls: list[dict[str, Any]] = []
        self._recorded: list[dict[str, Any]] | None = None

    def available(self) -> tuple[bool, str]:
        return True, "Offline mock provider: recorded answers only, nothing leaves this computer."

    # ------------------------------------------------------------------ API
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        images: list[bytes] | None = None,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "n_images": len(images or []),
                "json_mode": json_mode,
                "max_tokens": max_tokens,
            }
        )
        if self._queue is not None:
            index = len(self.calls) - 1
            if index >= len(self._queue):
                raise ProviderError(
                    f"MockProvider was given {len(self._queue)} response(s) but was called {index + 1} times."
                )
            return self._queue[index]
        return self._recorded_answer(prompt)

    # ------------------------------------------------------------------ recorded mode
    def recorded(self) -> list[dict[str, Any]]:
        """Recorded answers, in file-name order (loaded once)."""
        if self._recorded is None:
            entries = []
            for path in sorted(self.recorded_dir.glob("*.json")):
                try:
                    entry = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ProviderError(f"Recorded mock response {path.name} is not valid JSON: {exc}") from None
                if not isinstance(entry, dict) or "task" not in entry or "response" not in entry:
                    raise ProviderError(f"Recorded mock response {path.name} needs 'task' and 'response'.")
                entry["_file"] = path.name
                entries.append(entry)
            self._recorded = entries
        return self._recorded

    def _recorded_answer(self, prompt: str) -> str:
        task_match = _TASK.search(prompt)
        task = task_match.group(1) if task_match else ""
        current = _current_table(prompt)
        request_match = _REQUEST.search(prompt)
        request = (request_match.group(1) if request_match else prompt).lower()
        kind = current.get("kind") if current else None

        best: dict[str, Any] | None = None
        best_score = -1
        for entry in self.recorded():
            if entry["task"] != task:
                continue
            kinds = entry.get("table_kinds")
            if kinds and kind not in kinds:
                continue
            keywords = [str(k).lower() for k in entry.get("keywords", [])]
            if not all(k in request for k in keywords):
                continue
            if len(keywords) > best_score:
                best, best_score = entry, len(keywords)
        if best is not None:
            text = json.dumps(best["response"], ensure_ascii=False)
            if _PLACEHOLDER in text:
                if not current:
                    return json.dumps(_generic(task, current))
                text = text.replace(_PLACEHOLDER, json.dumps(current["name"], ensure_ascii=False)[1:-1])
            return text
        return json.dumps(_generic(task, current), ensure_ascii=False)


def _current_table(prompt: str) -> dict[str, Any] | None:
    """The summary of the current table (or of the only table) found in a prompt."""
    summaries = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith('{"name":'):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "kind" in item:
                summaries.append(item)
    match = _CURRENT.search(prompt)
    if match:
        for item in summaries:
            if item["name"] == match.group(1):
                return item
    return summaries[0] if summaries else None


def _generic(task: str, current: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic fallback answers that are valid for each task."""
    if task == "describe_figure":
        if current is None:
            return {"plot_type": "bar", "data": {"table": "My table"}}
        from ..prompts import default_binding

        return default_binding(current)
    if task == "extract_style":
        return {
            "plot_type": None,
            "colors": [],
            "background": None,
            "text_color": None,
            "has_top_right_spines": None,
            "grid": None,
            "font_category": "sans",
            "bar_edge": None,
            "error_bars": None,
            "points_shown": None,
            "legend_position": None,
        }
    if task == "critique_figure":
        return {"patch": {}, "done": True, "comment": "Mock provider: no changes suggested."}
    return {"ok": True, "note": "Mock provider: no recorded answer for this task."}
