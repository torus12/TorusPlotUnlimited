"""Pull a JSON object out of model or chatbot output.

Models wrap JSON in prose, markdown fences, "thinking" tags, comments and trailing commas.
:func:`extract_json` finds the first JSON *object* that parses, trying in order:

1. the whole text;
2. fenced code blocks (```json first, then any fence);
3. every balanced ``{...}`` span in the text (string-aware brace matching), left to right.

Each candidate is parsed strictly first and then once more after a conservative clean-up
(``//`` and ``/* */`` comments and trailing commas removed, outside strings only).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


class JSONExtractError(ValueError):
    """No JSON object could be found in the text."""


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text`` (raises :class:`JSONExtractError`)."""
    if not isinstance(text, str) or not text.strip():
        raise JSONExtractError("The answer is empty: expected one JSON object.")
    cleaned = _THINK.sub("", text).strip().lstrip("﻿")

    candidates: list[str] = [cleaned]
    fences = _FENCE.findall(cleaned)
    candidates += [body for lang, body in fences if lang.lower() in {"json", "json5", "jsonc"}]
    candidates += [body for lang, body in fences if lang.lower() not in {"json", "json5", "jsonc"}]
    for candidate in candidates:
        found = _parse_object(candidate.strip())
        if found is not None:
            return found
    for span in balanced_objects(cleaned):
        found = _parse_object(span)
        if found is not None:
            return found
    if "{" not in cleaned:
        raise JSONExtractError("No JSON object found: the answer contains no '{'.")
    raise JSONExtractError(
        "No valid JSON object found: check for missing quotes, commas or closing braces."
    )


def contains_json_object(text: str) -> bool:
    """True if :func:`extract_json` would succeed with a non-empty object."""
    try:
        return bool(extract_json(text))
    except JSONExtractError:
        return False


def _parse_object(candidate: str) -> dict[str, Any] | None:
    if not candidate.startswith("{"):
        return None
    for attempt in (candidate, strip_json_noise(candidate)):
        try:
            value = json.loads(attempt)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
        return None
    return None


def balanced_objects(text: str) -> list[str]:
    """Every top-level balanced ``{...}`` span, left to right (braces inside strings are ignored)."""
    spans: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        end = _match_brace(text, i)
        if end is None:
            i += 1
            continue
        spans.append(text[i : end + 1])
        i = end + 1
    return spans


def _match_brace(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def strip_json_noise(text: str) -> str:
    """Remove ``//`` / ``/* */`` comments and trailing commas that are outside strings."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif ch == ",":
            j = i + 1
            while j < n:
                if text[j].isspace():
                    j += 1
                elif text.startswith("//", j):
                    end = text.find("\n", j)
                    j = n if end < 0 else end
                elif text.startswith("/*", j):
                    end = text.find("*/", j + 2)
                    j = n if end < 0 else end + 2
                else:
                    break
            if j < n and text[j] in "}]":
                i += 1  # trailing comma: drop it
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)
