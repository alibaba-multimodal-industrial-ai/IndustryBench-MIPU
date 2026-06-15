"""Utility functions for parsing LLM responses."""

import json
import re
from typing import Any


def strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```...```) wrapping JSON output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def parse_json_response(text: str) -> dict | None:
    """Parse an LLM response string into a dict.

    Handles code-fenced JSON, raw JSON, and nested output structures.
    Returns None if parsing fails.
    """
    if not text:
        return None

    cleaned = strip_code_fence(text)
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(parsed, dict):
        return parsed
    return None
