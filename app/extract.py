"""Robust extraction of a JSON array of strings from a model's raw response.

The response may be clean JSON, wrapped in a markdown code fence, or padded
with prose. Try progressively: exact parse -> fence-stripped parse -> balanced
bracket scan -> regex. On total failure return status 'unparseable' with the
raw preserved so a better parser can retry later.
"""

import json
import re


def _strip_fence(raw):
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", raw, re.DOTALL)
    return m.group(1) if m else None


def _balanced_scan(raw):
    """Find the first '[' and walk to its matching ']', respecting strings and
    escapes. Returns the substring or None if unbalanced / not present."""
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def extract_json_array(raw):
    """Return (status, parsed|None) where status in {'ok','unparseable'}."""
    if not raw or not raw.strip():
        return "unparseable", None

    candidates = [raw, _strip_fence(raw), _balanced_scan(raw)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, list) and all(isinstance(s, str) for s in value):
            return "ok", value
        # A JSON array of objects/other types: still record as parsed? We only
        # want string arrays; treat non-conforming as unparseable.
        if isinstance(value, list):
            return "unparseable", None

    # Last resort: regex across the whole raw text.
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            value = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            value = None
        if isinstance(value, list) and all(isinstance(s, str) for s in value):
            return "ok", value
    return "unparseable", None