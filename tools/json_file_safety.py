from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


ILLEGAL_JSON_TOKEN_RE = re.compile(r'(?<!["\w])(?:NaN|Infinity|-Infinity)(?!["\w])')


def _raise_illegal_constant(value: str) -> None:
    raise ValueError(f"Illegal JSON constant: {value}")


def strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_raise_illegal_constant)


def find_illegal_json_tokens(text: str) -> list[str]:
    return sorted(set(match.group(0) for match in ILLEGAL_JSON_TOKEN_RE.finditer(text)))


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return [sanitize_for_json(item) for item in value]
    if value is None:
        return None
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return sanitize_for_json(value.item())
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dumps_json_strict(data: Any, *, ensure_ascii: bool = False, indent: int = 2) -> str:
    sanitized = sanitize_for_json(data)
    text = json.dumps(sanitized, ensure_ascii=ensure_ascii, indent=indent, allow_nan=False)
    tokens = find_illegal_json_tokens(text)
    if tokens:
        raise ValueError(f"Illegal JSON tokens after dump: {tokens}")
    strict_json_loads(text)
    return text


def write_json_strict(path: Path, data: Any, *, ensure_ascii: bool = False, indent: int = 2, encoding: str = "utf-8") -> None:
    text = dumps_json_strict(data, ensure_ascii=ensure_ascii, indent=indent)
    path.write_text(text, encoding=encoding)
    assert_json_file_strict(path)


def assert_json_file_strict(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    tokens = find_illegal_json_tokens(text)
    if tokens:
        raise ValueError(f"{path} contains illegal JSON tokens: {tokens}")
    strict_json_loads(text)

