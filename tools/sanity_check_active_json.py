import re
from pathlib import Path

from json_file_safety import find_illegal_json_tokens, strict_json_loads

BAD_VALUE_BY_FIELD_RE = re.compile(r'"([^"]+)"\s*:\s*(NaN|Infinity|-Infinity)')


def active_json_files() -> list[Path]:
    files = []
    for path in sorted(Path("data").glob("*.json")):
        name = path.name.lower()
        if "backup" in name:
            continue
        if name.endswith("_preview.json"):
            continue
        files.append(path)
    return files


def illegal_field_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field, token in BAD_VALUE_BY_FIELD_RE.findall(text):
        key = f"{field}={token}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    bad = []

    for p in active_json_files():
        text = p.read_text(encoding="utf-8-sig")
        tokens = find_illegal_json_tokens(text)
        if tokens:
            field_counts = illegal_field_counts(text)
            details = ", ".join(f"{field} x{count}" for field, count in sorted(field_counts.items()))
            suffix = f" ({details})" if details else ""
            bad.append((str(p), f"Illegal JSON tokens {tokens}{suffix}"))

        try:
            strict_json_loads(text)
        except Exception as e:
            bad.append((str(p), f"JSON parse error: {e}"))

    print("=== Active JSON sanity ===")
    if bad:
        for path, err in bad:
            print("BAD", path, err)
        return 1

    print(f"OK: {len(active_json_files())} aktive data/*.json parser strict og har ingen NaN/Infinity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
