import json
import re
from pathlib import Path

bad_tokens = re.compile(r'(?<![A-Za-z0-9_"\'])-Infinity(?![A-Za-z0-9_"\'])|(?<![A-Za-z0-9_"\'])Infinity(?![A-Za-z0-9_"\'])|(?<![A-Za-z0-9_"\'])NaN(?![A-Za-z0-9_"\'])')

bad = []

for p in sorted(Path("data").glob("*.json")):
    if "backup" in p.name.lower():
        continue

    text = p.read_text(encoding="utf-8-sig")

    hits = list(bad_tokens.finditer(text))
    if hits:
        bad.append((str(p), f"NaN/Infinity tokens: {len(hits)}"))
        continue

    try:
        json.loads(text)
    except Exception as e:
        bad.append((str(p), f"JSON parse error: {e}"))

print("=== Active JSON sanity ===")
if bad:
    for path, err in bad:
        print("BAD", path, err)
    raise SystemExit(1)

print("OK: Alle aktive data/*.json parser og har ingen NaN/Infinity.")
