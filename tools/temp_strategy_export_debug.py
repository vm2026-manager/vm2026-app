from pathlib import Path
import re

html = Path("index.html").read_text(encoding="utf-8", errors="replace")

patterns = [
    "USER_STRATEGIES",
    "activeStrategyKey",
    "strategy",
    "Vælg optimalt hold",
    "fill",
    "optimal",
    "selectOptimal",
    "optimize",
    "FORMATION",
    "formation",
    "optimalSquads",
    "bestSquad",
]

out = []

for pattern in patterns:
    out.append(f"\n\n=== {pattern} ===\n")
    for m in re.finditer(re.escape(pattern), html, flags=re.IGNORECASE):
        start = max(0, m.start() - 600)
        end = min(len(html), m.end() + 900)
        out.append(f"\n--- char {m.start()} ---\n")
        out.append(html[start:end])
        if out.count(f"\n\n=== {pattern} ===\n") > 1:
            pass

Path("strategy_export_debug.txt").write_text("\n".join(out[:300]), encoding="utf-8")
print("Skrev strategy_export_debug.txt")
