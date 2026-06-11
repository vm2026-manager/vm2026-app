from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    "TIME_AWARE_STRATEGY_BUTTONS_JS_START",
    "function renderStrategyButtons",
    "strategyButtons.innerHTML",
    "Object.keys",
    "ensureActiveStrategyVisible",
    "STRATEGY",
    "fixtures",
    "stage",
    "GROUP_ROUND_FIXTURE_COUNT",
]

seen = set()

for pat in patterns:
    print("\n" + "="*90)
    print("PATTERN:", pat)
    print("="*90)

    hits = [i for i, line in enumerate(lines) if pat.lower() in line.lower()]
    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:12]:
        start = max(0, i - 10)
        end = min(len(lines), i + 25)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
