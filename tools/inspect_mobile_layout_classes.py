from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    "squad-field",
    "player-card",
    "slot",
    "pitch",
    "field",
    "flag",
    "remove",
    "replace",
    "trade-row",
    "right-panel",
    "main-layout",
    "app-layout",
    "@media",
    "max-width: 700",
    "max-width: 820",
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

    for i in hits[:18]:
        start = max(0, i - 10)
        end = min(len(lines), i + 24)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
