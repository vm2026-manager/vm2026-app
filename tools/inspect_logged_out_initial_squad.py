from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    "localStorage",
    "saveState",
    "loadState",
    "STORAGE",
    "squad_json",
    "manual_bank_millions",
    "currentUser",
    "initAuth",
    "init()",
    "clearLocalSquadViewAfterLogout",
    "render();",
]

seen = set()

for pat in patterns:
    print("\n" + "="*90)
    print("PATTERN:", pat)
    print("="*90)

    rx = re.compile(re.escape(pat), re.IGNORECASE)
    hits = [i for i, line in enumerate(lines) if rx.search(line)]

    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:18]:
        start = max(0, i - 10)
        end = min(len(lines), i + 18)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
