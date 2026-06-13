from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    "FORMATION",
    "formations",
    "formationPositions",
    "slotPositions",
    "playerSlots",
    "top:",
    "left:",
    "renderPitch",
    "renderSquad",
    "position: absolute",
    "slot.style.top",
    "slot.style.left",
    "dataset.slot",
    "formationKey",
    "3-4-3",
    "4-4-2",
    "4-3-3",
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

    for i in hits[:20]:
        start = max(0, i - 12)
        end = min(len(lines), i + 28)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
