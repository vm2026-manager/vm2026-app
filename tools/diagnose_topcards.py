from pathlib import Path
import re

text = Path("index.html").read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

terms = [
    "Gns. start",
    "High risk",
    "avg",
    "risk",
    "start",
    "renderBudget",
    "getDisplayedBankLeft",
    "getBankLeft",
    "Spillere",
    "manualBank",
]

for term in terms:
    hits = [i for i, line in enumerate(lines, start=1) if term.lower() in line.lower()]
    print("\n" + "="*90)
    print(f"TERM: {term} | hits: {len(hits)}")
    for ln in hits[:20]:
        start = max(1, ln - 4)
        end = min(len(lines), ln + 6)
        print(f"\n--- around line {ln} ---")
        for n in range(start, end + 1):
            print(f"{n:5}: {lines[n-1]}")
