from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    r"manualBankInput",
    r"manual_bank_millions",
    r"manualBank",
    r"bankValue",
    r"spentValue",
    r"BUDGET",
    r"getBankLeft",
    r"getDisplayedBankLeft",
    r"getAutofillBankLeft",
    r"bankLeft",
    r"bank left",
    r"Restbank",
    r"save.*bank",
    r"load.*bank",
]

seen = set()

for pat in patterns:
    print("\n" + "="*90)
    print("PATTERN:", pat)
    print("="*90)

    hits = []
    rx = re.compile(pat, re.IGNORECASE)

    for i, line in enumerate(lines):
        if rx.search(line):
            hits.append(i)

    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:12]:
        key = (max(0, i-8), min(len(lines), i+14))
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(key[0], key[1]):
            print(f"{j+1:5}: {lines[j]}")
