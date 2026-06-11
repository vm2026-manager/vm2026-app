from pathlib import Path

p = Path("index.html")
text = p.read_text(encoding="utf-8")

terms = [
    "manual_bank",
    "manualBank",
    "getDisplayedBankLeft",
    "getAutofillBankLeft",
    "getBankLeft",
    "Aktuel bank",
    "fx 1,2",
    "bank"
]

lines = text.splitlines()

for term in terms:
    print("\n" + "="*80)
    print("TERM:", term)
    print("="*80)
    hits = [i for i, line in enumerate(lines) if term in line]
    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:8]:
        start = max(0, i - 8)
        end = min(len(lines), i + 12)
        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
