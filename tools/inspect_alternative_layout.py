from pathlib import Path

p = Path("index.html")
txt = p.read_text(encoding="utf-8")

terms = [
    "alternative",
    "alternativ",
    "alt-player",
    "player-alt",
    "candidate",
    "Vælg",
    "gebyr"
]

lines = txt.splitlines()
hits = []
for i, line in enumerate(lines, start=1):
    low = line.lower()
    if any(t.lower() in low for t in terms):
        hits.append((i, line[:220]))

print("=== relevante linjer ===")
for i, line in hits[-220:]:
    print(f"{i}: {line}")
