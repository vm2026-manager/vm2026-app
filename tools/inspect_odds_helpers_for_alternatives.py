from pathlib import Path

txt = Path("index.html").read_text(encoding="utf-8")
lines = txt.splitlines()

terms = [
    "nextOpponentText",
    "winChance",
    "winner",
    "chance",
    "vinder",
    "odds",
    "matchOdds",
    "getOpponent",
    "getNext"
]

for i, line in enumerate(lines, start=1):
    low = line.lower()
    if any(t.lower() in low for t in terms):
        print(f"{i}: {line[:220]}")
