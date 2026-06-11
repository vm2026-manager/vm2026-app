from pathlib import Path
import csv
import re

DATA = Path("data")

print("=== match_odds_probs struktur ===")
path = DATA / "match_odds_probs.csv"
with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("rows:", len(rows))
print("columns:", rows[0].keys() if rows else "INGEN RÆKKER")
print("sources:", sorted(set(r.get("source", "") for r in rows)))

print("\n=== Første 8 rækker ===")
for r in rows[:8]:
    print({k: r.get(k) for k in r.keys() if k.lower() in [
        "match_id", "home_team_id", "away_team_id", "home_team", "away_team",
        "home_odds", "draw_odds", "away_odds", "source",
        "odds_fetched_label", "odds_fetched_at"
    ]})

print("\n=== Søg efter relevante holdkombinationer ===")
pairs = [
    ("IRN", "NZL"),
    ("BEL", "IRN"),
    ("EGY", "IRN"),
    ("FRA", "SEN"),
    ("IRQ", "NOR"),
]

def row_text(r):
    return " ".join(str(v) for v in r.values())

for a, b in pairs:
    hits = [r for r in rows if a in row_text(r) and b in row_text(r)]
    print(f"\n{a}-{b}: {len(hits)} hit(s)")
    for r in hits[:3]:
        print({k: r.get(k) for k in r.keys() if k.lower() in [
            "match_id", "home_team_id", "away_team_id", "home_team", "away_team",
            "home_odds", "draw_odds", "away_odds", "source",
            "odds_fetched_label", "odds_fetched_at"
        ]})

print("\n=== index.html: tooltip/source/tidsstempel-kandidater ===")
html = Path("index.html").read_text(encoding="utf-8")
lines = html.splitlines()

patterns = [
    "odds_fetched_label",
    "odds_fetched_at",
    "source",
    "Unibet",
    "tooltip",
    "fixture",
    "oddsSource",
    "sourceLabel",
]

seen = set()
for pat in patterns:
    print("\n" + "="*80)
    print("PATTERN:", pat)
    print("="*80)
    rx = re.compile(re.escape(pat), re.IGNORECASE)
    hits = [i for i, line in enumerate(lines) if rx.search(line)]
    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:10]:
        start = max(0, i - 8)
        end = min(len(lines), i + 12)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
