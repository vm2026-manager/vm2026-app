import json
import csv
from pathlib import Path

DATA = Path("data")
TARGET_PLAYER = "Arya Yousefi"
TARGET_TEAM_CODES = {"IRN", "Iran", "IRAN"}
TARGET_OPP_NAMES = {"New Zealand", "New Zealand "}

def norm(s):
    return str(s or "").strip().lower()

def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

print("=== 1) Finder spiller i player_pool ===")
pool_path = DATA / "player_pool_v1.json"
pool_raw = read_json(pool_path)
players = pool_raw.get("players", pool_raw) if isinstance(pool_raw, dict) else pool_raw

matches = [
    p for p in players
    if isinstance(p, dict) and norm(p.get("player_name")) == norm(TARGET_PLAYER)
]

print("Antal Arya Yousefi i player_pool:", len(matches))
for p in matches:
    for k in [
        "player_id", "player_name", "team_id", "team_name", "flag_code",
        "position", "price", "match_1_opponent_team", "match_2_opponent_team",
        "match_3_opponent_team", "match_1_weighted_match_ev",
        "match_2_weighted_match_ev", "match_3_weighted_match_ev",
        "next_opponent", "next_match_opponent"
    ]:
        if k in p:
            print(f"{k}: {p.get(k)}")

print("\n=== 2) Søger Iran/New Zealand i fixtures-filer ===")
fixture_candidates = [
    DATA / "fixtures_group.csv",
    DATA / "fixtures.csv",
    DATA / "match_fixtures.csv",
    DATA / "worldcup_fixtures.csv",
]

for path in fixture_candidates:
    if not path.exists():
        continue

    rows = read_csv(path)
    print(f"\nFIL: {path} ({len(rows)} rækker)")
    if rows:
        print("Kolonner:", list(rows[0].keys()))

    hits = []
    for r in rows:
        blob = " | ".join(str(v) for v in r.values())
        if "Iran" in blob or "IRN" in blob or "New Zealand" in blob or "NZL" in blob:
            hits.append(r)

    print("Hits Iran/New Zealand:", len(hits))
    for r in hits[:20]:
        print(r)

print("\n=== 3) Søger Iran/New Zealand i odds-filer ===")
odds_candidates = [
    DATA / "match_odds.csv",
    DATA / "match_odds_probs.csv",
    DATA / "match_ev_inputs.csv",
    DATA / "fixture_strength_multipliers.csv",
]

for path in odds_candidates:
    if not path.exists():
        continue

    rows = read_csv(path)
    print(f"\nFIL: {path} ({len(rows)} rækker)")
    if rows:
        print("Kolonner:", list(rows[0].keys()))

    hits = []
    for r in rows:
        blob = " | ".join(str(v) for v in r.values())
        if "Iran" in blob or "IRN" in blob or "New Zealand" in blob or "NZL" in blob:
            hits.append(r)

    print("Hits Iran/New Zealand:", len(hits))
    for r in hits[:20]:
        print(r)

print("\n=== 4) Søger tooltip-/upcoming-funktioner i index.html ===")
index = Path("index.html")
text = index.read_text(encoding="utf-8", errors="replace")
terms = [
    "Ingen kommende gruppekampe",
    "Ingen kommende kampe",
    "Næste 3 kampe",
    "upcoming",
    "fixture",
    "matchOdds",
    "odds",
    "nextOpponent",
    "match_1_opponent_team",
]
lines = text.splitlines()
for term in terms:
    hits = [i for i, line in enumerate(lines, start=1) if term.lower() in line.lower()]
    print(f"\nTERM: {term} | hits: {len(hits)}")
    for ln in hits[:10]:
        start = max(1, ln - 4)
        end = min(len(lines), ln + 6)
        print(f"\n--- around line {ln} ---")
        for n in range(start, end + 1):
            print(f"{n:5}: {lines[n-1]}")

out = DATA / "audit_missing_odds_arya_yousefi.txt"
# Gemmer terminalens vigtigste output kræver ikke ekstra, men markerer filnavn til evt. senere.
print("\nDiagnose færdig. Kopiér outputtet herind.")
