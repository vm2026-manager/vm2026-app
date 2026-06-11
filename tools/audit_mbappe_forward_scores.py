import json
import csv
from pathlib import Path

DATA = Path("data")
POOL = DATA / "player_pool_v1.json"
EV = DATA / "player_ev_group_stage_v1.csv"

NAMES = [
    "Kylian Mbappe",
    "Kylian Mbappé",
    "Mikel Oyarzabal",
    "Michael Olise",
    "Kai Havertz",
    "Cody Gakpo",
    "Donyell Malen",
    "Breel Embolo",
    "Julian Alvarez",
    "Alexander Sørloth",
    "Harry Kane",
    "Cristiano Ronaldo",
]

FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "price",
    "holdet_price",
    "start_prob",
    "start_probability_pct",
    "weighted_group_stage_ev",
    "optimizer_ev",
    "display_score",
    "price_quality_ev",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "match_1_weighted_match_ev",
    "match_2_weighted_match_ev",
    "match_3_weighted_match_ev",
    "forward_micro_adjust",
    "forward_micro_adjust_multiplier",
    "forward_micro_adjust_note",
    "is_out",
    "holdet_is_out",
]

def norm(s):
    return str(s or "").lower().replace("é", "e").replace("ø", "o").strip()

def wanted(row):
    n = norm(row.get("player_name") or row.get("name"))
    return any(norm(x) == n for x in NAMES)

def show_row(source, row):
    print("\n" + "=" * 100)
    print(source, "|", row.get("player_name") or row.get("name"))
    for f in FIELDS:
        if f in row:
            print(f"{f}: {row.get(f)}")

raw = json.loads(POOL.read_text(encoding="utf-8"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

print("\n### PLAYER_POOL ###")
for row in players:
    if isinstance(row, dict) and wanted(row):
        show_row("POOL", row)

print("\n\n### EV CSV ###")
with EV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if wanted(row):
        show_row("EV", row)

print("\n\n### MBAPPE NAME SEARCH ###")
for row in players:
    if isinstance(row, dict) and "mbapp" in norm(row.get("player_name") or row.get("name")):
        show_row("POOL_MBAPPE_SEARCH", row)

for row in rows:
    if "mbapp" in norm(row.get("player_name") or row.get("name")):
        show_row("EV_MBAPPE_SEARCH", row)
