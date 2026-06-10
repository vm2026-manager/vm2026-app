import csv
from pathlib import Path

EV = Path("data/player_ev_group_stage_v1.csv")

SEARCHES = [
    "mbapp",
    "dembele",
    "dembélé",
    "vinici",
    "raul",
    "raúl",
    "jimenez",
    "jiménez",
    "arnautovic",
    "arnautović",
    "marko",
]

def norm(x):
    return str(x or "").lower().replace("é", "e").replace("è", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ć", "c").strip()

with EV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("=== NAME / PLAYER_ID SEARCH ===")
for s in SEARCHES:
    print()
    print("SEARCH:", s)
    found = []
    ns = norm(s)
    for r in rows:
        hay = " | ".join([
            str(r.get("player_id", "")),
            str(r.get("player_name", "")),
            str(r.get("name", "")),
            str(r.get("team_id", "")),
            str(r.get("team_name", "")),
            str(r.get("position", "")),
            str(r.get("holdet_position", "")),
        ])
        if ns in norm(hay):
            found.append(r)

    print("matches:", len(found))
    for r in found[:20]:
        print(
            "  id=", r.get("player_id"),
            "| name=", r.get("player_name") or r.get("name"),
            "| team=", r.get("team_id") or r.get("team_name"),
            "| pos=", r.get("position") or r.get("holdet_position"),
            "| price=", r.get("price") or r.get("price_estimate") or r.get("holdet_price"),
            "| start=", r.get("start_probability_pct") or r.get("start_prob") or r.get("start_security"),
            "| ev=", r.get("optimizer_ev") or r.get("weighted_group_stage_ev") or r.get("score"),
            "| out=", r.get("holdet_is_out") or r.get("is_out"),
        )
