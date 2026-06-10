import csv
import json
from pathlib import Path

DATA = Path("data")
EV = DATA / "player_ev_group_stage_v1.csv"
POOL = DATA / "player_pool_v1.json"

TARGETS = [
    "erling_haaland__nor",
    "harry_kane__eng",
    "kylian_mbappe__fra",
    "michael_olise__fra",
    "ousmane_dembele__fra",
    "mikel_oyarzabal__esp",
    "kai_havertz__ger",
    "vinicius_junior__bra",
    "raul_jimenez__mex",
    "marko_arnautovic__aut",
    "cristiano_ronaldo__por",
    "lionel_messi__arg",
    "jonathan_david__can",
]

def norm(x):
    return str(x or "").strip().lower()

def val(row, *keys):
    for k in keys:
        if row.get(k) not in [None, ""]:
            return row.get(k)
    return ""

def rows_from_csv():
    with EV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

rows = rows_from_csv()

print("=== EV CSV TARGET CHECK ===")
for tid in TARGETS:
    matches = [r for r in rows if norm(r.get("player_id")) == tid]
    print()
    print(tid, "matches:", len(matches))
    for r in matches[:3]:
        print("  name:", val(r, "player_name", "name"))
        print("  pos:", val(r, "position", "holdet_position"))
        print("  price:", val(r, "price", "price_estimate", "holdet_price"))
        print("  start:", val(r, "start_probability_pct", "start_prob", "start_security"))
        print("  strategy_score:", val(r, "strategy_score"))
        print("  optimizer_ev:", val(r, "optimizer_ev"))
        print("  weighted_group_stage_ev:", val(r, "weighted_group_stage_ev"))
        print("  round1_ev:", val(r, "round1_ev"))
        print("  is_out:", val(r, "holdet_is_out", "is_out"))
        print("  rebalance:", val(r, "forward_model_rebalance_version"), val(r, "forward_model_rebalance_multiplier"))
