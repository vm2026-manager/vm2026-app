import csv
from pathlib import Path

EV = Path("data/player_ev_group_stage_v1.csv")

TARGETS = [
    "erling_haaland__nor",
    "kylian_mbapp__fra",
    "ousmane_demb_l__fra",
    "cristiano_ronaldo__por",
    "vinicius_j_nior__bra",
    "ra_l_jim_nez__mex",
    "harry_kane__eng",
    "michael_olise__fra",
    "mikel_oyarzabal__esp",
]

INTERESTING = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "price",
    "start_probability_pct",
    "start_prob",
    "start_security",
    "optimizer_ev",
    "weighted_group_stage_ev",
    "round1_ev",
    "match_1_weighted_match_ev",
    "match_2_weighted_match_ev",
    "match_3_weighted_match_ev",
    "team_long_run_score",
    "long_run_score",
    "strategy_score",
    "display_score",
    "score",
    "captain_score",
    "forward_model_rebalance_version",
    "forward_model_rebalance_multiplier",
]

def norm(x):
    return str(x or "").strip().lower()

with EV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("=== FORWARD FIELD DIAGNOSE ===")
print("Columns in EV file:", len(rows[0].keys()) if rows else 0)
print()

for tid in TARGETS:
    matches = [r for r in rows if norm(r.get("player_id")) == tid]
    print("=" * 80)
    print(tid, "matches:", len(matches))
    for r in matches:
        for col in INTERESTING:
            if col in r:
                print(f"{col}: {r.get(col)}")
        print()
