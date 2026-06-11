import json
import csv
from pathlib import Path

print("=== Odds sanity ===")
with Path("data/match_odds_probs.csv").open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("match_odds_probs rows:", len(rows))
print("sources:", sorted(set(r.get("source","") for r in rows)))

for home, away in [("IRN","NZL"), ("BEL","IRN"), ("EGY","IRN"), ("FRA","SEN"), ("IRQ","NOR")]:
    hit = [r for r in rows if r.get("home") == home and r.get("away") == away]
    if hit:
        r = hit[0]
        print(
            f"{home}-{away}:",
            r.get("home_win_odds"),
            r.get("draw_odds"),
            r.get("away_win_odds"),
            r.get("source"),
            r.get("odds_fetched_label")
        )
    else:
        print(f"{home}-{away}: MANGLER")

print("\n=== Optimizer sanity ===")
opt_path = Path("data/optimal_squads_by_strategy.json")
data = json.loads(opt_path.read_text(encoding="utf-8"))

count = 0
problems = []

for strategy, formations in data.items():
    if not isinstance(formations, dict):
        continue

    for formation, squad_data in formations.items():
        if not isinstance(squad_data, dict):
            continue

        players = squad_data.get("players") or squad_data.get("squad") or []
        if players:
            count += 1

        price = squad_data.get("total_price") or squad_data.get("price") or squad_data.get("total_cost")
        if price is not None:
            try:
                p = float(price)
                if p > 50000000.01 and p > 50.01:
                    problems.append((strategy, formation, "budget", price))
            except Exception:
                pass

print("optimizer-hold fundet:", count)
print("problemer:", problems[:20] if problems else "ingen simple budgetproblemer fundet")

print("\n=== Data freshness ===")
fresh_path = Path("data/data_freshness.json")
if fresh_path.exists():
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    for k in ["unibet_odds_fetched_label", "unibet_odds_fetched_at", "match_odds_source", "match_odds_updated_at"]:
        print(k + ":", fresh.get(k))
