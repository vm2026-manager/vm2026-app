from pathlib import Path
import csv
import json

DATA = Path("data")

print("=== match_odds_probs ===")
with (DATA / "match_odds_probs.csv").open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("rows:", len(rows))
print("sources:", sorted(set(r.get("source", "") for r in rows)))

for match_id in ["IRN-NZL", "BEL-IRN", "EGY-IRN", "FRA-SEN", "IRQ-NOR"]:
    hit = next((r for r in rows if r.get("match_id") == match_id), None)
    if hit:
        print(
            match_id,
            hit.get("home_odds"),
            hit.get("draw_odds"),
            hit.get("away_odds"),
            hit.get("source"),
            hit.get("odds_fetched_label"),
        )
    else:
        print(match_id, "MANGLER")

print("")
print("=== data_freshness ===")
freshness = json.loads((DATA / "data_freshness.json").read_text(encoding="utf-8"))
for k in [
    "unibet_odds_fetched_label",
    "unibet_odds_fetched_at",
    "match_odds_source",
    "match_odds_updated_at",
]:
    print(k + ":", freshness.get(k))
