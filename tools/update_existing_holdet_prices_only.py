import json
import pandas as pd
from pathlib import Path
from datetime import datetime

data = Path("data")
pool_file = data / "player_pool_v1.json"
holdet_file = data / "holdet_players_game_616_flat.csv"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = data / f"player_pool_v1.backup_before_existing_price_update_{stamp}.json"
report = data / f"existing_player_price_update_report_{stamp}.csv"

pool = json.loads(pool_file.read_text(encoding="utf-8"))
holdet = pd.read_csv(holdet_file)

def norm_name(x):
    return str(x).strip().casefold()

def norm_pos(x):
    return str(x).strip().upper()

holdet_map = {}
for _, r in holdet.iterrows():
    key = (norm_name(r.get("player_name")), norm_pos(r.get("position")))
    holdet_map[key] = r

changes = []
updated = 0

for row in pool:
    key = (norm_name(row.get("player_name")), norm_pos(row.get("position")))
    h = holdet_map.get(key)
    if h is None:
        continue

    old = row.get("holdet_price", row.get("price"))
    new = h.get("price")

    try:
        old_num = float(old)
        new_num = float(new)
    except Exception:
        continue

    if abs(old_num - new_num) < 0.000001:
        continue

    changes.append({
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_name": row.get("team_name"),
        "position": row.get("position"),
        "old_holdet_price": old_num,
        "new_holdet_price": new_num,
        "diff": new_num - old_num,
    })

    row["holdet_price"] = int(new_num)
    row["price"] = int(new_num)
    row["price_estimate"] = int(new_num)
    updated += 1

backup.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
pool_file.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

pd.DataFrame(changes).to_csv(report, index=False, encoding="utf-8-sig")

print("Backup:", backup)
print("Report:", report)
print("Opdaterede eksisterende spillere:", updated)
print()
if changes:
    df = pd.DataFrame(changes).sort_values("diff", ascending=False)
    print(df.head(40).to_string(index=False))
