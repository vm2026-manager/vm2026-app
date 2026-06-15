import json
import pandas as pd
from pathlib import Path

data = Path("data")

# Find nyeste Holdet flat/csv-fil for game 616
candidates = sorted(
    list(data.glob("*holdet*616*flat*.csv")) +
    list(data.glob("*holdet*players*616*.csv")) +
    list(data.glob("holdet_players_game_616*.csv")),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not candidates:
    raise SystemExit("Fandt ingen Holdet CSV for game 616 i data-mappen.")

holdet_file = candidates[0]
pool_file = data / "player_pool_v1.json"

print("Holdet-fil:", holdet_file)

holdet = pd.read_csv(holdet_file)
pool = pd.read_json(pool_file)

# Normaliser kolonner
holdet["key"] = (
    holdet["player_name"].astype(str).str.strip().str.lower()
    + "||"
    + holdet["position"].astype(str).str.strip().str.upper()
)

pool["key"] = (
    pool["player_name"].astype(str).str.strip().str.lower()
    + "||"
    + pool["position"].astype(str).str.strip().str.upper()
)

# Find relevante prisfelter
holdet_price_col = "price" if "price" in holdet.columns else "holdet_price"
pool_price_col = "holdet_price" if "holdet_price" in pool.columns else "price"

merged = pool.merge(
    holdet[["key", "player_name", "team_name", "position", holdet_price_col, "is_out"]].rename(columns={
        "player_name": "holdet_name",
        "team_name": "holdet_team",
        "position": "holdet_position",
        holdet_price_col: "new_holdet_price",
        "is_out": "new_is_out",
    }),
    on="key",
    how="inner"
)

merged["old_price"] = pd.to_numeric(merged[pool_price_col], errors="coerce")
merged["new_price"] = pd.to_numeric(merged["new_holdet_price"], errors="coerce")
merged["price_diff"] = merged["new_price"] - merged["old_price"]

changes = merged[merged["price_diff"].abs() > 0.000001].copy()
changes = changes.sort_values(["price_diff", "player_name"], ascending=[False, True])

cols = [
    "player_name",
    "team_name",
    "position",
    "old_price",
    "new_price",
    "price_diff",
    "holdet_is_out",
    "new_is_out",
]

out = data / "existing_player_price_changes_latest.csv"
changes[cols].to_csv(out, index=False, encoding="utf-8-sig")

print()
print("Eksisterende spillere matchet:", len(merged))
print("Prisændringer:", len(changes))
print("Skrevet:", out)
print()

if len(changes):
    print(changes[cols].head(80).to_string(index=False))
else:
    print("Ingen prisændringer på eksisterende spillere.")
