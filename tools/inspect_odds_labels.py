import pandas as pd, json
from pathlib import Path

print("=== data_freshness.json ===")
p = Path("data/data_freshness.json")
print(p.read_text(encoding="utf-8") if p.exists() else "Mangler")

print("\n=== match_odds.csv Unibet labels ===")
df = pd.read_csv("data/match_odds.csv")
print(df[["source","odds_fetched_label"]].drop_duplicates().to_string(index=False))

print("\n=== match_odds_probs.csv labels ===")
df2 = pd.read_csv("data/match_odds_probs.csv")
print(df2[["source","odds_fetched_label"]].drop_duplicates().to_string(index=False))

print("\n=== fixture_strength_multipliers source sample ===")
df3 = pd.read_csv("data/fixture_strength_multipliers.csv")
print(df3[["match_id","home","away","source"]].head(12).to_string(index=False))
