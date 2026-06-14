import pandas as pd
from pathlib import Path

p = Path("data/match_odds.csv")
df = pd.read_csv(p)

print("Kolonner:")
print(df.columns.tolist())
print()
print("Antal rækker:", len(df))
print()
print("Første 15 rækker:")
print(df.head(15).to_string())
print()
print("Sidste 15 rækker:")
print(df.tail(15).to_string())
