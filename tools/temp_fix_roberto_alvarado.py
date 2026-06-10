import csv
from pathlib import Path

path = Path("data/start_signal_context_overrides.csv")

rows = []
if path.exists():
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
else:
    fieldnames = []

needed = [
    "player_name",
    "team",
    "start_prob",
    "conditional_start_prob",
    "reason",
]

for col in needed:
    if col not in fieldnames:
        fieldnames.append(col)

target_names = {"Roberto Alvarado", "R. Alvarado", "Alvarado"}
found = False

for row in rows:
    name = (row.get("player_name") or row.get("name") or "").strip()
    team = (row.get("team") or row.get("country") or "").strip()

    if name in target_names or (name == "Roberto Alvarado" and team in {"Mexico", "MEX"}):
        row["player_name"] = "Roberto Alvarado"
        row["team"] = "Mexico"
        row["start_prob"] = "0.78"
        row["conditional_start_prob"] = "0.86"
        row["reason"] = "Bold+ lineup: remove check-start warning; treated as likely starter"
        found = True

if not found:
    rows.append({
        "player_name": "Roberto Alvarado",
        "team": "Mexico",
        "start_prob": "0.78",
        "conditional_start_prob": "0.86",
        "reason": "Bold+ lineup: remove check-start warning; treated as likely starter",
    })

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Updated Roberto Alvarado override in", path)
