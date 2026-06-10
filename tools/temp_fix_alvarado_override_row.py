import csv
from pathlib import Path

path = Path("data/start_signal_context_overrides.csv")

with path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

# Sørg for at alle nødvendige kolonner findes
needed = [
    "player_id",
    "player_name",
    "team_id",
    "start_prob",
    "conditional_start_prob",
    "appearance_prob",
    "availability_prob",
    "availability_risk",
    "round_specific_rotation_risk",
    "source_note",
    "team",
    "reason",
]

for col in needed:
    if col not in fieldnames:
        fieldnames.append(col)

# Fjern alle gamle/fejlplacerede Roberto Alvarado-rækker
cleaned = []
for row in rows:
    name = (row.get("player_name") or "").strip().lower()
    pid = (row.get("player_id") or "").strip().lower()
    reason = (row.get("reason") or row.get("source_note") or "").strip().lower()

    is_alvarado = (
        "alvarado" in name
        or "alvarado" in pid
        or ("roberto alvarado" in reason)
    )

    if not is_alvarado:
        cleaned.append(row)

# Indsæt korrekt override-række
new_row = {col: "" for col in fieldnames}
new_row.update({
    "player_id": "roberto_alvarado__mex",
    "player_name": "Roberto Alvarado",
    "team_id": "MEX",
    "start_prob": "0.78",
    "conditional_start_prob": "0.86",
    "appearance_prob": "0.94",
    "availability_prob": "0.94",
    "availability_risk": "low_risk",
    "round_specific_rotation_risk": "medium",
    "source_note": "Bold+ expected Mexico XI: Roberto Alvarado treated as likely starter; remove check-start warning.",
    "team": "Mexico",
    "reason": "Bold+ expected Mexico XI: remove Tjek start warning.",
})

cleaned.append(new_row)

with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned)

print("Fixed Roberto Alvarado override row")
