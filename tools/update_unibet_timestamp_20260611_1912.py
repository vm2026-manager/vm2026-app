from pathlib import Path
from datetime import datetime
import csv
import json
import shutil

ROOT = Path(".")
DATA = ROOT / "data"

NEW_LABEL = "11.06 kl. 19:12"
NEW_FETCHED_AT = "2026-06-11T19:12:00+02:00"
RUN_UPDATED_AT = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

files = [
    DATA / "match_odds.csv",
    DATA / "match_odds_probs.csv",
    DATA / "data_freshness.json",
]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for path in files:
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup_before_unibet_timestamp_1912_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")

def update_csv(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        for col in ["odds_fetched_at", "odds_fetched_label"]:
            if col not in fieldnames:
                fieldnames.append(col)

        for row in reader:
            source = str(row.get("source", "")).strip().lower()
            if source == "unibet":
                row["odds_fetched_at"] = NEW_FETCHED_AT
                row["odds_fetched_label"] = NEW_LABEL
            rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    unibet_rows = sum(1 for r in rows if str(r.get("source", "")).strip().lower() == "unibet")
    print(f"Opdateret {path}: {unibet_rows} Unibet-rækker")

update_csv(DATA / "match_odds.csv")
update_csv(DATA / "match_odds_probs.csv")

freshness_path = DATA / "data_freshness.json"
if freshness_path.exists():
    freshness = json.loads(freshness_path.read_text(encoding="utf-8"))
else:
    freshness = {}

freshness["unibet_odds_fetched_label"] = NEW_LABEL
freshness["unibet_odds_fetched_at"] = NEW_FETCHED_AT
freshness["match_odds_source"] = "Unibet"
freshness["match_odds_updated_at"] = RUN_UPDATED_AT

freshness_path.write_text(
    json.dumps(freshness, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("")
print("OK: Unibet-tidsstempel opdateret.")
print(f"Label: {NEW_LABEL}")
print(f"Fetched at: {NEW_FETCHED_AT}")
print(f"Updated at: {RUN_UPDATED_AT}")
print("")
print("Bemærk: Selve odds-værdierne er ikke OCR-aflæst fra PNG'en. Kun tidsstemplet er opdateret sikkert.")
