from pathlib import Path
from datetime import datetime
import csv
import shutil

DATA = Path("data")
FILES = [
    DATA / "match_odds.csv",
    DATA / "match_odds_probs.csv",
]

NEW_LABEL = "14.06 kl. 12:31"
OLD_LABEL = "11.06 kl. 19:12"
SOURCE = "Unibet"


def clean(v):
    return "" if v is None else str(v).strip()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def match_key(row):
    mid = clean(row.get("match_id"))
    if mid:
        return ("match_id", mid)

    home = clean(row.get("home")).upper() or clean(row.get("home_team_id")).upper()
    away = clean(row.get("away")).upper() or clean(row.get("away_team_id")).upper()
    return ("pair", home, away)


for path in FILES:
    if not path.exists():
        print(f"Springer over, mangler: {path}")
        continue

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_before_dedupe_unibet_labels_{stamp}{path.suffix}")
    shutil.copy2(path, backup)

    rows, fields = read_csv(path)

    new_keys = {
        match_key(r)
        for r in rows
        if clean(r.get("source")) == SOURCE and clean(r.get("odds_fetched_label")) == NEW_LABEL
    }

    before = len(rows)
    out = []
    removed = 0

    for r in rows:
        is_old_unibet = clean(r.get("source")) == SOURCE and clean(r.get("odds_fetched_label")) == OLD_LABEL
        has_new_same_match = match_key(r) in new_keys

        if is_old_unibet and has_new_same_match:
            removed += 1
            continue

        out.append(r)

    write_csv(path, out, fields)

    print(f"{path}:")
    print(f"  backup: {backup.name}")
    print(f"  før: {before}")
    print(f"  efter: {len(out)}")
    print(f"  fjernede gamle Unibet-rækker med ny erstatning: {removed}")