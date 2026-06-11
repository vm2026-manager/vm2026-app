import csv
import json
from pathlib import Path
from datetime import datetime

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ev_backup = DATA / f"player_ev_group_stage_v1.backup_before_mbappe_longrun_nudge_{timestamp}.csv"
pool_backup = DATA / f"player_pool_v1.backup_before_mbappe_longrun_nudge_{timestamp}.json"

ev_backup.write_text(EV_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
pool_backup.write_text(POOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")

TARGET_ID = "kylian_mbapp__fra"

# Kun Mbappé. Målet er ikke lock, men at han kan komme med i enkelte long-run/premium-hold.
GLOBAL_MULT = 1.08
PRICE_QUALITY_MULT = 1.16
BASE_BEFORE_PRICE_QUALITY_MULT = 1.08

NOTE = "Long-run premium nudge: Mbappe skal kunne optræde i enkelte langsigtede/premiumvarianter uden at blive lock."

EXTRA_FIELDS = [
    "mbappe_longrun_nudge",
    "mbappe_longrun_nudge_version",
    "mbappe_longrun_nudge_note",
]

def norm(s):
    return str(s or "").strip().lower()

def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def fmt(v):
    return f"{v:.12g}"

def is_mbappe(row):
    return norm(row.get("player_id")) == TARGET_ID or "mbapp" in norm(row.get("player_name"))

def mult(row, fields, factor):
    for f in fields:
        if f in row:
            old = to_float(row.get(f))
            if old is not None:
                row[f] = fmt(old * factor)

def apply(row):
    if not is_mbappe(row):
        return False

    # Almindelig optimizer/base lidt op
    mult(row, [
        "weighted_group_stage_ev",
        "optimizer_ev",
        "display_score",
        "score",
        "strategy_score",
    ], GLOBAL_MULT)

    # De felter der især kan betyde noget for long-run/value-afvejning
    mult(row, [
        "price_quality_ev",
        "price_quality_raw_ev",
        "price_quality_appearance_scaled_ev",
        "price_quality_base_capped_ev",
    ], PRICE_QUALITY_MULT)

    mult(row, [
        "weighted_group_stage_ev_before_price_quality",
        "optimizer_ev_before_price_quality",
        "model_ev_before_price_quality",
    ], BASE_BEFORE_PRICE_QUALITY_MULT)

    row["mbappe_longrun_nudge"] = "true"
    row["mbappe_longrun_nudge_version"] = "v1"
    row["mbappe_longrun_nudge_note"] = NOTE
    return True

# EV CSV
with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames or [])
    rows = list(reader)

for f in EXTRA_FIELDS:
    if f not in fields:
        fields.append(f)

ev_changed = []
for row in rows:
    if apply(row):
        ev_changed.append(row.get("player_name"))

with EV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# Player pool JSON
raw = json.loads(POOL_PATH.read_text(encoding="utf-8"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

pool_changed = []
for row in players:
    if isinstance(row, dict) and apply(row):
        pool_changed.append(row.get("player_name"))

POOL_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

print("Mbappe long-run nudge applied")
print("EV backup:", ev_backup)
print("Pool backup:", pool_backup)
print("EV changed:", ev_changed)
print("Pool changed:", pool_changed)
