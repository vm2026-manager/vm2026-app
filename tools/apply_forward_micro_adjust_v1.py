import csv
import json
from pathlib import Path
from datetime import datetime

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ev_backup = DATA / f"player_ev_group_stage_v1.backup_before_forward_micro_adjust_{timestamp}.csv"
pool_backup = DATA / f"player_pool_v1.backup_before_forward_micro_adjust_{timestamp}.json"

ev_backup.write_text(EV_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
pool_backup.write_text(POOL_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

# Lille korrektion efter diversity-output:
# - Vini skal fortsat være lav.
# - Mbappé skal ind på noget long-run/premium, ikke være lock.
# - Havertz lidt op pga. start-9'er for Tyskland til 5,5 mio.
# - Malen/Yamal/Lautaro ned.
OVERRIDES = {
    "kylian_mbapp__fra": {
        "global": 1.10,
        "match_1": 0.98,
        "match_2": 1.10,
        "match_3": 1.08,
        "start": None,
        "note": "Lille løft: dyr premium, skal med på noget langsigtet uden at blive lock.",
    },
    "kai_havertz__ger": {
        "global": 1.08,
        "match_1": 1.10,
        "match_2": 1.06,
        "match_3": 1.04,
        "start": None,
        "note": "Lille løft: billig start-9'er for Tyskland.",
    },
    "donyell_malen__ned": {
        "global": 0.72,
        "match_1": 0.72,
        "match_2": 0.72,
        "match_3": 0.72,
        "start": None,
        "note": "Ned: bruger ønsker helst ikke Malen i reelt output.",
    },
    "lamine_yamal__esp": {
        "global": 0.78,
        "match_1": 0.76,
        "match_2": 0.78,
        "match_3": 0.78,
        "start": None,
        "note": "Ned: dyr og skal ikke fylde i output.",
    },
    "lautaro_mart_nez__arg": {
        "global": 0.86,
        "match_1": 0.86,
        "match_2": 0.86,
        "match_3": 0.86,
        "start": 0.58,
        "note": "Ned: startusikkerhed mod Julian Alvarez; bør ikke fylde for meget.",
    },
}

# Fallback-søgning hvis id'er afviger lidt
NAME_FALLBACKS = {
    "donyell_malen__ned": ("Donyell Malen", "NED"),
    "lamine_yamal__esp": ("Lamine Yamal", "ESP"),
    "lautaro_mart_nez__arg": ("Lautaro Martinez", "ARG"),
}

GLOBAL_EV_FIELDS = [
    "optimizer_ev",
    "weighted_group_stage_ev",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "price_quality_ev",
    "display_score",
    "score",
    "strategy_score",
]

MATCH_FIELDS = {
    "match_1": ["match_1_weighted_match_ev"],
    "match_2": ["match_2_weighted_match_ev"],
    "match_3": ["match_3_weighted_match_ev"],
}

START_PROB_FIELDS = ["start_prob", "conditional_start_prob", "start_security"]
START_PCT_FIELDS = ["start_probability_pct"]

EXTRA_FIELDS = [
    "forward_micro_adjust",
    "forward_micro_adjust_version",
    "forward_micro_adjust_multiplier",
    "forward_micro_adjust_note",
]

def norm(value):
    return str(value or "").strip().lower()

def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None

def fmt(value):
    return f"{value:.12g}"

def player_key(row):
    pid = norm(row.get("player_id"))
    if pid in OVERRIDES:
        return pid

    name = str(row.get("player_name") or row.get("name") or "").strip()
    team = str(row.get("team_id") or row.get("team") or "").strip().upper()

    for key, (fallback_name, fallback_team) in NAME_FALLBACKS.items():
        if name.lower() == fallback_name.lower() and team == fallback_team:
            return key

    return None

def apply_multiplier(row, fields, multiplier):
    for field in fields:
        old = to_float(row.get(field))
        if old is not None:
            row[field] = fmt(old * multiplier)

def apply_to_row(row):
    key = player_key(row)
    if key not in OVERRIDES:
        return False

    ov = OVERRIDES[key]

    apply_multiplier(row, GLOBAL_EV_FIELDS, ov["global"])
    apply_multiplier(row, MATCH_FIELDS["match_1"], ov["match_1"])
    apply_multiplier(row, MATCH_FIELDS["match_2"], ov["match_2"])
    apply_multiplier(row, MATCH_FIELDS["match_3"], ov["match_3"])

    if ov["start"] is not None:
        start = ov["start"]
        for field in START_PROB_FIELDS:
            if field in row:
                row[field] = fmt(start)
        for field in START_PCT_FIELDS:
            if field in row:
                row[field] = fmt(start * 100)

    row["forward_micro_adjust"] = "true"
    row["forward_micro_adjust_version"] = "v1"
    row["forward_micro_adjust_multiplier"] = (
        f"global={ov['global']};m1={ov['match_1']};m2={ov['match_2']};m3={ov['match_3']}"
    )
    row["forward_micro_adjust_note"] = ov["note"]

    return True

with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

for field in EXTRA_FIELDS:
    if field not in fieldnames:
        fieldnames.append(field)

changed = []
for row in rows:
    if apply_to_row(row):
        changed.append(row.get("player_name") or row.get("name") or row.get("player_id"))

with EV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

raw = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

pool_changed = []
for row in players:
    if isinstance(row, dict) and apply_to_row(row):
        pool_changed.append(row.get("player_name") or row.get("name") or row.get("player_id"))

POOL_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

print("Forward micro adjust applied")
print("EV backup:", ev_backup)
print("Pool backup:", pool_backup)
print("EV changed:", len(changed), changed)
print("Pool changed:", len(pool_changed), pool_changed)
