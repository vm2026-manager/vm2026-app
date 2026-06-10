import csv
import json
from pathlib import Path

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

# v5: kamp-/felt-specifik kalibrering.
# Hovedmål:
# - Haaland stærkere i runde 1.
# - Mbappé stadig premium, men ikke runde-1-lock mod Senegal.
# - Dembélé/Olise som alternativer, ikke locks.
# - Raul/Marko ikke value-locks.
OVERRIDES = {
    "erling_haaland__nor": {
        "global": 1.20,
        "match_1": 1.38,
        "match_2": 1.08,
        "match_3": 1.00,
        "start": None,
        "note": "Runde 1 mod Irak skal løftes tydeligt; ikke nødvendigvis long-run-lock.",
    },
    "kylian_mbapp__fra": {
        "global": 1.18,
        "match_1": 0.95,
        "match_2": 1.28,
        "match_3": 1.25,
        "start": None,
        "note": "Premium, men runde 1 mod Senegal skal ikke slå Haaland mod Irak.",
    },
    "ousmane_demb_l__fra": {
        "global": 1.25,
        "match_1": 1.02,
        "match_2": 1.28,
        "match_3": 1.22,
        "start": 0.70,
        "note": "Alternativ for Frankrig, men ikke fast makker til Mbappé.",
    },
    "michael_olise__fra": {
        "global": 1.42,
        "match_1": 1.00,
        "match_2": 1.22,
        "match_3": 1.18,
        "start": None,
        "note": "Langsigtet/Frankrig-alternativ, ikke runde-1-lock.",
    },
    "harry_kane__eng": {
        "global": 1.02,
        "match_1": 1.00,
        "match_2": 1.06,
        "match_3": 1.04,
        "start": None,
        "note": "Stærk gruppespiller, men ikke 7/7-lock.",
    },
    "vinicius_j_nior__bra": {
        "global": 0.96,
        "match_1": 0.94,
        "match_2": 1.00,
        "match_3": 1.02,
        "start": None,
        "note": "Stærk især fra runde 2/long-run, men ikke total-lock.",
    },
    "cristiano_ronaldo__por": {
        "global": 0.86,
        "match_1": 0.88,
        "match_2": 0.88,
        "match_3": 0.86,
        "start": None,
        "note": "Stadig relevant, men ikke automatisk locked.",
    },
    "lionel_messi__arg": {
        "global": 0.84,
        "match_1": 0.82,
        "match_2": 0.86,
        "match_3": 0.86,
        "start": None,
        "note": "Stadig relevant, men ikke gruppespils-lock.",
    },
    "jonathan_david__can": {
        "global": 0.68,
        "match_1": 0.68,
        "match_2": 0.68,
        "match_3": 0.68,
        "start": None,
        "note": "Canada/kvaldata overvurderer ham.",
    },
    "ra_l_jim_nez__mex": {
        "global": 0.62,
        "match_1": 0.58,
        "match_2": 0.64,
        "match_3": 0.64,
        "start": None,
        "note": "Valueangriber, men ikke standardvalg i næsten alle hold.",
    },
    "marko_arnautovi__aut": {
        "global": 0.66,
        "match_1": 0.62,
        "match_2": 0.66,
        "match_3": 0.66,
        "start": None,
        "note": "Valueangriber, men bør fylde markant mindre.",
    },
    "mikel_oyarzabal__esp": {
        "global": 1.00,
        "match_1": 1.00,
        "match_2": 1.04,
        "match_3": 1.02,
        "start": None,
        "note": "Spanien-angriber, især relevant over flere runder.",
    },
    "kai_havertz__ger": {
        "global": 1.18,
        "match_1": 1.15,
        "match_2": 1.08,
        "match_3": 1.04,
        "start": None,
        "note": "Tyskland/Curaçao-value, men ikke for dominerende.",
    },
    "nico_williams__esp": {
        "global": 0.55,
        "match_1": 0.55,
        "match_2": 0.55,
        "match_3": 0.55,
        "start": 0.56,
        "note": "Skades-/startusikkerhed.",
    },
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
    "forward_model_rebalance",
    "forward_model_rebalance_version",
    "forward_model_rebalance_multiplier",
    "forward_model_rebalance_note",
    "forward_model_rebalance_source",
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

def apply_multiplier(row, fields, multiplier):
    for field in fields:
        old = to_float(row.get(field))
        if old is not None:
            row[field] = fmt(old * multiplier)

def apply_to_row(row, source):
    player_id = norm(row.get("player_id"))
    if player_id not in OVERRIDES:
        return False

    ov = OVERRIDES[player_id]

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

    row["forward_model_rebalance"] = "true"
    row["forward_model_rebalance_version"] = "v5"
    row["forward_model_rebalance_multiplier"] = (
        f"global={ov['global']};m1={ov['match_1']};m2={ov['match_2']};m3={ov['match_3']}"
    )
    row["forward_model_rebalance_note"] = ov["note"]
    row["forward_model_rebalance_source"] = source

    return True

with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

for field in EXTRA_FIELDS:
    if field not in fieldnames:
        fieldnames.append(field)

ev_changed = 0
changed_names = []

for row in rows:
    if apply_to_row(row, "player_ev_group_stage_v1"):
        ev_changed += 1
        changed_names.append(row.get("player_name") or row.get("name") or row.get("player_id"))

with EV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

raw = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

pool_changed = 0
for row in players:
    if isinstance(row, dict) and apply_to_row(row, "player_pool_v1"):
        pool_changed += 1

POOL_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

print("Forward rebalance v5 applied")
print("EV rows changed:", ev_changed)
print("Pool rows changed:", pool_changed)
print("Changed EV players:")
for n in changed_names:
    print("-", n)
