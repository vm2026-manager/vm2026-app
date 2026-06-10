import csv
import json
from pathlib import Path

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

# v6: topscorer-odds-kalibreret.
# Vini markant ned. Mbappé dyr men skal med på noget. Haaland/Kane mere naturligt.
OVERRIDES = {
    "erling_haaland__nor": {
        "global": 1.22,
        "match_1": 1.42,
        "match_2": 1.08,
        "match_3": 1.00,
        "start": None,
        "note": "Stærk runde 1 mod Irak; skal være klart relevant næste runde.",
    },
    "kylian_mbapp__fra": {
        "global": 1.08,
        "match_1": 0.82,
        "match_2": 1.12,
        "match_3": 1.10,
        "start": None,
        "note": "Dyr premium. Skal med på noget, men ikke være lock.",
    },
    "harry_kane__eng": {
        "global": 1.08,
        "match_1": 1.02,
        "match_2": 1.12,
        "match_3": 1.08,
        "start": None,
        "note": "Topscorerprofil og sikker rolle; relevant især gruppespil.",
    },
    "vinicius_j_nior__bra": {
        "global": 0.72,
        "match_1": 0.70,
        "match_2": 0.74,
        "match_3": 0.76,
        "start": None,
        "note": "Topscorer-odds/rolle siger, at Vini ikke skal fylde særlig meget.",
    },
    "cristiano_ronaldo__por": {
        "global": 0.82,
        "match_1": 0.82,
        "match_2": 0.82,
        "match_3": 0.80,
        "start": None,
        "note": "Relevant, men v5 låste ham for meget i 1+2.",
    },
    "lionel_messi__arg": {
        "global": 0.82,
        "match_1": 0.80,
        "match_2": 0.84,
        "match_3": 0.84,
        "start": None,
        "note": "Relevant, men ikke gruppespils-lock.",
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
        "global": 0.60,
        "match_1": 0.56,
        "match_2": 0.62,
        "match_3": 0.62,
        "start": None,
        "note": "Valueangriber, men skal ikke standardvælges.",
    },
    "marko_arnautovi__aut": {
        "global": 0.62,
        "match_1": 0.58,
        "match_2": 0.62,
        "match_3": 0.62,
        "start": None,
        "note": "Valueangriber, men skal fylde mindre.",
    },
    "mikel_oyarzabal__esp": {
        "global": 0.88,
        "match_1": 0.88,
        "match_2": 0.92,
        "match_3": 0.90,
        "start": None,
        "note": "Relevant, men v5 gjorde ham long-run-lock.",
    },
    "michael_olise__fra": {
        "global": 1.35,
        "match_1": 0.96,
        "match_2": 1.14,
        "match_3": 1.10,
        "start": None,
        "note": "Frankrig-alternativ, især long-run/gruppefase.",
    },
    "ousmane_demb_l__fra": {
        "global": 1.18,
        "match_1": 0.94,
        "match_2": 1.14,
        "match_3": 1.08,
        "start": 0.70,
        "note": "Alternativ, men ikke fast makker til Mbappé.",
    },
    "kai_havertz__ger": {
        "global": 1.24,
        "match_1": 1.20,
        "match_2": 1.10,
        "match_3": 1.05,
        "start": None,
        "note": "Tyskland/Curaçao-value og topscorer-outsider.",
    },
    "nico_williams__esp": {
        "global": 0.50,
        "match_1": 0.50,
        "match_2": 0.50,
        "match_3": 0.50,
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
    row["forward_model_rebalance_version"] = "v6"
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

print("Forward rebalance v6 applied")
print("EV rows changed:", ev_changed)
print("Pool rows changed:", pool_changed)
print("Changed EV players:")
for n in changed_names:
    print("-", n)
