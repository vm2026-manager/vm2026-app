import csv
import json
from pathlib import Path

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

# v4: korrekte faktiske player_id'er, men mildere end v3b.
OVERRIDES = {
    # Ned: gamle locks/value-locks
    "cristiano_ronaldo__por": {"mult": 0.88, "start": None, "note": "Stadig relevant, men må ikke dominere 1+2."},
    "lionel_messi__arg": {"mult": 0.88, "start": None, "note": "Stadig relevant, men må ikke dominere gruppespil."},
    "jonathan_david__can": {"mult": 0.70, "start": None, "note": "Canada/kvaldata overvurderer ham i modellen."},
    "ra_l_jim_nez__mex": {"mult": 0.74, "start": None, "note": "Valueangriber, men ikke fast lås."},
    "marko_arnautovi__aut": {"mult": 0.78, "start": None, "note": "Valueangriber, men bør fylde mindre."},
    "nico_williams__esp": {"mult": 0.55, "start": 0.56, "note": "Skades-/startusikkerhed; skal ikke være long-run-lock."},
    "vinicius_j_nior__bra": {"mult": 0.96, "start": None, "note": "Stadig stærk long-run, men ikke for dominerende."},

    # Op: topangribere, men ikke som nye locks
    "erling_haaland__nor": {"mult": 1.12, "start": None, "note": "Stærk næste-runde-kaptajn, men ikke automatisk alle formationer."},
    "harry_kane__eng": {"mult": 1.00, "start": None, "note": "Stærk, men ikke ekstra boostet lock."},
    "kylian_mbapp__fra": {"mult": 1.42, "start": None, "note": "Skal være reel premiumkandidat, men v3b gjorde ham til 28/28-lock."},
    "michael_olise__fra": {"mult": 1.55, "start": None, "note": "Skal kunne konkurrere som Frankrig-angriber."},
    "ousmane_demb_l__fra": {"mult": 1.45, "start": 0.70, "note": "Hævet start og EV, men v3b gjorde ham for låst."},
    "mikel_oyarzabal__esp": {"mult": 0.98, "start": None, "note": "Relevant, men ikke long-run-lock."},
    "kai_havertz__ger": {"mult": 1.18, "start": None, "note": "Relevant value, men ikke for dominerende long-run."},
}

EV_FIELDS = [
    "strategy_score",
    "optimizer_ev",
    "weighted_group_stage_ev",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "price_quality_ev",
    "round1_ev",
    "match_1_weighted_match_ev",
    "match_2_weighted_match_ev",
    "match_3_weighted_match_ev",
    "display_score",
    "score",
]

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

def apply_to_row(row, source):
    player_id = norm(row.get("player_id"))
    if player_id not in OVERRIDES:
        return False

    ov = OVERRIDES[player_id]
    mult = ov["mult"]

    for field in EV_FIELDS:
        old = to_float(row.get(field))
        if old is not None:
            row[field] = fmt(old * mult)

    if ov["start"] is not None:
        start = ov["start"]
        for field in START_PROB_FIELDS:
            if field in row:
                row[field] = fmt(start)
        for field in START_PCT_FIELDS:
            if field in row:
                row[field] = fmt(start * 100)

    row["forward_model_rebalance"] = "true"
    row["forward_model_rebalance_version"] = "v4"
    row["forward_model_rebalance_multiplier"] = fmt(mult)
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

print("Forward rebalance v4 applied")
print("EV rows changed:", ev_changed)
print("Pool rows changed:", pool_changed)
print("Changed EV players:")
for n in changed_names:
    print("-", n)
