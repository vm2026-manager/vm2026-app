import csv
import json
from pathlib import Path

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

# V2: mildere boost til Haaland/Kane/Oyarzabal,
# hårdere løft til Mbappé/Olise/Dembélé/Havertz,
# og fortsat nedjustering af David/Messi/Ronaldo/Nico.
OVERRIDES = {
    "cristiano_ronaldo__por": {"mult": 0.90, "start": None, "note": "Stadig relevant, men må ikke låse alle hold."},
    "lionel_messi__arg": {"mult": 0.90, "start": None, "note": "Stadig relevant, men må ikke låse gruppespil."},
    "jonathan_david__can": {"mult": 0.76, "start": None, "note": "Ned fra tydelig overdominans i 1+2 og gruppespil."},
    "raul_jimenez__mex": {"mult": 0.76, "start": None, "note": "Valueangriber, men for dominerende i næste runde."},
    "marko_arnautovic__aut": {"mult": 0.82, "start": None, "note": "Valueangriber, men bør ikke blokere topangribere."},
    "nico_williams__esp": {"mult": 0.60, "start": 0.58, "note": "Skades-/startusikkerhed; skal markant ned især lang sigt."},

    "erling_haaland__nor": {"mult": 1.28, "start": None, "note": "Stærk runde 1-kaptajn, men ikke 7/7-lock."},
    "harry_kane__eng": {"mult": 1.12, "start": None, "note": "Skal være relevant, men ikke 7/7-lock i gruppespil."},
    "kylian_mbappe__fra": {"mult": 1.65, "start": None, "note": "Skal ind som reel premium/long-run-kandidat."},
    "mikel_oyarzabal__esp": {"mult": 1.08, "start": None, "note": "Relevant Spanien-angriber, men ikke 7/7-lock lang sigt."},
    "michael_olise__fra": {"mult": 1.60, "start": None, "note": "Skal kunne konkurrere som Frankrig-angriber."},
    "ousmane_dembele__fra": {"mult": 1.75, "start": 0.72, "note": "Hævet start og EV; skal frem som alternativ, ikke locked."},
    "kai_havertz__ger": {"mult": 1.45, "start": None, "note": "Tyskland/Curaçao og pris gør ham relevant."},
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
    row["forward_model_rebalance_version"] = "v2"
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
for row in rows:
    if apply_to_row(row, "player_ev_group_stage_v1"):
        ev_changed += 1

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

print("Forward rebalance v2 applied")
print("EV rows changed:", ev_changed)
print("Pool rows changed:", pool_changed)
