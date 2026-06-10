import csv
import json
from pathlib import Path

DATA = Path("data")
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"

# Justeringer efter angriber-rapporten:
# Ned: spillere der dominerer for meget.
# Op: topangribere der bør komme mere i spil.
OVERRIDES = {
    "cristiano_ronaldo__por": {
        "mult": 0.88,
        "start": None,
        "note": "Dominerer for meget i optimizer; stadig relevant, men ikke automatisk låst.",
    },
    "lionel_messi__arg": {
        "mult": 0.88,
        "start": None,
        "note": "Dominerer for meget i optimizer; stadig relevant, men ikke automatisk låst.",
    },
    "jonathan_david__can": {
        "mult": 0.72,
        "start": None,
        "note": "Overvurderet af model/kvaldata; Canada bør ikke dominere gruppespil så hårdt.",
    },
    "raul_jimenez__mex": {
        "mult": 0.82,
        "start": None,
        "note": "For høj EV ift. andre topangribere; stadig value, men ikke standardvalg.",
    },
    "marko_arnautovic__aut": {
        "mult": 0.90,
        "start": None,
        "note": "Fin value, men bør ikke blokere større topangribere så ofte.",
    },
    "nico_williams__esp": {
        "mult": 0.72,
        "start": 0.62,
        "note": "Skades-/startusikkerhed; for høj i lang sigt.",
    },
    "erling_haaland__nor": {
        "mult": 1.55,
        "start": None,
        "note": "Skal være stærk kaptajn-/topangriberkandidat mod Irak og generelt højere i modellen.",
    },
    "harry_kane__eng": {
        "mult": 1.32,
        "start": None,
        "note": "For lav relativt til rolle, pris og straffe.",
    },
    "kylian_mbappe__fra": {
        "mult": 1.38,
        "start": None,
        "note": "For lav i long-run/topspillerlogik; bør være reel premiumkandidat.",
    },
    "mikel_oyarzabal__esp": {
        "mult": 1.22,
        "start": None,
        "note": "Spanien-favorit og klar angriberrolle; bør være mere konkurrencedygtig.",
    },
    "michael_olise__fra": {
        "mult": 1.28,
        "start": None,
        "note": "For lav i modellen; skal kunne dukke op som Frankrig-angriber.",
    },
    "ousmane_dembele__fra": {
        "mult": 1.35,
        "start": 0.68,
        "note": "Hævet startsandsynlighed og EV, men stadig ikke locked.",
    },
    "kai_havertz__ger": {
        "mult": 1.25,
        "start": None,
        "note": "Tyskland mod Curaçao og lav pris gør ham mere relevant som angriber.",
    },
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

START_PROB_FIELDS = [
    "start_prob",
    "conditional_start_prob",
    "start_security",
]

START_PCT_FIELDS = [
    "start_probability_pct",
]

EXTRA_FIELDS = [
    "forward_model_rebalance",
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

    override = OVERRIDES[player_id]
    mult = override["mult"]

    for field in EV_FIELDS:
        old = to_float(row.get(field))
        if old is not None:
            row[field] = fmt(old * mult)

    start = override["start"]
    if start is not None:
        for field in START_PROB_FIELDS:
            if field in row:
                row[field] = fmt(start)
        for field in START_PCT_FIELDS:
            if field in row:
                row[field] = fmt(start * 100)

    row["forward_model_rebalance"] = "true"
    row["forward_model_rebalance_multiplier"] = fmt(mult)
    row["forward_model_rebalance_note"] = override["note"]
    row["forward_model_rebalance_source"] = source

    return True

# CSV
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

# JSON
raw = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

pool_changed = 0
for row in players:
    if isinstance(row, dict):
        if apply_to_row(row, "player_pool_v1"):
            pool_changed += 1

POOL_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

print("Forward rebalance applied")
print("EV rows changed:", ev_changed)
print("Pool rows changed:", pool_changed)
