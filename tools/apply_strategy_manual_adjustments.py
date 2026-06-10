import csv
import json
from pathlib import Path

DATA = Path("data")
ADJ_PATH = DATA / "strategy_manual_adjustments.csv"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"

STRATEGY_MULT_COL = {
    "next_round": "next_round_multiplier",
    "practical_start": "practical_start_multiplier",
    "group_stage": "group_stage_multiplier",
    "long_run": "long_run_multiplier",
    "round1_2": "practical_start_multiplier",
}

EV_FIELDS = [
    "weighted_group_stage_ev",
    "optimizer_ev",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "model_ev_before_price_quality",
    "price_quality_ev",
    "round1_ev",
    "round2_ev",
    "round3_ev",
    "match_1_weighted_match_ev",
    "match_2_weighted_match_ev",
    "match_3_weighted_match_ev",
    "strategy_score",
]


def read_adjustments():
    with ADJ_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}
    for row in rows:
        pid = (row.get("player_id") or "").strip()
        if not pid:
            continue
        out[pid] = row
    return out


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def fmt_float(value):
    if value is None:
        return ""
    return f"{value:.12g}"


def multiplier_for(adj, strategy_key=None):
    if strategy_key:
        col = STRATEGY_MULT_COL.get(strategy_key)
        if col:
            specific = to_float(adj.get(col))
            if specific is not None:
                return specific

    return to_float(adj.get("ev_multiplier"), 1.0) or 1.0


def adjust_number(value, mult):
    n = to_float(value)
    if n is None:
        return value
    return fmt_float(n * mult)


def apply_to_player_dict(player, adj, strategy_key=None):
    mult = multiplier_for(adj, strategy_key)

    for field in EV_FIELDS:
        if field in player and player[field] not in [None, ""]:
            player[field] = adjust_number(player[field], mult)

    start_prob = to_float(adj.get("start_prob"))
    if start_prob is not None:
        player["start_prob"] = start_prob
        player["start_probability_pct"] = round(start_prob * 100, 1)
        player["start_security"] = start_prob
        player["start_prob_source"] = "manual_strategy_adjustment"
        player["start_status"] = "manuel justering - tjek start/skadesrisiko"

    notes = []
    existing = str(player.get("source_note") or "").strip()
    if existing:
        notes.append(existing)

    note = str(adj.get("selection_note") or "").strip()
    if note:
        notes.append("Manual strategy adjustment: " + note)

    if notes:
        player["source_note"] = " | ".join(dict.fromkeys(notes))

    player["manual_strategy_adjustment"] = adj.get("adjustment_type") or "manual"
    player["manual_strategy_ev_multiplier"] = mult

    return player


def key_for_row(row):
    return str(row.get("player_id") or "").strip()


adjustments = read_adjustments()

# 1) player_pool_v1.json
pool_raw = json.loads(POOL_PATH.read_text(encoding="utf-8"))
if isinstance(pool_raw, dict):
    players = pool_raw.get("players") or pool_raw.get("data") or []
else:
    players = pool_raw

pool_hits = 0
for p in players:
    pid = key_for_row(p)
    if pid in adjustments:
        apply_to_player_dict(p, adjustments[pid])
        pool_hits += 1

POOL_PATH.write_text(json.dumps(pool_raw, ensure_ascii=False, indent=2), encoding="utf-8")

# 2) player_ev_group_stage_v1.csv
with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    ev_rows = list(reader)

for col in [
    "manual_strategy_adjustment",
    "manual_strategy_ev_multiplier",
    "source_note",
    "start_prob_source",
    "start_status",
    "start_probability_pct",
    "start_security",
]:
    if col not in fieldnames:
        fieldnames.append(col)

ev_hits = 0
for row in ev_rows:
    pid = key_for_row(row)
    if pid in adjustments:
        apply_to_player_dict(row, adjustments[pid])
        ev_hits += 1

with EV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ev_rows)

# 3) optimal_squads_by_strategy.json
strategy_raw = json.loads(STRATEGY_PATH.read_text(encoding="utf-8"))
strategy_hits = 0

for strategy_key, strategy_entry in strategy_raw.items():
    if not isinstance(strategy_entry, dict):
        continue

    for collection_key in ["best_squad"]:
        rows = strategy_entry.get(collection_key)
        if isinstance(rows, list):
            for row in rows:
                pid = key_for_row(row)
                if pid in adjustments:
                    apply_to_player_dict(row, adjustments[pid], strategy_key)
                    strategy_hits += 1

    formations = strategy_entry.get("squads_by_formation") or {}
    for formation, formation_entry in formations.items():
        if not isinstance(formation_entry, dict):
            continue

        rows = formation_entry.get("squad") or []
        for row in rows:
            pid = key_for_row(row)
            if pid in adjustments:
                apply_to_player_dict(row, adjustments[pid], strategy_key)
                strategy_hits += 1

STRATEGY_PATH.write_text(json.dumps(strategy_raw, ensure_ascii=False, indent=2), encoding="utf-8")

print("Manual strategy adjustments applied")
print("player_pool hits:", pool_hits)
print("player_ev hits:", ev_hits)
print("strategy squad row hits:", strategy_hits)
print("")
print("Bemærk: Dette nedjusterer værdierne i eksisterende strategihold.")
print("Hvis spillere helt skal erstattes i holdene, skal strategi-optimizer genkøres bagefter.")
