import csv
import json
from pathlib import Path
from datetime import datetime

DATA = Path("data")
HOLDET = DATA / "holdet_players_game_616_flat.csv"
POOL = DATA / "player_pool_v1.json"
EV = DATA / "player_ev_group_stage_v1.csv"
START = DATA / "player_start_security_nt.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

pool_backup = DATA / f"player_pool_v1.backup_before_add_agustin_giay_{timestamp}.json"
ev_backup = DATA / f"player_ev_group_stage_v1.backup_before_add_agustin_giay_{timestamp}.csv"
start_backup = DATA / f"player_start_security_nt.backup_before_add_agustin_giay_{timestamp}.csv"

pool_backup.write_text(POOL.read_text(encoding="utf-8"), encoding="utf-8")
ev_backup.write_text(EV.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
start_backup.write_text(START.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

def norm(s):
    return str(s or "").strip().lower()

def to_bool(v):
    return str(v).strip().lower() in ["true", "1", "yes", "ja"]

def price_mio_to_int(v):
    x = float(str(v).replace(",", "."))
    if x > 100000:
        return int(round(x))
    return int(round(x * 1000000))

with HOLDET.open("r", encoding="utf-8-sig", newline="") as f:
    holdet_rows = list(csv.DictReader(f))

giay = None
for r in holdet_rows:
    if norm(r.get("player_name")) == "agustin giay" and str(r.get("position")).upper() == "DEF":
        giay = r
        break

if not giay:
    raise SystemExit("Fandt ikke Agustin Giay i Holdet flatfilen.")

raw = json.loads(POOL.read_text(encoding="utf-8"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

exists = any(norm(p.get("player_name")) == "agustin giay" and str(p.get("position")).upper() == "DEF" for p in players)
if exists:
    print("Agustin Giay findes allerede i player_pool. Ingen tilføjelse.")
else:
    price = price_mio_to_int(giay.get("price") or giay.get("start_price") or 2500000)

    new_player = {
        "player_id": "agustin_giay__arg",
        "player_name": "Agustin Giay",
        "team_id": "ARG",
        "team_name": "Argentina",
        "flag_code": "ARG",
        "position": "DEF",
        "advance_pct": 0,
        "avg_points": 0,
        "nt_ev_score": 0,
        "blended_ev_score": 0,
        "display_score": 0,
        "price_estimate": price,
        "value_score": 0,
        "display_value": 0,
        "price": price,
        "price_source": "holdet_game_616",
        "position_source": "holdet_game_616",
        "holdet_game_id": 616,
        "holdet_player_id": giay.get("holdet_player_id"),
        "holdet_person_id": giay.get("holdet_person_id"),
        "holdet_team_id": giay.get("holdet_team_id"),
        "holdet_team_name": giay.get("team_name"),
        "holdet_position_id": giay.get("holdet_position_id"),
        "holdet_position": giay.get("position"),
        "holdet_start_price": price_mio_to_int(giay.get("start_price") or giay.get("price") or 2500000),
        "holdet_price": price,
        "holdet_is_out": True,
        "is_out": True,
        "has_holdet_vm_match": True,
        "official_holdet_master": True,
        "start_prob": 0,
        "start_prob_source": "holdet_is_out",
        "weighted_group_stage_ev": 0,
        "optimizer_ev": 0,
        "display_score_source": "holdet_is_out_zero_ev",
        "start_probability_pct": 0,
        "start_security": 0,
        "start_status": "Ude / ikke valgbar",
        "conditional_start_prob": 0,
        "availability_prob": 0,
        "availability_risk": "out",
        "availability_status": "holdet_is_out",
        "weighted_group_stage_ev_before_price_quality": 0,
        "price_quality_ev": 0,
        "model_ev_before_price_quality": 0,
        "optimizer_ev_before_price_quality": 0,
        "price_quality_raw_ev": 0,
        "price_quality_applied": False,
        "base_ev_source": "holdet_is_out_zero_ev",
        "match_1_weighted_match_ev": 0,
        "match_2_weighted_match_ev": 0,
        "match_3_weighted_match_ev": 0,
        "note": "Added from latest Holdet.dk game 616 list; holdet_is_out=True, EV/start set to 0.",
    }

    players.append(new_player)
    POOL.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Tilføjet til player_pool:", new_player["player_id"])

# Tilføj EV-række med 0, hvis EV-filen har player_id og spilleren ikke findes
with EV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    ev_fields = list(reader.fieldnames or [])
    ev_rows = list(reader)

if not any(norm(r.get("player_name")) == "agustin giay" and str(r.get("position")).upper() == "DEF" for r in ev_rows):
    new_ev = {field: "" for field in ev_fields}
    for field, value in {
        "player_id": "agustin_giay__arg",
        "player_name": "Agustin Giay",
        "team_id": "ARG",
        "team_name": "Argentina",
        "position": "DEF",
        "price": 2500000,
        "holdet_price": 2500000,
        "holdet_is_out": "True",
        "is_out": "True",
        "start_prob": 0,
        "conditional_start_prob": 0,
        "availability_prob": 0,
        "availability_risk": "out",
        "availability_status": "holdet_is_out",
        "weighted_group_stage_ev": 0,
        "optimizer_ev": 0,
        "display_score": 0,
        "price_quality_ev": 0,
        "weighted_group_stage_ev_before_price_quality": 0,
        "optimizer_ev_before_price_quality": 0,
        "match_1_weighted_match_ev": 0,
        "match_2_weighted_match_ev": 0,
        "match_3_weighted_match_ev": 0,
    }.items():
        if field in new_ev:
            new_ev[field] = value
    ev_rows.append(new_ev)

    with EV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ev_fields)
        writer.writeheader()
        writer.writerows(ev_rows)

    print("Tilføjet 0-række til EV-fil.")
else:
    print("Agustin Giay findes allerede i EV-fil. Ingen tilføjelse.")

# Tilføj start-security 0-række, hvis relevant
with START.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    start_fields = list(reader.fieldnames or [])
    start_rows = list(reader)

if not any(norm(r.get("player_name")) == "agustin giay" and str(r.get("position")).upper() == "DEF" for r in start_rows):
    new_start = {field: "" for field in start_fields}
    for field, value in {
        "player_id": "agustin_giay__arg",
        "player_name": "Agustin Giay",
        "team_id": "ARG",
        "team_name": "Argentina",
        "position": "DEF",
        "start_prob": 0,
        "conditional_start_prob": 0,
        "start_probability_pct": 0,
        "start_security": 0,
        "availability_prob": 0,
        "availability_risk": "out",
        "availability_status": "holdet_is_out",
        "start_status": "Ude / ikke valgbar",
        "holdet_is_out": "True",
        "is_out": "True",
    }.items():
        if field in new_start:
            new_start[field] = value
    start_rows.append(new_start)

    with START.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=start_fields)
        writer.writeheader()
        writer.writerows(start_rows)

    print("Tilføjet 0-række til start-security-fil.")
else:
    print("Agustin Giay findes allerede i start-security-fil. Ingen tilføjelse.")

print()
print("Backups:")
print(pool_backup)
print(ev_backup)
print(start_backup)
