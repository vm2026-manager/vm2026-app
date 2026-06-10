import csv
import json
from pathlib import Path
from collections import Counter

DATA = Path("data")
OUT = DATA / "strategy_squad_exports"
OUT.mkdir(exist_ok=True)

POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"

BUDGET = 50_000_000
MAX_PER_TEAM = 4

# To topangribere pr. hold bygges herfra.
# Rækkefølgen er bevidst: de store/populære profiler skal have reel adgang.
TOP_FORWARD_ORDER = [
    "erling_haaland__nor",
    "harry_kane__eng",
    "kylian_mbappe__fra",
    "mikel_oyarzabal__esp",
    "michael_olise__fra",
    "ousmane_dembele__fra",
    "kai_havertz__ger",
    "vinicius_junior__bra",
    "julian_alvarez__arg",
    "cody_gakpo__ned",
    "lionel_messi__arg",
    "cristiano_ronaldo__por",
    "jonathan_david__can",
    "raul_jimenez__mex",
    "marko_arnautovic__aut",
]

# Dembélé skal ikke stå med 36 %.
DEMBO_IDS = {
    "ousmane_dembele__fra",
    "ousmane_dembélé__fra",
}

DEMBO_NAMES = {
    "ousmane dembele",
    "ousmane dembélé",
}

DEMBO_START_PROB = 0.68
DEMBO_EV_MULTIPLIER = 1.10

POS_ORDER = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

FORMATION_COUNTS = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

# Formationer med kun 1 angriber kan ikke have 2 FWD-slots.
# Her prioriterer vi 1 topangriber, ellers bliver formationen umulig.
MIN_TOP_FORWARDS_BY_FORMATION = {
    "4-5-1": 1,
    "5-4-1": 1,
}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def fmt(value):
    if value is None:
        return ""
    return f"{value:.12g}"


def pid(row):
    return str(row.get("player_id") or "").strip()


def pname(row):
    return str(row.get("player_name") or row.get("name") or "").strip()


def norm_name(value):
    return str(value or "").lower().replace("é", "e").replace("è", "e").replace("á", "a").replace("í", "i").replace("ó", "o").strip()


def team(row):
    return str(row.get("team_id") or row.get("team") or row.get("team_name") or "").strip()


def pos(row):
    return str(row.get("position") or row.get("holdet_position") or "").strip().upper()


def price(row):
    n = to_float(row.get("price") or row.get("price_estimate") or row.get("holdet_price"), 0)
    if 0 < n < 1000:
        n *= 1_000_000
    return int(round(n))


def score(row, strategy_key=None):
    # Strategi-specifik prioritet, men med fallback.
    keys_by_strategy = {
        "next_round": ["round1_ev", "match_1_weighted_match_ev", "strategy_score", "optimizer_ev", "weighted_group_stage_ev", "display_score", "score"],
        "practical_start": ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"],
        "round1_2": ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"],
        "group_stage": ["weighted_group_stage_ev", "optimizer_ev", "strategy_score", "display_score", "score"],
        "long_run": ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "display_score", "score"],
    }

    keys = keys_by_strategy.get(strategy_key, ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"])

    for key in keys:
        value = to_float(row.get(key), None)
        if value is not None:
            return value

    return 0.0


def start_pct(row):
    for key in ["start_probability_pct", "start_prob", "start_security"]:
        value = to_float(row.get(key), None)
        if value is not None and value > 0:
            return value * 100 if value <= 1 else value
    return 0.0


def is_out(row):
    return str(row.get("holdet_is_out") or row.get("is_out") or "").strip().lower() in {"true", "1", "yes", "ja"}


def is_dembele(row):
    return pid(row) in DEMBO_IDS or norm_name(pname(row)) in {norm_name(x) for x in DEMBO_NAMES}


def adjust_dembele_row(row):
    if not is_dembele(row):
        return False

    old_start = start_pct(row)

    row["start_prob"] = DEMBO_START_PROB
    row["start_probability_pct"] = round(DEMBO_START_PROB * 100, 1)
    row["start_security"] = DEMBO_START_PROB
    row["start_status"] = "manuel justering - sandsynlig men ikke sikker starter"
    row["start_prob_source"] = "manual_dembele_adjustment"

    ev_fields = [
        "strategy_score",
        "optimizer_ev",
        "weighted_group_stage_ev",
        "round1_ev",
        "match_1_weighted_match_ev",
        "match_2_weighted_match_ev",
        "match_3_weighted_match_ev",
        "display_score",
        "score",
    ]

    for field in ev_fields:
        value = to_float(row.get(field), None)
        if value is not None:
            row[field] = fmt(value * DEMBO_EV_MULTIPLIER)

    note = "Manual adjustment: Dembélé hæves fra lav startsandsynlighed; stadig ikke locked, men skal være relevant som topangriber."
    old_note = str(row.get("source_note") or "").strip()
    row["source_note"] = (old_note + " | " + note).strip(" |")
    row["manual_dembele_adjustment"] = True
    row["manual_dembele_old_start_pct"] = old_start

    return True


def row_key(row):
    return pid(row) or f"{norm_name(pname(row))}|{team(row).lower()}"


def team_counts(rows):
    counts = Counter()
    for row in rows:
        counts[team(row)] += 1
    return counts


def squad_price(rows):
    return sum(price(row) for row in rows)


def valid_squad(rows, formation):
    if len(rows) != 11:
        return False

    if squad_price(rows) > BUDGET:
        return False

    ids = [pid(row) for row in rows if pid(row)]
    if len(ids) != len(set(ids)):
        return False

    counts = Counter(pos(row) for row in rows)
    wanted = FORMATION_COUNTS.get(formation)
    if wanted:
        for p, n in wanted.items():
            if counts[p] != n:
                return False

    if max(team_counts(rows).values() or [0]) > MAX_PER_TEAM:
        return False

    return True


def load_pool_lookup_and_players():
    raw = load_json(POOL_PATH)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw

    lookup = {}
    for p in players:
        if not isinstance(p, dict):
            continue

        adjust_dembele_row(p)

        keys = [
            pid(p),
            str(p.get("holdet_player_id") or ""),
            f"{norm_name(pname(p))}|{str(p.get('team_id') or '').lower()}",
            f"{norm_name(pname(p))}|{str(p.get('team_name') or '').lower()}",
        ]

        for key in keys:
            if key:
                lookup[key] = p

    write_json(POOL_PATH, raw)
    return lookup, players


POOL_LOOKUP, POOL_PLAYERS = load_pool_lookup_and_players()


def enrich(row):
    keys = [
        pid(row),
        str(row.get("holdet_player_id") or ""),
        f"{norm_name(pname(row))}|{str(row.get('team_id') or '').lower()}",
        f"{norm_name(pname(row))}|{str(row.get('team_name') or '').lower()}",
    ]

    base = {}
    for key in keys:
        if key and key in POOL_LOOKUP:
            base = POOL_LOOKUP[key]
            break

    merged = dict(base)
    merged.update(row or {})
    adjust_dembele_row(merged)
    return merged


def load_ev_candidates():
    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for col in [
        "manual_dembele_adjustment",
        "manual_dembele_old_start_pct",
        "start_prob",
        "start_probability_pct",
        "start_security",
        "start_status",
        "start_prob_source",
        "source_note",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    changed = 0
    for row in rows:
        if adjust_dembele_row(row):
            changed += 1

    with EV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidates = [enrich(row) for row in rows]
    candidates = [
        row for row in candidates
        if not is_out(row)
        and pos(row) in {"GK", "DEF", "MID", "FWD"}
        and price(row) > 0
    ]

    # Dedupér og behold bedste version
    best = {}
    for row in candidates:
        key = row_key(row)
        if not key:
            continue
        if key not in best or score(row) > score(best[key]):
            best[key] = row

    return list(best.values()), changed


ALL_CANDIDATES, DEMBELE_EV_HITS = load_ev_candidates()


def top_forward_rank(row, strategy_key):
    player_id = pid(row)

    if player_id in TOP_FORWARD_ORDER:
        top_bonus = 100 - TOP_FORWARD_ORDER.index(player_id)
    else:
        top_bonus = 0

    # Dembélé må godt komme i spil, men skal ikke overtage alt.
    if is_dembele(row):
        top_bonus += 8

    return (
        top_bonus,
        score(row, strategy_key),
        start_pct(row),
        -price(row),
    )


def candidate_rank(row, strategy_key):
    value_bonus = 0
    if pid(row) in {
        "nathaniel_brown__ger",
        "antonee_robinson__usa",
        "cesar_montes__mex",
        "silvan_widmer__sui",
        "philipp_lienhart__aut",
        "roberto_alvarado__mex",
        "brian_gutierrez__mex",
        "john_yeboah__ecu",
        "antonio_nusa__nor",
    }:
        value_bonus += 2

    # Billige spillere skal kunne finansiere topangribere
    price_bonus = max(0, (5_000_000 - price(row)) / 1_000_000) * 0.30

    return (
        score(row, strategy_key) + value_bonus + price_bonus,
        start_pct(row),
        -price(row),
    )


def can_add(row, chosen, formation):
    if pid(row) in {pid(x) for x in chosen if pid(x)}:
        return False

    if start_pct(row) < 55:
        return False

    wanted = FORMATION_COUNTS[formation]
    current_pos_count = Counter(pos(x) for x in chosen)

    if current_pos_count[pos(row)] >= wanted.get(pos(row), 0):
        return False

    counts = team_counts(chosen)
    if counts[team(row)] >= MAX_PER_TEAM:
        return False

    if squad_price(chosen) + price(row) > BUDGET:
        return False

    return True


def build_squad_with_top_forwards(strategy_key, formation, old_rows):
    wanted = FORMATION_COUNTS.get(formation)
    if not wanted:
        return old_rows, "unknown_formation"

    min_top = MIN_TOP_FORWARDS_BY_FORMATION.get(formation, 2)
    fwd_slots = wanted["FWD"]
    min_top = min(min_top, fwd_slots)

    chosen = []

    # Kandidatlister
    top_forwards = [
        row for row in ALL_CANDIDATES
        if pos(row) == "FWD"
        and pid(row) in TOP_FORWARD_ORDER
        and start_pct(row) >= 55
    ]
    top_forwards.sort(key=lambda r: top_forward_rank(r, strategy_key), reverse=True)

    all_by_pos = {
        p: [row for row in ALL_CANDIDATES if pos(row) == p and start_pct(row) >= 55]
        for p in ["GK", "DEF", "MID", "FWD"]
    }

    for p in all_by_pos:
        all_by_pos[p].sort(key=lambda r: candidate_rank(r, strategy_key), reverse=True)

    # 1) Lås først topangribere
    for row in top_forwards:
        if len([x for x in chosen if pos(x) == "FWD" and pid(x) in TOP_FORWARD_ORDER]) >= min_top:
            break
        if can_add(row, chosen, formation):
            chosen.append(dict(row))

    if len([x for x in chosen if pos(x) == "FWD" and pid(x) in TOP_FORWARD_ORDER]) < min_top:
        return old_rows, "could_not_lock_top_forwards"

    # 2) Fyld resten greedy - men med budgetbuffer.
    # Først prøver vi med bedste spillere. Hvis det fejler, prøver vi med value-bias.
    for p in ["GK", "DEF", "MID", "FWD"]:
        while Counter(pos(x) for x in chosen)[p] < wanted[p]:
            added = False

            for row in all_by_pos[p]:
                if can_add(row, chosen, formation):
                    chosen.append(dict(row))
                    added = True
                    break

            if not added:
                return old_rows, f"could_not_fill_{p}"

    if valid_squad(chosen, formation):
        chosen.sort(key=lambda r: (POS_ORDER.get(pos(r), 99), pname(r)))
        return chosen, "rebuilt_top_forwards_first"

    return old_rows, "invalid_after_build"


strategy_data = load_json(STRATEGY_PATH)
changes = []

for strategy_key, strategy_entry in strategy_data.items():
    formations = strategy_entry.get("squads_by_formation") or {}

    for formation, formation_entry in formations.items():
        old_rows = [enrich(row) for row in (formation_entry.get("squad") or [])]

        if not old_rows:
            continue

        new_rows, status = build_squad_with_top_forwards(strategy_key, formation, old_rows)

        old_names = {pname(row) for row in old_rows}
        new_names = {pname(row) for row in new_rows}

        formation_entry["squad"] = new_rows
        formation_entry["manual_top_forwards_first_status"] = status

        changes.append({
            "strategy": strategy_key,
            "formation": formation,
            "status": status,
            "old_price_m": round(squad_price(old_rows) / 1_000_000, 1),
            "new_price_m": round(squad_price(new_rows) / 1_000_000, 1),
            "top_forwards": ", ".join(
                pname(row) for row in new_rows
                if pos(row) == "FWD" and pid(row) in TOP_FORWARD_ORDER
            ),
            "in": ", ".join(sorted(new_names - old_names)),
            "out": ", ".join(sorted(old_names - new_names)),
        })

write_json(STRATEGY_PATH, strategy_data)

audit_path = OUT / "top_forwards_first_changes.csv"
with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "strategy",
        "formation",
        "status",
        "old_price_m",
        "new_price_m",
        "top_forwards",
        "in",
        "out",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(changes)

print("Top-forwards-first applied")
print("Dembélé EV rows adjusted:", DEMBELE_EV_HITS)
print("Audit:", audit_path)

for row in changes:
    print(
        f"{row['strategy']} {row['formation']}: "
        f"{row['status']} | {row['new_price_m']} mio. | top: {row['top_forwards']}"
    )
