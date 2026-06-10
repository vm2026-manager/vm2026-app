import csv
import json
from pathlib import Path
from collections import Counter

DATA = Path("data")
OUT = DATA / "strategy_squad_exports"
OUT.mkdir(exist_ok=True)

STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"

BUDGET = 50_000_000
MAX_PER_TEAM = 4

FORMATION_COUNTS = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

POS_ORDER = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

# Rotation frem for altid Haaland + Kane
PAIR_ROTATION = {
    "next_round": {
        "3-4-3": ["erling_haaland__nor", "mikel_oyarzabal__esp"],
        "3-5-2": ["harry_kane__eng", "michael_olise__fra"],
        "4-3-3": ["erling_haaland__nor", "kylian_mbappe__fra"],
        "4-4-2": ["harry_kane__eng", "mikel_oyarzabal__esp"],
        "5-3-2": ["erling_haaland__nor", "cristiano_ronaldo__por"],
    },
    "round1_2": {
        "3-4-3": ["erling_haaland__nor", "vinicius_junior__bra"],
        "3-5-2": ["harry_kane__eng", "mikel_oyarzabal__esp"],
        "4-3-3": ["kylian_mbappe__fra", "michael_olise__fra"],
        "4-4-2": ["erling_haaland__nor", "harry_kane__eng"],
        "5-3-2": ["vinicius_junior__bra", "mikel_oyarzabal__esp"],
    },
    "practical_start": {
        "3-4-3": ["erling_haaland__nor", "vinicius_junior__bra"],
        "3-5-2": ["harry_kane__eng", "mikel_oyarzabal__esp"],
        "4-3-3": ["kylian_mbappe__fra", "michael_olise__fra"],
        "4-4-2": ["erling_haaland__nor", "harry_kane__eng"],
        "5-3-2": ["vinicius_junior__bra", "mikel_oyarzabal__esp"],
    },
    "group_stage": {
        "3-4-3": ["vinicius_junior__bra", "kylian_mbappe__fra"],
        "3-5-2": ["harry_kane__eng", "michael_olise__fra"],
        "4-3-3": ["mikel_oyarzabal__esp", "ousmane_dembele__fra"],
        "4-4-2": ["erling_haaland__nor", "vinicius_junior__bra"],
        "5-3-2": ["kylian_mbappe__fra", "harry_kane__eng"],
    },
    "long_run": {
        "3-4-3": ["vinicius_junior__bra", "kylian_mbappe__fra"],
        "3-5-2": ["harry_kane__eng", "mikel_oyarzabal__esp"],
        "4-3-3": ["erling_haaland__nor", "michael_olise__fra"],
        "4-4-2": ["kylian_mbappe__fra", "vinicius_junior__bra"],
        "5-3-2": ["harry_kane__eng", "ousmane_dembele__fra"],
    },
}

SINGLE_ROTATION = {
    "4-5-1": [
        "erling_haaland__nor",
        "harry_kane__eng",
        "kylian_mbappe__fra",
        "mikel_oyarzabal__esp",
        "vinicius_junior__bra",
    ],
    "5-4-1": [
        "harry_kane__eng",
        "kylian_mbappe__fra",
        "erling_haaland__nor",
        "michael_olise__fra",
        "vinicius_junior__bra",
    ],
}

FALLBACK_TOP_FORWARDS = [
    "erling_haaland__nor",
    "harry_kane__eng",
    "kylian_mbappe__fra",
    "mikel_oyarzabal__esp",
    "michael_olise__fra",
    "ousmane_dembele__fra",
    "vinicius_junior__bra",
    "kai_havertz__ger",
    "julian_alvarez__arg",
    "cody_gakpo__ned",
    "lionel_messi__arg",
    "cristiano_ronaldo__por",
    "jonathan_david__can",
    "raul_jimenez__mex",
    "marko_arnautovic__aut",
]

DEMBO_IDS = {"ousmane_dembele__fra", "ousmane_dembélé__fra"}
DEMBO_START_PROB = 0.68


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


def pid(row):
    return str(row.get("player_id") or "").strip()


def name(row):
    return str(row.get("player_name") or row.get("name") or "").strip()


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
    keys = ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"]
    if strategy_key == "next_round":
        keys = ["round1_ev", "match_1_weighted_match_ev"] + keys

    for key in keys:
        n = to_float(row.get(key), None)
        if n is not None:
            return n
    return 0.0


def start_pct(row):
    for key in ["start_probability_pct", "start_prob", "start_security"]:
        n = to_float(row.get(key), None)
        if n is not None and n > 0:
            return n * 100 if n <= 1 else n
    return 0.0


def is_out(row):
    return str(row.get("holdet_is_out") or row.get("is_out") or "").strip().lower() in {"true", "1", "yes", "ja"}


def adjust_dembele(row):
    if pid(row) in DEMBO_IDS:
        row["start_prob"] = DEMBO_START_PROB
        row["start_probability_pct"] = round(DEMBO_START_PROB * 100, 1)
        row["start_security"] = DEMBO_START_PROB
        row["start_prob_source"] = "manual_dembele_adjustment"
    return row


def squad_price(rows):
    return sum(price(r) for r in rows)


def team_counts(rows):
    c = Counter()
    for r in rows:
        c[team(r)] += 1
    return c


def load_pool_lookup():
    raw = load_json(POOL_PATH)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw
    lookup = {}

    for p in players:
        if not isinstance(p, dict):
            continue
        adjust_dembele(p)
        keys = [
            pid(p),
            str(p.get("holdet_player_id") or ""),
            f"{name(p).lower()}|{str(p.get('team_id') or '').lower()}",
            f"{name(p).lower()}|{str(p.get('team_name') or '').lower()}",
        ]
        for k in keys:
            if k:
                lookup[k] = p

    write_json(POOL_PATH, raw)
    return lookup


POOL = load_pool_lookup()


def enrich(row):
    keys = [
        pid(row),
        str(row.get("holdet_player_id") or ""),
        f"{name(row).lower()}|{str(row.get('team_id') or '').lower()}",
        f"{name(row).lower()}|{str(row.get('team_name') or '').lower()}",
    ]

    base = {}
    for k in keys:
        if k and k in POOL:
            base = POOL[k]
            break

    merged = dict(base)
    merged.update(row or {})
    return adjust_dembele(merged)


def load_candidates():
    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [enrich(r) for r in csv.DictReader(f)]

    rows = [
        r for r in rows
        if not is_out(r)
        and pos(r) in {"GK", "DEF", "MID", "FWD"}
        and price(r) > 0
        and start_pct(r) >= 55
    ]

    best = {}
    for r in rows:
        key = pid(r) or f"{name(r).lower()}|{team(r).lower()}"
        if key not in best or score(r) > score(best[key]):
            best[key] = r

    return list(best.values())


ALL = load_candidates()
BY_ID = {pid(r): r for r in ALL if pid(r)}


def valid_add(row, chosen, formation):
    if pid(row) in {pid(x) for x in chosen if pid(x)}:
        return False

    wanted = FORMATION_COUNTS[formation]
    if Counter(pos(x) for x in chosen)[pos(row)] >= wanted[pos(row)]:
        return False

    if team_counts(chosen)[team(row)] >= MAX_PER_TEAM:
        return False

    if squad_price(chosen) + price(row) > BUDGET:
        return False

    return True


def valid_squad(rows, formation, min_budget):
    if len(rows) != 11:
        return False

    if squad_price(rows) > BUDGET:
        return False

    if squad_price(rows) < min_budget:
        return False

    ids = [pid(r) for r in rows if pid(r)]
    if len(ids) != len(set(ids)):
        return False

    counts = Counter(pos(r) for r in rows)
    wanted = FORMATION_COUNTS[formation]
    for p, n in wanted.items():
        if counts[p] != n:
            return False

    if max(team_counts(rows).values() or [0]) > MAX_PER_TEAM:
        return False

    return True


def rank(row, strategy_key, chosen):
    # Brug budgettet aktivt: efter topangribere skal vi både bruge gode og ikke alt for billige spillere.
    value_bonus = 0
    if pid(row) in {
        "nathaniel_brown__ger",
        "antonee_robinson__usa",
        "silvan_widmer__sui",
        "cesar_montes__mex",
        "john_yeboah__ecu",
        "antonio_nusa__nor",
        "roberto_alvarado__mex",
        "james_rodriguez__col",
    }:
        value_bonus = 1.0

    return (
        score(row, strategy_key) + value_bonus,
        start_pct(row),
        price(row),
    )


def get_locked_forward_ids(strategy_key, formation):
    fwd_slots = FORMATION_COUNTS[formation]["FWD"]

    if fwd_slots == 1:
        rotation = SINGLE_ROTATION.get(formation, FALLBACK_TOP_FORWARDS)
        idx = ["next_round", "round1_2", "practical_start", "group_stage", "long_run"].index(strategy_key) if strategy_key in ["next_round", "round1_2", "practical_start", "group_stage", "long_run"] else 0
        return [rotation[idx % len(rotation)]]

    pair = PAIR_ROTATION.get(strategy_key, {}).get(formation)
    if pair:
        return pair[:min(2, fwd_slots)]

    return FALLBACK_TOP_FORWARDS[:min(2, fwd_slots)]


def build(strategy_key, formation):
    wanted = FORMATION_COUNTS[formation]
    min_budget = 47_000_000 if wanted["FWD"] == 1 else 48_500_000

    locked_ids = get_locked_forward_ids(strategy_key, formation)

    # Prøv først de planlagte låste angribere. Hvis det ikke kan fyldes tæt nok på budget, løsnes kravet gradvist.
    attempts = [
        locked_ids,
        locked_ids[:1],
        [],
    ]

    for lock_ids in attempts:
        chosen = []

        ok_locks = True
        for lock_id in lock_ids:
            row = BY_ID.get(lock_id)
            if not row or pos(row) != "FWD":
                ok_locks = False
                break
            if not valid_add(row, chosen, formation):
                ok_locks = False
                break
            chosen.append(dict(row))

        if not ok_locks:
            continue

        # Først fyld ud efter ren score
        for p in ["GK", "DEF", "MID", "FWD"]:
            while Counter(pos(x) for x in chosen)[p] < wanted[p]:
                candidates = [r for r in ALL if pos(r) == p]
                candidates.sort(key=lambda r: rank(r, strategy_key, chosen), reverse=True)

                added = False
                for cand in candidates:
                    if valid_add(cand, chosen, formation):
                        chosen.append(dict(cand))
                        added = True
                        break

                if not added:
                    break

        if valid_squad(chosen, formation, min_budget):
            return sorted(chosen, key=lambda r: (POS_ORDER[pos(r)], name(r))), "rotating_pair_budget_ok", lock_ids

        # Hvis holdet er valid men bruger for få penge, prøv at opgradere billigste non-locked spillere
        if len(chosen) == 11 and squad_price(chosen) <= BUDGET:
            chosen = upgrade_budget(chosen, formation, strategy_key, lock_ids, min_budget)
            if valid_squad(chosen, formation, min_budget):
                return sorted(chosen, key=lambda r: (POS_ORDER[pos(r)], name(r))), "rotating_pair_after_upgrades", lock_ids

    # Fallback: accepter lavere budget frem for fejl
    for lock_ids in attempts:
        chosen = []
        for lock_id in lock_ids:
            row = BY_ID.get(lock_id)
            if row and pos(row) == "FWD" and valid_add(row, chosen, formation):
                chosen.append(dict(row))

        for p in ["GK", "DEF", "MID", "FWD"]:
            while Counter(pos(x) for x in chosen)[p] < wanted[p]:
                candidates = [r for r in ALL if pos(r) == p]
                candidates.sort(key=lambda r: rank(r, strategy_key, chosen), reverse=True)
                for cand in candidates:
                    if valid_add(cand, chosen, formation):
                        chosen.append(dict(cand))
                        break
                else:
                    break

        if len(chosen) == 11 and squad_price(chosen) <= BUDGET:
            return sorted(chosen, key=lambda r: (POS_ORDER[pos(r)], name(r))), "fallback_low_budget", lock_ids

    return [], "failed", []


def upgrade_budget(rows, formation, strategy_key, lock_ids, min_budget):
    locked = set(lock_ids)
    current = list(rows)

    improved = True
    while improved and squad_price(current) < min_budget:
        improved = False
        ids = {pid(r) for r in current}

        for i, old in enumerate(list(current)):
            if pid(old) in locked:
                continue

            candidates = [r for r in ALL if pos(r) == pos(old) and pid(r) not in ids and price(r) > price(old)]
            candidates.sort(key=lambda r: (score(r, strategy_key), price(r), start_pct(r)), reverse=True)

            for cand in candidates:
                trial = list(current)
                trial[i] = dict(cand)

                if len({pid(r) for r in trial if pid(r)}) != 11:
                    continue
                if squad_price(trial) > BUDGET:
                    continue
                if max(team_counts(trial).values() or [0]) > MAX_PER_TEAM:
                    continue

                current = trial
                improved = True
                break

            if improved:
                break

    return current


strategy_data = load_json(STRATEGY_PATH)
audit = []

for strategy_key, entry in strategy_data.items():
    formations = entry.get("squads_by_formation") or {}

    for formation, formation_entry in formations.items():
        if formation not in FORMATION_COUNTS:
            continue

        old_rows = [enrich(r) for r in formation_entry.get("squad", [])]
        new_rows, status, locked = build(strategy_key, formation)

        if new_rows:
            formation_entry["squad"] = new_rows
            formation_entry["manual_rotating_top_pair_status"] = status
            formation_entry["manual_locked_top_forwards"] = locked

        old_names = {name(r) for r in old_rows}
        new_names = {name(r) for r in new_rows}

        audit.append({
            "strategy": strategy_key,
            "formation": formation,
            "status": status,
            "locked": ", ".join(locked),
            "budget_m": round(squad_price(new_rows) / 1_000_000, 1) if new_rows else 0,
            "top_forwards": ", ".join(name(r) for r in new_rows if pos(r) == "FWD"),
            "in": ", ".join(sorted(new_names - old_names)),
            "out": ", ".join(sorted(old_names - new_names)),
        })

write_json(STRATEGY_PATH, strategy_data)

audit_path = OUT / "rotating_top_forward_pairs_changes.csv"
with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["strategy", "formation", "status", "locked", "budget_m", "top_forwards", "in", "out"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(audit)

print("Rotating top-forward pairs applied")
print("Audit:", audit_path)
for r in audit:
    print(f"{r['strategy']} {r['formation']}: {r['status']} | {r['budget_m']} mio. | locked: {r['locked']} | FWD: {r['top_forwards']}")
