import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

DATA = Path("data")
STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"

BUDGET = 50_000_000
MAX_PER_TEAM = 4

# Spredning: de her må ikke dominere alle formationer i samme strategi
PLAYER_LIMITS_BY_STRATEGY = {
    "lionel_messi__arg": 2,
    "cristiano_ronaldo__por": 2,
    "jonathan_david__can": 3,
}

MESSI_RONALDO = {"lionel_messi__arg", "cristiano_ronaldo__por"}

FORMATION_ORDER = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]

POS_ORDER = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def price(row):
    n = to_float(
        row.get("price")
        or row.get("price_estimate")
        or row.get("holdet_price")
        or 0
    )
    if n < 1000 and n > 0:
        n *= 1_000_000
    return int(round(n))


def player_id(row):
    return str(row.get("player_id") or "").strip()


def team_id(row):
    return str(row.get("team_id") or row.get("team") or row.get("team_name") or "").strip()


def position(row):
    return str(row.get("position") or row.get("holdet_position") or "").strip().upper()


def score(row):
    candidates = [
        row.get("strategy_score"),
        row.get("optimizer_ev"),
        row.get("weighted_group_stage_ev"),
        row.get("round1_ev"),
        row.get("display_score"),
        row.get("score"),
    ]
    for value in candidates:
        n = to_float(value, None)
        if n is not None:
            return n
    return 0.0


def start_pct(row):
    candidates = [
        row.get("start_probability_pct"),
        row.get("start_prob"),
        row.get("start_security"),
    ]
    for value in candidates:
        n = to_float(value, None)
        if n is not None and n > 0:
            return n * 100 if n <= 1 else n
    return 0.0


def is_out(row):
    value = str(row.get("holdet_is_out") or row.get("is_out") or "").strip().lower()
    return value in {"true", "1", "yes", "ja"}


def merge_row(base, extra):
    merged = dict(base or {})
    merged.update(extra or {})
    return merged


def load_player_pool_lookup():
    raw = load_json(POOL_PATH)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw

    lookup = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        keys = [
            player_id(p),
            str(p.get("holdet_player_id") or ""),
            f"{str(p.get('player_name') or '').strip().lower()}|{str(p.get('team_id') or '').strip().lower()}",
            f"{str(p.get('player_name') or '').strip().lower()}|{str(p.get('team_name') or '').strip().lower()}",
        ]
        for key in keys:
            if key:
                lookup[key] = p
    return lookup


POOL = load_player_pool_lookup()


def enrich(row):
    keys = [
        player_id(row),
        str(row.get("holdet_player_id") or ""),
        f"{str(row.get('player_name') or '').strip().lower()}|{str(row.get('team_id') or '').strip().lower()}",
        f"{str(row.get('player_name') or '').strip().lower()}|{str(row.get('team_name') or '').strip().lower()}",
    ]

    base = {}
    for key in keys:
        if key and key in POOL:
            base = POOL[key]
            break

    return merge_row(base, row)


def load_ev_candidates():
    candidates = []
    if not EV_PATH.exists():
        return candidates

    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = enrich(row)
            if position(row) == "FWD" and not is_out(row):
                candidates.append(row)

    return candidates


def row_key(row):
    return player_id(row) or f"{row.get('player_name')}|{team_id(row)}"


def squad_price(rows):
    return sum(price(r) for r in rows)


def team_counts(rows):
    c = Counter()
    for r in rows:
        c[team_id(r)] += 1
    return c


def can_use_candidate(candidate, squad_rows, outgoing_row, strategy_counts):
    cid = player_id(candidate)
    if not cid:
        return False

    if cid == player_id(outgoing_row):
        return False

    if cid in {player_id(r) for r in squad_rows}:
        return False

    if position(candidate) != "FWD":
        return False

    if is_out(candidate):
        return False

    if start_pct(candidate) and start_pct(candidate) < 55:
        return False

    # Undgå at replacement bare sætter Messi/Ronaldo tilbage
    if cid in MESSI_RONALDO:
        return False

    # Jonathan David må gerne bruges, men ikke over sin egen kvote
    if cid in PLAYER_LIMITS_BY_STRATEGY and strategy_counts[cid] >= PLAYER_LIMITS_BY_STRATEGY[cid]:
        return False

    new_price = squad_price(squad_rows) - price(outgoing_row) + price(candidate)
    if new_price > BUDGET:
        return False

    counts = team_counts(squad_rows)
    old_team = team_id(outgoing_row)
    new_team = team_id(candidate)

    counts[old_team] -= 1
    counts[new_team] += 1

    if counts[new_team] > MAX_PER_TEAM:
        return False

    return True


def candidate_sort_key(row):
    # Giv lidt plus til stærke alternativer, men uden at låse os til samme tre spillere
    pid = player_id(row)
    diversity_bonus = 0.0

    if pid in {
        "vinicius_junior__bra",
        "kylian_mbappe__fra",
        "erling_haaland__nor",
        "harry_kane__eng",
        "kai_havertz__ger",
        "mikel_oyarzabal__esp",
        "raul_jimenez__mex",
        "marko_arnautovic__aut",
        "mohamed_salah__egy",
        "omar_marmoush__egy",
        "breel_embolo__sui",
    }:
        diversity_bonus = 0.25

    return (
        score(row) + diversity_bonus,
        start_pct(row),
        -price(row),
    )


strategy_data = load_json(STRATEGY_PATH)
global_ev_candidates = load_ev_candidates()

changes = []

for strategy_key, strategy_entry in strategy_data.items():
    formations = strategy_entry.get("squads_by_formation") or {}
    if not formations:
        continue

    # Kandidater fra samme strategis egne hold + EV-filen
    strategy_candidates = []

    for formation_entry in formations.values():
        for row in formation_entry.get("squad") or []:
            row = enrich(row)
            if position(row) == "FWD" and not is_out(row):
                strategy_candidates.append(row)

    strategy_candidates.extend(global_ev_candidates)

    # Dedup candidates
    dedup = {}
    for c in strategy_candidates:
        key = row_key(c)
        if not key:
            continue
        if key not in dedup or candidate_sort_key(c) > candidate_sort_key(dedup[key]):
            dedup[key] = c

    strategy_candidates = sorted(dedup.values(), key=candidate_sort_key, reverse=True)

    strategy_counts = Counter()

    formation_names = sorted(
        formations.keys(),
        key=lambda f: FORMATION_ORDER.index(f) if f in FORMATION_ORDER else 99
    )

    for formation in formation_names:
        formation_entry = formations[formation]
        squad_rows = [enrich(r) for r in (formation_entry.get("squad") or [])]

        if not squad_rows:
            continue

        # Premium-angribere i dette hold
        premium_rows = [r for r in squad_rows if player_id(r) in PLAYER_LIMITS_BY_STRATEGY]

        # Behold højest scorede premium inden for reglerne; skift resten
        premium_rows_sorted = sorted(premium_rows, key=score, reverse=True)
        kept_messi_ronaldo_in_squad = False

        for premium in premium_rows_sorted:
            pid = player_id(premium)
            limit = PLAYER_LIMITS_BY_STRATEGY.get(pid)
            should_replace = False

            if limit is not None and strategy_counts[pid] >= limit:
                should_replace = True

            if pid in MESSI_RONALDO:
                if kept_messi_ronaldo_in_squad:
                    should_replace = True
                elif not should_replace:
                    kept_messi_ronaldo_in_squad = True

            if not should_replace:
                strategy_counts[pid] += 1
                continue

            replacement = None
            for candidate in strategy_candidates:
                if can_use_candidate(candidate, squad_rows, premium, strategy_counts):
                    replacement = candidate
                    break

            if not replacement:
                strategy_counts[pid] += 1
                changes.append({
                    "strategy": strategy_key,
                    "formation": formation,
                    "out": premium.get("player_name"),
                    "in": "",
                    "reason": "No valid replacement found",
                })
                continue

            replacement_row = dict(replacement)
            replacement_row["manual_diversity_replacement"] = True
            replacement_row["manual_diversity_replacement_reason"] = (
                f"Replaced {premium.get('player_name')} to vary premium forwards "
                f"within strategy {strategy_key}."
            )

            for idx, row in enumerate(squad_rows):
                if player_id(row) == pid:
                    squad_rows[idx] = replacement_row
                    break

            strategy_counts[player_id(replacement_row)] += 1

            changes.append({
                "strategy": strategy_key,
                "formation": formation,
                "out": premium.get("player_name"),
                "in": replacement_row.get("player_name"),
                "reason": "Premium forward diversity",
            })

        # Sortér pænt tilbage efter position
        squad_rows.sort(key=lambda r: (POS_ORDER.get(position(r), 99), str(r.get("player_name") or "")))
        formation_entry["squad"] = squad_rows
        formation_entry["manual_forward_diversity_applied"] = True

write_json(STRATEGY_PATH, strategy_data)

audit_path = DATA / "strategy_squad_exports" / "forward_diversity_changes.csv"
with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["strategy", "formation", "out", "in", "reason"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(changes)

print("Forward diversity applied")
print("Changes:", len(changes))
print("Audit:", audit_path)

for c in changes:
    print(f"{c['strategy']} {c['formation']}: {c['out']} -> {c['in'] or 'INGEN ERSTATNING'}")
