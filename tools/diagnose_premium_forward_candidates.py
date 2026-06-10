import csv
import json
from pathlib import Path
from collections import Counter

DATA = Path("data")
STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"

BUDGET = 50_000_000
MAX_PER_TEAM = 4

WATCH_NAMES = [
    "Kylian Mbappe",
    "Kylian Mbappé",
    "Harry Kane",
    "Erling Haaland",
    "Mikel Oyarzabal",
    "Michael Olise",
    "Ousmane Dembele",
    "Ousmane Dembélé",
    "Dani Olmo",
    "Lamine Yamal",
    "Kai Havertz",
    "Julian Alvarez",
    "Julián Álvarez",
    "Vinicius Junior",
    "Vinícius Júnior",
    "Cody Gakpo",
    "Raul Jimenez",
    "Raúl Jiménez",
    "Marko Arnautovic",
]

PREMIUM_OUT = {
    "lionel_messi__arg",
    "cristiano_ronaldo__por",
    "jonathan_david__can",
}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def norm(text):
    return str(text or "").lower().replace("é", "e").replace("í", "i").replace("á", "a").replace("ú", "u").replace("ó", "o")


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def price(row):
    n = to_float(row.get("price") or row.get("price_estimate") or row.get("holdet_price") or 0)
    if 0 < n < 1000:
        n *= 1_000_000
    return int(round(n))


def price_m(row):
    p = price(row)
    return f"{p/1_000_000:.1f}".replace(".", ",") + " mio." if p else ""


def pid(row):
    return str(row.get("player_id") or "").strip()


def team(row):
    return str(row.get("team_id") or row.get("team_name") or row.get("team") or "").strip()


def pos(row):
    return str(row.get("position") or row.get("holdet_position") or "").strip().upper()


def score(row):
    for key in ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"]:
        val = to_float(row.get(key), None)
        if val is not None:
            return val
    return 0.0


def start_pct(row):
    for key in ["start_probability_pct", "start_prob", "start_security"]:
        val = to_float(row.get(key), None)
        if val is not None and val > 0:
            return val * 100 if val <= 1 else val
    return 0.0


def is_out(row):
    return str(row.get("holdet_is_out") or row.get("is_out") or "").strip().lower() in {"true", "1", "yes", "ja"}


def load_pool():
    raw = load_json(POOL_PATH)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw
    lookup = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        keys = [
            pid(p),
            str(p.get("holdet_player_id") or ""),
            f"{norm(p.get('player_name'))}|{norm(p.get('team_id'))}",
            f"{norm(p.get('player_name'))}|{norm(p.get('team_name'))}",
        ]
        for k in keys:
            if k:
                lookup[k] = p
    return lookup


POOL = load_pool()


def enrich(row):
    keys = [
        pid(row),
        str(row.get("holdet_player_id") or ""),
        f"{norm(row.get('player_name'))}|{norm(row.get('team_id'))}",
        f"{norm(row.get('player_name'))}|{norm(row.get('team_name'))}",
    ]

    base = {}
    for k in keys:
        if k and k in POOL:
            base = POOL[k]
            break

    merged = dict(base)
    merged.update(row)
    return merged


def load_ev_players():
    rows = []
    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(enrich(row))
    return rows


def squad_price(rows):
    return sum(price(r) for r in rows)


def team_counts(rows):
    c = Counter()
    for r in rows:
        c[team(r)] += 1
    return c


def find_watch_candidates(all_players):
    found = []
    for p in all_players:
        name = norm(p.get("player_name"))
        if any(norm(w) in name for w in WATCH_NAMES):
            found.append(p)

    # dedup
    out = {}
    for p in found:
        key = pid(p) or norm(p.get("player_name"))
        if key not in out or score(p) > score(out[key]):
            out[key] = p
    return sorted(out.values(), key=lambda r: score(r), reverse=True)


def explain_candidate(candidate, squad_rows, outgoing):
    reasons = []

    if pos(candidate) != "FWD":
        reasons.append(f"position={pos(candidate)} ikke FWD")

    if is_out(candidate):
        reasons.append("is_out/holdet_is_out")

    if start_pct(candidate) and start_pct(candidate) < 55:
        reasons.append(f"start lav: {start_pct(candidate):.0f}%")

    if pid(candidate) in {pid(r) for r in squad_rows}:
        reasons.append("allerede i holdet")

    new_price = squad_price(squad_rows) - price(outgoing) + price(candidate)
    if new_price > BUDGET:
        reasons.append(f"for dyrt: {new_price/1_000_000:.1f} mio.")

    counts = team_counts(squad_rows)
    old_team = team(outgoing)
    new_team = team(candidate)
    counts[old_team] -= 1
    counts[new_team] += 1
    if counts[new_team] > MAX_PER_TEAM:
        reasons.append(f"max {MAX_PER_TEAM} fra {new_team}")

    if not reasons:
        reasons.append("OK")

    return " | ".join(reasons)


strategy_data = load_json(STRATEGY_PATH)
all_players = load_ev_players()
watch_candidates = find_watch_candidates(all_players)

out_rows = []

for strategy_key, strategy_entry in strategy_data.items():
    formations = strategy_entry.get("squads_by_formation") or {}

    for formation, formation_entry in formations.items():
        squad_rows = [enrich(r) for r in (formation_entry.get("squad") or [])]
        outgoing_rows = [r for r in squad_rows if pid(r) in PREMIUM_OUT]

        if not outgoing_rows:
            continue

        for outgoing in outgoing_rows:
            for cand in watch_candidates:
                out_rows.append({
                    "strategy": strategy_key,
                    "formation": formation,
                    "outgoing": outgoing.get("player_name"),
                    "outgoing_price": price_m(outgoing),
                    "candidate": cand.get("player_name"),
                    "candidate_team": cand.get("team_name") or cand.get("team_id"),
                    "candidate_pos": pos(cand),
                    "candidate_price": price_m(cand),
                    "candidate_start": round(start_pct(cand), 1),
                    "candidate_score": round(score(cand), 3),
                    "reason": explain_candidate(cand, squad_rows, outgoing),
                })

out_path = DATA / "strategy_squad_exports" / "premium_forward_candidate_diagnostics.csv"
with out_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "strategy",
        "formation",
        "outgoing",
        "outgoing_price",
        "candidate",
        "candidate_team",
        "candidate_pos",
        "candidate_price",
        "candidate_start",
        "candidate_score",
        "reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(out_rows)

print("Skrev:", out_path)
print("Kandidater fundet:")
for c in watch_candidates:
    print(f"- {c.get('player_name')} | {c.get('team_name') or c.get('team_id')} | {pos(c)} | {price_m(c)} | start {start_pct(c):.0f}% | score {score(c):.3f}")
