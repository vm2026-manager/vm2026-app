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

OUT_MD = OUT / "top_forward_alternatives.md"
OUT_CSV = OUT / "top_forward_alternatives.csv"

BUDGET = 50_000_000
MAX_PER_TEAM = 4

SKIP_STRATEGIES = {"round1_2"}

FORMATION_COUNTS = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

STRATEGY_NAMES = {
    "next_round": "Næste runde",
    "practical_start": "1. + 2. runde",
    "group_stage": "Gruppespil",
    "long_run": "Lang sigt",
}

POS_DK = {
    "GK": "Mål",
    "DEF": "For",
    "MID": "Mid",
    "FWD": "Ang",
}

POS_ORDER = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

TARGET_PACKAGES = [
    ("Haaland + Kane", ["erling_haaland__nor", "harry_kane__eng"]),
    ("Haaland + Mbappé", ["erling_haaland__nor", "kylian_mbappe__fra"]),
    ("Kane + Oyarzabal", ["harry_kane__eng", "mikel_oyarzabal__esp"]),
    ("Mbappé + Olise", ["kylian_mbappe__fra", "michael_olise__fra"]),
    ("Vini + Olise", ["vinicius_junior__bra", "michael_olise__fra"]),
    ("Vini + Oyarzabal", ["vinicius_junior__bra", "mikel_oyarzabal__esp"]),
    ("Ronaldo + Haaland", ["cristiano_ronaldo__por", "erling_haaland__nor"]),
    ("Messi + Vini", ["lionel_messi__arg", "vinicius_junior__bra"]),
    ("Dembélé + Mbappé", ["ousmane_dembele__fra", "kylian_mbappe__fra"]),
    ("Dembélé + Oyarzabal", ["ousmane_dembele__fra", "mikel_oyarzabal__esp"]),
]

SINGLE_TARGETS = [
    ("Haaland", ["erling_haaland__nor"]),
    ("Kane", ["harry_kane__eng"]),
    ("Mbappé", ["kylian_mbappe__fra"]),
    ("Oyarzabal", ["mikel_oyarzabal__esp"]),
    ("Vini", ["vinicius_junior__bra"]),
    ("Olise", ["michael_olise__fra"]),
    ("Dembélé", ["ousmane_dembele__fra"]),
    ("Ronaldo", ["cristiano_ronaldo__por"]),
    ("Messi", ["lionel_messi__arg"]),
]

DEMBO_IDS = {"ousmane_dembele__fra", "ousmane_dembélé__fra"}
DEMBO_START_PROB = 0.68


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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


def country(row):
    return str(row.get("team_name") or row.get("team_id") or row.get("team") or "").strip()


def pos(row):
    return str(row.get("position") or row.get("holdet_position") or "").strip().upper()


def price(row):
    n = to_float(row.get("price") or row.get("price_estimate") or row.get("holdet_price"), 0)
    if 0 < n < 1000:
        n *= 1_000_000
    return int(round(n))


def price_m(row):
    return price(row) / 1_000_000


def price_fmt(value):
    return f"{value:.1f}".replace(".", ",") + " mio."


def start_pct(row):
    if pid(row) in DEMBO_IDS:
        return DEMBO_START_PROB * 100

    for key in ["start_probability_pct", "start_prob", "start_security"]:
        n = to_float(row.get(key), None)
        if n is not None and n > 0:
            return n * 100 if n <= 1 else n

    return 0.0


def score(row, strategy_key=None):
    keys = ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"]

    if strategy_key == "next_round":
        keys = ["round1_ev", "match_1_weighted_match_ev"] + keys

    for key in keys:
        n = to_float(row.get(key), None)
        if n is not None:
            return n

    return 0.0


def is_out(row):
    return str(row.get("holdet_is_out") or row.get("is_out") or "").strip().lower() in {"true", "1", "yes", "ja"}


def squad_price(rows):
    return sum(price(r) for r in rows)


def team_counts(rows):
    c = Counter()
    for r in rows:
        c[team(r)] += 1
    return c


def valid_squad(rows, formation):
    if len(rows) != 11:
        return False

    if squad_price(rows) > BUDGET:
        return False

    ids = [pid(r) for r in rows if pid(r)]
    if len(ids) != len(set(ids)):
        return False

    wanted = FORMATION_COUNTS[formation]
    counts = Counter(pos(r) for r in rows)

    for p, n in wanted.items():
        if counts[p] != n:
            return False

    if max(team_counts(rows).values() or [0]) > MAX_PER_TEAM:
        return False

    return True


def load_pool_lookup():
    raw = load_json(POOL_PATH)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw

    lookup = {}

    for p in players:
        if not isinstance(p, dict):
            continue

        keys = [
            pid(p),
            str(p.get("holdet_player_id") or ""),
            f"{name(p).lower()}|{str(p.get('team_id') or '').lower()}",
            f"{name(p).lower()}|{str(p.get('team_name') or '').lower()}",
        ]

        for key in keys:
            if key:
                lookup[key] = p

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
    for key in keys:
        if key and key in POOL:
            base = POOL[key]
            break

    merged = dict(base)
    merged.update(row or {})

    if pid(merged) in DEMBO_IDS:
        merged["start_prob"] = DEMBO_START_PROB
        merged["start_probability_pct"] = round(DEMBO_START_PROB * 100, 1)
        merged["start_security"] = DEMBO_START_PROB

    return merged


def load_candidates():
    rows = []

    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = enrich(row)

            if is_out(row):
                continue

            if pos(row) not in {"GK", "DEF", "MID", "FWD"}:
                continue

            if price(row) <= 0:
                continue

            if start_pct(row) < 55:
                continue

            rows.append(row)

    best = {}

    for row in rows:
        key = pid(row) or f"{name(row).lower()}|{team(row).lower()}"

        if key not in best or score(row) > score(best[key]):
            best[key] = row

    return list(best.values())


ALL_CANDIDATES = load_candidates()
BY_ID = {pid(r): r for r in ALL_CANDIDATES if pid(r)}


def candidate_rank(row, strategy_key):
    value_bonus = 0

    if pid(row) in {
        "nathaniel_brown__ger",
        "antonee_robinson__usa",
        "cesar_montes__mex",
        "silvan_widmer__sui",
        "philipp_lienhart__aut",
        "roberto_alvarado__mex",
        "john_yeboah__ecu",
        "antonio_nusa__nor",
        "brian_gutierrez__mex",
    }:
        value_bonus += 1.5

    return (
        score(row, strategy_key) + value_bonus,
        start_pct(row),
        -price(row),
    )


def sort_squad(rows):
    return sorted(rows, key=lambda r: (POS_ORDER.get(pos(r), 99), name(r)))


def can_add(row, chosen, formation):
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


def build_from_package(strategy_key, formation, package_name, package_ids):
    wanted = FORMATION_COUNTS[formation]

    if wanted["FWD"] < len(package_ids):
        return None, "formation_has_too_few_forward_slots"

    chosen = []

    for target_id in package_ids:
        row = BY_ID.get(target_id)

        if not row:
            return None, f"missing_target:{target_id}"

        if pos(row) != "FWD":
            return None, f"target_not_fwd:{target_id}"

        if not can_add(row, chosen, formation):
            return None, f"cannot_add_target:{target_id}"

        chosen.append(dict(row))

    for p in ["GK", "DEF", "MID", "FWD"]:
        while Counter(pos(x) for x in chosen)[p] < wanted[p]:
            candidates = [r for r in ALL_CANDIDATES if pos(r) == p]
            candidates.sort(key=lambda r: candidate_rank(r, strategy_key), reverse=True)

            added = False
            for cand in candidates:
                if can_add(cand, chosen, formation):
                    chosen.append(dict(cand))
                    added = True
                    break

            if not added:
                return None, f"could_not_fill:{p}"

    if not valid_squad(chosen, formation):
        return None, "invalid_squad"

    return sort_squad(chosen), "ok"


def total_score(rows, strategy_key):
    return sum(score(r, strategy_key) for r in rows)


def summarize_forwards(rows):
    return ", ".join(name(r) for r in rows if pos(r) == "FWD")


def row_line(row):
    return (
        f"| {POS_DK.get(pos(row), pos(row))} | {name(row)} | {country(row)} | "
        f"{price_fmt(price_m(row))} | {round(start_pct(row))}% | "
        f"{str(round(score(row), 3)).replace('.', ',')} |"
    )


strategy_data = load_json(STRATEGY_PATH)

summary_rows = []
md = []

md.append("# Alternative topangriber-hold")
md.append("")
md.append("Dette er en separat test-rapport. Den ændrer ikke `optimal_squads_by_strategy.json`.")
md.append("Formålet er at sammenligne basehold med alternative angriberpakker.")
md.append("")

for strategy_key, strategy_entry in strategy_data.items():
    if strategy_key in SKIP_STRATEGIES:
        continue

    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key)
    formations = strategy_entry.get("squads_by_formation") or {}

    md.append(f"# {strategy_name}")
    md.append("")

    for formation, formation_entry in formations.items():
        if formation not in FORMATION_COUNTS:
            continue

        base = [enrich(r) for r in (formation_entry.get("squad") or [])]
        if not base:
            continue

        base_score = total_score(base, strategy_key)
        base_budget = squad_price(base) / 1_000_000

        packages = SINGLE_TARGETS if FORMATION_COUNTS[formation]["FWD"] == 1 else TARGET_PACKAGES

        variants = []

        for package_name, package_ids in packages:
            squad, status = build_from_package(strategy_key, formation, package_name, package_ids)

            if squad is None:
                continue

            diff = total_score(squad, strategy_key) - base_score

            variants.append({
                "package": package_name,
                "squad": squad,
                "score": total_score(squad, strategy_key),
                "diff": diff,
                "budget": squad_price(squad) / 1_000_000,
                "forwards": summarize_forwards(squad),
                "in": sorted({name(r) for r in squad} - {name(r) for r in base}),
                "out": sorted({name(r) for r in base} - {name(r) for r in squad}),
            })

        variants.sort(key=lambda v: (v["diff"], v["score"]), reverse=True)
        variants = variants[:5]

        md.append(f"## {strategy_name} – {formation}")
        md.append("")
        md.append(f"**Base:** {price_fmt(base_budget)} | score {str(round(base_score, 3)).replace('.', ',')} | angreb: {summarize_forwards(base)}")
        md.append("")

        summary_rows.append({
            "strategy": strategy_name,
            "formation": formation,
            "variant": "BASE",
            "budget_m": round(base_budget, 1),
            "score": round(base_score, 3),
            "diff_vs_base": 0,
            "forwards": summarize_forwards(base),
            "in": "",
            "out": "",
        })

        md.append("| Variant | Budget | Score | Diff | Angribere | Ind | Ud |")
        md.append("|---|---:|---:|---:|---|---|---|")

        for v in variants:
            md.append(
                f"| {v['package']} | {price_fmt(v['budget'])} | "
                f"{str(round(v['score'], 3)).replace('.', ',')} | "
                f"{str(round(v['diff'], 3)).replace('.', ',')} | "
                f"{v['forwards']} | {', '.join(v['in'])} | {', '.join(v['out'])} |"
            )

            summary_rows.append({
                "strategy": strategy_name,
                "formation": formation,
                "variant": v["package"],
                "budget_m": round(v["budget"], 1),
                "score": round(v["score"], 3),
                "diff_vs_base": round(v["diff"], 3),
                "forwards": v["forwards"],
                "in": ", ".join(v["in"]),
                "out": ", ".join(v["out"]),
            })

        md.append("")

        # Skriv fulde hold for de 2 bedste varianter
        for v in variants[:2]:
            md.append(f"### Fuldt hold: {v['package']}")
            md.append("")
            md.append(f"Budget: {price_fmt(v['budget'])} | score: {str(round(v['score'], 3)).replace('.', ',')} | diff: {str(round(v['diff'], 3)).replace('.', ',')}")
            md.append("")
            md.append("| Pos | Spiller | Land | Pris | Start | EV |")
            md.append("|---|---|---:|---:|---:|---:|")

            for row in v["squad"]:
                md.append(row_line(row))

            md.append("")

OUT_MD.write_text("\n".join(md), encoding="utf-8")

with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "strategy",
        "formation",
        "variant",
        "budget_m",
        "score",
        "diff_vs_base",
        "forwards",
        "in",
        "out",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(summary_rows)

print("FÆRDIG")
print("Markdown:", OUT_MD)
print("CSV:", OUT_CSV)
print("Antal rækker i summary:", len(summary_rows))
