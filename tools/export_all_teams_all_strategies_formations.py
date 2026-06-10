import json
import csv
from pathlib import Path
from datetime import datetime

DATA = Path("data")
OUT = DATA / "strategy_squad_exports"
OUT.mkdir(exist_ok=True)

PLAYER_POOL = DATA / "player_pool_v1.json"

SQUAD_FILES = [
    DATA / "optimal_squads_by_formation.json",
    DATA / "clean_market_optimizer_squads.json",
    DATA / "generated_5_squads_from_bruttoliste.json",
]

POS_ORDER = {"GK": 1, "Mål": 1, "DEF": 2, "For": 2, "MID": 3, "Mid": 3, "FWD": 4, "Ang": 4}
POS_DK = {"GK": "Mål", "MÅL": "Mål", "DEF": "For", "Forsvar": "For", "MID": "Mid", "Midtbane": "Mid", "FWD": "Ang", "Angriber": "Ang"}

STRATEGY_DK = {
    "next_round": "Næste runde",
    "round1": "Næste runde",
    "round_1": "Næste runde",
    "rounds_1_2": "1. + 2. runde",
    "round_1_2": "1. + 2. runde",
    "group_stage": "Gruppespil",
    "group": "Gruppespil",
    "long_run": "Lang sigt",
    "long_term": "Lang sigt",
}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_players():
    if not PLAYER_POOL.exists():
        return {}

    raw = load_json(PLAYER_POOL)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw

    lookup = {}
    for p in players:
        if not isinstance(p, dict):
            continue

        keys = [
            str(p.get("player_id") or ""),
            str(p.get("holdet_player_id") or ""),
            f"{str(p.get('player_name') or '').lower()}|{str(p.get('team_id') or '').lower()}",
            f"{str(p.get('player_name') or '').lower()}|{str(p.get('team_name') or '').lower()}",
        ]

        for k in keys:
            if k:
                lookup[k] = p

    return lookup


PLAYER_LOOKUP = load_players()


def norm_pos(pos):
    return POS_DK.get(str(pos or "").strip(), str(pos or "").strip())


def pos_sort(pos):
    return POS_ORDER.get(norm_pos(pos), 99)


def price_fmt(value):
    try:
        n = float(value)
    except Exception:
        return ""

    if n > 100000:
        n = n / 1000000

    return f"{n:.1f}".replace(".", ",") + " mio."


def pct_fmt(value):
    try:
        n = float(value)
    except Exception:
        return ""

    if n <= 1:
        n *= 100

    return f"{round(n)}%"


def enrich(p):
    if not isinstance(p, dict):
        return {}

    keys = [
        str(p.get("player_id") or p.get("id") or ""),
        str(p.get("holdet_player_id") or ""),
        f"{str(p.get('player_name') or p.get('name') or '').lower()}|{str(p.get('team_id') or p.get('team') or '').lower()}",
        f"{str(p.get('player_name') or p.get('name') or '').lower()}|{str(p.get('team_name') or '').lower()}",
    ]

    base = {}
    for k in keys:
        if k in PLAYER_LOOKUP:
            base = PLAYER_LOOKUP[k]
            break

    merged = dict(base)
    merged.update(p)
    return merged


def is_formation(text):
    parts = str(text).split("-")
    return len(parts) == 3 and all(x.isdigit() for x in parts)


def get_players(node):
    if isinstance(node, dict):
        for k in ["players", "squad", "selected_players", "lineup", "team"]:
            v = node.get(k)
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v

    if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        return node

    return None


def walk(node, path=None):
    path = path or []

    players = get_players(node)
    if players:
        yield path, players, node if isinstance(node, dict) else {}
        return

    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                yield from walk(v, path + [str(k)])

    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                yield from walk(v, path + [str(i + 1)])


def infer_strategy(path, meta):
    for k in ["strategy", "strategy_key", "strategy_id", "strategy_name"]:
        if isinstance(meta, dict) and meta.get(k):
            raw = str(meta.get(k))
            return STRATEGY_DK.get(raw, raw)

    joined = " ".join(path).lower()
    for key, label in STRATEGY_DK.items():
        if key in joined:
            return label

    return "Standard"


def infer_formation(path, meta):
    for k in ["formation", "formation_key"]:
        if isinstance(meta, dict) and meta.get(k):
            return str(meta.get(k))

    for p in path:
        if is_formation(p):
            return p

    return "Ukendt formation"


def get_score(meta):
    for k in ["score", "total_score", "strategy_score", "weighted_score", "optimizer_ev", "total_ev", "raw_ev"]:
        if isinstance(meta, dict) and meta.get(k) not in [None, ""]:
            return meta.get(k)
    return ""


rows = []

for file in SQUAD_FILES:
    if not file.exists():
        continue

    raw = load_json(file)

    for path, players, meta in walk(raw):
        strategy = infer_strategy(path, meta)
        formation = infer_formation(path, meta)
        score = get_score(meta)

        enriched = [enrich(p) for p in players]
        enriched.sort(key=lambda p: (pos_sort(p.get("position") or p.get("holdet_position")), str(p.get("player_name") or p.get("name") or "")))

        for i, p in enumerate(enriched, 1):
            rows.append({
                "Kilde": file.name,
                "Strategi": strategy,
                "Formation": formation,
                "Score": score,
                "Nr": i,
                "Pos": norm_pos(p.get("position") or p.get("holdet_position")),
                "Spiller": p.get("player_name") or p.get("name") or "",
                "Land": p.get("team_name") or p.get("team_id") or p.get("team") or "",
                "Pris": price_fmt(p.get("price") or p.get("price_estimate") or p.get("holdet_price")),
                "Start": pct_fmt(p.get("start_probability_pct") or p.get("start_prob") or p.get("start_security")),
                "EV": p.get("optimizer_ev") or p.get("weighted_group_stage_ev") or "",
                "player_id": p.get("player_id") or "",
            })


rows.sort(key=lambda r: (r["Strategi"], r["Formation"], pos_sort(r["Pos"]), r["Nr"]))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUT / f"alle_hold_alle_strategier_formationer_{timestamp}.csv"
md_path = OUT / f"alle_hold_alle_strategier_formationer_{timestamp}.md"

with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["Ingen data"])
    writer.writeheader()
    writer.writerows(rows)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Alle hold – alle strategier og formationer\n\n")

    current = None
    for r in rows:
        group = (r["Strategi"], r["Formation"], r["Kilde"])
        if group != current:
            current = group
            f.write(f"\n## {r['Strategi']} – {r['Formation']}\n\n")
            f.write(f"Kilde: `{r['Kilde']}`\n\n")
            if r["Score"] != "":
                f.write(f"Score: {r['Score']}\n\n")
            f.write("| Pos | Spiller | Land | Pris | Start | EV |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")

        f.write(f"| {r['Pos']} | {r['Spiller']} | {r['Land']} | {r['Pris']} | {r['Start']} | {r['EV']} |\n")


print("\nFÆRDIG")
print("CSV:", csv_path)
print("Markdown:", md_path)
print("Antal spillerrækker:", len(rows))

print("\nKORT OVERSIGT:")
last = None
for r in rows:
    group = (r["Strategi"], r["Formation"])
    if group != last:
        last = group
        print(f"\n=== {r['Strategi']} – {r['Formation']} ===")
    print(f"{r['Pos']:>3} | {r['Spiller']:<24} | {r['Land']:<10} | {r['Pris']:>8} | Start {r['Start']:>4}")
