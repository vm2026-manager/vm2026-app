import json
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
DATA = ROOT / "data"
OUT_DIR = DATA / "strategy_squad_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_POOL_PATH = DATA / "player_pool_v1.json"

CANDIDATE_FILES = [
    DATA / "optimal_squads_by_formation.json",
    DATA / "clean_market_optimizer_squads.json",
    DATA / "generated_5_squads_from_bruttoliste.json",
]

POSITION_ORDER = {
    "GK": 1,
    "Mål": 1,
    "MÅL": 1,
    "Maal": 1,
    "DEF": 2,
    "For": 2,
    "Forsvar": 2,
    "MID": 3,
    "Mid": 3,
    "Midtbane": 3,
    "FWD": 4,
    "Ang": 4,
    "Angriber": 4,
}

POSITION_DK = {
    "GK": "Mål",
    "Mål": "Mål",
    "MÅL": "Mål",
    "Maal": "Mål",
    "DEF": "For",
    "For": "For",
    "Forsvar": "For",
    "MID": "Mid",
    "Mid": "Mid",
    "Midtbane": "Mid",
    "FWD": "Ang",
    "Ang": "Ang",
    "Angriber": "Ang",
}

STRATEGY_LABELS = {
    "next_round": "Næste runde",
    "round_1": "Næste runde",
    "round1": "Næste runde",
    "rounds_1_2": "1. + 2. runde",
    "round_1_2": "1. + 2. runde",
    "r1_r2": "1. + 2. runde",
    "group_stage": "Gruppespil",
    "group": "Gruppespil",
    "long_run": "Lang sigt",
    "long_term": "Lang sigt",
}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def player_key(player):
    if not isinstance(player, dict):
        return ""
    return str(
        player.get("player_id")
        or player.get("id")
        or player.get("holdet_player_id")
        or ""
    )


def load_player_pool():
    if not PLAYER_POOL_PATH.exists():
        return {}

    raw = load_json(PLAYER_POOL_PATH)
    if isinstance(raw, dict):
        players = raw.get("players") or raw.get("data") or []
    else:
        players = raw

    lookup = {}
    for p in players:
        if not isinstance(p, dict):
            continue

        keys = {
            str(p.get("player_id") or ""),
            str(p.get("holdet_player_id") or ""),
            f'{str(p.get("player_name") or "").strip().lower()}|{str(p.get("team_id") or p.get("team_name") or "").strip().lower()}',
        }

        for key in keys:
            if key:
                lookup[key] = p

    return lookup


PLAYER_LOOKUP = load_player_pool()


def enrich_player(p):
    if not isinstance(p, dict):
        return {}

    pid = str(p.get("player_id") or p.get("id") or "")
    hid = str(p.get("holdet_player_id") or "")
    name_team = f'{str(p.get("player_name") or p.get("name") or "").strip().lower()}|{str(p.get("team_id") or p.get("team_name") or p.get("team") or "").strip().lower()}'

    base = {}
    for key in [pid, hid, name_team]:
        if key and key in PLAYER_LOOKUP:
            base = PLAYER_LOOKUP[key]
            break

    merged = dict(base)
    merged.update(p)
    return merged


def norm_position(pos):
    pos = str(pos or "").strip()
    return POSITION_DK.get(pos, pos or "")


def pos_sort(pos):
    return POSITION_ORDER.get(str(pos or "").strip(), 99)


def money_m(value):
    try:
        n = float(value)
    except Exception:
        return ""

    if n > 100000:
        n = n / 1000000

    return f"{n:.1f}".replace(".", ",") + " mio."


def pct(value):
    try:
        n = float(value)
    except Exception:
        return ""

    if n <= 1:
        n = n * 100

    return f"{round(n)}%"


def get_players_from_node(node):
    if isinstance(node, dict):
        for key in ["players", "squad", "selected_players", "lineup", "team"]:
            value = node.get(key)
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                return value

    if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        return node

    return None


def looks_like_formation(value):
    text = str(value)
    return bool(text.count("-") == 2 and all(part.isdigit() for part in text.split("-")))


def walk(node, path_parts=None):
    if path_parts is None:
        path_parts = []

    players = get_players_from_node(node)
    if players:
        meta = {}
        if isinstance(node, dict):
            meta = node

        yield path_parts, players, meta
        return

    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"players", "squad", "selected_players", "lineup", "team"}:
                continue
            if isinstance(value, (dict, list)):
                yield from walk(value, path_parts + [str(key)])

    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, (dict, list)):
                yield from walk(value, path_parts + [str(i + 1)])


def infer_strategy(path_parts, meta):
    for key in ["strategy", "strategy_key", "strategy_id", "strategy_name"]:
        if isinstance(meta, dict) and meta.get(key):
            raw = str(meta.get(key))
            return STRATEGY_LABELS.get(raw, raw)

    for part in path_parts:
        low = part.lower()
        if low in STRATEGY_LABELS:
            return STRATEGY_LABELS[low]
        for key, label in STRATEGY_LABELS.items():
            if key in low:
                return label

    return "Ukendt strategi"


def infer_formation(path_parts, meta):
    for key in ["formation", "formation_key"]:
        if isinstance(meta, dict) and meta.get(key):
            return str(meta.get(key))

    for part in path_parts:
        if looks_like_formation(part):
            return part

    return "Ukendt formation"


def squad_score(meta):
    for key in [
        "score",
        "total_score",
        "strategy_score",
        "weighted_score",
        "ev",
        "optimizer_ev",
        "total_ev",
        "raw_ev",
    ]:
        if isinstance(meta, dict) and meta.get(key) not in [None, ""]:
            return meta.get(key)
    return ""


def collect_from_file(path):
    if not path.exists():
        return []

    raw = load_json(path)
    rows = []

    for path_parts, players, meta in walk(raw):
        strategy = infer_strategy(path_parts, meta)
        formation = infer_formation(path_parts, meta)
        score = squad_score(meta)

        enriched = [enrich_player(p) for p in players]

        enriched.sort(
            key=lambda p: (
                pos_sort(norm_position(p.get("position") or p.get("holdet_position"))),
                str(p.get("player_name") or p.get("name") or ""),
            )
        )

        for idx, p in enumerate(enriched, start=1):
            pos = norm_position(p.get("position") or p.get("holdet_position"))
            price = p.get("price") or p.get("price_estimate") or p.get("holdet_price")
            start = p.get("start_probability_pct") or p.get("start_prob") or p.get("start_security")
            ev = p.get("optimizer_ev") or p.get("weighted_group_stage_ev") or p.get("round1_ev") or ""

            rows.append({
                "source_file": path.name,
                "strategy": strategy,
                "formation": formation,
                "squad_score": score,
                "slot_no": idx,
                "position": pos,
                "player_name": p.get("player_name") or p.get("name") or "",
                "team": p.get("team_name") or p.get("team_id") or p.get("team") or "",
                "price": money_m(price),
                "start": pct(start),
                "ev": ev,
                "player_id": p.get("player_id") or "",
            })

    return rows


all_rows = []

for path in CANDIDATE_FILES:
    all_rows.extend(collect_from_file(path))

# Ekstra: scan relevante squad-json-filer, hvis nye strategifiler er kommet til
for path in DATA.glob("*.json"):
    name = path.name.lower()
    if path in CANDIDATE_FILES:
        continue
    if any(token in name for token in ["squad", "formation", "strategy", "optimizer"]):
        all_rows.extend(collect_from_file(path))

# fjern dubletter
seen = set()
deduped = []
for row in all_rows:
    key = (
        row["source_file"],
        row["strategy"],
        row["formation"],
        row["slot_no"],
        row["player_id"],
        row["player_name"],
        row["team"],
    )
    if key in seen:
        continue
    seen.add(key)
    deduped.append(row)

strategy_order = {
    "Næste runde": 1,
    "1. + 2. runde": 2,
    "Gruppespil": 3,
    "Lang sigt": 4,
    "Ukendt strategi": 9,
}

deduped.sort(
    key=lambda r: (
        strategy_order.get(r["strategy"], 8),
        r["strategy"],
        r["formation"],
        pos_sort(r["position"]),
        r["slot_no"],
    )
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUT_DIR / f"all_strategy_squads_by_position_{timestamp}.csv"
md_path = OUT_DIR / f"all_strategy_squads_by_position_{timestamp}.md"

fieldnames = [
    "source_file",
    "strategy",
    "formation",
    "squad_score",
    "slot_no",
    "position",
    "player_name",
    "team",
    "price",
    "start",
    "ev",
    "player_id",
]

with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Alle strategi-hold efter position\n\n")

    current_group = None
    for row in deduped:
        group = (row["source_file"], row["strategy"], row["formation"])
        if group != current_group:
            current_group = group
            f.write(f"\n## {row['strategy']} – {row['formation']} ({row['source_file']})\n\n")
            if row["squad_score"] != "":
                f.write(f"Score: {row['squad_score']}\n\n")
            f.write("| Pos | Spiller | Land | Pris | Start | EV |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")

        f.write(
            f"| {row['position']} | {row['player_name']} | {row['team']} | "
            f"{row['price']} | {row['start']} | {row['ev']} |\n"
        )

print("Skrev CSV:", csv_path)
print("Skrev Markdown:", md_path)
print("Antal spillerrækker:", len(deduped))
