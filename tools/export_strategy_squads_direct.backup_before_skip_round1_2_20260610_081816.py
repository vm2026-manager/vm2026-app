import json
import csv
from pathlib import Path
from datetime import datetime

DATA = Path("data")
OUT = DATA / "strategy_squad_exports"
OUT.mkdir(exist_ok=True)

STRATEGY_FILE = DATA / "optimal_squads_by_strategy.json"
PLAYER_POOL_FILE = DATA / "player_pool_v1.json"

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

POS_ORDER = {
    "Mål": 1,
    "For": 2,
    "Mid": 3,
    "Ang": 4,
}


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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
        n = n * 100

    return f"{round(n)}%"


def ev_fmt(value):
    try:
        n = float(value)
    except Exception:
        return ""
    return f"{n:.3f}".replace(".", ",")


def norm_pos(value):
    value = str(value or "").strip().upper()
    return POS_DK.get(value, value)


def pos_sort(value):
    return POS_ORDER.get(norm_pos(value), 99)


def load_player_lookup():
    raw = load_json(PLAYER_POOL_FILE)
    players = raw.get("players", raw) if isinstance(raw, dict) else raw

    lookup = {}

    for p in players:
        if not isinstance(p, dict):
            continue

        keys = [
            str(p.get("player_id") or ""),
            str(p.get("holdet_player_id") or ""),
            f"{str(p.get('player_name') or '').strip().lower()}|{str(p.get('team_id') or '').strip().lower()}",
            f"{str(p.get('player_name') or '').strip().lower()}|{str(p.get('team_name') or '').strip().lower()}",
        ]

        for key in keys:
            if key:
                lookup[key] = p

    return lookup


PLAYER_LOOKUP = load_player_lookup()


def enrich(row):
    row = dict(row or {})

    keys = [
        str(row.get("player_id") or ""),
        str(row.get("holdet_player_id") or ""),
        f"{str(row.get('player_name') or '').strip().lower()}|{str(row.get('team_id') or '').strip().lower()}",
        f"{str(row.get('player_name') or '').strip().lower()}|{str(row.get('team_name') or '').strip().lower()}",
    ]

    base = {}
    for key in keys:
        if key and key in PLAYER_LOOKUP:
            base = PLAYER_LOOKUP[key]
            break

    merged = dict(base)
    merged.update(row)
    return merged


if not STRATEGY_FILE.exists():
    raise SystemExit(f"Mangler fil: {STRATEGY_FILE}")

strategy_data = load_json(STRATEGY_FILE)

rows = []

for strategy_key, strategy_entry in strategy_data.items():
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key)

    formations = strategy_entry.get("squads_by_formation") or {}

    for formation, formation_entry in formations.items():
        squad_rows = formation_entry.get("squad") or []

        if not squad_rows:
            continue

        summary = formation_entry.get("summary") or {}
        formation_score = (
            formation_entry.get("score")
            or formation_entry.get("strategy_score")
            or summary.get("score")
            or summary.get("strategy_score")
            or ""
        )

        enriched_rows = [enrich(r) for r in squad_rows]

        enriched_rows.sort(
            key=lambda p: (
                pos_sort(p.get("position")),
                str(p.get("player_name") or ""),
            )
        )

        for i, p in enumerate(enriched_rows, start=1):
            price = (
                p.get("price")
                or p.get("price_estimate")
                or p.get("holdet_price")
            )

            start = (
                p.get("start_probability_pct")
                or p.get("start_prob")
                or p.get("start_security")
            )

            ev = (
                p.get("strategy_score")
                or p.get("optimizer_ev")
                or p.get("weighted_group_stage_ev")
                or p.get("round1_ev")
                or ""
            )

            rows.append({
                "Strategi": strategy_name,
                "Strategi_key": strategy_key,
                "Formation": formation,
                "Formation_score": formation_score,
                "Nr": i,
                "Pos": norm_pos(p.get("position")),
                "Spiller": p.get("player_name") or "",
                "Land": p.get("team_name") or p.get("team_id") or "",
                "Pris": price_fmt(price),
                "Start": pct_fmt(start),
                "EV": ev_fmt(ev),
                "player_id": p.get("player_id") or "",
            })


rows.sort(
    key=lambda r: (
        list(STRATEGY_NAMES.values()).index(r["Strategi"])
        if r["Strategi"] in STRATEGY_NAMES.values()
        else 99,
        r["Formation"],
        pos_sort(r["Pos"]),
        r["Nr"],
    )
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUT / f"strategier_alle_formationer_direkte_{timestamp}.csv"
md_path = OUT / f"strategier_alle_formationer_direkte_{timestamp}.md"

fieldnames = [
    "Strategi",
    "Strategi_key",
    "Formation",
    "Formation_score",
    "Nr",
    "Pos",
    "Spiller",
    "Land",
    "Pris",
    "Start",
    "EV",
    "player_id",
]

with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Alle hold – alle strategier og formationer\n\n")
    current = None

    for r in rows:
        group = (r["Strategi"], r["Formation"])

        if group != current:
            current = group
            f.write(f"\n## {r['Strategi']} – {r['Formation']}\n\n")
            if r["Formation_score"] != "":
                f.write(f"Score: {r['Formation_score']}\n\n")
            f.write("| Pos | Spiller | Land | Pris | Start | EV |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")

        f.write(
            f"| {r['Pos']} | {r['Spiller']} | {r['Land']} | "
            f"{r['Pris']} | {r['Start']} | {r['EV']} |\n"
        )

print("")
print("FÆRDIG")
print("CSV:", csv_path)
print("Markdown:", md_path)
print("Antal spillerrækker:", len(rows))

print("")
print("OVERSIGT")
current = None
for r in rows:
    group = (r["Strategi"], r["Formation"])
    if group != current:
        current = group
        print(f"\n=== {r['Strategi']} – {r['Formation']} ===")
    print(f"{r['Pos']:>3} | {r['Spiller']:<24} | {r['Land']:<12} | {r['Pris']:>8} | Start {r['Start']:>4} | EV {r['EV']}")
