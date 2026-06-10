import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

DATA = Path("data")
OUT = DATA / "strategy_squad_exports"
OUT.mkdir(exist_ok=True)

STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"

OUT_MD = OUT / "top_forward_package_swap_report.md"
OUT_CSV = OUT / "top_forward_package_swap_report.csv"

SKIP_STRATEGIES = {"round1_2"}

STRATEGY_NAMES = {
    "next_round": "Næste runde",
    "practical_start": "1. + 2. runde",
    "group_stage": "Gruppespil",
    "long_run": "Lang sigt",
}

def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def name(row):
    return str(row.get("player_name") or row.get("name") or "").strip()

def pos(row):
    return str(row.get("position") or row.get("holdet_position") or "").strip().upper()

def price(row):
    value = row.get("price") or row.get("price_estimate") or row.get("holdet_price") or 0
    try:
        n = float(str(value).replace(",", "."))
    except Exception:
        return 0.0
    if 0 < n < 1000:
        return n
    return n / 1_000_000

def start(row):
    for key in ["start_probability_pct", "start_prob", "start_security"]:
        value = row.get(key)
        if value not in [None, ""]:
            try:
                n = float(str(value).replace(",", "."))
                return n * 100 if n <= 1 else n
            except Exception:
                pass
    return 0.0

def ev(row):
    for key in ["strategy_score", "optimizer_ev", "weighted_group_stage_ev", "round1_ev", "display_score", "score"]:
        value = row.get(key)
        if value not in [None, ""]:
            try:
                return float(str(value).replace(",", "."))
            except Exception:
                pass
    return 0.0

data = load_json(STRATEGY_PATH)

rows = []
forward_counter = Counter()
forward_by_strategy = defaultdict(Counter)

md = []
md.append("# Topangriber-rapport")
md.append("")
md.append("Rapporten ændrer ikke data. Den viser kun, hvilke angribere der går igen i de nuværende strategi-hold.")
md.append("")

for strategy_key, strategy_entry in data.items():
    if strategy_key in SKIP_STRATEGIES:
        continue

    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key)
    md.append(f"## {strategy_name}")
    md.append("")

    formations = strategy_entry.get("squads_by_formation", {})

    for formation, formation_entry in formations.items():
        squad = formation_entry.get("squad", [])
        forwards = [p for p in squad if pos(p) == "FWD"]

        forward_names = [name(p) for p in forwards]
        budget = sum(price(p) for p in squad)

        for f in forwards:
            forward_counter[name(f)] += 1
            forward_by_strategy[strategy_name][name(f)] += 1

        rows.append({
            "strategy": strategy_name,
            "formation": formation,
            "budget_m": round(budget, 1),
            "forwards": ", ".join(forward_names),
        })

        md.append(f"**{formation}** — budget {str(round(budget, 1)).replace('.', ',')} mio.")
        md.append("")
        md.append("| Angriber | Pris | Start | EV |")
        md.append("|---|---:|---:|---:|")

        for f in forwards:
            md.append(
                f"| {name(f)} | {str(round(price(f), 1)).replace('.', ',')} mio. | "
                f"{round(start(f))}% | {str(round(ev(f), 3)).replace('.', ',')} |"
            )

        md.append("")

md.append("# Samlet angriber-frekvens")
md.append("")
md.append("| Spiller | Antal hold |")
md.append("|---|---:|")

for player, count in forward_counter.most_common():
    md.append(f"| {player} | {count} |")

md.append("")
md.append("# Angriber-frekvens pr. strategi")
md.append("")

for strategy_name, counter in forward_by_strategy.items():
    md.append(f"## {strategy_name}")
    md.append("")
    md.append("| Spiller | Antal formationer |")
    md.append("|---|---:|")
    for player, count in counter.most_common():
        md.append(f"| {player} | {count} |")
    md.append("")

OUT_MD.write_text("\n".join(md), encoding="utf-8")

with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["strategy", "formation", "budget_m", "forwards"], delimiter=";")
    writer.writeheader()
    writer.writerows(rows)

print("FÆRDIG")
print("Markdown:", OUT_MD)
print("CSV:", OUT_CSV)
print("Antal hold:", len(rows))
