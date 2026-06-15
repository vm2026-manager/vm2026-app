import json
from pathlib import Path

pool = json.loads(Path("data/player_pool_v1.json").read_text(encoding="utf-8"))

names = [
    "Nathaniel Brown",
    "Nico Schlotterbeck",
    "Kai Havertz",
    "Jamal Musiala",
    "Florian Wirtz",
    "Gregor Kobel",
    "Roberto Alvarado",
    "Erling Haaland",
]

for name in names:
    rows = [p for p in pool if p.get("player_name") == name]
    if not rows:
        print(name, "IKKE FUNDET")
        continue
    p = rows[0]
    print(
        name,
        p.get("team_name"),
        p.get("position"),
        "price=", p.get("price"),
        "holdet_price=", p.get("holdet_price"),
        "price_estimate=", p.get("price_estimate"),
        "start_price=", p.get("holdet_start_price"),
    )
