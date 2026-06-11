import json
import csv
from pathlib import Path
from datetime import datetime

DATA = Path("data")
HOLDET = DATA / "holdet_players_game_616_flat.csv"
POOL = DATA / "player_pool_v1.json"

def norm(s):
    return str(s or "").strip().lower()

def key_name_team_pos(row):
    name = norm(row.get("player_name") or row.get("name"))
    team = norm(row.get("team_id") or row.get("team") or row.get("team_name"))
    pos = norm(row.get("position"))
    return (name, team, pos)

def to_price_mio(row):
    for col in ["price_millions", "price_mio", "price", "value", "current_value"]:
        if col in row and str(row.get(col, "")).strip() != "":
            v = str(row[col]).replace(",", ".").strip()
            try:
                x = float(v)
                # Holdet flat kan være i hele kroner, fx 3000000
                if x > 100000:
                    return round(x / 1000000, 2)
                return round(x, 2)
            except:
                pass
    return None

def to_bool(v):
    return str(v).strip().lower() in ["true", "1", "yes", "ja"]

# Load Holdet flat
with HOLDET.open("r", encoding="utf-8-sig", newline="") as f:
    holdet_rows = list(csv.DictReader(f))

# Load player pool
raw = json.loads(POOL.read_text(encoding="utf-8"))
pool_rows = raw.get("players", raw) if isinstance(raw, dict) else raw

holdet_by_key = {}
for r in holdet_rows:
    holdet_by_key[key_name_team_pos(r)] = r

pool_by_key = {}
for r in pool_rows:
    if isinstance(r, dict):
        pool_by_key[key_name_team_pos(r)] = r

new_in_holdet = []
missing_from_holdet = []
price_changes = []
position_or_team_name_issues = []
is_out_changes = []

for k, h in holdet_by_key.items():
    if k not in pool_by_key:
        new_in_holdet.append(h)

for k, p in pool_by_key.items():
    if k not in holdet_by_key:
        missing_from_holdet.append(p)

for k, h in holdet_by_key.items():
    p = pool_by_key.get(k)
    if not p:
        continue

    hp = to_price_mio(h)
    pp = to_price_mio(p)
    if hp is not None and pp is not None and abs(hp - pp) >= 0.01:
        price_changes.append({
            "player_name": h.get("player_name") or h.get("name"),
            "team": h.get("team_name") or h.get("team_id"),
            "position": h.get("position"),
            "pool_price": pp,
            "holdet_price": hp,
            "diff": round(hp - pp, 2),
        })

    h_out = to_bool(h.get("is_out"))
    p_out = to_bool(p.get("is_out") or p.get("holdet_is_out"))
    if h_out != p_out:
        is_out_changes.append({
            "player_name": h.get("player_name") or h.get("name"),
            "team": h.get("team_name") or h.get("team_id"),
            "position": h.get("position"),
            "pool_is_out": p_out,
            "holdet_is_out": h_out,
            "pool_price": pp,
            "holdet_price": hp,
        })

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = DATA / f"holdet_player_pool_diff_{timestamp}.csv"
out_md = DATA / f"holdet_player_pool_diff_{timestamp}.md"

with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["type","player_name","team","position","pool_price","holdet_price","diff","pool_is_out","holdet_is_out"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    for r in new_in_holdet:
        w.writerow({
            "type": "new_in_holdet",
            "player_name": r.get("player_name") or r.get("name"),
            "team": r.get("team_name") or r.get("team_id"),
            "position": r.get("position"),
            "holdet_price": to_price_mio(r),
            "holdet_is_out": to_bool(r.get("is_out")),
        })

    for r in missing_from_holdet:
        w.writerow({
            "type": "missing_from_holdet",
            "player_name": r.get("player_name") or r.get("name"),
            "team": r.get("team_name") or r.get("team_id"),
            "position": r.get("position"),
            "pool_price": to_price_mio(r),
            "pool_is_out": to_bool(r.get("is_out") or r.get("holdet_is_out")),
        })

    for r in price_changes:
        r["type"] = "price_change"
        w.writerow(r)

    for r in is_out_changes:
        r["type"] = "is_out_change"
        w.writerow(r)

def table(rows, cols, max_rows=80):
    if not rows:
        return "_Ingen._\n"
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows[:max_rows]:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Viser {max_rows} af {len(rows)}._")
    return "\n".join(lines) + "\n"

md = []
md.append("# Holdet.dk diff mod player_pool\n")
md.append(f"- Holdet-rækker: {len(holdet_rows)}")
md.append(f"- Player pool-rækker: {len(pool_rows)}")
md.append(f"- Nye i Holdet: {len(new_in_holdet)}")
md.append(f"- Mangler fra Holdet: {len(missing_from_holdet)}")
md.append(f"- Prisændringer: {len(price_changes)}")
md.append(f"- is_out-ændringer: {len(is_out_changes)}")
md.append("")
md.append("## Nye i Holdet")
md.append(table(
    [{"player_name": r.get("player_name") or r.get("name"), "team": r.get("team_name") or r.get("team_id"), "position": r.get("position"), "holdet_price": to_price_mio(r), "holdet_is_out": to_bool(r.get("is_out"))} for r in new_in_holdet],
    ["player_name","team","position","holdet_price","holdet_is_out"]
))
md.append("## Mangler fra Holdet")
md.append(table(
    [{"player_name": r.get("player_name") or r.get("name"), "team": r.get("team_name") or r.get("team_id"), "position": r.get("position"), "pool_price": to_price_mio(r), "pool_is_out": to_bool(r.get("is_out") or r.get("holdet_is_out"))} for r in missing_from_holdet],
    ["player_name","team","position","pool_price","pool_is_out"]
))
md.append("## Prisændringer")
md.append(table(price_changes, ["player_name","team","position","pool_price","holdet_price","diff"]))
md.append("## is_out-ændringer")
md.append(table(is_out_changes, ["player_name","team","position","pool_is_out","holdet_is_out","pool_price","holdet_price"]))

out_md.write_text("\n".join(md), encoding="utf-8")

print("Diff skrevet:")
print(out_md)
print(out_csv)
print()
print("Kort status:")
print("Nye i Holdet:", len(new_in_holdet))
print("Mangler fra Holdet:", len(missing_from_holdet))
print("Prisændringer:", len(price_changes))
print("is_out-ændringer:", len(is_out_changes))
