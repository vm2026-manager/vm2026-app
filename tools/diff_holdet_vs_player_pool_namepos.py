import json
import csv
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DATA = Path("data")
HOLDET = DATA / "holdet_players_game_616_flat.csv"
POOL = DATA / "player_pool_v1.json"

def clean(s):
    s = str(s or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().replace("ø", "o").replace("æ", "ae").replace("å", "aa")

def val(row, names):
    for n in names:
        if n in row and str(row.get(n, "")).strip():
            return row.get(n)
    return ""

def pos(row):
    return str(val(row, ["position", "pos", "providerPositionId"])).strip().upper()

def name(row):
    return str(val(row, ["player_name", "name", "full_name"])).strip()

def team_any(row):
    return str(val(row, ["team_id", "team", "team_name", "country", "national_team"])).strip()

def price_mio(row):
    for col in ["price_millions", "price_mio", "price", "value", "current_value"]:
        if col in row and str(row.get(col, "")).strip():
            try:
                x = float(str(row[col]).replace(",", "."))
                return round(x / 1000000, 2) if x > 100000 else round(x, 2)
            except:
                pass
    return None

def is_out(row):
    return str(val(row, ["is_out", "holdet_is_out", "unavailable"])).strip().lower() in ["true", "1", "yes", "ja"]

def holdet_id(row):
    return str(val(row, ["holdet_id", "player_id", "id", "externalId", "external_id", "person_id"])).strip()

def pool_holdet_id(row):
    return str(val(row, ["holdet_id", "holdet_player_id", "externalId", "external_id", "provider_player_id"])).strip()

with HOLDET.open("r", encoding="utf-8-sig", newline="") as f:
    holdet_rows = list(csv.DictReader(f))

raw = json.loads(POOL.read_text(encoding="utf-8"))
pool_rows = raw.get("players", raw) if isinstance(raw, dict) else raw
pool_rows = [r for r in pool_rows if isinstance(r, dict)]

print("Holdet kolonner:")
print(list(holdet_rows[0].keys()))
print()
print("Pool eksempelkolonner:")
print(list(pool_rows[0].keys())[:80])
print()

print("Holdet team-eksempler:", sorted(set(team_any(r) for r in holdet_rows))[:30])
print("Pool team-eksempler:", sorted(set(team_any(r) for r in pool_rows))[:30])
print()

# Match først på rent navn+position for at se om diffen reelt er lille
holdet_np = defaultdict(list)
pool_np = defaultdict(list)

for r in holdet_rows:
    holdet_np[(clean(name(r)), pos(r))].append(r)

for r in pool_rows:
    pool_np[(clean(name(r)), pos(r))].append(r)

common_np = set(holdet_np) & set(pool_np)
new_np = set(holdet_np) - set(pool_np)
missing_np = set(pool_np) - set(holdet_np)

print("Match på navn+position:")
print("Fælles:", len(common_np))
print("Nye:", len(new_np))
print("Mangler:", len(missing_np))
print()

# Reelle nye/mangler ud fra navn+position
new_rows = [holdet_np[k][0] for k in sorted(new_np)]
missing_rows = [pool_np[k][0] for k in sorted(missing_np)]

# For fælles navn+position: find pris/is_out-ændringer
price_changes = []
out_changes = []
team_text_diffs = []

for k in sorted(common_np):
    h = holdet_np[k][0]
    p = pool_np[k][0]

    hp = price_mio(h)
    pp = price_mio(p)
    if hp is not None and pp is not None and abs(hp - pp) >= 0.01:
        price_changes.append({
            "player_name": name(h),
            "position": pos(h),
            "pool_team": team_any(p),
            "holdet_team": team_any(h),
            "pool_price": pp,
            "holdet_price": hp,
            "diff": round(hp - pp, 2),
        })

    ho = is_out(h)
    po = is_out(p)
    if ho != po:
        out_changes.append({
            "player_name": name(h),
            "position": pos(h),
            "pool_team": team_any(p),
            "holdet_team": team_any(h),
            "pool_is_out": po,
            "holdet_is_out": ho,
            "pool_price": pp,
            "holdet_price": hp,
        })

    if clean(team_any(h)) != clean(team_any(p)):
        team_text_diffs.append({
            "player_name": name(h),
            "position": pos(h),
            "pool_team": team_any(p),
            "holdet_team": team_any(h),
        })

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_md = DATA / f"holdet_player_pool_diff_namepos_{timestamp}.md"
out_csv = DATA / f"holdet_player_pool_diff_namepos_{timestamp}.csv"

def table(rows, cols, max_rows=120):
    if not rows:
        return "_Ingen._\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows[:max_rows]:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Viser {max_rows} af {len(rows)}._")
    return "\n".join(lines) + "\n"

new_show = [{
    "player_name": name(r),
    "team": team_any(r),
    "position": pos(r),
    "holdet_price": price_mio(r),
    "holdet_is_out": is_out(r),
} for r in new_rows]

missing_show = [{
    "player_name": name(r),
    "team": team_any(r),
    "position": pos(r),
    "pool_price": price_mio(r),
    "pool_is_out": is_out(r),
} for r in missing_rows]

md = []
md.append("# Holdet.dk diff mod player_pool – navn+position")
md.append("")
md.append(f"- Holdet-rækker: {len(holdet_rows)}")
md.append(f"- Player pool-rækker: {len(pool_rows)}")
md.append(f"- Fælles på navn+position: {len(common_np)}")
md.append(f"- Reelt nye på navn+position: {len(new_rows)}")
md.append(f"- Reelt mangler på navn+position: {len(missing_rows)}")
md.append(f"- Prisændringer: {len(price_changes)}")
md.append(f"- is_out-ændringer: {len(out_changes)}")
md.append(f"- Teamtekst-forskelle blandt fælles: {len(team_text_diffs)}")
md.append("")
md.append("## Reelt nye i Holdet")
md.append(table(new_show, ["player_name", "team", "position", "holdet_price", "holdet_is_out"]))
md.append("## Reelt mangler fra Holdet")
md.append(table(missing_show, ["player_name", "team", "position", "pool_price", "pool_is_out"]))
md.append("## Prisændringer")
md.append(table(price_changes, ["player_name", "position", "pool_team", "holdet_team", "pool_price", "holdet_price", "diff"]))
md.append("## is_out-ændringer")
md.append(table(out_changes, ["player_name", "position", "pool_team", "holdet_team", "pool_is_out", "holdet_is_out", "pool_price", "holdet_price"]))
md.append("## Teamtekst-forskelle, stikprøve")
md.append(table(team_text_diffs, ["player_name", "position", "pool_team", "holdet_team"], max_rows=40))

out_md.write_text("\n".join(md), encoding="utf-8")

with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    fields = ["type", "player_name", "team", "position", "pool_team", "holdet_team", "pool_price", "holdet_price", "diff", "pool_is_out", "holdet_is_out"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in new_show:
        w.writerow({"type": "new_in_holdet", **r})
    for r in missing_show:
        w.writerow({"type": "missing_from_holdet", **r})
    for r in price_changes:
        w.writerow({"type": "price_change", **r})
    for r in out_changes:
        w.writerow({"type": "is_out_change", **r})

print()
print("Skrevet:")
print(out_md)
print(out_csv)
