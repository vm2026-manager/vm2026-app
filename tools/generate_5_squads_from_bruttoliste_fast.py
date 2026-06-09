from pathlib import Path
import pandas as pd
import json
import re
import unicodedata
import math
from collections import Counter

BRUTTO_PATH = Path("data/bruttoliste_med_score_og_start_filled.csv")
if not BRUTTO_PATH.exists():
    BRUTTO_PATH = Path("data/bruttoliste_med_score_og_start.csv")

POOL_PATH = Path("data/player_pool_v1.json")
EV_PATH = Path("data/player_ev_group_stage_v1.csv")

OUT_MD = Path("data/generated_5_squads_from_bruttoliste.md")
OUT_CSV = Path("data/generated_5_squads_from_bruttoliste.csv")
OUT_JSON = Path("data/generated_5_squads_from_bruttoliste.json")

BUDGET = 50.0
MAX_PER_TEAM = 4
BEAM_WIDTH = 6000

EXCLUDE_NAMES = [
    "Messi",
    "Lionel Messi",
    "Alvarez",
    "Álvarez",
    "Julián Álvarez",
    "Julian Alvarez",
    "Bruno Fernandes",
    "Lamine Yamal",
]

FORMATIONS = [
    ("3-4-3", {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}),
    ("4-3-3", {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}),
]

OBJECTIVES = [
    ("Hold 1 - Første runde", "round1", 0.0),
    ("Hold 2 - Hele vejen", "longrun", 1.5),
    ("Hold 3 - Balanceret", "balanced", 3.0),
    ("Hold 4 - Value", "value", 4.0),
    ("Hold 5 - Alternativ", "alt", 6.0),
]

def norm(s):
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = s.replace("ø", "o").replace("æ", "ae").replace("å", "a")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())

def to_num(x):
    s = str(x or "").strip().replace(",", ".").replace("%", "")
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return 0.0
    try:
        n = float(s)
        return 0.0 if math.isnan(n) else n
    except Exception:
        return 0.0

def price_to_float(x):
    s = str(x or "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0

def pct_to_float(x):
    n = to_num(x)
    if n > 1:
        n /= 100
    return max(0.0, min(1.0, n))

def fmt(x, d=1):
    return f"{float(x):.{d}f}".replace(".", ",")

def minmax(vals):
    vals = list(vals)
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0 for _ in vals]
    return [(v - lo) / (hi - lo) * 100 for v in vals]

def pos_to_code(x):
    k = norm(x)
    raw = str(x or "").strip().lower()

    if k.startswith("midtbane"):
        return "MID"
    if k.startswith("forsvar"):
        return "DEF"
    if k.startswith("angriber"):
        return "FWD"

    # Håndter Mål, M?l, MÃ¥l, Maal osv.
    if k in ["mal", "m l", "ml", "maal"] or raw.startswith("m?") or raw.startswith("mÃ"):
        return "GK"

    return None

def markdown_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)

brutto = pd.read_csv(BRUTTO_PATH, dtype=str).fillna("")
pool = pd.DataFrame(json.loads(POOL_PATH.read_text(encoding="utf-8")))
ev = pd.read_csv(EV_PATH, dtype=str).fillna("")

brutto["name_key"] = brutto["Spiller"].map(norm)
brutto["team_key"] = brutto["Land"].map(norm)
pool["name_key"] = pool["player_name"].map(norm)
pool["team_key"] = pool["team_name"].map(norm)
ev["name_key"] = ev["player_name"].map(norm)
ev["team_key"] = ev["team_name"].map(norm)

aliases = {
    "andrew robertson": ["andy robertson", "andrew robertson"],
    "nmecha": ["felix nmecha", "nmecha"],
}

def aliases_for(name_key):
    return list(dict.fromkeys([norm(x) for x in ([name_key] + aliases.get(name_key, [])) if norm(x)]))

def find_one(df, name_key, team_key):
    for a in aliases_for(name_key):
        m = df[(df["name_key"] == a) & (df["team_key"] == team_key)]
        if len(m) == 1:
            return m.iloc[0]
    for a in aliases_for(name_key):
        m = df[df["name_key"] == a]
        if len(m) == 1:
            return m.iloc[0]
    return None

exclude_keys = [norm(x) for x in EXCLUDE_NAMES]
players = []

for _, b in brutto.iterrows():
    nk = b["name_key"]
    tk = b["team_key"]

    if any(ex and ex in nk for ex in exclude_keys):
        continue

    p = find_one(pool, nk, tk)
    e = find_one(ev, nk, tk)

    if p is None:
        continue

    if str(p.get("is_out", p.get("holdet_is_out", ""))).lower() == "true":
        continue

    pos = pos_to_code(b["Position"])
    if not pos:
        continue

    price = price_to_float(b["Pris"])
    if price <= 0:
        continue

    score = to_num(p.get("display_score", b.get("Score", "")))
    start = pct_to_float(b.get("Start", p.get("start_prob", "")))

    if e is not None:
        r1_ev = to_num(e.get("match_1_weighted_match_ev", e.get("match_1_total_ev_next_match", "")))
        group_ev = to_num(e.get("weighted_group_stage_ev", e.get("optimizer_ev", "")))
        price_quality = to_num(e.get("price_quality_ev", ""))
    else:
        r1_ev = group_ev = price_quality = 0.0

    players.append({
        "name": str(b["Spiller"]),
        "model_name": str(p.get("player_name", b["Spiller"])),
        "team": str(p.get("team_name", b["Land"])),
        "pos": pos,
        "price": price,
        "score": score,
        "start": start,
        "r1_ev": r1_ev,
        "group_ev": group_ev,
        "price_quality": price_quality,
    })

if not players:
    raise RuntimeError("Ingen spillere fundet.")

for metric in ["score", "r1_ev", "group_ev", "price_quality"]:
    scaled = minmax([p[metric] for p in players])
    for p, s in zip(players, scaled):
        p["n_" + metric] = s

for p in players:
    p["n_start"] = p["start"] * 100

def obj_value(p, mode):
    if mode == "round1":
        return 0.50*p["n_r1_ev"] + 0.25*p["n_score"] + 0.15*p["n_start"] + 0.10*p["n_price_quality"]
    if mode == "longrun":
        return 0.40*p["n_score"] + 0.35*p["n_group_ev"] + 0.15*p["n_start"] + 0.10*p["n_price_quality"]
    if mode == "balanced":
        return 0.30*p["n_r1_ev"] + 0.30*p["n_group_ev"] + 0.25*p["n_score"] + 0.15*p["n_start"]
    if mode == "value":
        return 0.40*p["n_price_quality"] + 0.25*p["n_score"] + 0.20*p["n_r1_ev"] + 0.15*p["n_start"]
    return 0.25*p["n_r1_ev"] + 0.25*p["n_group_ev"] + 0.25*p["n_score"] + 0.25*p["n_start"]

for p in players:
    for _, mode, _ in OBJECTIVES:
        p["obj_" + mode] = obj_value(p, mode)

by_pos = {pos: [p for p in players if p["pos"] == pos] for pos in ["GK", "DEF", "MID", "FWD"]}

print("Kandidatantal pr. position:")
for pos in ["GK", "DEF", "MID", "FWD"]:
    print(pos, len(by_pos[pos]))

if any(len(by_pos[pos]) == 0 for pos in ["GK", "DEF", "MID", "FWD"]):
    raise RuntimeError("Mindst én position har 0 kandidater. Stopper.")

def cheapest_remaining_cost(needs, counts):
    cost = 0.0
    for pos, need in needs.items():
        rem = need - counts.get(pos, 0)
        if rem <= 0:
            continue
        cheapest = sorted(by_pos[pos], key=lambda x: x["price"])[:rem]
        if len(cheapest) < rem:
            return 9999
        cost += sum(p["price"] for p in cheapest)
    return cost

def build_for_formation(mode, used_counts, diversity_penalty, formation_name, needs):
    slot_order = []
    # GK først, derefter dyre/knappe positioner
    for pos in ["GK", "FWD", "DEF", "MID"]:
        slot_order += [pos] * needs[pos]

    beam = [{
        "players": [],
        "price": 0.0,
        "value": 0.0,
        "names": set(),
        "teams": Counter(),
        "counts": Counter(),
    }]

    for slot_pos in slot_order:
        new_beam = []
        candidates = sorted(
            by_pos[slot_pos],
            key=lambda p: p["obj_" + mode] - used_counts[p["model_name"]] * diversity_penalty,
            reverse=True
        )

        for state in beam:
            for p in candidates:
                if p["model_name"] in state["names"]:
                    continue
                if state["teams"][p["team"]] >= MAX_PER_TEAM:
                    continue
                if state["counts"][p["pos"]] >= needs[p["pos"]]:
                    continue

                new_price = state["price"] + p["price"]
                if new_price > BUDGET:
                    continue

                new_counts = Counter(state["counts"])
                new_counts[p["pos"]] += 1

                # Simpelt budget-prune: der skal være råd til billigste resterende spillere.
                if new_price + cheapest_remaining_cost(needs, new_counts) > BUDGET:
                    continue

                add_value = p["obj_" + mode]
                add_value -= used_counts[p["model_name"]] * diversity_penalty

                new_state = {
                    "players": state["players"] + [p],
                    "price": new_price,
                    "value": state["value"] + add_value,
                    "names": set(state["names"]) | {p["model_name"]},
                    "teams": Counter(state["teams"]),
                    "counts": new_counts,
                }
                new_state["teams"][p["team"]] += 1

                # Lille bonus for at bruge budget, men ikke så meget at det ødelægger holdet.
                rank_value = new_state["value"] + new_state["price"] * 0.5
                new_state["rank_value"] = rank_value

                new_beam.append(new_state)

        if not new_beam:
            return None

        new_beam.sort(key=lambda s: s["rank_value"], reverse=True)
        beam = new_beam[:BEAM_WIDTH]

    finals = []
    for s in beam:
        if len(s["players"]) != 11:
            continue
        if s["price"] > BUDGET:
            continue
        if any(v > MAX_PER_TEAM for v in s["teams"].values()):
            continue

        # Straf ubrugt budget.
        s["final_value"] = s["value"] - (BUDGET - s["price"]) * 2.0
        if formation_name == "3-4-3":
            s["final_value"] += 2.0
        finals.append(s)

    if not finals:
        return None

    finals.sort(key=lambda s: s["final_value"], reverse=True)
    best = finals[0]

    return {
        "formation": formation_name,
        "players": best["players"],
        "price": best["price"],
        "value": best["final_value"],
    }

def build_squad(mode, used_counts, diversity_penalty):
    options = []

    for formation_name, needs in FORMATIONS:
        s = build_for_formation(mode, used_counts, diversity_penalty, formation_name, needs)
        if s is not None:
            options.append(s)

    if not options:
        raise RuntimeError(f"Kunne ikke bygge hold for {mode}")

    options.sort(key=lambda x: x["value"], reverse=True)
    return options[0]

used_counts = Counter()
squads = []

for name, mode, diversity_penalty in OBJECTIVES:
    s = build_squad(mode, used_counts, diversity_penalty)
    used_counts.update(p["model_name"] for p in s["players"])
    squads.append({
        "name": name,
        "mode": mode,
        "formation": s["formation"],
        "players": s["players"],
        "price": s["price"],
    })

pos_da = {"GK": "Mål", "DEF": "Forsvar", "MID": "Midtbane", "FWD": "Angriber"}
pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

all_rows = []
md = []

for s in squads:
    md.append(f"## {s['name']}")
    md.append("")
    md.append(f"Formation: **{s['formation']}**  ")
    md.append(f"Pris: **{fmt(s['price'])} mio.**")
    md.append("")

    rows = []
    for p in sorted(s["players"], key=lambda x: pos_order[x["pos"]]):
        row = {
            "Spiller": p["name"],
            "Land": p["team"],
            "Position": pos_da[p["pos"]],
            "Pris": fmt(p["price"]),
            "Score": fmt(p["score"]),
            "Start": f"{p['start']*100:.0f}%",
            "R1_EV": fmt(p["r1_ev"], 2),
            "Gruppe_EV": fmt(p["group_ev"], 2),
        }
        rows.append(row)
        all_rows.append({"Hold": s["name"], "Formation": s["formation"], **row})

    md.append(markdown_table(rows, ["Spiller", "Land", "Position", "Pris", "Score", "Start", "R1_EV", "Gruppe_EV"]))
    md.append("")

pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
OUT_MD.write_text("\n".join(md), encoding="utf-8")
OUT_JSON.write_text(json.dumps(squds if False else squads, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print()
print("OK: 5 hold genereret fra bruttolisten.")
print("Kandidater brugt:", len(players))
print("Formationer: 3-4-3 eller 4-3-3")
print("Ekskluderet:", ", ".join(EXCLUDE_NAMES))
print("Skrevet:")
print(OUT_MD)
print(OUT_CSV)
print(OUT_JSON)
print()

for s in squads:
    print(f"{s['name']} | {s['formation']} | {s['price']:.1f} mio.")
    for p in sorted(s["players"], key=lambda x: pos_order[x["pos"]]):
        print(f"  {p['pos']:3} {p['name']:<24} {p['team']:<10} {p['price']:.1f} | score {p['score']:.1f} | start {p['start']*100:.0f}%")
    print()
