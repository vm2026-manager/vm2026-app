import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA = Path("data")
STRATEGY_PATH = DATA / "optimal_squads_by_strategy.json"
POOL_PATH = DATA / "player_pool_v1.json"

raw = json.loads(STRATEGY_PATH.read_text(encoding="utf-8"))
pool_raw = json.loads(POOL_PATH.read_text(encoding="utf-8"))
players = pool_raw.get("players", pool_raw) if isinstance(pool_raw, dict) else pool_raw

by_id = {str(p.get("player_id")): p for p in players if isinstance(p, dict)}
mbappe_ids = [pid for pid, p in by_id.items() if "mbapp" in str(p.get("player_name", "")).lower()]

print("Mbappé ids:", mbappe_ids)

def player_name(pid):
    return by_id.get(str(pid), {}).get("player_name", str(pid))

def player_price(pid):
    p = by_id.get(str(pid), {})
    return float(p.get("price") or p.get("holdet_price") or 0) / 1_000_000

def player_pos(pid):
    return by_id.get(str(pid), {}).get("position", "")

def extract_player_ids(obj):
    ids = []
    if isinstance(obj, dict):
        for key in ["player_id", "id"]:
            if key in obj and str(obj[key]) in by_id:
                ids.append(str(obj[key]))
        for v in obj.values():
            ids += extract_player_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            ids += extract_player_ids(item)
    return ids

rows = []

def walk(node, path=()):
    if isinstance(node, dict):
        ids = extract_player_ids(node)
        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) >= 11:
            strategy = path[0] if len(path) > 0 else ""
            formation = path[1] if len(path) > 1 else ""
            names = [player_name(pid) for pid in unique_ids]
            has_mbappe = any(pid in mbappe_ids for pid in unique_ids)
            fwd_names = [player_name(pid) for pid in unique_ids if player_pos(pid) == "FWD"]
            price = sum(player_price(pid) for pid in unique_ids)
            score = node.get("score") or node.get("total_score") or node.get("strategy_score") or node.get("ev") or ""
            rows.append({
                "strategy": strategy,
                "formation": formation,
                "has_mbappe": has_mbappe,
                "price": round(price, 1),
                "score": score,
                "fwds": ", ".join(fwd_names),
                "players": ", ".join(names),
            })
        for k, v in node.items():
            walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            walk(item, path + (str(i),))

walk(raw)

out = DATA / "strategy_squad_exports" / f"mbappe_presence_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
out.parent.mkdir(parents=True, exist_ok=True)

lines = ["# Mbappé presence audit", ""]
lines.append(f"Mbappé ids: `{', '.join(mbappe_ids)}`")
lines.append("")
lines.append(f"Fundne hold: {len(rows)}")
lines.append(f"Hold med Mbappé: {sum(1 for r in rows if r['has_mbappe'])}")
lines.append("")
lines.append("| Strategi | Formation | Mbappé | Pris | Score | Angribere |")
lines.append("|---|---|---:|---:|---:|---|")
for r in rows:
    lines.append(f"| {r['strategy']} | {r['formation']} | {'JA' if r['has_mbappe'] else 'NEJ'} | {r['price']} | {r['score']} | {r['fwds']} |")

out.write_text("\n".join(lines), encoding="utf-8")

print("Skrevet:", out)
print("Hold med Mbappé:", sum(1 for r in rows if r["has_mbappe"]))
