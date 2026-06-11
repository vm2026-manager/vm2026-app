import json
import csv
from pathlib import Path
from collections import Counter

DATA = Path("data")

checks = []

def ok(label, value):
    checks.append((label, "OK", value))

def warn(label, value):
    checks.append((label, "TJEK", value))

# 1) Player pool
pool_path = DATA / "player_pool_v1.json"
pool_raw = json.loads(pool_path.read_text(encoding="utf-8"))
players = pool_raw.get("players", pool_raw) if isinstance(pool_raw, dict) else pool_raw

ok("player_pool rows", len(players))

giay = [
    p for p in players
    if isinstance(p, dict)
    and str(p.get("player_name", "")).lower() == "agustin giay"
]
if len(giay) == 1:
    p = giay[0]
    ok("Agustin Giay i player_pool", f"{p.get('team_id')} {p.get('position')} price={p.get('price')} holdet_is_out={p.get('holdet_is_out')} optimizer_ev={p.get('optimizer_ev')}")
else:
    warn("Agustin Giay i player_pool", f"antal={len(giay)}")

# 2) EV-fil
ev_path = DATA / "player_ev_group_stage_v1.csv"
with ev_path.open("r", encoding="utf-8-sig", newline="") as f:
    ev_rows = list(csv.DictReader(f))

ok("EV rows", len(ev_rows))

giay_ev = [
    r for r in ev_rows
    if str(r.get("player_name", "")).lower() == "agustin giay"
]
if len(giay_ev) == 1:
    r = giay_ev[0]
    ok("Agustin Giay i EV", f"{r.get('team_id')} {r.get('position')} holdet_is_out={r.get('holdet_is_out')} optimizer_ev={r.get('optimizer_ev')}")
else:
    warn("Agustin Giay i EV", f"antal={len(giay_ev)}")

# 3) Optimizer-output
opt_path = DATA / "optimal_squads_by_strategy.json"
opt_text = opt_path.read_text(encoding="utf-8")
if "Agustin Giay" in opt_text or "agustin_giay" in opt_text:
    warn("Agustin Giay i optimizer-output", "FUNDET - bør undersøges")
else:
    ok("Agustin Giay i optimizer-output", "ikke fundet")

# 4) Mbappé premium auditfiler
exports = DATA / "strategy_squad_exports"
mbappe_reports = sorted(exports.glob("mbappe_premium_long_run_report.*"))
if mbappe_reports:
    ok("Mbappé premium auditfiler", ", ".join(p.name for p in mbappe_reports))
else:
    warn("Mbappé premium auditfiler", "ikke fundet")

# 5) Nyeste direkte strategi-eksport
direct_exports = sorted(exports.glob("strategier_alle_formationer_direkte_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
if direct_exports:
    ok("Nyeste direkte strategy-export", direct_exports[0].name)
else:
    warn("Nyeste direkte strategy-export", "ikke fundet")

# 6) Nyeste Holdet-diff
diffs = sorted(DATA.glob("holdet_player_pool_diff_namepos_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
if diffs:
    latest = diffs[0]
    text = latest.read_text(encoding="utf-8")
    wanted = [
        "Reelt nye på navn+position: 0",
        "Reelt mangler på navn+position: 0",
        "Prisændringer: 0",
        "is_out-ændringer: 0",
    ]
    missing = [w for w in wanted if w not in text]
    if missing:
        warn("Nyeste Holdet-diff", f"{latest.name}; mangler forventet tekst: {missing}")
    else:
        ok("Nyeste Holdet-diff", latest.name)
else:
    warn("Nyeste Holdet-diff", "ikke fundet")

print("\n=== VM2026 simpel statuskontrol ===\n")
for label, status, value in checks:
    print(f"[{status}] {label}: {value}")

out = DATA / "simple_project_status_check.txt"
out.write_text(
    "\n".join(f"[{status}] {label}: {value}" for label, status, value in checks),
    encoding="utf-8"
)
print(f"\nSkrevet: {out}")
