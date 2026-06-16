import csv
import shutil
from pathlib import Path
from datetime import datetime

worldcup_path = Path("data/worldcup_outright_odds.csv")
manual_path = Path("data/team_market_odds_manual.csv")

# Manual-fil bruger ALG, mens worldcup/fixtures typisk bruger DZA for Algeria.
TEAM_ID_ALIASES = {
    "ALG": "DZA",
}

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = manual_path.with_name(f"team_market_odds_manual.backup_before_worldcup_sync_alias_{stamp}.csv")
shutil.copy2(manual_path, backup_path)

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

worldcup_rows = read_csv(worldcup_path)
manual_rows = read_csv(manual_path)

worldcup_by_team = {
    (r.get("team_id") or "").strip().upper(): r
    for r in worldcup_rows
    if (r.get("team_id") or "").strip()
}

changed = []
missing = []

for r in manual_rows:
    manual_team_id = (r.get("team_id") or "").strip().upper()
    lookup_team_id = TEAM_ID_ALIASES.get(manual_team_id, manual_team_id)

    w = worldcup_by_team.get(lookup_team_id)
    if not w:
        missing.append({
            "manual_team_id": manual_team_id,
            "lookup_team_id": lookup_team_id,
            "team_name": r.get("team_name") or "",
            "reason": "no worldcup row",
        })
        continue

    new_odds = (w.get("model_win_odds") or "").strip()
    if not new_odds:
        missing.append({
            "manual_team_id": manual_team_id,
            "lookup_team_id": lookup_team_id,
            "team_name": r.get("team_name") or "",
            "reason": "missing model_win_odds",
        })
        continue

    old_odds = (r.get("winner_odds") or "").strip()
    if old_odds != new_odds:
        changed.append({
            "manual_team_id": manual_team_id,
            "lookup_team_id": lookup_team_id,
            "team_name": r.get("team_name") or w.get("team_name") or "",
            "old_winner_odds": old_odds,
            "new_winner_odds": new_odds,
        })
        r["winner_odds"] = new_odds

fieldnames = list(manual_rows[0].keys())
with manual_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manual_rows)

audit_path = Path(f"data/team_market_odds_manual_worldcup_sync_audit_{stamp}.csv")
with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames_audit = ["manual_team_id", "lookup_team_id", "team_name", "old_winner_odds", "new_winner_odds"]
    writer = csv.DictWriter(f, fieldnames=fieldnames_audit)
    writer.writeheader()
    writer.writerows(changed)

print("Backup:", backup_path)
print("Audit:", audit_path)
print("Manual rows:", len(manual_rows))
print("Worldcup rows:", len(worldcup_rows))
print("Changed winner_odds:", len(changed))
print("Missing/problem rows:", len(missing))

if missing:
    print("Missing:")
    for x in missing:
        print(x)

print("")
print("Stikprøver efter sync:")
for t in ["ESP", "FRA", "ENG", "POR", "BRA", "ARG", "GER", "NOR", "ALG"]:
    manual = next((r for r in manual_rows if (r.get("team_id") or "").strip().upper() == t), {})
    lookup = TEAM_ID_ALIASES.get(t, t)
    worldcup = worldcup_by_team.get(lookup, {})
    print(t, "manual=", manual.get("winner_odds"), "worldcup_lookup=", lookup, "worldcup=", worldcup.get("model_win_odds"))
