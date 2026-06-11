import json
import csv
from pathlib import Path
from datetime import datetime

DATA = Path("data")
POOL = DATA / "player_pool_v1.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = DATA / f"player_pool_v1.backup_before_sync_new_players_match_context_{timestamp}.json"
backup.write_text(POOL.read_text(encoding="utf-8"), encoding="utf-8")
print("Backup:", backup)

raw = json.loads(POOL.read_text(encoding="utf-8"))
players = raw.get("players", raw) if isinstance(raw, dict) else raw

TARGETS = {
    "lutsharel_geertruida__ned": "NED",
    "ederson__bra": "BRA",
}

MATCH_PREFIXES = (
    "match_1_", "match_2_", "match_3_",
    "round1_", "round2_", "round3_",
)

def has_match_context(p):
    keys = p.keys()
    return any(k.startswith(MATCH_PREFIXES) for k in keys) and any(
        p.get(f"match_{i}_opponent") or
        p.get(f"match_{i}_opponent_team") or
        p.get(f"match_{i}_opponent_team_id") or
        p.get(f"round{i}_opponent") or
        p.get(f"round{i}_opponent_team")
        for i in [1, 2, 3]
    )

def donor_for_team(team_id, exclude_id):
    candidates = [
        p for p in players
        if str(p.get("team_id", "")).upper() == team_id
        and str(p.get("player_id", "")) != exclude_id
        and not p.get("holdet_is_out", False)
        and has_match_context(p)
    ]
    if not candidates:
        return None

    # Foretræk en spiller med flest match-felter
    candidates.sort(
        key=lambda p: sum(1 for k, v in p.items() if k.startswith(MATCH_PREFIXES) and v not in ("", None)),
        reverse=True
    )
    return candidates[0]

changed = []

for player_id, team_id in TARGETS.items():
    target = next((p for p in players if str(p.get("player_id")) == player_id), None)
    if not target:
        print("MANGLER target:", player_id)
        continue

    if has_match_context(target):
        print("Har allerede kampkontekst:", player_id)
        continue

    donor = donor_for_team(team_id, player_id)
    if not donor:
        print("INGEN donor fundet for:", player_id, team_id)
        continue

    copied = []
    for k, v in donor.items():
        if k.startswith(MATCH_PREFIXES):
            target[k] = v
            copied.append(k)

    changed.append((player_id, donor.get("player_id"), len(copied)))
    print(f"Synket {player_id} fra donor {donor.get('player_id')} ({len(copied)} felter)")

if isinstance(raw, dict) and "players" in raw:
    raw["players"] = players
    POOL.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    POOL.write_text(json.dumps(players, ensure_ascii=False, indent=2), encoding="utf-8")

print()
print("Ændringer:")
for row in changed:
    print(row)

if not changed:
    print("Ingen ændringer lavet.")
