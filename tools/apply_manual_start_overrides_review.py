from __future__ import annotations

import json
import unicodedata
import shutil
from datetime import datetime
from pathlib import Path

from json_file_safety import write_json_strict

DATA = Path("data")
POOL_PATH = DATA / "player_pool_v1.json"

OVERRIDES = [
    {
        "name": "Julian Ryerson",
        "team_id": "NOR",
        "start_prob": 0.88,
        "conditional_start_prob": 0.93,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: secure Norway starter; frequent 90-minute national-team usage; round 1 vs Iraq is strong value; offensive/assist upside from fullback role.",
    },
    {
        "name": "Konrad Laimer",
        "team_id": "AUT",
        "start_prob": 0.86,
        "conditional_start_prob": 0.91,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: looks like secure Austria starter; raised after Transfermarkt usage review and expert-start audit.",
    },
    {
        "name": "John McGinn",
        "team_id": "SCO",
        "start_prob": 0.88,
        "conditional_start_prob": 0.93,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: secure Scotland starter; raised after Transfermarkt usage review and expert-start audit.",
    },
    {
        "name": "Marko Arnautovic",
        "team_id": "AUT",
        "start_prob": 0.82,
        "conditional_start_prob": 0.88,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "medium",
        "note": "Manual review: expected Austria starter and strong round 1 value vs Jordan; some rotation/minute risk retained.",
    },
    {
        "name": "Philipp Lienhart",
        "team_id": "AUT",
        "start_prob": 0.84,
        "conditional_start_prob": 0.90,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: looks like Austria starter; raised after Transfermarkt usage review and expert-start audit.",
    },
    {
        "name": "Stefan Posch",
        "team_id": "AUT",
        "start_prob": 0.84,
        "conditional_start_prob": 0.90,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: looks like fairly secure Austria starter; added value from notable national-team goal/assist output.",
    },
    {
        "name": "Arda Guler",
        "team_id": "TUR",
        "start_prob": 0.88,
        "conditional_start_prob": 0.93,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: should be treated as secure Turkey starter after Transfermarkt usage review.",
    },
    {
        "name": "Nuno Mendes",
        "team_id": "POR",
        "start_prob": 0.88,
        "conditional_start_prob": 0.93,
        "availability_risk": "low_risk",
        "round_specific_rotation_risk": "low",
        "note": "Manual review: should be treated as secure Portugal starter; June friendly minutes likely affected by post-Champions-League-final management rather than real start uncertainty.",
    },
]

WATCH_ONLY = [
    {
        "name": "David Moller Wolfe",
        "team_id": "NOR",
        "note": "Manual review: usage is more ambiguous; do not raise blindly. Keep as watch/manual-check unless later expert-start data confirms.",
    }
]


def norm(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().replace("ø", "o").replace("ö", "o").replace("ü", "u").replace("æ", "ae").replace("å", "a").strip()


def find_player(players: list[dict], name: str, team_id: str) -> dict | None:
    target_name = norm(name)
    target_team = norm(team_id)
    matches = [
        p for p in players
        if norm(p.get("player_name")) == target_name and norm(p.get("team_id")) == target_team
    ]
    if matches:
        return matches[0]

    # fallback: partial name match, still constrained by team
    matches = [
        p for p in players
        if target_name in norm(p.get("player_name")) and norm(p.get("team_id")) == target_team
    ]
    return matches[0] if matches else None


def main() -> None:
    if not POOL_PATH.exists():
        raise FileNotFoundError(POOL_PATH)

    players = json.loads(POOL_PATH.read_text(encoding="utf-8"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = POOL_PATH.with_suffix(f".backup_before_manual_start_overrides_{stamp}.json")
    shutil.copy2(POOL_PATH, backup)

    print("Manual start overrides")
    print("----------------------")
    print("Backup:", backup)
    print()

    changed = []
    missing = []

    for item in OVERRIDES:
        player = find_player(players, item["name"], item["team_id"])
        if not player:
            missing.append(item)
            continue

        before = {
            "start_prob": player.get("start_prob"),
            "conditional_start_prob": player.get("conditional_start_prob"),
            "availability_risk": player.get("availability_risk"),
            "round_specific_rotation_risk": player.get("round_specific_rotation_risk"),
        }

        player["start_prob"] = item["start_prob"]
        player["conditional_start_prob"] = item["conditional_start_prob"]
        player["availability_risk"] = item["availability_risk"]
        player["round_specific_rotation_risk"] = item["round_specific_rotation_risk"]

        old_note = str(player.get("source_note") or "").strip()
        new_note = item["note"]
        if old_note and new_note not in old_note:
            player["source_note"] = old_note + " | " + new_note
        else:
            player["source_note"] = new_note

        changed.append((player, before, item))

    # Watch-only note, no numeric override
    for item in WATCH_ONLY:
        player = find_player(players, item["name"], item["team_id"])
        if not player:
            missing.append(item)
            continue
        old_note = str(player.get("source_note") or "").strip()
        new_note = item["note"]
        if new_note not in old_note:
            player["source_note"] = (old_note + " | " + new_note).strip(" |")

    write_json_strict(POOL_PATH, players)

    print("CHANGED")
    for player, before, item in changed:
        print(
            f"{player.get('player_name'):<24} {player.get('team_id'):<3} {player.get('position'):<3} "
            f"start {before['start_prob']} -> {player.get('start_prob')} | "
            f"cond {before['conditional_start_prob']} -> {player.get('conditional_start_prob')} | "
            f"risk {before['availability_risk']} -> {player.get('availability_risk')}"
        )

    if missing:
        print()
        print("MISSING / NOT MATCHED")
        for item in missing:
            print(item["team_id"], item["name"])

    print()
    print("Wrote:", POOL_PATH)


if __name__ == "__main__":
    main()
