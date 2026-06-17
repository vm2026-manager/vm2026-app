from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOOLS_DIR = Path(__file__).resolve().parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from holdet_players_api import fetch_holdet_players, flatten_holdet_payload

POOL_PATH = DATA_DIR / "player_pool_v1.json"
HOLDET_PATH = DATA_DIR / "holdet_players_game_616_flat.csv"
RAW_PATH = DATA_DIR / "holdet_players_game_616_raw.json"
FLAT_JSON_PATH = DATA_DIR / "holdet_players_game_616_flat.json"
DEFAULT_GAME_ID = 616


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_int(value: Any) -> int | None:
    raw = txt(value).replace(",", ".")
    if not raw:
        return None
    try:
        return int(round(float(raw)))
    except ValueError:
        return None


def is_active_player(player: dict[str, Any]) -> bool:
    return not bool(player.get("holdet_is_out"))


def refresh_holdet_files(game_id: int) -> None:
    payload = fetch_holdet_players(game_id)
    df = flatten_holdet_payload(payload)

    RAW_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    df.to_csv(HOLDET_PATH, index=False, encoding="utf-8-sig")
    FLAT_JSON_PATH.write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=int, default=DEFAULT_GAME_ID)
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    if not args.no_refresh:
        refresh_holdet_files(args.game_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATA_DIR / f"player_pool_v1.backup_before_existing_price_update_{timestamp}.json"
    audit_path = DATA_DIR / f"existing_player_price_update_audit_{timestamp}.json"
    changes_csv_path = DATA_DIR / f"existing_player_price_changes_{timestamp}.csv"

    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(pool, list):
        raise ValueError("player_pool_v1.json skal indeholde en liste")

    with HOLDET_PATH.open(encoding="utf-8-sig", newline="") as f:
        holdet_rows = list(csv.DictReader(f))

    backup_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    holdet_by_player_id = {}
    holdet_by_person_id = {}
    ignored_new_holdet = []
    duplicate_holdet_player_ids = set()
    duplicate_holdet_person_ids = set()

    for row in holdet_rows:
        holdet_player_id = txt(row.get("holdet_player_id"))
        holdet_person_id = txt(row.get("holdet_person_id"))

        if holdet_player_id:
            if holdet_player_id in holdet_by_player_id:
                duplicate_holdet_player_ids.add(holdet_player_id)
            else:
                holdet_by_player_id[holdet_player_id] = row

        if holdet_person_id:
            if holdet_person_id in holdet_by_person_id:
                duplicate_holdet_person_ids.add(holdet_person_id)
            else:
                holdet_by_person_id[holdet_person_id] = row

    matched_players = 0
    price_changes = []
    pool_without_match = []
    position_mismatches = []
    team_mismatches = []
    name_mismatches = []

    used_holdet_player_ids: set[str] = set()

    for player in pool:
        holdet_player_id = txt(player.get("holdet_player_id"))
        holdet_person_id = txt(player.get("holdet_person_id"))
        holdet_row = None
        match_method = ""

        if holdet_player_id and holdet_player_id not in duplicate_holdet_player_ids:
            holdet_row = holdet_by_player_id.get(holdet_player_id)
            if holdet_row is not None:
                match_method = "holdet_player_id"

        if holdet_row is None and not holdet_player_id and holdet_person_id and holdet_person_id not in duplicate_holdet_person_ids:
            holdet_row = holdet_by_person_id.get(holdet_person_id)
            if holdet_row is not None:
                match_method = "holdet_person_id"

        if holdet_row is None:
            pool_without_match.append(
                {
                    "player_id": txt(player.get("player_id")),
                    "player_name": txt(player.get("player_name")),
                    "team_id": txt(player.get("team_id")),
                    "position": txt(player.get("position")),
                    "holdet_player_id": holdet_player_id,
                    "holdet_person_id": holdet_person_id,
                }
            )
            continue

        matched_players += 1
        used_holdet_player_ids.add(txt(holdet_row.get("holdet_player_id")))

        pool_name = txt(player.get("player_name"))
        holdet_name = txt(holdet_row.get("player_name"))
        if pool_name.casefold() != holdet_name.casefold():
            name_mismatches.append(
                {
                    "player_id": txt(player.get("player_id")),
                    "holdet_player_id": txt(holdet_row.get("holdet_player_id")),
                    "pool_name": pool_name,
                    "holdet_name": holdet_name,
                }
            )

        pool_position = txt(player.get("position")).upper()
        holdet_position = txt(holdet_row.get("position")).upper()
        if pool_position != holdet_position:
            position_mismatches.append(
                {
                    "player_id": txt(player.get("player_id")),
                    "holdet_player_id": txt(holdet_row.get("holdet_player_id")),
                    "player_name": pool_name,
                    "pool_position": pool_position,
                    "holdet_position": holdet_position,
                }
            )

        pool_team = txt(player.get("holdet_team_name") or player.get("team_name"))
        holdet_team = txt(holdet_row.get("team_name"))
        if pool_team and holdet_team and pool_team.casefold() != holdet_team.casefold():
            team_mismatches.append(
                {
                    "player_id": txt(player.get("player_id")),
                    "holdet_player_id": txt(holdet_row.get("holdet_player_id")),
                    "player_name": pool_name,
                    "pool_team": pool_team,
                    "holdet_team": holdet_team,
                }
            )

        old_price = to_int(player.get("price"))
        new_price = to_int(holdet_row.get("price"))
        old_holdet_price = to_int(player.get("holdet_price"))

        if new_price is None:
            continue

        if old_price != new_price or old_holdet_price != new_price:
            price_changes.append(
                {
                    "player_id": txt(player.get("player_id")),
                    "holdet_player_id": txt(holdet_row.get("holdet_player_id")),
                    "player_name": pool_name,
                    "team_name": txt(player.get("team_name")),
                    "position": pool_position,
                    "match_method": match_method,
                    "old_price": old_price,
                    "old_holdet_price": old_holdet_price,
                    "new_price": new_price,
                    "diff": None if old_price is None else new_price - old_price,
                }
            )

            player["holdet_price"] = new_price
            player["price"] = new_price
            player["price_estimate"] = new_price

    for row in holdet_rows:
        holdet_player_id = txt(row.get("holdet_player_id"))
        if holdet_player_id and holdet_player_id in used_holdet_player_ids:
            continue
        ignored_new_holdet.append(
            {
                "holdet_player_id": holdet_player_id,
                "holdet_person_id": txt(row.get("holdet_person_id")),
                "player_name": txt(row.get("player_name")),
                "team_name": txt(row.get("team_name")),
                "position": txt(row.get("position")),
                "price": to_int(row.get("price")),
                "is_out": txt(row.get("is_out")),
                "reason": "not_in_player_pool",
            }
        )

    price_changes_sorted_up = sorted(price_changes, key=lambda item: (item["diff"] if item["diff"] is not None else -10**12), reverse=True)
    price_changes_sorted_down = sorted(price_changes, key=lambda item: (item["diff"] if item["diff"] is not None else 10**12))

    POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    with changes_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "player_id",
            "holdet_player_id",
            "player_name",
            "team_name",
            "position",
            "match_method",
            "old_price",
            "old_holdet_price",
            "new_price",
            "diff",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(price_changes)

    audit = {
        "holdet_rows": len(holdet_rows),
        "player_pool_rows": len(pool),
        "safe_matches": matched_players,
        "price_changes": len(price_changes),
        "duplicate_holdet_player_ids": sorted(duplicate_holdet_player_ids),
        "duplicate_holdet_person_ids": sorted(duplicate_holdet_person_ids),
        "largest_price_increases": price_changes_sorted_up[:20],
        "largest_price_decreases": price_changes_sorted_down[:20],
        "ignored_new_holdet_players": ignored_new_holdet,
        "player_pool_without_safe_holdet_match": pool_without_match,
        "name_mismatches": name_mismatches,
        "position_mismatches": position_mismatches,
        "team_mismatches": team_mismatches,
        "backup_path": str(backup_path),
        "changes_csv_path": str(changes_csv_path),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Backup:", backup_path)
    print("Audit:", audit_path)
    print("Changes CSV:", changes_csv_path)
    print("Raw JSON:", RAW_PATH)
    print("Flat CSV:", HOLDET_PATH)
    print("Flat JSON:", FLAT_JSON_PATH)
    print("Holdet rows:", len(holdet_rows))
    print("Player pool rows:", len(pool))
    print("Safe matches:", matched_players)
    print("Price changes:", len(price_changes))
    print("Ignored new Holdet players:", len(ignored_new_holdet))
    print("Player pool without safe Holdet match:", len(pool_without_match))
    print("Name mismatches:", len(name_mismatches))
    print("Position mismatches:", len(position_mismatches))
    print("Team mismatches:", len(team_mismatches))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
