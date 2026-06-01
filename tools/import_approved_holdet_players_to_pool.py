from __future__ import annotations

import csv
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

APPROVAL_PATH = DATA_DIR / "holdet_new_players_approval.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
IMPORT_REPORT_PATH = DATA_DIR / "holdet_new_players_import_report.csv"

REPORT_FIELDS = [
    "action",
    "holdet_player_id",
    "player_name",
    "team_id",
    "position",
    "price",
    "reason",
]

TEAM_ALIASES = {
    "HOLDET_584": "CZE",
    "HOLDET_767": "CIV",
}

SOURCE_TAG = f"holdet_new_player_import_{datetime.now().strftime('%Y_%m_%d')}"


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def slug(value: Any) -> str:
    text = norm(value)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_price(value: Any) -> int:
    cleaned = re.sub(r"[^\d]", "", txt(value))
    return int(cleaned) if cleaned else 0


def canonical_team(value: Any) -> str:
    team = txt(value).upper()
    return TEAM_ALIASES.get(team, team)


def position(value: Any) -> str:
    return txt(value).upper()


def approved(value: Any) -> bool:
    return txt(value).upper() == "JA"


def identity_key(name: Any, team_id: Any, player_position: Any) -> tuple[str, str, str]:
    return norm(name), canonical_team(team_id), position(player_position)


def make_player_id(name: str, team_id: str, holdet_player_id: str, existing_ids: set[str]) -> str:
    base = f"{slug(name)}__{slug(team_id)}"
    if base and base not in existing_ids:
        return base
    fallback = f"{base}__holdet_{slug(holdet_player_id)}" if base else f"holdet_{slug(holdet_player_id)}"
    candidate = fallback
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{fallback}_{suffix}"
        suffix += 1
    return candidate


def build_player(row: dict[str, str], existing_ids: set[str]) -> dict[str, Any]:
    holdet_player_id_text = txt(row.get("holdet_player_id"))
    player_name = txt(row.get("player_name"))
    team_id = canonical_team(row.get("team_id"))
    player_position = position(row.get("position"))
    price = parse_price(row.get("price"))
    player_id = make_player_id(player_name, team_id, holdet_player_id_text, existing_ids)
    holdet_player_id: int | str = int(holdet_player_id_text) if holdet_player_id_text.isdigit() else holdet_player_id_text

    return {
        "player_id": player_id,
        "old_player_id": player_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_name": txt(row.get("team_name")),
        "position": player_position,
        "price": price,
        "price_estimate": price,
        "price_source": SOURCE_TAG,
        "position_source": SOURCE_TAG,
        "source": SOURCE_TAG,
        "holdet_player_id": holdet_player_id,
        "holdet_person_id": None,
        "holdet_team_id": None,
        "holdet_team_name": txt(row.get("team_name")),
        "holdet_position": player_position,
        "holdet_position_id": None,
        "holdet_price": price,
        "holdet_start_price": price,
        "holdet_is_out": None,
        "holdet_game_id": 616,
        "official_holdet_master": True,
        "has_holdet_vm_match": True,
        "start_price": price,
        "start_prob": 0.48,
        "start_security": 0.48,
        "start_probability_pct": 48,
        "conditional_start_prob": 0.48,
        "availability_prob": 1.0,
        "availability_risk": "unknown",
        "availability_status": "unknown",
        "start_prob_source": "holdet_new_player_import_default",
        "start_status": "ukendt / kræver review",
        "display_score": 0,
        "display_score_source": SOURCE_TAG,
        "display_value": 0,
        "value_score": 0,
        "avg_points": 0,
        "nt_ev_score": 0,
        "weighted_group_stage_ev": 0,
        "optimizer_ev": 0,
        "blended_ev_score": 0,
        "advance_pct": None,
        "copied_from_old_player_pool": False,
    }


def main() -> int:
    if not APPROVAL_PATH.exists():
        print(f"FEJL: Mangler {APPROVAL_PATH.relative_to(PROJECT_ROOT)}")
        return 1
    if not PLAYER_POOL_PATH.exists():
        print(f"FEJL: Mangler {PLAYER_POOL_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    with APPROVAL_PATH.open(encoding="utf-8-sig", newline="") as f:
        approval_rows = list(csv.DictReader(f))

    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        player_pool = json.load(f)

    existing_holdet_ids = {
        txt(player.get("holdet_player_id"))
        for player in player_pool
        if txt(player.get("holdet_player_id"))
    }
    existing_identity_keys = {
        identity_key(player.get("player_name"), player.get("team_id"), player.get("position"))
        for player in player_pool
    }
    existing_player_ids = {
        txt(player.get("player_id"))
        for player in player_pool
        if txt(player.get("player_id"))
    }

    approved_rows = [row for row in approval_rows if approved(row.get("approved_for_player_pool"))]
    report_rows: list[dict[str, Any]] = []
    imported = 0
    skipped_duplicates = 0

    for row in approved_rows:
        holdet_player_id = txt(row.get("holdet_player_id"))
        player_name = txt(row.get("player_name"))
        team_id = canonical_team(row.get("team_id"))
        player_position = position(row.get("position"))
        price = parse_price(row.get("price"))
        key = identity_key(player_name, team_id, player_position)

        if holdet_player_id and holdet_player_id in existing_holdet_ids:
            skipped_duplicates += 1
            report_rows.append(
                {
                    "action": "skipped_duplicate",
                    "holdet_player_id": holdet_player_id,
                    "player_name": player_name,
                    "team_id": team_id,
                    "position": player_position,
                    "price": price,
                    "reason": "duplicate_holdet_player_id",
                }
            )
            continue

        if key in existing_identity_keys:
            skipped_duplicates += 1
            report_rows.append(
                {
                    "action": "skipped_duplicate",
                    "holdet_player_id": holdet_player_id,
                    "player_name": player_name,
                    "team_id": team_id,
                    "position": player_position,
                    "price": price,
                    "reason": "duplicate_name_team_position",
                }
            )
            continue

        player = build_player(row, existing_player_ids)
        player_pool.append(player)
        existing_holdet_ids.add(holdet_player_id)
        existing_identity_keys.add(key)
        existing_player_ids.add(player["player_id"])
        imported += 1
        report_rows.append(
            {
                "action": "imported",
                "holdet_player_id": holdet_player_id,
                "player_name": player_name,
                "team_id": team_id,
                "position": player_position,
                "price": price,
                "reason": "approved_for_player_pool=JA",
            }
        )

    backup_path = DATA_DIR / f"player_pool_v1.backup_before_holdet_new_players_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(PLAYER_POOL_PATH, backup_path)

    PLAYER_POOL_PATH.write_text(
        json.dumps(player_pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with IMPORT_REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Approval-rækker læst: {len(approval_rows)}")
    print(f"Godkendte JA-rækker: {len(approved_rows)}")
    print(f"Importeret: {imported}")
    print(f"Sprunget over som dublet: {skipped_duplicates}")
    print(f"Ny total i player_pool: {len(player_pool)}")
    print(f"Backup: {backup_path.relative_to(PROJECT_ROOT)}")
    print(f"Rapport: {IMPORT_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
