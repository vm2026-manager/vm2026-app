#!/usr/bin/env python3
"""
Upsert one player suspension row in data/player_suspensions.csv.

Example:
python tools/upsert_player_suspension.py --player-id nathan_ngoy__bel --player-name "Nathan Ngoy" --team-id BEL --round 3 --reason "Direkte rødt kort mod Iran, 66'"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path


DEFAULT_CSV_PATH = Path("data/player_suspensions.csv")
FIELDNAMES = [
    "player_id",
    "player_name",
    "team_id",
    "status",
    "suspended_next_match",
    "suspension_matches_total",
    "suspension_matches_served",
    "suspension_round",
    "reason",
    "source_note",
    "active",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tilføj eller opdater én karantænerække i data/player_suspensions.csv."
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Sti til player_suspensions.csv")
    parser.add_argument("--player-id", default="", help="Valgfrit player_id. Hvis angivet, er det primær matchnøgle.")
    parser.add_argument("--player-name", required=True, help="Spillernavn")
    parser.add_argument("--team-id", required=True, help="Landekode/team_id, fx BEL")
    parser.add_argument("--round", required=True, dest="suspension_round", help="Runden karantænen gælder for")
    parser.add_argument("--reason", required=True, help="Årsag/beskrivelse")
    parser.add_argument("--matches-total", default="1", help="Samlet antal kampes karantæne. Default 1")
    parser.add_argument("--active", default="1", help="1 for aktiv, 0 for inaktiv. Default 1")
    parser.add_argument(
        "--source-note",
        default="Manuel opdatering via upsert_player_suspension.py",
        help="Kildenote til CSV-rækken",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "").replace("'", "")
    return " ".join(text.lower().strip().split())


def normalize_team_id(value: str) -> str:
    return str(value or "").strip().upper()


def match_key(player_name: str, team_id: str) -> tuple[str, str]:
    return normalize_name(player_name), normalize_team_id(team_id)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-fil findes ikke: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]
    return rows


def ensure_headers(rows: list[dict[str, str]]) -> None:
    for row in rows:
        for field in FIELDNAMES:
            row.setdefault(field, "")


def backup_file(csv_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.stem}.backup_before_upsert_{timestamp}{csv_path.suffix}")
    shutil.copy2(csv_path, backup_path)
    return backup_path


def find_existing_row_index(
    rows: list[dict[str, str]],
    player_id: str,
    player_name: str,
    team_id: str,
) -> int | None:
    normalized_player_id = str(player_id or "").strip().lower()
    if normalized_player_id:
        for idx, row in enumerate(rows):
            if str(row.get("player_id", "")).strip().lower() == normalized_player_id:
                return idx

    wanted_key = match_key(player_name, team_id)
    for idx, row in enumerate(rows):
        if match_key(row.get("player_name", ""), row.get("team_id", "")) == wanted_key:
            return idx

    return None


def build_row(args: argparse.Namespace, existing_row: dict[str, str] | None) -> dict[str, str]:
    row = dict(existing_row or {})
    row.update(
        {
            "player_id": str(args.player_id or row.get("player_id", "")).strip(),
            "player_name": str(args.player_name).strip(),
            "team_id": normalize_team_id(args.team_id),
            "status": "suspended",
            "suspended_next_match": "1",
            "suspension_matches_total": str(args.matches_total).strip(),
            "suspension_matches_served": str(row.get("suspension_matches_served", "") or "0").strip(),
            "suspension_round": str(args.suspension_round).strip(),
            "reason": str(args.reason).strip(),
            "source_note": str(args.source_note).strip(),
            "active": str(args.active).strip(),
        }
    )
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return {field: row.get(field, "") for field in FIELDNAMES}


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDNAMES} for row in rows])


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)

    rows = load_rows(csv_path)
    ensure_headers(rows)

    existing_idx = find_existing_row_index(rows, args.player_id, args.player_name, args.team_id)
    existing_row = rows[existing_idx] if existing_idx is not None else None
    next_row = build_row(args, existing_row)

    backup_path = backup_file(csv_path)

    if existing_idx is None:
        rows.append(next_row)
        action = "added"
    else:
        rows[existing_idx] = next_row
        action = "updated"

    write_rows(csv_path, rows)

    print(f"backup: {backup_path}")
    print(f"row_count: {len(rows)}")
    print(f"action: {action}")
    print("player:")
    for field in FIELDNAMES:
      print(f"  {field}: {next_row.get(field, '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
