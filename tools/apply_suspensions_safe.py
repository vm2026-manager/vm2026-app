#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUSPENSIONS_CSV = ROOT / "data" / "player_suspensions.csv"
PLAYER_POOL_JSON = ROOT / "data" / "player_pool_v1.json"
PLAYER_EV_CSV = ROOT / "data" / "player_ev_group_stage_v1.csv"
PLAYER_START_SECURITY_CSV = ROOT / "data" / "player_start_security_nt.csv"
AUDIT_CSV = ROOT / "data" / "suspensions_apply_audit.csv"
AUDIT_MD = ROOT / "data" / "suspensions_apply_audit.md"


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "").replace("'", "")
    return " ".join(text.lower().strip().split())


def normalize_team_id(value: str) -> str:
    return str(value or "").strip().upper()


def is_truthy(value) -> bool:
    normalized = str(value or "").strip().lower()
    return value is True or normalized in {"1", "true", "yes", "ja"}


def match_key(player_name: str, team_id: str) -> tuple[str, str]:
    return normalize_name(player_name), normalize_team_id(team_id)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_before_suspensions_apply_{timestamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def to_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number_like(existing_value, numeric_value: float) -> str:
    if existing_value in (None, ""):
        return "0"
    text = str(existing_value)
    if "." in text:
        decimals = len(text.split(".")[-1])
        return f"{numeric_value:.{decimals}f}"
    return str(int(round(numeric_value)))


def build_pool_indexes(rows: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_id = {}
    by_name_team = {}
    for row in rows:
        player_id = str(row.get("player_id") or "").strip().lower()
        if player_id:
            by_id[player_id] = row
        key = match_key(row.get("player_name", ""), row.get("team_id", ""))
        if key[0] and key[1]:
            by_name_team[key] = row
    return by_id, by_name_team


def build_csv_indexes(rows: list[dict[str, str]], player_id_field: str = "player_id") -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    by_id = {}
    by_name_team = {}
    for row in rows:
        player_id = str(row.get(player_id_field) or row.get("\ufeffplayer_id") or "").strip().lower()
        if player_id:
            by_id[player_id] = row
        key = match_key(row.get("player_name", ""), row.get("team_id", ""))
        if key[0] and key[1]:
            by_name_team[key] = row
    return by_id, by_name_team


def find_match(player_id: str, player_name: str, team_id: str, by_id: dict, by_name_team: dict):
    normalized_player_id = str(player_id or "").strip().lower()
    if normalized_player_id and normalized_player_id in by_id:
        return by_id[normalized_player_id]
    return by_name_team.get(match_key(player_name, team_id))


def safe_zero_dict_fields(row: dict, fields: list[str], changed_fields: list[str]) -> None:
    for field in fields:
        if field in row:
            old = row.get(field)
            if old not in (0, 0.0, "0", "0.0", "0.00", ""):
                changed_fields.append(field)
            row[field] = 0 if not isinstance(old, str) else format_number_like(old, 0.0)


def safe_zero_csv_fields(row: dict[str, str], fields: list[str], changed_fields: list[str]) -> None:
    for field in fields:
        if field in row:
            old = row.get(field, "")
            if old not in ("", "0", "0.0", "0.00", "0.0000"):
                changed_fields.append(field)
            row[field] = format_number_like(old, 0.0)


def update_player_pool_row(row: dict) -> tuple[dict[str, str], list[str]]:
    before = {
        "start_prob": str(row.get("start_prob", "")),
        "availability_status": str(row.get("availability_status", "")),
        "optimizer_ev": str(row.get("optimizer_ev", "")),
    }
    changed = []
    row["holdet_is_out"] = False
    row["availability_status"] = "suspended"
    row["availability_risk"] = "suspended"
    row["start_status"] = "suspended"
    safe_zero_dict_fields(
        row,
        [
            "start_prob",
            "conditional_start_prob",
            "availability_prob",
            "appearance_prob",
            "minute_share",
            "start_probability_pct",
            "start_security",
        ],
        changed,
    )
    after = {
        "start_prob": str(row.get("start_prob", "")),
        "availability_status": str(row.get("availability_status", "")),
        "optimizer_ev": str(row.get("optimizer_ev", "")),
    }
    return {"before": before, "after": after}, sorted(set(changed + ["availability_status", "availability_risk", "start_status", "holdet_is_out"]))


def update_start_security_row(row: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    before = {
        "start_prob": str(row.get("start_prob", "")),
        "availability_status": str(row.get("availability_status", "")),
    }
    changed = []
    row["availability_status"] = "suspended"
    row["availability_risk"] = "suspended"
    row["start_status"] = "suspended"
    safe_zero_csv_fields(
        row,
        [
            "start_prob",
            "conditional_start_prob",
            "availability_prob",
            "appearance_prob",
            "start_probability_pct",
            "start_security",
        ],
        changed,
    )
    after = {
        "start_prob": str(row.get("start_prob", "")),
        "availability_status": str(row.get("availability_status", "")),
    }
    return {"before": before, "after": after}, sorted(set(changed + ["availability_status", "availability_risk", "start_status"]))


def update_ev_row(row: dict[str, str], suspension_round: str) -> tuple[dict[str, str], list[str], str]:
    before = {
        "start_prob": str(row.get("start_prob", "")),
        "optimizer_ev": str(row.get("optimizer_ev", "")),
        "weighted_group_stage_ev": str(row.get("weighted_group_stage_ev", "")),
    }
    changed = []
    warning = "unchanged_with_warning"

    if "start_prob" in row:
        safe_zero_csv_fields(row, ["start_prob", "minute_share"], changed)

    round_no = str(suspension_round or "").strip()
    if round_no in {"1", "2", "3"}:
        round_fields = [
            f"match_{round_no}_goal_ev",
            f"match_{round_no}_assist_ev",
            f"match_{round_no}_shots_on_target_ev",
            f"match_{round_no}_clean_sheet_ev",
            f"match_{round_no}_card_ev",
            f"match_{round_no}_result_ev",
            f"match_{round_no}_team_scores_ev",
            f"match_{round_no}_opponent_scores_ev",
            f"match_{round_no}_on_pitch_ev",
            f"match_{round_no}_start_minutes_ev",
            f"match_{round_no}_total_ev_next_match",
            f"match_{round_no}_weighted_match_ev",
            f"round{round_no}_ev",
            f"round{round_no}_captain_growth",
            f"round{round_no}_clean_sheet_prob",
        ]
        existing_round_fields = [field for field in round_fields if field in row]
        if existing_round_fields:
            safe_zero_csv_fields(row, existing_round_fields, changed)
            warning = "round_component_fields_zeroed; aggregate_ev_left_unchanged"

    after = {
        "start_prob": str(row.get("start_prob", "")),
        "optimizer_ev": str(row.get("optimizer_ev", "")),
        "weighted_group_stage_ev": str(row.get("weighted_group_stage_ev", "")),
    }
    return {"before": before, "after": after}, sorted(set(changed)), warning


def write_audit(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "player_id",
        "player_name",
        "team_id",
        "suspension_round",
        "matched",
        "old_start_prob",
        "new_start_prob",
        "old_availability_status",
        "new_availability_status",
        "old_ev",
        "new_ev",
        "files_changed",
        "fields_changed",
        "warning",
    ]
    with AUDIT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Suspensions Safe Apply Audit",
        "",
        f"- Aktive karantæner: {len(rows)}",
        f"- Matched: {sum(1 for row in rows if row['matched'] == 'yes')}",
        f"- Unmatched: {sum(1 for row in rows if row['matched'] == 'no')}",
        "",
        "| Spiller | Team | Runde | Matched | Gammel start_prob | Ny start_prob | Gammel status | Ny status | Gammel EV | Ny EV | Filer ændret | Warning |",
        "|---|---|---:|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['player_name']} | {row['team_id']} | {row['suspension_round']} | {row['matched']} | "
            f"{row['old_start_prob'] or '-'} | {row['new_start_prob'] or '-'} | "
            f"{row['old_availability_status'] or '-'} | {row['new_availability_status'] or '-'} | "
            f"{row['old_ev'] or '-'} | {row['new_ev'] or '-'} | {row['files_changed'] or '-'} | {row['warning'] or '-'} |"
        )
    AUDIT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    suspensions_fieldnames, suspensions_rows = load_csv(SUSPENSIONS_CSV)
    active_suspensions = [row for row in suspensions_rows if is_truthy(row.get("active"))]

    player_pool = json.loads(PLAYER_POOL_JSON.read_text(encoding="utf-8"))
    ev_fieldnames, ev_rows = load_csv(PLAYER_EV_CSV)
    start_fieldnames, start_rows = load_csv(PLAYER_START_SECURITY_CSV)

    pool_by_id, pool_by_name_team = build_pool_indexes(player_pool)
    ev_by_id, ev_by_name_team = build_csv_indexes(ev_rows)
    start_by_id, start_by_name_team = build_csv_indexes(start_rows)

    audit_rows = []
    matched = 0
    unmatched = 0
    player_pool_changed = False
    ev_changed = False
    start_changed = False

    pool_backup = backup_file(PLAYER_POOL_JSON)
    ev_backup = backup_file(PLAYER_EV_CSV)
    start_backup = backup_file(PLAYER_START_SECURITY_CSV)

    for suspension in active_suspensions:
        suspension_player_id = str(suspension.get("player_id") or "").strip()
        suspension_player_name = str(suspension.get("player_name") or "").strip()
        suspension_team_id = normalize_team_id(suspension.get("team_id"))
        suspension_round = str(suspension.get("suspension_round") or "").strip()

        pool_row = find_match(suspension_player_id, suspension_player_name, suspension_team_id, pool_by_id, pool_by_name_team)
        ev_row = find_match(suspension_player_id, suspension_player_name, suspension_team_id, ev_by_id, ev_by_name_team)
        start_row = find_match(suspension_player_id, suspension_player_name, suspension_team_id, start_by_id, start_by_name_team)

        if not pool_row and not ev_row and not start_row:
            unmatched += 1
            audit_rows.append({
                "player_id": suspension_player_id,
                "player_name": suspension_player_name,
                "team_id": suspension_team_id,
                "suspension_round": suspension_round,
                "matched": "no",
                "old_start_prob": "",
                "new_start_prob": "",
                "old_availability_status": "",
                "new_availability_status": "",
                "old_ev": "",
                "new_ev": "",
                "files_changed": "",
                "fields_changed": "",
                "warning": "unmatched_active_suspension",
            })
            continue

        matched += 1
        changed_files = []
        changed_fields = []
        warning_parts = []

        pool_before_after = {"before": {}, "after": {}}
        if pool_row:
          pool_before_after, pool_fields = update_player_pool_row(pool_row)
          changed_fields.extend([f"player_pool:{field}" for field in pool_fields])
          changed_files.append("player_pool_v1.json")
          player_pool_changed = True

        start_before_after = {"before": {}, "after": {}}
        if start_row:
          start_before_after, start_fields = update_start_security_row(start_row)
          changed_fields.extend([f"player_start_security_nt.csv:{field}" for field in start_fields])
          changed_files.append("player_start_security_nt.csv")
          start_changed = True

        ev_before_after = {"before": {}, "after": {}}
        if ev_row:
          ev_before_after, ev_fields, ev_warning = update_ev_row(ev_row, suspension_round)
          changed_fields.extend([f"player_ev_group_stage_v1.csv:{field}" for field in ev_fields])
          changed_files.append("player_ev_group_stage_v1.csv")
          ev_changed = True
          if ev_warning:
              warning_parts.append(ev_warning)
        else:
          warning_parts.append("no_ev_row_matched")

        old_start_prob = (
            pool_before_after["before"].get("start_prob")
            or start_before_after["before"].get("start_prob")
            or ev_before_after["before"].get("start_prob")
            or ""
        )
        new_start_prob = (
            pool_before_after["after"].get("start_prob")
            or start_before_after["after"].get("start_prob")
            or ev_before_after["after"].get("start_prob")
            or ""
        )
        old_status = (
            pool_before_after["before"].get("availability_status")
            or start_before_after["before"].get("availability_status")
            or ""
        )
        new_status = (
            pool_before_after["after"].get("availability_status")
            or start_before_after["after"].get("availability_status")
            or ""
        )
        old_ev = (
            pool_before_after["before"].get("optimizer_ev")
            or ev_before_after["before"].get("optimizer_ev")
            or ""
        )
        new_ev = (
            pool_before_after["after"].get("optimizer_ev")
            or ev_before_after["after"].get("optimizer_ev")
            or ""
        )

        audit_rows.append({
            "player_id": suspension_player_id,
            "player_name": suspension_player_name,
            "team_id": suspension_team_id,
            "suspension_round": suspension_round,
            "matched": "yes",
            "old_start_prob": old_start_prob,
            "new_start_prob": new_start_prob,
            "old_availability_status": old_status,
            "new_availability_status": new_status,
            "old_ev": old_ev,
            "new_ev": new_ev if new_ev != old_ev else "unchanged_with_warning",
            "files_changed": ", ".join(dict.fromkeys(changed_files)),
            "fields_changed": ", ".join(dict.fromkeys(changed_fields)),
            "warning": "; ".join(dict.fromkeys(warning_parts)),
        })

    if player_pool_changed:
        PLAYER_POOL_JSON.write_text(json.dumps(player_pool, ensure_ascii=False, indent=2), encoding="utf-8")
    if ev_changed:
        write_csv(PLAYER_EV_CSV, ev_fieldnames, ev_rows)
    if start_changed:
        write_csv(PLAYER_START_SECURITY_CSV, start_fieldnames, start_rows)

    write_audit(audit_rows)

    print(f"active_suspensions: {len(active_suspensions)}")
    print(f"matched: {matched}")
    print(f"unmatched: {unmatched}")
    print(f"backups: {pool_backup.name}, {ev_backup.name}, {start_backup.name}")
    print(f"wrote_audit_csv: {AUDIT_CSV}")
    print(f"wrote_audit_md: {AUDIT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
