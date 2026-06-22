#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUSPENSIONS_CSV = ROOT / "data" / "player_suspensions.csv"
PLAYER_POOL_JSON = ROOT / "data" / "player_pool_v1.json"
PLAYER_EV_CSV = ROOT / "data" / "player_ev_group_stage_v1.csv"
OPTIMAL_SQUADS_JSON = ROOT / "data" / "optimal_squads_by_strategy.json"
OUT_CSV = ROOT / "data" / "suspensions_model_impact_audit.csv"
OUT_MD = ROOT / "data" / "suspensions_model_impact_audit.md"


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_player_pool_indexes(rows: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_id: dict[str, dict] = {}
    by_name_team: dict[tuple[str, str], dict] = {}
    for row in rows:
        player_id = str(row.get("player_id") or "").strip().lower()
        if player_id:
            by_id[player_id] = row
        name_team = match_key(row.get("player_name", ""), row.get("team_id", ""))
        if name_team[0] and name_team[1]:
            by_name_team[name_team] = row
    return by_id, by_name_team


def build_ev_indexes(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_name_team: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        player_id = str(row.get("player_id") or "").strip().lower()
        if player_id:
            by_id[player_id] = row
        name_team = match_key(row.get("player_name", ""), row.get("team_id", ""))
        if name_team[0] and name_team[1]:
            by_name_team[name_team] = row
    return by_id, by_name_team


def collect_optimizer_presence(data: dict) -> tuple[set[str], set[tuple[str, str]]]:
    ids: set[str] = set()
    names: set[tuple[str, str]] = set()

    def visit_row(row: dict):
        player_id = str(row.get("player_id") or "").strip().lower()
        if player_id:
            ids.add(player_id)
        key = match_key(row.get("player_name", ""), row.get("team_id", ""))
        if key[0] and key[1]:
            names.add(key)

    if not isinstance(data, dict):
        return ids, names

    for strategy_value in data.values():
        if not isinstance(strategy_value, dict):
            continue
        for row in strategy_value.get("best_squad", []) or []:
            if isinstance(row, dict):
                visit_row(row)
        for formation_value in (strategy_value.get("squads_by_formation", {}) or {}).values():
            if not isinstance(formation_value, dict):
                continue
            for row in formation_value.get("squad", []) or []:
                if isinstance(row, dict):
                    visit_row(row)
    return ids, names


def to_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def recommend_action(found_in_pool: bool, found_in_ev: bool, found_in_optimizer: bool) -> str:
    actions = []
    if found_in_pool:
        actions.append("set next-match start_prob to 0")
        actions.append("mark availability_status suspended")
        actions.append("keep holdet_is_out false")
    if found_in_ev or found_in_optimizer:
        actions.append("exclude from optimizer for affected round")
    return "; ".join(actions) if actions else "manual review - no model match found"


def main() -> int:
    suspensions = [row for row in load_csv(SUSPENSIONS_CSV) if is_truthy(row.get("active"))]
    player_pool = load_json(PLAYER_POOL_JSON)
    ev_rows = load_csv(PLAYER_EV_CSV)
    optimal_squads = load_json(OPTIMAL_SQUADS_JSON) if OPTIMAL_SQUADS_JSON.exists() else {}

    pool_by_id, pool_by_name_team = build_player_pool_indexes(player_pool)
    ev_by_id, ev_by_name_team = build_ev_indexes(ev_rows)
    optimizer_ids, optimizer_name_team = collect_optimizer_presence(optimal_squads)

    audit_rows: list[dict[str, str]] = []
    unmatched: list[str] = []
    matched_pool_count = 0
    matched_ev_count = 0
    matched_optimizer_count = 0

    for suspension in suspensions:
        suspension_player_id = str(suspension.get("player_id") or "").strip().lower()
        suspension_name = str(suspension.get("player_name") or "").strip()
        suspension_team_id = normalize_team_id(suspension.get("team_id"))
        key = match_key(suspension_name, suspension_team_id)

        pool_row = (
            pool_by_id.get(suspension_player_id)
            if suspension_player_id
            else None
        ) or pool_by_name_team.get(key)

        ev_row = (
            ev_by_id.get(suspension_player_id)
            if suspension_player_id
            else None
        ) or ev_by_name_team.get(key)

        found_in_optimizer = (
            (suspension_player_id in optimizer_ids if suspension_player_id else False)
            or (key in optimizer_name_team)
        )

        if pool_row:
            matched_pool_count += 1
        else:
            unmatched.append(f"{suspension_name} ({suspension_team_id})")

        if ev_row:
            matched_ev_count += 1
        if found_in_optimizer:
            matched_optimizer_count += 1

        audit_rows.append({
            "player_id": suspension.get("player_id", ""),
            "player_name": suspension_name,
            "team_id": suspension_team_id,
            "suspension_round": str(suspension.get("suspension_round", "")),
            "matches_total": str(suspension.get("suspension_matches_total", "")),
            "found_in_player_pool": "yes" if pool_row else "no",
            "current_start_prob": str(pool_row.get("start_prob", "") if pool_row else ""),
            "current_availability_status": str(pool_row.get("availability_status", "") if pool_row else ""),
            "current_holdet_is_out": str(pool_row.get("holdet_is_out", "") if pool_row else ""),
            "current_ev": str(
                (ev_row.get("weighted_group_stage_ev") if ev_row and ev_row.get("weighted_group_stage_ev") not in ("", None) else pool_row.get("weighted_group_stage_ev", "") if pool_row else "")
            ),
            "current_optimizer_ev": str(
                (ev_row.get("optimizer_ev") if ev_row and ev_row.get("optimizer_ev") not in ("", None) else pool_row.get("optimizer_ev", "") if pool_row else "")
            ),
            "found_in_ev": "yes" if ev_row else "no",
            "found_in_optimizer_squads": "yes" if found_in_optimizer else "no",
            "reason": str(suspension.get("reason", "")),
            "recommended_action": recommend_action(bool(pool_row), bool(ev_row), found_in_optimizer),
        })

    fieldnames = [
        "player_id",
        "player_name",
        "team_id",
        "suspension_round",
        "matches_total",
        "found_in_player_pool",
        "current_start_prob",
        "current_availability_status",
        "current_holdet_is_out",
        "current_ev",
        "current_optimizer_ev",
        "found_in_ev",
        "found_in_optimizer_squads",
        "reason",
        "recommended_action",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    md_lines = [
        "# Suspensions Model Impact Audit",
        "",
        f"- Aktive karantæner: {len(suspensions)}",
        f"- Matchet i player_pool: {matched_pool_count}",
        f"- Fundet i EV-fil: {matched_ev_count}",
        f"- Fundet i optimizer-squads: {matched_optimizer_count}",
        f"- Unmatched: {', '.join(unmatched) if unmatched else 'Ingen'}",
        "",
        "## Spillere",
        "",
        "| Spiller | Team | Runde | start_prob | availability_status | holdet_is_out | EV | optimizer_ev | I optimizer? | Anbefalet handling |",
        "|---|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in audit_rows:
        md_lines.append(
            f"| {row['player_name']} | {row['team_id']} | {row['suspension_round']} | "
            f"{row['current_start_prob'] or '-'} | {row['current_availability_status'] or '-'} | "
            f"{row['current_holdet_is_out'] or '-'} | {row['current_ev'] or '-'} | "
            f"{row['current_optimizer_ev'] or '-'} | {row['found_in_optimizer_squads']} | "
            f"{row['recommended_action']} |"
        )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"active_suspensions: {len(suspensions)}")
    print(f"matched_in_player_pool: {matched_pool_count}")
    print(f"found_in_ev: {matched_ev_count}")
    print(f"found_in_optimizer_squads: {matched_optimizer_count}")
    print(f"unmatched: {', '.join(unmatched) if unmatched else 'none'}")
    print(f"wrote_csv: {OUT_CSV}")
    print(f"wrote_md: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
