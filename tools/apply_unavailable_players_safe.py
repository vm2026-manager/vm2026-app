from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sync_final_ev_to_player_pool import sync_final_ev_to_pool


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_PATH = DATA / "player_unavailable_overrides.csv"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
SECURITY_PATH = DATA / "player_start_security_nt.csv"
AUDIT_PATH = DATA / "player_unavailable_audit.csv"

POOL_ZERO_FIELDS = [
    "start_prob", "conditional_start_prob", "start_probability_pct", "start_security", "start_pct",
    "weighted_group_stage_ev", "optimizer_ev", "weighted_group_stage_ev_before_price_quality",
    "model_ev_before_price_quality", "optimizer_ev_before_price_quality", "price_quality_ev",
    "price_quality_raw_ev", "price_quality_appearance_scaled_ev", "price_quality_base_capped_ev",
    "round1_ev", "round2_ev", "round3_ev", "nt_ev_score", "blended_ev_score", "display_score",
    "value_score", "display_value", "avg_points",
]
EV_ZERO_FIELDS = [
    "start_prob", "conditional_start_prob", "start_probability_pct", "start_security", "minute_share",
    "weighted_group_stage_ev", "optimizer_ev", "total_ev_group_stage",
    "weighted_group_stage_ev_before_price_quality", "model_ev_before_price_quality",
    "optimizer_ev_before_price_quality", "price_quality_ev", "price_quality_raw_ev",
    "price_quality_appearance_scaled_ev", "price_quality_base_capped_ev",
]
for round_number in (1, 2, 3):
    EV_ZERO_FIELDS.extend(
        f"match_{round_number}_{suffix}"
        for suffix in [
            "goal_ev", "assist_ev", "shots_on_target_ev", "clean_sheet_ev", "card_ev", "result_ev",
            "team_scores_ev", "opponent_scores_ev", "on_pitch_ev", "start_minutes_ev",
            "total_ev_next_match", "weighted_match_ev",
        ]
    )


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def truthy(value: Any) -> bool:
    return text(value).casefold() in {"true", "1", "yes", "ja"}


def number(value: Any) -> float:
    try:
        return float(text(value).replace(",", "."))
    except ValueError:
        return 0.0


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def require_unique(rows: list[dict[str, Any]], label: str) -> None:
    counts = Counter(text(row.get("player_id")) for row in rows if text(row.get("player_id")))
    duplicates = [player_id for player_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} has duplicate player_id values: {', '.join(duplicates)}")


def backup(paths: list[Path], stamp: str) -> list[Path]:
    outputs = []
    for path in paths:
        if path.exists():
            target = path.with_name(f"{path.stem}.backup_before_unavailable_sync_{stamp}{path.suffix}")
            shutil.copy2(path, target)
            outputs.append(target)
    return outputs


def set_out_fields(row: dict[str, Any], source: dict[str, str], zero_fields: list[str]) -> None:
    string_values = isinstance(row.get("holdet_is_out"), str)
    row["holdet_is_out"] = "True" if string_values else True
    row["is_out"] = "True" if string_values else True
    row["availability_prob"] = "0" if isinstance(row.get("availability_prob"), str) else 0
    row["availability_risk"] = "out"
    row["availability_status"] = "out_of_tournament"
    row["start_status"] = "ude af VM"
    row["start_prob_source"] = "manual_unavailable_override"
    row["base_ev_source"] = "out_of_tournament_manual"
    row["price_quality_applied"] = "False" if isinstance(row.get("price_quality_applied"), str) else False
    row["price_quality_method"] = "not_applied"
    row["source_note"] = text(source.get("source_note"))
    for field in zero_fields:
        if field in row:
            row[field] = "0" if isinstance(row.get(field), str) else 0


def apply_by_id(
    rows: list[dict[str, Any]], sources: list[dict[str, str]], zero_fields: list[str], label: str
) -> None:
    by_id = {text(row.get("player_id")): row for row in rows}
    missing = [text(source["player_id"]) for source in sources if text(source["player_id"]) not in by_id]
    if missing:
        raise ValueError(f"{label} missing unavailable players: {', '.join(missing)}")
    for source in sources:
        set_out_fields(by_id[text(source["player_id"])], source, zero_fields)


def sync_security(
    fields: list[str], rows: list[dict[str, Any]], players: list[dict[str, Any]], target_ids: set[str]
) -> None:
    wanted_fields = [
        "start_prob", "conditional_start_prob", "start_probability_pct", "start_security",
        "start_prob_source", "start_status", "availability_prob", "availability_risk",
        "availability_status",
    ]
    sync_fields = [field for field in wanted_fields if field in fields]
    pool_by_id = {text(player.get("player_id")): player for player in players}
    security_by_id = {text(row.get("player_id")): row for row in rows}
    missing = sorted(target_ids - set(security_by_id))
    if missing:
        raise ValueError(f"Start-security CSV missing unavailable players: {', '.join(missing)}")
    for player_id in target_ids:
        for field in sync_fields:
            security_by_id[player_id][field] = pool_by_id[player_id].get(field, "")


def build_audit(
    sources: list[dict[str, str]], players: list[dict[str, Any]], ev_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pool_by_id = {text(row.get("player_id")): row for row in players}
    ev_by_id = {text(row.get("player_id")): row for row in ev_rows}
    audit = []
    for source in sources:
        player_id = text(source["player_id"])
        pool = pool_by_id[player_id]
        ev = ev_by_id[player_id]
        out = truthy(pool.get("holdet_is_out")) or truthy(pool.get("is_out"))
        start_zero = number(pool.get("start_prob")) == 0 and number(pool.get("conditional_start_prob")) == 0
        ev_zero = number(ev.get("weighted_group_stage_ev")) == 0 and number(ev.get("optimizer_ev")) == 0
        display_fields = ["display_score", "display_value", "value_score", "nt_ev_score", "blended_ev_score"]
        display_zero = all(number(pool.get(field)) == 0 for field in display_fields)
        audit.append(
            {
                "player_id": player_id,
                "player_name": text(pool.get("player_name")),
                "holdet_is_out": text(pool.get("holdet_is_out")),
                "is_out": text(pool.get("is_out")),
                "start_prob": number(pool.get("start_prob")),
                "conditional_start_prob": number(pool.get("conditional_start_prob")),
                "weighted_group_stage_ev": number(ev.get("weighted_group_stage_ev")),
                "optimizer_ev": number(ev.get("optimizer_ev")),
                **{field: number(pool.get(field)) for field in display_fields},
                "optimizer_eligible": "no" if out else "yes",
                "frontend_selectable": "no" if out else "yes",
                "check_status": "ok" if out and start_zero and ev_zero and display_zero else "failed",
            }
        )
    return audit


def main() -> int:
    _, sources = read_csv(SOURCE_PATH)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    ev_fields, ev_rows = read_csv(EV_PATH)
    security_fields, security_rows = read_csv(SECURITY_PATH)
    require_unique(sources, "Unavailable source")
    require_unique(pool, "Player pool")
    require_unique(ev_rows, "EV CSV")
    target_ids = {text(source["player_id"]) for source in sources}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_outputs = [
        DATA / "optimal_squads_by_strategy.json",
        DATA / "strategy_comparison_report.csv",
        DATA / "strategy_formation_comparison_report.csv",
        DATA / "current_strategy_context.json",
        DATA / "strategy_cleanup_report.md",
    ]
    backups = backup([POOL_PATH, EV_PATH, SECURITY_PATH] + optimizer_outputs, stamp)

    apply_by_id(pool, sources, POOL_ZERO_FIELDS, "Player pool")
    apply_by_id(ev_rows, sources, EV_ZERO_FIELDS, "EV CSV")
    write_json(POOL_PATH, pool)
    write_csv(EV_PATH, ev_fields, ev_rows)
    sync_final_ev_to_pool()

    pool = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    sync_security(security_fields, security_rows, pool, target_ids)
    write_csv(SECURITY_PATH, security_fields, security_rows)
    subprocess.run([sys.executable, str(ROOT / "tools" / "optimize_squad_group_stage.py")], cwd=ROOT, check=True)

    audit = build_audit(sources, pool, ev_rows)
    write_csv(AUDIT_PATH, list(audit[0]), audit)
    if any(row["check_status"] != "ok" for row in audit):
        raise RuntimeError("Unavailable-player audit failed")

    print("Unavailable player synchronization")
    print("----------------------------------")
    print(f"Persistent source: {SOURCE_PATH.relative_to(ROOT)}")
    print(f"Players applied: {len(audit)}")
    print(f"Backups created: {len(backups)}")
    for row in audit:
        print(
            f"{row['player_name']}: out=yes, start=0, EV=0, display=0, "
            "optimizer_eligible=no, frontend_selectable=no"
        )
    print(f"Wrote: {AUDIT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
