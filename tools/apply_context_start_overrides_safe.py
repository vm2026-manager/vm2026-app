from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from repair_ev_price_quality_consistency import (
    FORMULA_TOLERANCE,
    formula_diff,
    formula_expected,
    reserve_safe_price_quality,
)
from sync_final_ev_to_player_pool import sync_final_ev_to_pool


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OVERRIDE_PATH = DATA / "start_signal_context_overrides.csv"
POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
SECURITY_PATH = DATA / "player_start_security_nt.csv"
SOURCE = "manual_context_override_safe"
MAX_EV_SCALE_UP = 1.50
MIN_EV_SCALE_DOWN = 0.50

KEY_PLAYER_IDS = [
    "neymar_jr__bra",
    "kylian_mbapp__fra",
    "harry_kane__eng",
    "lautaro_mart_nez__arg",
    "juli_n_lvarez__arg",
    "alisson_becker__bra",
    "ederson_moraes__bra",
    "wesley_franca__bra",
    "christoph_baumgartner__aut",
    "lennart_karl__ger",
]

POOL_CONTEXT_FIELDS = [
    "conditional_start_prob",
    "appearance_prob",
    "availability_prob",
    "availability_risk",
    "availability_status",
    "round_specific_rotation_risk",
]
EV_CONTEXT_FIELDS = POOL_CONTEXT_FIELDS + ["source_note"]
SCALE_FIELDS = [
    "model_ev_before_price_quality",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "total_ev_group_stage",
]
for round_number in (1, 2, 3):
    SCALE_FIELDS.extend(
        [
            f"match_{round_number}_weighted_match_ev",
            f"match_{round_number}_total_ev_next_match",
        ]
    )


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def number(value: Any) -> float | None:
    raw = text(value).replace(",", ".")
    if not raw:
        return None
    try:
        result = float(raw)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def fmt(value: float) -> str:
    return f"{value:.12f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def require_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    counts = Counter(text(row.get("player_id")) for row in rows if text(row.get("player_id")))
    duplicates = sorted(player_id for player_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate player_id values: {', '.join(duplicates[:10])}")


def load_overrides() -> list[dict[str, str]]:
    fields, rows = read_csv(OVERRIDE_PATH)
    required = {"player_id", "start_prob", "conditional_start_prob"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Override CSV is missing columns: {', '.join(sorted(missing))}")
    require_unique_ids(rows, "Override CSV")
    for row in rows:
        player_id = text(row.get("player_id"))
        if not player_id:
            raise ValueError("Override CSV contains a blank player_id")
        for field in ("start_prob", "conditional_start_prob"):
            value = number(row.get(field))
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{player_id}: {field} must be between 0 and 1")
    return rows


def backup_files(paths: list[Path], stamp: str) -> list[Path]:
    backups = []
    for path in paths:
        if not path.exists():
            continue
        backup = path.with_name(f"{path.stem}.backup_before_context_start_overrides_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        backups.append(backup)
    return backups


def start_status(start_prob: float, current: Any) -> str:
    current_text = text(current)
    suffix = " - Transfermarkt landshold" if "Transfermarkt" in current_text else ""
    if start_prob >= 0.90:
        return "sikker starter" + suffix
    if start_prob >= 0.70:
        return "sandsynlig starter" + suffix
    return "rotation/usikker" + suffix


def apply_pool_overrides(
    players: list[dict[str, Any]], overrides: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]], list[dict[str, Any]]]:
    by_id = {text(player.get("player_id")): player for player in players}
    changes: dict[str, tuple[float, float]] = {}
    audit = []
    for override in overrides:
        player_id = text(override["player_id"])
        player = by_id.get(player_id)
        if not player:
            audit.append(
                {
                    "player_id": player_id,
                    "player_name": text(override.get("player_name")),
                    "team_id": text(override.get("team_id")),
                    "status": "inactive_not_in_player_pool",
                }
            )
            continue
        old_start = number(player.get("start_prob")) or 0.0
        new_start = number(override.get("start_prob"))
        new_conditional = number(override.get("conditional_start_prob"))
        if new_start is None:
            new_start = old_start
        if new_conditional is None:
            new_conditional = number(player.get("conditional_start_prob")) or new_start

        player["start_prob"] = round(new_start, 6)
        player["conditional_start_prob"] = round(new_conditional, 6)
        player["start_prob_source"] = SOURCE
        player["start_security"] = round(new_start, 6)
        player["start_probability_pct"] = round(new_start * 100, 1)
        player["start_pct"] = round(new_start * 100, 1)
        player["start_status"] = start_status(new_start, player.get("start_status"))
        for field in POOL_CONTEXT_FIELDS:
            if text(override.get(field)):
                value = number(override[field])
                player[field] = round(value, 6) if value is not None else text(override[field])
        if text(override.get("source_note")):
            player["source_note"] = text(override["source_note"])

        changes[player_id] = (old_start, new_start)
        audit.append(
            {
                "player_id": player_id,
                "player_name": text(player.get("player_name")),
                "team_id": text(player.get("team_id")),
                "old_start": old_start,
                "new_start": new_start,
                "conditional": new_conditional,
                "status": "applied",
            }
        )
    return players, changes, audit


def ev_scale(old_start: float, new_start: float) -> float:
    if new_start <= 0:
        return 0.0
    if old_start <= 0:
        return 1.0
    return max(MIN_EV_SCALE_DOWN, min(MAX_EV_SCALE_UP, new_start / old_start))


def apply_ev_overrides(
    fields: list[str],
    rows: list[dict[str, str]],
    overrides: list[dict[str, str]],
    changes: dict[str, tuple[float, float]],
) -> tuple[list[str], list[dict[str, Any]]]:
    by_id = {text(row.get("player_id")): row for row in rows}
    missing = [player_id for player_id in changes if player_id not in by_id]
    if missing:
        raise ValueError(f"Overrides not found in EV CSV: {', '.join(missing)}")
    override_by_id = {text(row["player_id"]): row for row in overrides}

    for player_id, (old_start, new_start) in changes.items():
        row = by_id[player_id]
        override = override_by_id[player_id]
        scale = ev_scale(old_start, new_start)
        start_changed = abs(old_start - new_start) > 0.000001
        if start_changed:
            for field in SCALE_FIELDS:
                value = number(row.get(field))
                if value is not None:
                    row[field] = fmt(value * scale)
            base = (
                number(row.get("model_ev_before_price_quality"))
                or number(row.get("weighted_group_stage_ev_before_price_quality"))
                or 0.0
            )
            raw_price_quality = number(row.get("price_quality_raw_ev")) or 0.0
            price_quality = reserve_safe_price_quality(raw_price_quality, base, new_start)
            final_ev = formula_expected(base, price_quality)
            row["price_quality_appearance_scaled_ev"] = fmt(
                raw_price_quality * max(0.0, min(1.0, new_start / 0.70))
            )
            row["price_quality_base_capped_ev"] = fmt(
                min(raw_price_quality, max(0.15, 1.50 * max(base, 0.0)))
            )
            row["price_quality_ev"] = fmt(price_quality)
            row["weighted_group_stage_ev"] = fmt(final_ev)
            row["optimizer_ev"] = fmt(final_ev)
            row["price_quality_method"] = (
                "raw_for_likely_starter"
                if new_start >= 0.70
                else "appearance_scaled_then_base_capped"
            )
        row["start_prob"] = fmt(new_start)
        row["conditional_start_prob"] = fmt(
            number(override.get("conditional_start_prob"))
            or number(row.get("conditional_start_prob"))
            or new_start
        )
        row["start_prob_source"] = SOURCE
        row["start_security"] = fmt(new_start)
        row["start_probability_pct"] = fmt(new_start * 100)
        row["start_status"] = start_status(new_start, row.get("start_status"))
        for field in EV_CONTEXT_FIELDS:
            if text(override.get(field)):
                row[field] = text(override[field])

    for field in EV_CONTEXT_FIELDS + ["start_security", "start_probability_pct", "start_status"]:
        if any(text(row.get(field)) for row in rows) and field not in fields:
            fields.append(field)
    return fields, rows


def sync_start_security(
    fields: list[str], rows: list[dict[str, str]], players: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    pool_by_id = {text(player.get("player_id")): player for player in players}
    for field in [
        "start_probability_pct",
        "start_security",
        "start_prob",
        "start_prob_source",
        "start_status",
        "conditional_start_prob",
        "appearance_prob",
        "availability_prob",
        "availability_risk",
        "availability_status",
    ]:
        if field not in fields:
            fields.append(field)
    for row in rows:
        player = pool_by_id.get(text(row.get("player_id")))
        if not player:
            continue
        for field in fields:
            if field in player and player[field] is not None:
                row[field] = player[field]
    return fields, rows


def run_checks(
    players: list[dict[str, Any]],
    ev_rows: list[dict[str, Any]],
    security_rows: list[dict[str, Any]],
) -> None:
    pool_by_id = {text(row.get("player_id")): row for row in players}
    ev_by_id = {text(row.get("player_id")): row for row in ev_rows}
    security_by_id = {text(row.get("player_id")): row for row in security_rows}
    stale = []
    for player_id, player in pool_by_id.items():
        security = security_by_id.get(player_id)
        if not security:
            continue
        if abs((number(player.get("start_prob")) or 0.0) - (number(security.get("start_prob")) or 0.0)) > 0.0001:
            stale.append(player_id)
    if stale:
        raise RuntimeError(f"Stale frontend start percentages remain: {', '.join(stale[:10])}")

    formula_mismatches = [
        text(row.get("player_id"))
        for row in ev_rows
        if abs(formula_diff(row)) > FORMULA_TOLERANCE
    ]
    if formula_mismatches:
        raise RuntimeError(f"Price-quality formula mismatches remain: {', '.join(formula_mismatches[:10])}")

    final_ev_fields = ("optimizer_ev", "weighted_group_stage_ev", "price_quality_ev")
    mismatches = []
    for player_id, ev in ev_by_id.items():
        pool = pool_by_id.get(player_id)
        if not pool:
            continue
        for field in final_ev_fields:
            if abs((number(pool.get(field)) or 0.0) - (number(ev.get(field)) or 0.0)) > 0.0001:
                mismatches.append(f"{player_id}:{field}")
    if mismatches:
        raise RuntimeError(f"Player pool/EV mismatches remain: {', '.join(mismatches[:10])}")


def print_audit(players: list[dict[str, Any]]) -> None:
    by_id = {text(player.get("player_id")): player for player in players}
    print("\nKey-player audit")
    print("----------------")
    for player_id in KEY_PLAYER_IDS:
        player = by_id.get(player_id)
        if not player:
            print(f"MISSING {player_id}")
            continue
        print(
            f"{text(player.get('player_name')):<24} {text(player.get('team_id')):<3} "
            f"start={number(player.get('start_prob')) or 0:.4f} "
            f"cond={number(player.get('conditional_start_prob')) or 0:.4f} "
            f"EV={number(player.get('optimizer_ev')) or 0:.4f} "
            f"source={text(player.get('start_prob_source'))}"
        )


def main() -> int:
    overrides = load_overrides()
    pool_rows = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(pool_rows, list):
        raise ValueError("player_pool_v1.json must contain a list")
    ev_fields, ev_rows = read_csv(EV_PATH)
    security_fields, security_rows = read_csv(SECURITY_PATH)
    require_unique_ids(pool_rows, "Player pool")
    require_unique_ids(ev_rows, "EV CSV")
    require_unique_ids(security_rows, "Start-security CSV")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_outputs = [
        DATA / "optimal_squads_by_strategy.json",
        DATA / "strategy_comparison_report.csv",
        DATA / "strategy_formation_comparison_report.csv",
        DATA / "current_strategy_context.json",
        DATA / "strategy_cleanup_report.md",
    ]
    backups = backup_files([POOL_PATH, EV_PATH, SECURITY_PATH] + optimizer_outputs, stamp)

    pool_rows, changes, audit = apply_pool_overrides(pool_rows, overrides)
    ev_fields, ev_rows = apply_ev_overrides(ev_fields, ev_rows, overrides, changes)
    write_csv_atomic(EV_PATH, ev_fields, ev_rows)
    write_json_atomic(POOL_PATH, pool_rows)

    sync_result = sync_final_ev_to_pool()
    pool_rows = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    security_fields, security_rows = sync_start_security(security_fields, security_rows, pool_rows)
    write_csv_atomic(SECURITY_PATH, security_fields, security_rows)
    run_checks(pool_rows, ev_rows, security_rows)

    subprocess.run([sys.executable, str(ROOT / "tools" / "optimize_squad_group_stage.py")], cwd=ROOT, check=True)

    print("Safe context start override synchronization")
    print("-------------------------------------------")
    applied = sum(row.get("status") == "applied" for row in audit)
    inactive = [row for row in audit if row.get("status") != "applied"]
    print(f"Overrides applied: {applied}")
    print(f"Inactive overrides retained in source CSV: {len(inactive)}")
    for row in inactive:
        print(f"  inactive: {row['player_id']} ({row['player_name']}, {row['team_id']})")
    print(f"Final EV exact-ID sync matches: {sync_result['matched']}")
    print(f"Backups created: {len(backups)}")
    print("Sanity checks: frontend start percentages OK; price-quality consistency OK; optimizer OK")
    print_audit(pool_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
