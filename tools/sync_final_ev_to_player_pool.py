from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from json_file_safety import write_json_strict


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"
AUDIT_CSV = DATA / "player_pool_vs_final_ev_audit.csv"
AUDIT_MD = DATA / "player_pool_vs_final_ev_audit.md"

FINAL_EV_FIELDS = [
    "optimizer_ev",
    "weighted_group_stage_ev",
    "weighted_group_stage_ev_before_price_quality",
    "price_quality_ev",
    "model_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "price_quality_raw_ev",
    "price_quality_appearance_scaled_ev",
    "price_quality_base_capped_ev",
    "price_quality_weight",
    "price_quality_spread_multiplier",
    "price_quality_applied",
    "price_quality_method",
    "base_ev_source",
]

NUMERIC_FIELDS = {
    "optimizer_ev",
    "weighted_group_stage_ev",
    "weighted_group_stage_ev_before_price_quality",
    "price_quality_ev",
    "model_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "price_quality_raw_ev",
    "price_quality_appearance_scaled_ev",
    "price_quality_base_capped_ev",
    "price_quality_weight",
    "price_quality_spread_multiplier",
}

SANITY_IDS = {
    "unai_sim_n__esp",
    "jules_kound__fra",
    "leo_pereira__bra",
    "patrick_agyemang__usa",
    "noni_madueke__eng",
    "joan_garcia__esp",
}
SANITY_NAMES = {
    "Unai Simon",
    "Jules Kounde",
    "Jules Koundé",
    "Leo Pereira",
    "Patrick Agyemang",
    "Noni Madueke",
    "Joan Garcia",
}


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def number(value: Any) -> float | None:
    raw = text(value).replace(",", ".")
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def normalized_value(field: str, value: Any) -> Any:
    if field in NUMERIC_FIELDS:
        return number(value)
    if field == "price_quality_applied":
        return text(value).casefold() in {"true", "1", "yes", "ja"}
    return value if value is not None else ""


def values_match(field: str, left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if field in NUMERIC_FIELDS:
        a = number(left)
        b = number(right)
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) <= tolerance
    return normalized_value(field, left) == normalized_value(field, right)


def duplicate_ids(rows: list[dict[str, Any]]) -> set[str]:
    counts = Counter(text(row.get("player_id")) for row in rows if text(row.get("player_id")))
    return {player_id for player_id, count in counts.items() if count > 1}


def mismatch_counts(
    pool_rows: list[dict[str, Any]],
    ev_by_id: dict[str, dict[str, Any]],
    blocked_ids: set[str],
) -> dict[str, int]:
    counts = {field: 0 for field in FINAL_EV_FIELDS}
    for player in pool_rows:
        player_id = text(player.get("player_id"))
        ev = ev_by_id.get(player_id)
        if not ev or player_id in blocked_ids:
            continue
        for field in FINAL_EV_FIELDS:
            if not values_match(field, player.get(field), ev.get(field)):
                counts[field] += 1
    return counts


def optimizer_threshold_counts(
    pool_rows: list[dict[str, Any]],
    ev_by_id: dict[str, dict[str, Any]],
    blocked_ids: set[str],
) -> tuple[int, int]:
    over_001 = 0
    over_010 = 0
    for player in pool_rows:
        player_id = text(player.get("player_id"))
        ev = ev_by_id.get(player_id)
        if not ev or player_id in blocked_ids:
            continue
        pool_value = number(player.get("optimizer_ev")) or 0.0
        ev_value = number(ev.get("optimizer_ev")) or 0.0
        diff = abs(pool_value - ev_value)
        over_001 += diff > 0.001
        over_010 += diff > 0.10
    return over_001, over_010


def write_audit(
    audit_rows: list[dict[str, Any]],
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    before_thresholds: tuple[int, int],
    after_thresholds: tuple[int, int],
    pool_duplicates: set[str],
    ev_duplicates: set[str],
    unmatched_pool: int,
    unmatched_ev: int,
    backup: Path,
) -> None:
    fields = [
        "player_id",
        "player_name",
        "team_id",
        "position",
        "match_status",
        "field",
        "pool_value_before",
        "ev_value",
        "pool_value_after",
        "diff_before",
        "diff_after",
        "repair_status",
    ]
    with AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    sanity = [
        row
        for row in audit_rows
        if row["field"] == "optimizer_ev"
        and (row["player_id"] in SANITY_IDS or row["player_name"] in SANITY_NAMES)
    ]
    lines = [
        "# Player pool vs final EV audit",
        "",
        "## Autoritet og retning",
        "",
        "- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.",
        "- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.",
        "- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.",
        "",
        "## Før/efter",
        "",
        f"- `optimizer_ev` forskel > 0.001: {before_thresholds[0]} -> {after_thresholds[0]}",
        f"- `optimizer_ev` forskel > 0.10: {before_thresholds[1]} -> {after_thresholds[1]}",
        f"- Dublerede player_id i pool: {len(pool_duplicates)}",
        f"- Dublerede player_id i EV: {len(ev_duplicates)}",
        f"- Pool-rækker uden exact EV-match: {unmatched_pool}",
        f"- EV-rækker uden exact pool-match: {unmatched_ev}",
        f"- Backup: `{backup.relative_to(ROOT)}`",
        "",
        "## Feltmismatches",
        "",
        "| Felt | Før | Efter |",
        "| --- | ---: | ---: |",
    ]
    for field in FINAL_EV_FIELDS:
        lines.append(f"| {field} | {before_counts[field]} | {after_counts[field]} |")

    lines.extend(
        [
            "",
            "## Sanity",
            "",
            "| Spiller | Hold | Pool før | EV | Pool efter | Status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sanity:
        lines.append(
            f"| {row['player_name']} | {row['team_id']} | {row['pool_value_before']} | "
            f"{row['ev_value']} | {row['pool_value_after']} | {row['repair_status']} |"
        )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_final_ev_to_pool() -> dict[str, Any]:
    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        ev_rows = list(csv.DictReader(handle))
    pool_rows = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(pool_rows, list):
        raise ValueError("player_pool_v1.json skal være en liste")

    missing_fields = [field for field in FINAL_EV_FIELDS if field not in (ev_rows[0] if ev_rows else {})]
    if missing_fields:
        raise ValueError(f"EV-filen mangler autoritative slutfelter: {', '.join(missing_fields)}")

    pool_duplicates = duplicate_ids(pool_rows)
    ev_duplicates = duplicate_ids(ev_rows)
    blocked_ids = pool_duplicates | ev_duplicates
    ev_by_id = {
        text(row.get("player_id")): row
        for row in ev_rows
        if text(row.get("player_id")) and text(row.get("player_id")) not in blocked_ids
    }
    pool_ids = {text(row.get("player_id")) for row in pool_rows if text(row.get("player_id"))}
    ev_ids = {text(row.get("player_id")) for row in ev_rows if text(row.get("player_id"))}

    before_counts = mismatch_counts(pool_rows, ev_by_id, blocked_ids)
    before_thresholds = optimizer_threshold_counts(pool_rows, ev_by_id, blocked_ids)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = POOL_PATH.with_name(f"player_pool_v1.backup_before_final_ev_sync_{timestamp}.json")
    shutil.copy2(POOL_PATH, backup)

    audit_rows = []
    matched = 0
    for player in pool_rows:
        player_id = text(player.get("player_id"))
        ev = ev_by_id.get(player_id)
        if player_id in blocked_ids:
            match_status = "blocked_duplicate_player_id"
        elif not ev:
            match_status = "no_exact_ev_match"
        else:
            match_status = "exact_player_id"
            matched += 1

        if not ev:
            continue
        for field in FINAL_EV_FIELDS:
            before = player.get(field)
            authoritative = normalized_value(field, ev.get(field))
            before_num = number(before)
            ev_num = number(authoritative)
            diff_before = (
                abs((before_num or 0.0) - (ev_num or 0.0))
                if field in NUMERIC_FIELDS
                else (0 if values_match(field, before, authoritative) else 1)
            )
            player[field] = authoritative
            diff_after = 0
            audit_rows.append(
                {
                    "player_id": player_id,
                    "player_name": text(player.get("player_name")),
                    "team_id": text(player.get("team_id")),
                    "position": text(player.get("position")),
                    "match_status": match_status,
                    "field": field,
                    "pool_value_before": text(before),
                    "ev_value": text(authoritative),
                    "pool_value_after": text(player.get(field)),
                    "diff_before": f"{diff_before:.6f}",
                    "diff_after": f"{diff_after:.6f}",
                    "repair_status": "updated" if not values_match(field, before, authoritative) else "already_equal",
                }
            )

    write_json_strict(POOL_PATH, pool_rows)
    after_counts = mismatch_counts(pool_rows, ev_by_id, blocked_ids)
    after_thresholds = optimizer_threshold_counts(pool_rows, ev_by_id, blocked_ids)
    write_audit(
        audit_rows,
        before_counts,
        after_counts,
        before_thresholds,
        after_thresholds,
        pool_duplicates,
        ev_duplicates,
        len(pool_ids - ev_ids),
        len(ev_ids - pool_ids),
        backup,
    )
    return {
        "matched": matched,
        "blocked_duplicates": len(blocked_ids),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_thresholds": before_thresholds,
        "after_thresholds": after_thresholds,
        "backup": backup,
    }


def main() -> int:
    result = sync_final_ev_to_pool()
    print("Final EV -> player pool synchronization")
    print("----------------------------------------")
    print(f"Exact player_id matches: {result['matched']}")
    print(f"Blocked duplicate IDs: {result['blocked_duplicates']}")
    print(f"optimizer_ev diff > 0.001: {result['before_thresholds'][0]} -> {result['after_thresholds'][0]}")
    print(f"optimizer_ev diff > 0.10: {result['before_thresholds'][1]} -> {result['after_thresholds'][1]}")
    print(f"Backup: {result['backup'].relative_to(ROOT)}")
    print(f"Wrote: {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {AUDIT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
