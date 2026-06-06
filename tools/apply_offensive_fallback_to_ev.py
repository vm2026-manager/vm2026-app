from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
DATA = ROOT / "data"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_offensive_fallback_production_dry_run as dry_run
from sync_final_ev_to_player_pool import (
    FINAL_EV_FIELDS,
    mismatch_counts,
    sync_final_ev_to_pool,
)


EV_PATH = DATA / "player_ev_group_stage_v1.csv"
POOL_PATH = DATA / "player_pool_v1.json"
EXPERIMENT_PATH = DATA / "offensive_share_fallback_experiment.csv"
PRODUCTION_STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
OUT_CSV = DATA / "offensive_fallback_production_audit.csv"
OUT_MD = DATA / "offensive_fallback_production_audit.md"

MAX_FALLBACK_PLAYERS = 45
MAX_LIFTS_OVER_ONE = 5
MAX_FALLBACK_PER_STRATEGY = 2
MAX_MATCH_SUM_ERROR = 1e-6
FORMULA_TOLERANCE = 0.001

PROVENANCE_FIELDS = [
    "offensive_fallback_applied",
    "offensive_fallback_source",
    "offensive_fallback_variant_a_base_ev",
    "offensive_fallback_variant_b_base_ev",
    "offensive_fallback_hybrid_base_ev",
    "offensive_fallback_base_ev_lift",
    "offensive_fallback_cap_reason",
    "offensive_fallback_confidence",
    "offensive_fallback_reason_flags",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return txt(value).casefold() in {"true", "1", "yes", "ja"}


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_ev() -> tuple[list[str], list[dict[str, str]]]:
    with EV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_ev_temp(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    output_fields = list(fields)
    for field in PROVENANCE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=output_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def duplicate_ids(rows: list[dict[str, Any]]) -> set[str]:
    counts = Counter(txt(row.get("player_id")) for row in rows if txt(row.get("player_id")))
    return {player_id for player_id, count in counts.items() if count > 1}


def all_shares_missing(row: dict[str, Any]) -> bool:
    return all(
        not txt(row.get(column))
        or txt(row.get(column)).casefold() in {"nan", "na", "null"}
        for column in ("goal_share_norm", "assist_share_norm", "sot_share_norm")
    )


def all_shares_present(row: dict[str, Any]) -> bool:
    return all(
        txt(row.get(column))
        and txt(row.get(column)).casefold() not in {"nan", "na", "null"}
        for column in ("goal_share_norm", "assist_share_norm", "sot_share_norm")
    )


def eligible_source_row(row: dict[str, Any]) -> bool:
    return (
        txt(row.get("position")).upper() in {"MID", "FWD"}
        and number(row.get("start_prob")) >= 0.70
        and all_shares_missing(row)
        and txt(row.get("round_context_source"))
        == "distributed_from_existing_optimizer_ev"
    )


def mark_provenance(
    proposed_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    candidate_meta: dict[str, dict[str, Any]],
) -> None:
    original_by_id = {txt(row.get("player_id")): row for row in original_rows}
    for row in proposed_rows:
        player_id = txt(row.get("player_id"))
        meta = candidate_meta.get(player_id)
        if not meta or not meta["fallback_applied"]:
            continue
        original = original_by_id[player_id]
        row["offensive_fallback_applied"] = "True"
        row["offensive_fallback_source"] = "hybrid_variant_a_with_variant_b_team_cap_v1"
        row["offensive_fallback_variant_a_base_ev"] = fmt(
            number(meta["variant_a_base_ev"])
        )
        row["offensive_fallback_variant_b_base_ev"] = fmt(
            number(meta["variant_b_base_ev"])
        )
        row["offensive_fallback_hybrid_base_ev"] = fmt(
            number(meta["hybrid_base_ev"])
        )
        row["offensive_fallback_base_ev_lift"] = fmt(
            number(meta["hybrid_base_ev"]) - number(meta["current_base_ev"])
        )
        row["offensive_fallback_cap_reason"] = txt(meta["cap_reason"])
        row["offensive_fallback_confidence"] = txt(meta["confidence"])
        row["offensive_fallback_reason_flags"] = txt(meta["reason_flags"])
        if not eligible_source_row(original):
            raise ValueError(
                f"Fallback provenance attempted for ineligible player {player_id}"
            )


def formula_stats(rows: list[dict[str, Any]]) -> tuple[int, float]:
    mismatches = 0
    max_diff = 0.0
    for row in rows:
        base = number(
            row.get("model_ev_before_price_quality")
            or row.get("weighted_group_stage_ev_before_price_quality")
        )
        price_quality = number(row.get("price_quality_ev"))
        actual = number(row.get("optimizer_ev"))
        diff = abs(actual - (0.55 * base + 0.45 * price_quality))
        max_diff = max(max_diff, diff)
        mismatches += diff > FORMULA_TOLERANCE
    return mismatches, max_diff


def match_sum_error(
    rows: list[dict[str, Any]],
    applied_ids: set[str],
) -> float:
    maximum = 0.0
    for row in rows:
        if txt(row.get("player_id")) not in applied_ids:
            continue
        match_sum = sum(
            number(row.get(f"match_{match_no}_weighted_match_ev"))
            for match_no in (1, 2, 3)
        )
        base = number(row.get("weighted_group_stage_ev_before_price_quality"))
        maximum = max(maximum, abs(match_sum - base))
    return maximum


def simulate_pool_mismatches(
    proposed_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> tuple[int, set[str], set[str]]:
    ev_duplicates = duplicate_ids(proposed_rows)
    pool_duplicates = duplicate_ids(pool_rows)
    blocked = ev_duplicates | pool_duplicates
    ev_by_id = {
        txt(row.get("player_id")): row
        for row in proposed_rows
        if txt(row.get("player_id")) and txt(row.get("player_id")) not in blocked
    }
    simulated = [dict(player) for player in pool_rows]
    for player in simulated:
        player_id = txt(player.get("player_id"))
        ev = ev_by_id.get(player_id)
        if not ev:
            continue
        for field in FINAL_EV_FIELDS:
            player[field] = ev.get(field)
    counts = mismatch_counts(simulated, ev_by_id, blocked)
    return sum(counts.values()), ev_duplicates, pool_duplicates


def build_audit_rows(
    original_rows: list[dict[str, Any]],
    proposed_rows: list[dict[str, Any]],
    candidate_meta: dict[str, dict[str, Any]],
    match_changes: dict[str, dict[int, tuple[float, float]]],
) -> list[dict[str, Any]]:
    player_rows = dry_run.build_player_report(
        original_rows, proposed_rows, candidate_meta, match_changes
    )
    original_by_id = {txt(row.get("player_id")): row for row in original_rows}
    for row in player_rows:
        source = original_by_id[row["player_id"]]
        row["shares_status"] = (
            "already_shares_excluded"
            if all_shares_present(source)
            else "missing_shares"
        )
        row["round_context_source"] = txt(source.get("round_context_source"))
    return player_rows


def stop_checks(
    original_rows: list[dict[str, Any]],
    proposed_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    candidate_meta: dict[str, dict[str, Any]],
    optimizer_comparison: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    applied_ids = {
        player_id
        for player_id, meta in candidate_meta.items()
        if meta["fallback_applied"]
    }
    formula_mismatches, max_formula_diff = formula_stats(proposed_rows)
    max_sum_error = match_sum_error(proposed_rows, applied_ids)
    pool_mismatches, ev_duplicates, pool_duplicates = simulate_pool_mismatches(
        proposed_rows, pool_rows
    )
    lowered = sum(
        row["fallback_applied"] == "yes" and number(row["base_ev_lift"]) < -1e-9
        for row in audit_rows
    )
    existing_share_applied = sum(
        row["fallback_applied"] == "yes"
        and row["shares_status"] == "already_shares_excluded"
        for row in audit_rows
    )
    lifts_over_one = sum(
        row["fallback_applied"] == "yes"
        and number(row["optimizer_ev_lift"]) > 1.0
        for row in audit_rows
    )
    max_strategy_fallback = max(
        (int(row["fallback_player_count"]) for row in optimizer_comparison),
        default=0,
    )

    checks = [
        {
            "criterion": "fallback_players_le_45",
            "actual": len(applied_ids),
            "limit": MAX_FALLBACK_PLAYERS,
            "passed": len(applied_ids) <= MAX_FALLBACK_PLAYERS,
        },
        {
            "criterion": "optimizer_lifts_over_1_le_5",
            "actual": lifts_over_one,
            "limit": MAX_LIFTS_OVER_ONE,
            "passed": lifts_over_one <= MAX_LIFTS_OVER_ONE,
        },
        {
            "criterion": "fallback_per_strategy_le_2",
            "actual": max_strategy_fallback,
            "limit": MAX_FALLBACK_PER_STRATEGY,
            "passed": max_strategy_fallback <= MAX_FALLBACK_PER_STRATEGY,
        },
        {
            "criterion": "price_quality_formula_mismatches_eq_0",
            "actual": formula_mismatches,
            "limit": 0,
            "passed": formula_mismatches == 0,
        },
        {
            "criterion": "prospective_pool_ev_mismatches_eq_0",
            "actual": pool_mismatches,
            "limit": 0,
            "passed": pool_mismatches == 0,
        },
        {
            "criterion": "match_sum_error_le_1e_6",
            "actual": max_sum_error,
            "limit": MAX_MATCH_SUM_ERROR,
            "passed": max_sum_error <= MAX_MATCH_SUM_ERROR,
        },
        {
            "criterion": "lowered_players_eq_0",
            "actual": lowered,
            "limit": 0,
            "passed": lowered == 0,
        },
        {
            "criterion": "existing_share_players_applied_eq_0",
            "actual": existing_share_applied,
            "limit": 0,
            "passed": existing_share_applied == 0,
        },
        {
            "criterion": "duplicate_ev_ids_eq_0",
            "actual": len(ev_duplicates),
            "limit": 0,
            "passed": not ev_duplicates,
        },
        {
            "criterion": "duplicate_pool_ids_eq_0",
            "actual": len(pool_duplicates),
            "limit": 0,
            "passed": not pool_duplicates,
        },
    ]
    stats = {
        "applied_ids": applied_ids,
        "existing_share_players_checked": sum(
            all_shares_present(row) for row in original_rows
        ),
        "existing_share_players_applied": existing_share_applied,
        "formula_mismatches": formula_mismatches,
        "max_formula_diff": max_formula_diff,
        "max_match_sum_error": max_sum_error,
        "pool_mismatches": pool_mismatches,
        "lifts_over_one": lifts_over_one,
        "max_strategy_fallback": max_strategy_fallback,
    }
    return checks, stats


def write_audit(
    audit_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    stats: dict[str, Any],
    optimizer_comparison: list[dict[str, Any]],
    backups: dict[str, Path],
    commit_status: str,
) -> None:
    write_fields = list(audit_rows[0]) if audit_rows else []
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    applied = [row for row in audit_rows if row["fallback_applied"] == "yes"]
    top = sorted(applied, key=lambda row: number(row["optimizer_ev_lift"]), reverse=True)
    sanity_norms = {dry_run.norm(name) for name in dry_run.SANITY_NAMES}
    sanity = [
        row
        for row in audit_rows
        if dry_run.norm(row["player_name"]) in sanity_norms
    ]
    lines = [
        "# Offensive Fallback Production Audit",
        "",
        f"- Commit-status: **{commit_status}**",
        f"- Kandidater: {len([row for row in audit_rows if row['fallback_candidate'] == 'yes'])}",
        f"- Faktisk ændrede spillere: {len(applied)}",
        f"- Spillere med eksisterende shares kontrolleret/udeladt: {stats['existing_share_players_checked']}",
        f"- Spillere med eksisterende shares fejlagtigt ændret: {stats['existing_share_players_applied']}",
        f"- Base-EV-løft >= 0,25: {sum(number(row['base_ev_lift']) >= 0.25 for row in applied)}",
        f"- Optimizer-EV-løft > 0,25: {sum(number(row['optimizer_ev_lift']) > 0.25 for row in applied)}",
        f"- Optimizer-EV-løft > 1,00: {stats['lifts_over_one']}",
        f"- Maksimal kampfordelings-sumfejl: {stats['max_match_sum_error']:.12g}",
        f"- Price-quality-formelmismatches: {stats['formula_mismatches']}",
        f"- Maksimal price-quality-formeldifference: {stats['max_formula_diff']:.12g}",
        f"- Pool/EV-finalmismatches efter simuleret/udført sync: {stats['pool_mismatches']}",
        "",
        "## Stopkriterier",
        "",
        "| Kriterium | Faktisk | Grænse | Bestået |",
        "| --- | ---: | ---: | --- |",
    ]
    for check in checks:
        lines.append(
            f"| {check['criterion']} | {check['actual']} | {check['limit']} | "
            f"{'ja' if check['passed'] else 'nej'} |"
        )
    lines.extend(
        [
            "",
            "## Backups",
            "",
            *[
                f"- {name}: `{path.relative_to(ROOT)}`"
                for name, path in backups.items()
            ],
            "",
            "## Top 30 løft",
            "",
            *dry_run.markdown_table(
                top,
                [
                    "player_name",
                    "team_id",
                    "position",
                    "start_prob",
                    "current_base_ev",
                    "hybrid_base_ev",
                    "base_ev_lift",
                    "current_optimizer_ev",
                    "dry_run_optimizer_ev",
                    "optimizer_ev_lift",
                    "cap_reason",
                    "confidence",
                ],
                30,
            ),
            "",
            "## Sanity",
            "",
            *dry_run.markdown_table(
                sanity,
                [
                    "player_name",
                    "shares_status",
                    "fallback_candidate",
                    "fallback_applied",
                    "current_base_ev",
                    "hybrid_base_ev",
                    "current_optimizer_ev",
                    "dry_run_optimizer_ev",
                    "cap_reason",
                ],
            ),
            "",
            "## Optimizer før/efter",
            "",
            *dry_run.markdown_table(
                optimizer_comparison,
                [
                    "strategy",
                    "formation_before",
                    "formation_after",
                    "price_before",
                    "price_after",
                    "ev_before",
                    "ev_after",
                    "score_before",
                    "score_after",
                    "players_in",
                    "players_out",
                    "fallback_player_count",
                    "fallback_players",
                    "high_risk_before",
                    "high_risk_after",
                ],
            ),
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    fields, original_rows = read_ev()
    pool_rows = json.loads(POOL_PATH.read_text(encoding="utf-8-sig"))
    experiments = pd.read_csv(EXPERIMENT_PATH, low_memory=False)

    already_applied = {
        txt(row.get("player_id"))
        for row in original_rows
        if truthy(row.get("offensive_fallback_applied"))
    }
    if already_applied:
        raise RuntimeError(
            "Offensiv fallback er allerede anvendt på "
            f"{len(already_applied)} spillere; scriptet stopper for at undgå dobbeltløft."
        )

    proposed_rows, match_changes, candidate_meta = dry_run.price_quality_dry_run(
        original_rows, experiments
    )
    mark_provenance(proposed_rows, original_rows, candidate_meta)
    audit_rows = build_audit_rows(
        original_rows, proposed_rows, candidate_meta, match_changes
    )
    applied_ids = {
        player_id
        for player_id, meta in candidate_meta.items()
        if meta["fallback_applied"]
    }
    dry_players = dry_run.build_optimizer_players(proposed_rows, candidate_meta)
    dry_results = dry_run.run_optimizer_dry(dry_players, applied_ids)
    production_results = json.loads(
        PRODUCTION_STRATEGIES_PATH.read_text(encoding="utf-8-sig")
    )
    optimizer_comparison = dry_run.strategy_comparison(
        production_results, dry_results, applied_ids
    )
    checks, stats = stop_checks(
        original_rows,
        proposed_rows,
        audit_rows,
        candidate_meta,
        optimizer_comparison,
        pool_rows,
    )
    failed = [check for check in checks if not check["passed"]]
    if failed:
        write_audit(
            audit_rows,
            checks,
            stats,
            optimizer_comparison,
            {},
            "blocked_preflight_no_production_files_changed",
        )
        raise RuntimeError(
            "Stopkriterier brudt: "
            + ", ".join(check["criterion"] for check in failed)
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ev_backup = EV_PATH.with_name(
        f"player_ev_group_stage_v1.backup_before_offensive_fallback_{timestamp}.csv"
    )
    pool_backup = POOL_PATH.with_name(
        f"player_pool_v1.backup_before_offensive_fallback_{timestamp}.json"
    )
    temp_path = EV_PATH.with_suffix(".offensive_fallback.tmp")
    shutil.copy2(EV_PATH, ev_backup)
    shutil.copy2(POOL_PATH, pool_backup)
    backups = {"EV": ev_backup, "player_pool": pool_backup}

    try:
        write_ev_temp(temp_path, fields, proposed_rows)
        temp_path.replace(EV_PATH)
        sync_result = sync_final_ev_to_pool()
        stats["pool_mismatches"] = sum(sync_result["after_counts"].values())
        if stats["pool_mismatches"] != 0:
            raise RuntimeError(
                f"Pool/EV mismatches efter sync: {stats['pool_mismatches']}"
            )
        write_audit(
            audit_rows,
            checks,
            stats,
            optimizer_comparison,
            backups,
            "committed_all_preflight_checks_passed",
        )
    except Exception:
        shutil.copy2(ev_backup, EV_PATH)
        shutil.copy2(pool_backup, POOL_PATH)
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print("Offensive fallback production integration")
    print("-----------------------------------------")
    print(f"Candidates: {len(candidate_meta)}")
    print(f"Applied fallback players: {len(applied_ids)}")
    print(f"Optimizer lifts > 1.00: {stats['lifts_over_one']}")
    print(f"Max match sum error: {stats['max_match_sum_error']:.12g}")
    print(f"Price-quality formula mismatches: {stats['formula_mismatches']}")
    print(f"Pool/EV mismatches after sync: {stats['pool_mismatches']}")
    print(f"EV backup: {ev_backup.relative_to(ROOT)}")
    print(f"Pool backup: {pool_backup.relative_to(ROOT)}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
