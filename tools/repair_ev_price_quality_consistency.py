from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
OUT_CSV = DATA / "player_ev_price_quality_consistency_audit.csv"
OUT_MD = DATA / "player_ev_price_quality_consistency_audit.md"
PRICE_DIAG_PATH = DATA / "price_quality_ev_diagnostics.csv"

PRICE_QUALITY_WEIGHT = 0.45
PRICE_QUALITY_SPREAD_MULTIPLIER = 1.35
FORMULA_TOLERANCE = 0.001

BAUMGARTNER_ID = "christoph_baumgartner__aut"

BASE_COMPONENTS = [
    "goal_ev",
    "assist_ev",
    "shots_on_target_ev",
    "clean_sheet_ev",
    "card_ev",
    "result_ev",
    "team_scores_ev",
    "opponent_scores_ev",
    "on_pitch_ev",
    "start_minutes_ev",
]

ZERO_EV_COLUMNS = [
    "weighted_group_stage_ev",
    "optimizer_ev",
    "total_ev_group_stage",
    "model_ev_before_price_quality",
    "weighted_group_stage_ev_before_price_quality",
    "optimizer_ev_before_price_quality",
    "price_quality_ev",
    "price_quality_weight",
    "price_quality_spread_multiplier",
]

AUDIT_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "component_weighted_sum",
    "base_ev_before_price_quality",
    "price_quality_ev",
    "expected_final_ev",
    "actual_final_ev",
    "formula_diff",
    "component_source",
    "base_ev_source",
    "repair_status",
]

SANITY_NAMES = {
    "Erling Haaland",
    "Harry Kane",
    "Antonio Nusa",
    "Alexander Sørloth",
    "Martin Ødegaard",
    "Donyell Malen",
    "Kerim Alajbegovic",
    "Kylian Mbappe",
    "Raphinha",
    "Jules Kounde",
    "Manuel Neuer",
    "Christoph Baumgartner",
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    return txt(value).casefold() in {"true", "1", "yes", "ja"}


def get_price_m(row: dict[str, Any]) -> float:
    for col in ["price_m", "price_estimate_m", "price_mio"]:
        value = to_float(row.get(col), -1.0)
        if value >= 0:
            return value
    for col in ["price", "price_estimate", "holdet_price"]:
        value = to_float(row.get(col), -1.0)
        if value >= 0:
            return value / 1_000_000 if value > 1000 else value
    return 0.0


def component_weighted_sum(row: dict[str, Any]) -> float:
    return sum(to_float(row.get(f"match_{idx}_weighted_match_ev")) for idx in [1, 2, 3])


def has_base_components_for_round(row: dict[str, Any], idx: int) -> bool:
    return any(txt(row.get(f"match_{idx}_{component}")) for component in BASE_COMPONENTS)


def has_complete_components(row: dict[str, Any]) -> bool:
    return all(has_base_components_for_round(row, idx) for idx in [1, 2, 3])


def is_baumgartner(row: dict[str, Any]) -> bool:
    return txt(row.get("player_id")) == BAUMGARTNER_ID or txt(row.get("player_name")) == "Christoph Baumgartner"


def zero_baumgartner_ev_row(row: dict[str, Any]) -> None:
    row["holdet_is_out"] = "True"
    row["start_prob"] = "0"
    row["start_prob_source"] = "manual_out_baumgartner_2026_06_04"
    row["minute_share"] = "0"
    for key in ZERO_EV_COLUMNS:
        row[key] = "0"
    for idx in [1, 2, 3]:
        for suffix in BASE_COMPONENTS + ["total_ev_next_match", "weighted_match_ev"]:
            key = f"match_{idx}_{suffix}"
            if key in row:
                row[key] = "0"
    row["price_quality_applied"] = "False"
    row["base_ev_source"] = "out_of_tournament_manual"


def classify_and_base(row: dict[str, Any]) -> tuple[float, str, str, str]:
    if is_baumgartner(row) or truthy(row.get("holdet_is_out")):
        return 0.0, "out_of_tournament", "out_of_tournament_manual", "out_of_tournament_zeroed"

    component_sum = component_weighted_sum(row)
    if has_complete_components(row) and component_sum > 0:
        return component_sum, "complete_components", "component_weighted_sum", "component_base_rebuilt"

    aggregate_ev = max(
        to_float(row.get("weighted_group_stage_ev")),
        to_float(row.get("optimizer_ev")),
        component_sum,
    )
    if aggregate_ev > 0:
        return aggregate_ev, "aggregate_only", "preserved_aggregate_ev", "aggregate_base_preserved"

    return 0.0, "no_ev_source", "no_ev_source", "no_ev_source_zeroed"


def formula_expected(base: float, price_quality: float) -> float:
    return (1.0 - PRICE_QUALITY_WEIGHT) * base + PRICE_QUALITY_WEIGHT * price_quality


def formula_diff(row: dict[str, Any]) -> float:
    base = to_float(row.get("model_ev_before_price_quality") or row.get("weighted_group_stage_ev_before_price_quality"))
    pq = to_float(row.get("price_quality_ev"))
    actual = to_float(row.get("weighted_group_stage_ev"))
    return actual - formula_expected(base, pq)


def audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        base = to_float(row.get("model_ev_before_price_quality") or row.get("weighted_group_stage_ev_before_price_quality"))
        pq = to_float(row.get("price_quality_ev"))
        expected = formula_expected(base, pq)
        actual = to_float(row.get("weighted_group_stage_ev"))
        component_source = txt(row.get("component_source"))
        base_source = txt(row.get("base_ev_source"))
        if not component_source or not base_source:
            _, component_source, base_source, status = classify_and_base(row)
        else:
            status = txt(row.get("repair_status"))
        out.append(
            {
                "player_id": txt(row.get("player_id")),
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "position": txt(row.get("position")),
                "component_weighted_sum": fmt(component_weighted_sum(row)),
                "base_ev_before_price_quality": fmt(base),
                "price_quality_ev": fmt(pq),
                "expected_final_ev": fmt(expected),
                "actual_final_ev": fmt(actual),
                "formula_diff": fmt(actual - expected),
                "component_source": component_source,
                "base_ev_source": base_source,
                "repair_status": status,
            }
        )
    return out


def audit_stats(audit: list[dict[str, Any]]) -> dict[str, float]:
    diffs = [abs(to_float(row["formula_diff"])) for row in audit]
    return {
        "formula_mismatches": sum(1 for diff in diffs if diff > FORMULA_TOLERANCE),
        "max_formula_diff": max(diffs) if diffs else 0.0,
        "base_total": sum(to_float(row["base_ev_before_price_quality"]) for row in audit),
        "final_total": sum(to_float(row["actual_final_ev"]) for row in audit),
    }


def rank_pct(values: list[tuple[int, float]]) -> dict[int, float]:
    if not values:
        return {}
    sorted_values = sorted(values, key=lambda item: item[1])
    out: dict[int, float] = {}
    n = len(sorted_values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1][1] == sorted_values[i][1]:
            j += 1
        avg_rank = ((i + 1) + (j + 1)) / 2.0
        pct = avg_rank / n
        for k in range(i, j + 1):
            out[sorted_values[k][0]] = pct
        i = j + 1
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def apply_price_quality_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    work = [dict(row) for row in rows]
    meta: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(work):
        if is_baumgartner(row):
            zero_baumgartner_ev_row(row)
        base, component_source, base_source, status = classify_and_base(row)
        row["price_m"] = fmt(get_price_m(row))
        row["model_ev_before_price_quality"] = fmt(base)
        row["weighted_group_stage_ev_before_price_quality"] = fmt(base)
        row["optimizer_ev_before_price_quality"] = fmt(base)
        row["component_source"] = component_source
        row["base_ev_source"] = base_source
        row["repair_status"] = status
        meta[idx] = {"base": base, "eligible": base > 0 and component_source != "out_of_tournament"}

    by_position: dict[str, list[int]] = {}
    for idx, row in enumerate(work):
        if meta[idx]["eligible"]:
            by_position.setdefault(txt(row.get("position")).upper(), []).append(idx)

    for position, indices in by_position.items():
        price_ranks = rank_pct([(idx, to_float(work[idx].get("price_m"))) for idx in indices])
        ev_values = [meta[idx]["base"] for idx in indices]
        ev_p20 = quantile(ev_values, 0.20)
        ev_p90 = quantile(ev_values, 0.90)
        if ev_p90 <= ev_p20:
            median = quantile(ev_values, 0.50)
            ev_p20 = median
            ev_p90 = max(ev_values) if ev_values else median
        spread = max(ev_p90 - ev_p20, 0.85) * PRICE_QUALITY_SPREAD_MULTIPLIER

        for idx in indices:
            row = work[idx]
            pct = price_ranks[idx]
            price_quality = ev_p20 + pct * spread
            base = meta[idx]["base"]
            final = formula_expected(base, price_quality)
            row["price_rank_pct_position"] = fmt(pct)
            row["price_quality_ev"] = fmt(price_quality)
            row["weighted_group_stage_ev"] = fmt(final)
            row["optimizer_ev"] = fmt(final)
            row["price_quality_weight"] = fmt(PRICE_QUALITY_WEIGHT)
            row["price_quality_spread_multiplier"] = fmt(PRICE_QUALITY_SPREAD_MULTIPLIER)
            row["price_quality_applied"] = "True"

    for idx, row in enumerate(work):
        if meta[idx]["eligible"]:
            continue
        row["price_rank_pct_position"] = txt(row.get("price_rank_pct_position")) or "0"
        row["price_quality_ev"] = "0"
        row["weighted_group_stage_ev"] = "0"
        row["optimizer_ev"] = "0"
        row["price_quality_weight"] = fmt(PRICE_QUALITY_WEIGHT)
        row["price_quality_spread_multiplier"] = fmt(PRICE_QUALITY_SPREAD_MULTIPLIER)
        row["price_quality_applied"] = "False"

    return work


def ensure_fields(fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
    wanted = [
        "component_source",
        "base_ev_source",
        "repair_status",
        "price_m",
        "price_rank_pct_position",
        "model_ev_before_price_quality",
        "weighted_group_stage_ev_before_price_quality",
        "optimizer_ev_before_price_quality",
        "price_quality_ev",
        "price_quality_weight",
        "price_quality_spread_multiplier",
        "price_quality_applied",
    ]
    out = list(fields)
    for col in wanted:
        if col not in out:
            out.append(col)
    for row in rows:
        for col in out:
            row.setdefault(col, "")
    return out


def load_pool() -> list[dict[str, Any]]:
    data = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("player_pool_v1.json skal vaere en liste")
    return data


def update_baumgartner_pool(players: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    for player in players:
        if txt(player.get("player_id")) != BAUMGARTNER_ID:
            continue
        before = dict(player)
        player["holdet_is_out"] = True
        player["start_prob"] = 0.0
        player["start_prob_source"] = "manual_out_baumgartner_2026_06_04"
        player["start_probability_pct"] = 0
        player["start_security"] = 0.0
        player["start_status"] = "ude af VM"
        player["conditional_start_prob"] = 0.0
        player["availability_prob"] = 0.0
        player["availability_risk"] = "out"
        player["availability_status"] = "out_of_tournament"
        player["weighted_group_stage_ev"] = 0.0
        player["optimizer_ev"] = 0.0
        player["manual_data_note"] = "Ude af VM; nulstillet som valgbaar modelkandidat 2026-06-04"
        after = dict(player)
        break
    return before, after


def write_pool(players: list[dict[str, Any]]) -> None:
    PLAYER_POOL_PATH.write_text(json.dumps(players, ensure_ascii=False, indent=2), encoding="utf-8")


def sanity_table(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_by_id = {txt(row.get("player_id")): row for row in before}
    out = []
    for row in after:
        if txt(row.get("player_name")) not in SANITY_NAMES:
            continue
        old = before_by_id.get(txt(row.get("player_id")), {})
        out.append(
            {
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "start_prob_before": txt(old.get("start_prob")),
                "start_prob_after": txt(row.get("start_prob")),
                "minute_share_before": txt(old.get("minute_share")),
                "minute_share_after": txt(row.get("minute_share")),
                "component_sum_before": fmt(component_weighted_sum(old)),
                "component_sum_after": fmt(component_weighted_sum(row)),
                "base_before": txt(old.get("weighted_group_stage_ev_before_price_quality") or old.get("model_ev_before_price_quality")),
                "base_after": txt(row.get("weighted_group_stage_ev_before_price_quality")),
                "price_quality_before": txt(old.get("price_quality_ev")),
                "price_quality_after": txt(row.get("price_quality_ev")),
                "final_before": txt(old.get("weighted_group_stage_ev")),
                "final_after": txt(row.get("weighted_group_stage_ev")),
                "optimizer_before": txt(old.get("optimizer_ev")),
                "optimizer_after": txt(row.get("optimizer_ev")),
            }
        )
    return out


def md_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> list[str]:
    subset = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in subset:
        lines.append("| " + " | ".join(txt(row.get(field)) for field in fields) + " |")
    return lines


def write_md(
    before_audit: list[dict[str, Any]],
    after_audit: list[dict[str, Any]],
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    ev_backup: Path,
    pool_backup: Path,
    baum_before: dict[str, Any],
    baum_after: dict[str, Any],
) -> None:
    before_stats = audit_stats(before_audit)
    after_stats = audit_stats(after_audit)
    source_counts = Counter(row["component_source"] for row in after_audit)
    sanity = sanity_table(before_rows, after_rows)

    lines = [
        "# Player EV Price-Quality Consistency Audit",
        "",
        "Price-quality-laget er genberegnet efter start_prob- og komponentrebuild. Optimizer, strategi-output og frontend er ikke genkoert.",
        "",
        "## Rodarsag",
        "",
        "`repair_ev_components_after_start_prob_repair.py` opdaterede matchkomponenter, men price-quality metadata og slut-EV byggede stadig paa gamle `*_before_price_quality`-kolonner. `apply_price_quality_to_ev.py` var idempotent ved at foretraekke gamle basekolonner, hvilket efter komponentrebuild blev forkert.",
        "",
        "## Haandtering",
        "",
        "- Komplette komponentraekker: base-EV = `match_1_weighted_match_ev + match_2_weighted_match_ev + match_3_weighted_match_ev`.",
        "- Aggregate-only-raekker: eksisterende aggregerede EV bevares som base; der opfindes ikke basekomponenter.",
        "- No-EV-source-raekker: base, price-quality og final EV holdes paa 0.",
        "- Christoph Baumgartner: markeret ude og nulstillet som valgbaar modelkandidat.",
        "- Price-quality-formlen er uændret: `0.55 * model_ev_before_price_quality + 0.45 * price_quality_ev`.",
        "",
        "## Counts",
        "",
        f"- Komplette komponentraekker efter: {source_counts.get('complete_components', 0)}",
        f"- Aggregate-only-raekker efter: {source_counts.get('aggregate_only', 0)}",
        f"- No-EV-source-raekker efter: {source_counts.get('no_ev_source', 0)}",
        f"- Out-of-tournament efter: {source_counts.get('out_of_tournament', 0)}",
        f"- Price-quality formelmismatches foer: {before_stats['formula_mismatches']}",
        f"- Price-quality formelmismatches efter: {after_stats['formula_mismatches']}",
        f"- Maks formelafvigelse foer: {before_stats['max_formula_diff']:.6f}",
        f"- Maks formelafvigelse efter: {after_stats['max_formula_diff']:.6f}",
        f"- Base-EV total foer: {before_stats['base_total']:.3f}",
        f"- Base-EV total efter: {after_stats['base_total']:.3f}",
        f"- Slut-EV total foer: {before_stats['final_total']:.3f}",
        f"- Slut-EV total efter: {after_stats['final_total']:.3f}",
        f"- EV-backup: `{ev_backup.relative_to(ROOT)}`",
        f"- Player-pool-backup: `{pool_backup.relative_to(ROOT)}`",
        "",
        "## Christoph Baumgartner",
        "",
        f"- Foer: holdet_is_out={baum_before.get('holdet_is_out')}, start_prob={baum_before.get('start_prob')}, EV={baum_before.get('weighted_group_stage_ev')}",
        f"- Efter: holdet_is_out={baum_after.get('holdet_is_out')}, start_prob={baum_after.get('start_prob')}, EV={baum_after.get('weighted_group_stage_ev')}",
        "",
        "## Sanity-spillere",
        "",
        *md_table(
            sanity,
            [
                "player_name",
                "team_id",
                "start_prob_before",
                "start_prob_after",
                "component_sum_before",
                "component_sum_after",
                "base_before",
                "base_after",
                "price_quality_before",
                "price_quality_after",
                "final_before",
                "final_after",
                "optimizer_before",
                "optimizer_after",
            ],
        ),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    fields, rows = read_csv(EV_PATH)
    before_rows = [dict(row) for row in rows]
    before_audit = audit_rows(before_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ev_backup = EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_price_quality_consistency_{timestamp}.csv")
    pool_backup = PLAYER_POOL_PATH.with_name(f"player_pool_v1.backup_before_baumgartner_out_{timestamp}.json")
    shutil.copy2(EV_PATH, ev_backup)
    shutil.copy2(PLAYER_POOL_PATH, pool_backup)

    players = load_pool()
    baum_before, baum_after = update_baumgartner_pool(players)
    write_pool(players)

    repaired = apply_price_quality_consistency(rows)
    final_fields = ensure_fields(fields, repaired)
    repaired.sort(key=lambda row: (to_float(row.get("weighted_group_stage_ev")), to_float(row.get("total_ev_group_stage"))), reverse=True)
    write_csv(EV_PATH, final_fields, repaired)

    after_audit = audit_rows(repaired)
    write_csv(OUT_CSV, AUDIT_FIELDS, after_audit)
    write_md(before_audit, after_audit, before_rows, repaired, ev_backup, pool_backup, baum_before, baum_after)

    diag_cols = [
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "price_m",
        "price_rank_pct_position",
        "model_ev_before_price_quality",
        "price_quality_ev",
        "weighted_group_stage_ev",
        "price_quality_weight",
        "price_quality_spread_multiplier",
        "price_quality_applied",
        "component_source",
        "base_ev_source",
        "repair_status",
    ]
    write_csv(PRICE_DIAG_PATH, [col for col in diag_cols if col in final_fields], repaired)

    before_stats = audit_stats(before_audit)
    after_stats = audit_stats(after_audit)
    print("EV price-quality consistency repair")
    print("-----------------------------------")
    print(f"Formula mismatches before: {before_stats['formula_mismatches']}")
    print(f"Formula mismatches after: {after_stats['formula_mismatches']}")
    print(f"Max formula diff before: {before_stats['max_formula_diff']:.6f}")
    print(f"Max formula diff after: {after_stats['max_formula_diff']:.6f}")
    print(f"Base EV total before: {before_stats['base_total']:.3f}")
    print(f"Base EV total after: {after_stats['base_total']:.3f}")
    print(f"Final EV total before: {before_stats['final_total']:.3f}")
    print(f"Final EV total after: {after_stats['final_total']:.3f}")
    print(f"EV backup: {ev_backup.relative_to(ROOT)}")
    print(f"Pool backup: {pool_backup.relative_to(ROOT)}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
