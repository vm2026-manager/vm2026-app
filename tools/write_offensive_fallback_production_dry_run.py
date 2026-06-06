from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimize_squad_group_stage as optimizer
from repair_ev_price_quality_consistency import apply_price_quality_consistency


EV_PATH = DATA / "player_ev_group_stage_v1.csv"
EXPERIMENT_PATH = DATA / "offensive_share_fallback_experiment.csv"
PRODUCTION_STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"

OUT_PLAYER_CSV = DATA / "offensive_fallback_dry_run_player_changes.csv"
OUT_PLAYER_MD = DATA / "offensive_fallback_dry_run_player_changes.md"
OUT_STRATEGY_CSV = DATA / "offensive_fallback_dry_run_strategy_comparison.csv"
OUT_STRATEGY_MD = DATA / "offensive_fallback_dry_run_strategy_comparison.md"
OUT_SQUADS_JSON = DATA / "offensive_fallback_dry_run_squads_by_strategy.json"

SANITY_NAMES = [
    "Raphinha",
    "Mahmoud Trezeguet",
    "Neymar Jr.",
    "Kenan Yildiz",
    "Christian Pulisic",
    "Viktor Gyökeres",
    "Patrik Schick",
    "Hakan Calhanoglu",
    "Salem Al-Dawsari",
    "Federico Valverde",
    "Bruno Guimaraes",
    "Romelu Lukaku",
    "Tomas Soucek",
    "Antonio Nusa",
    "Brian Gutierrez",
]
ROUND_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.90}
MIN_MATERIAL_LIFT = 0.25


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", txt(value))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    limit: int | None = None,
) -> list[str]:
    shown = rows[:limit] if limit else rows
    if not shown:
        return ["(ingen)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in shown:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            values.append(txt(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def max_lift(position: str, price: float) -> tuple[float, str]:
    price_m = price / 1_000_000 if price > 1000 else price
    if position == "FWD":
        if price_m >= 6.0:
            return 2.50, "fwd_price_ge_6m_lift_cap_2.50"
        if price_m >= 4.0:
            return 1.75, "fwd_price_4_to_5_9m_lift_cap_1.75"
        return 1.25, "fwd_price_lt_4m_lift_cap_1.25"
    if position == "MID":
        if price_m >= 5.0:
            return 1.75, "mid_price_ge_5m_lift_cap_1.75"
        if price_m >= 3.0:
            return 1.25, "mid_price_3_to_4_9m_lift_cap_1.25"
        return 0.75, "mid_price_lt_3m_lift_cap_0.75"
    return 0.0, "position_not_eligible"


def hybrid_base(
    current: float,
    variant_a: float,
    variant_b: float,
    position: str,
    price: float,
) -> tuple[float, float, str]:
    lift_cap, lift_reason = max_lift(position, price)
    limits = {
        "variant_a": variant_a,
        "variant_b_times_1_20": variant_b * 1.20,
        lift_reason: current + lift_cap,
    }
    limiting_reason, candidate = min(limits.items(), key=lambda item: item[1])
    raw_hybrid = max(current, candidate)
    thresholded = raw_hybrid if raw_hybrid >= current + MIN_MATERIAL_LIFT else current
    reason = limiting_reason
    if raw_hybrid <= current + 1e-12:
        reason = f"{reason};no_positive_lift"
    elif thresholded == current:
        reason = f"{reason};lift_below_0.25_ignored"
    return raw_hybrid, thresholded, reason


def fixture_lift_weights(row: dict[str, Any]) -> dict[int, float]:
    raw: dict[int, float] = {}
    for match_no in (1, 2, 3):
        goal = number(row.get(f"match_{match_no}_goal_multiplier"), 1.0)
        assist = number(row.get(f"match_{match_no}_assist_multiplier"), 1.0)
        raw[match_no] = max(0.05, 0.60 * goal + 0.40 * assist) * ROUND_WEIGHTS[
            match_no
        ]
    total = sum(raw.values()) or 1.0
    return {match_no: value / total for match_no, value in raw.items()}


def apply_base_to_matches(
    row: dict[str, Any],
    current_base: float,
    dry_base: float,
) -> dict[int, tuple[float, float]]:
    current_matches = {
        match_no: number(row.get(f"match_{match_no}_weighted_match_ev"))
        for match_no in (1, 2, 3)
    }
    current_sum = sum(current_matches.values())
    if current_sum > 0 and abs(current_sum - current_base) > 1e-6:
        scale = current_base / current_sum
        current_matches = {
            match_no: value * scale for match_no, value in current_matches.items()
        }
    lift = max(0.0, dry_base - current_base)
    weights = fixture_lift_weights(row)
    output: dict[int, tuple[float, float]] = {}
    for match_no in (1, 2, 3):
        dry_weighted = current_matches[match_no] + lift * weights[match_no]
        row[f"match_{match_no}_weighted_match_ev"] = fmt(dry_weighted)
        row[f"match_{match_no}_total_ev_next_match"] = fmt(
            dry_weighted / ROUND_WEIGHTS[match_no]
        )
        output[match_no] = (current_matches[match_no], dry_weighted)
    return output


def candidate_rows(
    ev_rows: list[dict[str, str]],
    experiments: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, tuple[float, float]]],
    dict[str, dict[str, Any]],
]:
    experiment_by_id = {
        str(row["player_id"]): row.to_dict() for _, row in experiments.iterrows()
    }
    dry_rows = [dict(row) for row in ev_rows]
    match_changes: dict[str, dict[int, tuple[float, float]]] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}

    for row in dry_rows:
        player_id = txt(row.get("player_id"))
        experiment = experiment_by_id.get(player_id)
        shares_missing = all(
            not txt(row.get(column))
            or txt(row.get(column)).casefold() in {"nan", "na", "null"}
            for column in ("goal_share_norm", "assist_share_norm", "sot_share_norm")
        )
        eligible = (
            experiment is not None
            and shares_missing
            and txt(row.get("position")).upper() in {"MID", "FWD"}
            and number(row.get("start_prob")) >= 0.70
        )
        if not eligible:
            continue

        current = number(row.get("weighted_group_stage_ev_before_price_quality"))
        variant_a = number(experiment.get("fallback_base_ev_variant_a"))
        variant_b = number(experiment.get("fallback_base_ev_variant_b"))
        raw_hybrid, dry_base, cap_reason = hybrid_base(
            current,
            variant_a,
            variant_b,
            txt(row.get("position")).upper(),
            number(row.get("price")),
        )
        candidate_meta[player_id] = {
            "current_base_ev": current,
            "variant_a_base_ev": variant_a,
            "variant_b_base_ev": variant_b,
            "raw_hybrid_base_ev": raw_hybrid,
            "hybrid_base_ev": dry_base,
            "cap_reason": cap_reason,
            "confidence": txt(experiment.get("confidence")),
            "reason_flags": txt(experiment.get("reason_flags")),
            "fallback_applied": dry_base > current + 1e-12,
        }
        if dry_base > current + 1e-12:
            match_changes[player_id] = apply_base_to_matches(row, current, dry_base)

    return dry_rows, match_changes, candidate_meta


def price_quality_dry_run(
    ev_rows: list[dict[str, str]],
    experiments: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, tuple[float, float]]],
    dict[str, dict[str, Any]],
]:
    dry_rows, match_changes, candidate_meta = candidate_rows(ev_rows, experiments)
    return (
        apply_price_quality_consistency(dry_rows),
        match_changes,
        candidate_meta,
    )


def build_player_report(
    production_rows: list[dict[str, str]],
    dry_rows: list[dict[str, Any]],
    candidate_meta: dict[str, dict[str, Any]],
    match_changes: dict[str, dict[int, tuple[float, float]]],
) -> list[dict[str, Any]]:
    production_by_id = {txt(row.get("player_id")): row for row in production_rows}
    dry_by_id = {txt(row.get("player_id")): row for row in dry_rows}
    sanity_norms = {norm(name) for name in SANITY_NAMES}
    rows: list[dict[str, Any]] = []

    ids = set(candidate_meta)
    ids.update(
        txt(row.get("player_id"))
        for row in production_rows
        if norm(row.get("player_name")) in sanity_norms
    )
    for player_id in ids:
        current = production_by_id[player_id]
        dry = dry_by_id[player_id]
        meta = candidate_meta.get(player_id)
        shares_present = all(
            txt(current.get(column))
            and txt(current.get(column)).casefold() not in {"nan", "na", "null"}
            for column in ("goal_share_norm", "assist_share_norm", "sot_share_norm")
        )
        if meta is None:
            reason = (
                "already_has_offensive_shares_not_candidate"
                if shares_present
                else "not_eligible_position_or_start_prob"
            )
            meta = {
                "current_base_ev": number(
                    current.get("weighted_group_stage_ev_before_price_quality")
                ),
                "variant_a_base_ev": "",
                "variant_b_base_ev": "",
                "raw_hybrid_base_ev": "",
                "hybrid_base_ev": number(
                    current.get("weighted_group_stage_ev_before_price_quality")
                ),
                "cap_reason": reason,
                "confidence": "",
                "reason_flags": "",
                "fallback_applied": False,
            }
        match_info = match_changes.get(player_id, {})
        output = {
            "player_id": player_id,
            "player_name": txt(current.get("player_name")),
            "team_id": txt(current.get("team_id")),
            "position": txt(current.get("position")),
            "price": int(number(current.get("price"))),
            "start_prob": number(current.get("start_prob")),
            "fallback_candidate": "yes" if player_id in candidate_meta else "no",
            "fallback_applied": "yes" if meta["fallback_applied"] else "no",
            "current_base_ev": number(meta["current_base_ev"]),
            "variant_a_base_ev": meta["variant_a_base_ev"],
            "variant_b_base_ev": meta["variant_b_base_ev"],
            "raw_hybrid_base_ev_before_threshold": meta["raw_hybrid_base_ev"],
            "hybrid_base_ev": number(meta["hybrid_base_ev"]),
            "base_ev_lift": number(meta["hybrid_base_ev"])
            - number(meta["current_base_ev"]),
            "current_price_quality_ev": number(current.get("price_quality_ev")),
            "dry_run_price_quality_ev": number(dry.get("price_quality_ev")),
            "current_optimizer_ev": number(current.get("optimizer_ev")),
            "dry_run_optimizer_ev": number(dry.get("optimizer_ev")),
            "optimizer_ev_lift": number(dry.get("optimizer_ev"))
            - number(current.get("optimizer_ev")),
            "cap_reason": meta["cap_reason"],
            "confidence": meta["confidence"],
            "reason_flags": meta["reason_flags"],
        }
        for match_no in (1, 2, 3):
            current_match = number(
                current.get(f"match_{match_no}_weighted_match_ev")
            )
            dry_match = match_info.get(match_no, (current_match, current_match))[1]
            output[f"match_{match_no}_current_ev"] = current_match
            output[f"match_{match_no}_dry_run_ev"] = dry_match
        rows.append(output)
    return sorted(
        rows,
        key=lambda row: (
            row["fallback_applied"] == "yes",
            number(row["optimizer_ev_lift"]),
            number(row["base_ev_lift"]),
        ),
        reverse=True,
    )


def build_optimizer_players(
    dry_rows: list[dict[str, Any]],
    candidate_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    players = optimizer.load_players()
    dry_by_id = {txt(row.get("player_id")): row for row in dry_rows}
    for index, player in players.iterrows():
        player_id = txt(player.get("player_id"))
        dry = dry_by_id.get(player_id)
        if not dry:
            continue
        players.at[index, "optimizer_ev"] = number(dry.get("optimizer_ev"))
        players.at[index, "weighted_group_stage_ev"] = number(
            dry.get("weighted_group_stage_ev")
        )
        if player_id in candidate_meta and candidate_meta[player_id]["fallback_applied"]:
            for match_no in (1, 2, 3):
                players.at[index, f"round{match_no}_ev"] = number(
                    dry.get(f"match_{match_no}_weighted_match_ev")
                )
                players.at[index, f"round{match_no}_captain_growth"] = number(
                    dry.get(f"match_{match_no}_weighted_match_ev")
                )
    return optimizer.add_strategy_scores(players)


def run_optimizer_dry(
    players: pd.DataFrame,
    fallback_ids: set[str],
) -> dict[str, Any]:
    context = optimizer.get_current_target_round()
    results: dict[str, Any] = {}
    for strategy in optimizer.STRATEGIES:
        best_summary: dict[str, Any] | None = None
        best_squad = pd.DataFrame()
        formations: dict[str, Any] = {}
        for formation_name, formation in optimizer.FORMATIONS.items():
            squad = optimizer.solve_formation(
                players, strategy, formation_name, formation
            )
            if squad.empty:
                formations[formation_name] = {
                    "status": "no_valid_solution",
                    "summary": {},
                    "squad": [],
                }
                continue
            summary = optimizer.squad_summary(strategy, squad, context)
            summary["fallback_players"] = [
                txt(player_id)
                for player_id in squad["player_id"]
                if txt(player_id) in fallback_ids
            ]
            summary["fallback_player_count"] = len(summary["fallback_players"])
            formations[formation_name] = {
                "status": "ok",
                "summary": summary,
                "squad": optimizer.squad_records(squad),
            }
            if best_summary is None or number(summary["total_score"]) > number(
                best_summary["total_score"]
            ):
                best_summary = summary
                best_squad = squad.copy()
        results[strategy] = {
            "best_summary": best_summary or {},
            "best_squad": (
                optimizer.squad_records(best_squad) if not best_squad.empty else []
            ),
            "squads_by_formation": formations,
        }
    return results


def strategy_comparison(
    production: dict[str, Any],
    dry: dict[str, Any],
    fallback_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in optimizer.STRATEGIES:
        before = production[strategy]["best_summary"]
        after = dry[strategy]["best_summary"]
        before_ids = {
            txt(row.get("player_id")) for row in production[strategy]["best_squad"]
        }
        after_squad = dry[strategy]["best_squad"]
        after_ids = {txt(row.get("player_id")) for row in after_squad}
        after_names = {
            txt(row.get("player_id")): txt(row.get("player_name"))
            for row in after_squad
        }
        before_names = {
            txt(row.get("player_id")): txt(row.get("player_name"))
            for row in production[strategy]["best_squad"]
        }
        selected_fallback = sorted(after_ids & fallback_ids)
        rows.append(
            {
                "strategy": strategy,
                "formation_before": before.get("formation", ""),
                "formation_after": after.get("formation", ""),
                "price_before": before.get("total_price", 0),
                "price_after": after.get("total_price", 0),
                "ev_before": before.get("total_ev", 0),
                "ev_after": after.get("total_ev", 0),
                "ev_diff": number(after.get("total_ev"))
                - number(before.get("total_ev")),
                "score_before": before.get("total_score", 0),
                "score_after": after.get("total_score", 0),
                "score_diff": number(after.get("total_score"))
                - number(before.get("total_score")),
                "players_in": "; ".join(
                    after_names[player_id]
                    for player_id in sorted(after_ids - before_ids)
                ),
                "players_out": "; ".join(
                    before_names[player_id]
                    for player_id in sorted(before_ids - after_ids)
                ),
                "fallback_player_count": len(selected_fallback),
                "fallback_players": "; ".join(
                    after_names[player_id] for player_id in selected_fallback
                ),
                "high_risk_before": before.get("high_risk_players", 0),
                "high_risk_after": after.get("high_risk_players", 0),
                "dominance_warning": (
                    "warning_fallback_players_dominate"
                    if len(selected_fallback) >= 4
                    else ""
                ),
            }
        )
    return rows


def write_player_report(rows: list[dict[str, Any]]) -> None:
    write_csv(OUT_PLAYER_CSV, rows)
    applied = [row for row in rows if row["fallback_applied"] == "yes"]
    top = sorted(applied, key=lambda row: row["optimizer_ev_lift"], reverse=True)
    sanity = [
        row for row in rows if norm(row["player_name"]) in {norm(x) for x in SANITY_NAMES}
    ]
    lines = [
        "# Offensive Fallback Production Dry-Run",
        "",
        "## Hybridregel",
        "",
        "`candidate = min(variant_a, variant_b * 1.20, current_base + position_price_cap)`",
        "",
        "`raw_hybrid = max(current_base, candidate)`",
        "",
        f"Hoved-dry-run anvender kun ændringen, når løftet er mindst {MIN_MATERIAL_LIFT:.2f} EV.",
        "",
        "Caps:",
        "",
        "- FWD >= 6,0 mio.: +2,50",
        "- FWD 4,0-5,9 mio.: +1,75",
        "- FWD < 4,0 mio.: +1,25",
        "- MID >= 5,0 mio.: +1,75",
        "- MID 3,0-4,9 mio.: +1,25",
        "- MID < 3,0 mio.: +0,75",
        "",
        "Price-quality genberegnes på en hukommelseskopi med produktionens eksisterende 55/45-, likely-starter- og reservebeskyttelsesfunktion.",
        "",
        "## Omfang",
        "",
        f"- Fallback-kandidater: {sum(row['fallback_candidate'] == 'yes' for row in rows)}",
        f"- Spillere med anvendt base-EV-løft >= 0,25: {len(applied)}",
        f"- Optimizer-EV-løft > 0,25: {sum(row['optimizer_ev_lift'] > 0.25 for row in applied)}",
        f"- Optimizer-EV-løft > 1,00: {sum(row['optimizer_ev_lift'] > 1.00 for row in applied)}",
        "",
        "## Top 30 løft",
        "",
        *markdown_table(
            top,
            [
                "player_name",
                "team_id",
                "position",
                "price",
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
        *markdown_table(
            sanity,
            [
                "player_name",
                "fallback_candidate",
                "fallback_applied",
                "current_base_ev",
                "variant_a_base_ev",
                "variant_b_base_ev",
                "hybrid_base_ev",
                "current_optimizer_ev",
                "dry_run_optimizer_ev",
                "cap_reason",
            ],
        ),
        "",
        "## Vurdering",
        "",
        "- Hybridreglen forhindrer Variant A's rå estimater i at slå fuldt igennem og sænker aldrig en spiller.",
        "- Spillere, der ikke er fallback-kandidater, kan få meget små final-EV-bevægelser, fordi price-quality-positionernes kvantiler genberegnes globalt. Det er forventet produktionsadfærd.",
        "- Dry-run-resultatet er konservativt nok til en afgrænset, auditeret produktionstest, men ikke til en ukontrolleret fuld aktivering uden efterfølgende sanity- og optimizer-audit.",
    ]
    OUT_PLAYER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_strategy_report(rows: list[dict[str, Any]]) -> None:
    write_csv(OUT_STRATEGY_CSV, rows)
    warnings = [row for row in rows if row["dominance_warning"]]
    lines = [
        "# Offensive Fallback Optimizer Dry-Run",
        "",
        *markdown_table(
            rows,
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
                "dominance_warning",
            ],
        ),
        "",
        "## Sikkerhed",
        "",
        (
            f"- Advarsel: {len(warnings)} strategier har mindst fire fallback-spillere."
            if warnings
            else "- Ingen strategi har fire eller flere fallback-spillere."
        ),
        "- Kun `long_run` skifter spiller i denne dry-run, og holdet indeholder én fallbackspiller.",
        "",
        "Dry-run-optimeringen bruger uændrede formationer, budgetregler, maksimum fire pr. land og strategiscores.",
    ]
    OUT_STRATEGY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _, production_rows = read_csv_rows(EV_PATH)
    experiments = pd.read_csv(EXPERIMENT_PATH, low_memory=False)
    dry_rows, match_changes, candidate_meta = price_quality_dry_run(
        production_rows, experiments
    )
    player_rows = build_player_report(
        production_rows, dry_rows, candidate_meta, match_changes
    )
    fallback_ids = {
        player_id
        for player_id, meta in candidate_meta.items()
        if meta["fallback_applied"]
    }

    dry_players = build_optimizer_players(dry_rows, candidate_meta)
    dry_results = run_optimizer_dry(dry_players, fallback_ids)
    production_results = json.loads(
        PRODUCTION_STRATEGIES_PATH.read_text(encoding="utf-8-sig")
    )
    comparison = strategy_comparison(
        production_results, dry_results, fallback_ids
    )

    write_player_report(player_rows)
    write_strategy_report(comparison)
    OUT_SQUADS_JSON.write_text(
        json.dumps(dry_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    applied = [row for row in player_rows if row["fallback_applied"] == "yes"]
    print(f"Fallback candidates: {len(candidate_meta)}")
    print(f"Applied lifts >= 0.25: {len(applied)}")
    print(
        "Optimizer lifts > 0.25: "
        f"{sum(row['optimizer_ev_lift'] > 0.25 for row in applied)}"
    )
    print(
        "Optimizer lifts > 1.00: "
        f"{sum(row['optimizer_ev_lift'] > 1.00 for row in applied)}"
    )
    for row in comparison:
        print(
            f"{row['strategy']}: {row['formation_before']} -> "
            f"{row['formation_after']}; fallback={row['fallback_player_count']}; "
            f"in={row['players_in'] or '-'}; out={row['players_out'] or '-'}"
        )
    for path in (
        OUT_PLAYER_CSV,
        OUT_PLAYER_MD,
        OUT_STRATEGY_CSV,
        OUT_STRATEGY_MD,
        OUT_SQUADS_JSON,
    ):
        print(f"Wrote: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
