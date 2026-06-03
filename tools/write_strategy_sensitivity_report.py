from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
DISPLAY_NAMES_PATH = DATA / "strategy_display_names.json"
CONTEXT_PATH = DATA / "current_strategy_context.json"

OUT_MD = DATA / "strategy_sensitivity_report.md"
OUT_CSV = DATA / "strategy_sensitivity_report.csv"

STRATEGY_ORDER = ["next_round", "round1_2", "group_stage", "long_run"]
LOW_CONDITIONAL_THRESHOLD = 0.75
SMALL_SELECTED_MARGIN_THRESHOLD = 0.35

CSV_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "price",
    "picked_count",
    "picked_strategies",
    "picked_display_names",
    "robustness_bucket",
    "avg_ev",
    "avg_strategy_score",
    "min_conditional_start_prob",
    "max_conditional_start_prob",
    "availability_risks",
    "manual_warnings",
    "captain_in_strategies",
    "nearest_selected_same_position_margin",
    "small_selected_margin",
]


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


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return txt(value)


def md(value: Any) -> str:
    return txt(value).replace("|", "\\|")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md(value) for value in row) + " |")
    return "\n".join(lines)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def display_name(strategy: str, display_names: dict[str, str], strategies: dict[str, Any]) -> str:
    if strategy in display_names:
        return txt(display_names[strategy])
    summary = strategies.get(strategy, {}).get("best_summary", {})
    return txt(summary.get("display_name_da")) or strategy


def robustness_bucket(picked_count: int) -> str:
    if picked_count == 4:
        return "core_4_of_4"
    if picked_count == 3:
        return "strong_3_of_4"
    if picked_count == 2:
        return "shared_2_of_4"
    return "single_strategy"


def manual_warnings(players: list[dict[str, Any]]) -> list[str]:
    warnings: set[str] = set()
    for player in players:
        risk = txt(player.get("availability_risk"))
        cond = to_float(player.get("conditional_start_prob"))
        if risk == "high_risk":
            warnings.add("high_risk")
        if cond and cond < LOW_CONDITIONAL_THRESHOLD:
            warnings.add("low_conditional_start")
        for key in ["manual_status", "manual_start_status", "manual_note", "manual_role_note", "manual_captain_note"]:
            value = txt(player.get(key))
            if value:
                warnings.add(f"{key}:{value}")
    return sorted(warnings)


def nearest_same_position_margins(strategies: dict[str, Any]) -> dict[tuple[str, str], float]:
    margins: dict[tuple[str, str], float] = {}
    for strategy, item in strategies.items():
        by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for player in item.get("best_squad", []):
            by_position[txt(player.get("position"))].append(player)

        for position_players in by_position.values():
            sorted_players = sorted(
                position_players,
                key=lambda player: to_float(player.get("strategy_score")),
                reverse=True,
            )
            for idx, player in enumerate(sorted_players):
                score = to_float(player.get("strategy_score"))
                neighbor_scores: list[float] = []
                if idx > 0:
                    neighbor_scores.append(to_float(sorted_players[idx - 1].get("strategy_score")))
                if idx + 1 < len(sorted_players):
                    neighbor_scores.append(to_float(sorted_players[idx + 1].get("strategy_score")))
                if neighbor_scores:
                    margins[(strategy, txt(player.get("player_id")))] = min(abs(score - other) for other in neighbor_scores)
    return margins


def aggregate_rows(strategies: dict[str, Any], display_names: dict[str, str]) -> list[dict[str, Any]]:
    picks: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    captain_by_strategy: dict[str, str] = {}
    margins = nearest_same_position_margins(strategies)

    for strategy in STRATEGY_ORDER:
        item = strategies.get(strategy, {})
        summary = item.get("best_summary", {})
        captain_by_strategy[strategy] = txt(summary.get("recommended_captain"))
        for player in item.get("best_squad", []):
            picks[txt(player.get("player_id"))].append((strategy, player))

    rows: list[dict[str, Any]] = []
    for player_id, entries in picks.items():
        strategies_for_player = [strategy for strategy, _ in entries]
        players = [player for _, player in entries]
        first = players[0]
        picked_display_names = [display_name(strategy, display_names, strategies) for strategy in strategies_for_player]
        ev_values = [to_float(player.get("optimizer_ev")) for player in players]
        score_values = [to_float(player.get("strategy_score")) for player in players]
        cond_values = [to_float(player.get("conditional_start_prob")) for player in players]
        risk_values = sorted({txt(player.get("availability_risk")) or "unknown" for player in players})
        captain_strategies = [
            display_name(strategy, display_names, strategies)
            for strategy in strategies_for_player
            if captain_by_strategy.get(strategy) == txt(first.get("player_name"))
        ]
        selected_margins = [
            margins[(strategy, player_id)]
            for strategy in strategies_for_player
            if (strategy, player_id) in margins
        ]
        nearest_margin = min(selected_margins) if selected_margins else 0.0

        rows.append(
            {
                "player_id": player_id,
                "player_name": txt(first.get("player_name")),
                "team_id": txt(first.get("team_id")),
                "position": txt(first.get("position")),
                "price": int(to_float(first.get("price"))) if txt(first.get("price")) else "",
                "picked_count": len(entries),
                "picked_strategies": "; ".join(strategies_for_player),
                "picked_display_names": "; ".join(picked_display_names),
                "robustness_bucket": robustness_bucket(len(entries)),
                "avg_ev": fmt(sum(ev_values) / len(ev_values), 6),
                "avg_strategy_score": fmt(sum(score_values) / len(score_values), 6),
                "min_conditional_start_prob": fmt(min(cond_values), 4),
                "max_conditional_start_prob": fmt(max(cond_values), 4),
                "availability_risks": "; ".join(risk_values),
                "manual_warnings": "; ".join(manual_warnings(players)),
                "captain_in_strategies": "; ".join(captain_strategies),
                "nearest_selected_same_position_margin": fmt(nearest_margin, 6),
                "small_selected_margin": "yes" if nearest_margin and nearest_margin <= SMALL_SELECTED_MARGIN_THRESHOLD else "",
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -int(row["picked_count"]),
            -to_float(row["avg_strategy_score"]),
            txt(row["position"]),
            txt(row["player_name"]),
        ),
    )


def player_table_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[list[Any]]:
    selected = rows[:limit] if limit else rows
    return [
        [
            row["player_name"],
            row["team_id"],
            row["position"],
            row["picked_count"],
            row["robustness_bucket"],
            row["avg_ev"],
            row["min_conditional_start_prob"],
            row["availability_risks"],
            row["manual_warnings"],
            row["small_selected_margin"],
        ]
        for row in selected
    ]


def write_markdown(rows: list[dict[str, Any]], strategies: dict[str, Any], context: dict[str, Any]) -> None:
    bucket_counts = defaultdict(int)
    for row in rows:
        bucket_counts[row["robustness_bucket"]] += 1

    warning_rows = [row for row in rows if row["manual_warnings"]]
    small_margin_rows = [row for row in rows if row["small_selected_margin"]]

    captain_rows = []
    for strategy in STRATEGY_ORDER:
        item = strategies.get(strategy, {})
        summary = item.get("best_summary", {})
        captain_rows.append(
            [
                summary.get("display_name_da") or strategy,
                summary.get("recommended_captain", ""),
                fmt(summary.get("captain_expected_growth"), 3),
                summary.get("captain_reason", ""),
            ]
        )

    lines = [
        "# Strategy Sensitivity Report",
        "",
        "Focused report for robust picks, fragile picks, and manual follow-up flags across the four user-facing strategies.",
        "",
        "## Context",
        "",
        f"- generated_at: {context.get('generated_at', '')}",
        f"- target_round: {context.get('target_round', '')}",
        f"- next_round_display_name: {context.get('next_round_display_name', '')}",
        "",
        "## Robustness Counts",
        "",
        table(
            ["Bucket", "Players"],
            [
                ["core_4_of_4", bucket_counts["core_4_of_4"]],
                ["strong_3_of_4", bucket_counts["strong_3_of_4"]],
                ["shared_2_of_4", bucket_counts["shared_2_of_4"]],
                ["single_strategy", bucket_counts["single_strategy"]],
            ],
        ),
        "",
        "## Robust Picks",
        "",
        table(
            ["Player", "Team", "Pos", "Picked", "Bucket", "Avg EV", "Min cond", "Risk", "Warnings", "Small margin"],
            player_table_rows([row for row in rows if int(row["picked_count"]) >= 3]),
        ),
        "",
        "## Manual Follow-Up",
        "",
        table(
            ["Player", "Team", "Pos", "Picked", "Bucket", "Avg EV", "Min cond", "Risk", "Warnings", "Small margin"],
            player_table_rows(warning_rows),
        ),
        "",
        "## Small Selected Margins",
        "",
        "This is a selected-squad sensitivity proxy: it flags players whose score is close to another selected teammate in the same position group. It is not a full replacement-candidate margin.",
        "",
        table(
            ["Player", "Team", "Pos", "Picked", "Bucket", "Avg EV", "Min cond", "Risk", "Warnings", "Small margin"],
            player_table_rows(small_margin_rows),
        ),
        "",
        "## Captain Check",
        "",
        table(["Strategy", "Captain", "Expected growth", "Reason"], captain_rows),
        "",
        "## All Selected Players",
        "",
        table(
            ["Player", "Team", "Pos", "Picked", "Bucket", "Avg EV", "Min cond", "Risk", "Warnings", "Small margin"],
            player_table_rows(rows),
        ),
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    strategies = read_json(STRATEGIES_PATH)
    display_names = read_json(DISPLAY_NAMES_PATH) if DISPLAY_NAMES_PATH.exists() else {}
    context = read_json(CONTEXT_PATH) if CONTEXT_PATH.exists() else {}

    rows = aggregate_rows(strategies, display_names)
    write_csv(OUT_CSV, rows)
    write_markdown(rows, strategies, context)

    bucket_counts = defaultdict(int)
    for row in rows:
        bucket_counts[row["robustness_bucket"]] += 1

    print(f"Skrevet: {OUT_CSV.relative_to(ROOT)}")
    print(f"Skrevet: {OUT_MD.relative_to(ROOT)}")
    print(
        "Robustness: "
        + "; ".join(
            f"{bucket}={bucket_counts[bucket]}"
            for bucket in ["core_4_of_4", "strong_3_of_4", "shared_2_of_4", "single_strategy"]
        )
    )
    print(f"Manual follow-up: {sum(1 for row in rows if row['manual_warnings'])}")
    print(f"Small selected margins: {sum(1 for row in rows if row['small_selected_margin'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
