from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

STRATEGIES_JSON_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA_DIR / "strategy_comparison_report.csv"
PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
OUT_PATH = DATA_DIR / "strategy_sanity_report.md"


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
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


def risk_counts(squad: list[dict[str, Any]]) -> str:
    counts = Counter(txt(player.get("availability_risk")) or "unknown" for player in squad)
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def team_counts(squad: list[dict[str, Any]]) -> str:
    counts = Counter(txt(player.get("team_id")) for player in squad)
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def player_rows(squad: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            player.get("player_name", ""),
            player.get("team_id", ""),
            player.get("position", ""),
            player.get("price", ""),
            fmt(player.get("optimizer_ev"), 3),
            fmt(player.get("strategy_score"), 3),
            fmt(player.get("start_prob"), 3),
            fmt(player.get("conditional_start_prob"), 3),
            player.get("availability_risk", ""),
        ]
        for player in squad
    ]


def names(squad: list[dict[str, Any]]) -> set[str]:
    return {txt(player.get("player_name")) for player in squad}


def player_lookup(squad: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {txt(player.get("player_name")): player for player in squad}


def main() -> int:
    with STRATEGIES_JSON_PATH.open(encoding="utf-8") as f:
        strategies = json.load(f)

    with COMPARISON_PATH.open(encoding="utf-8-sig", newline="") as f:
        comparison_rows = list(csv.DictReader(f))

    with PLAYER_EV_PATH.open(encoding="utf-8-sig", newline="") as f:
        ev_rows = list(csv.DictReader(f))

    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        player_pool = json.load(f)

    balanced_squad = strategies["balanced"]["best_squad"]
    balanced_names = names(balanced_squad)
    all_name_counts: Counter[str] = Counter()
    for item in strategies.values():
        all_name_counts.update(names(item["best_squad"]))

    summary_rows = []
    for row in comparison_rows:
        summary_rows.append(
            [
                row["strategy"],
                row["formation"],
                f"{int(float(row['total_price'])):,}",
                fmt(row["total_score"], 3),
                fmt(row["total_ev"], 3),
                row["high_risk_players"],
            ]
        )

    overlap_rows = []
    for strategy, item in strategies.items():
        strategy_names = names(item["best_squad"])
        overlap = strategy_names & balanced_names
        only_strategy = strategy_names - balanced_names
        only_balanced = balanced_names - strategy_names
        overlap_rows.append(
            [
                strategy,
                len(overlap),
                ", ".join(sorted(only_strategy)) or "-",
                ", ".join(sorted(only_balanced)) or "-",
            ]
        )

    unique_players = []
    for strategy, item in strategies.items():
        for player in item["best_squad"]:
            name = txt(player.get("player_name"))
            if all_name_counts[name] == 1:
                unique_players.append([strategy, name, player.get("team_id", ""), player.get("position", ""), fmt(player.get("strategy_score"), 3)])

    availability_rows = []
    for strategy, item in strategies.items():
        squad = item["best_squad"]
        summary = item["best_summary"]
        availability_rows.append(
            [
                strategy,
                fmt(summary.get("avg_start_prob"), 4),
                fmt(summary.get("avg_conditional_start_prob"), 4),
                risk_counts(squad),
            ]
        )

    team_rows = [
        [strategy, team_counts(item["best_squad"])]
        for strategy, item in strategies.items()
    ]

    def pulled_in(strategy: str) -> list[list[Any]]:
        squad = strategies[strategy]["best_squad"]
        lookup = player_lookup(squad)
        return player_rows([lookup[name] for name in sorted(names(squad) - balanced_names)])

    clean_sheet_same_as_balanced = names(strategies["clean_sheet_stack"]["best_squad"]) == balanced_names
    fixture_overlap = len(names(strategies["fixture_attack"]["best_squad"]) & balanced_names)
    safe_overlap = len(names(strategies["safe_starters"]["best_squad"]) & balanced_names)

    lines = [
        "# Strategy Sanity Report",
        "",
        "Denne rapport sammenligner strategi-presets uden at ændre optimizer, EV, player_pool eller UI.",
        "",
        "## 1. Bedste Hold Pr. Strategi",
        "",
        table(["Strategi", "Formation", "Pris", "Total score", "Total EV", "High risk"], summary_rows),
        "",
        "## 2. Spillerliste Pr. Strategi",
        "",
    ]

    for strategy, item in strategies.items():
        lines.extend(
            [
                f"### {strategy}",
                "",
                table(
                    ["Spiller", "Hold", "Pos", "Pris", "EV", "Strategy score", "Start", "Conditional", "Risk"],
                    player_rows(item["best_squad"]),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 3. Overlap Mod Balanced",
            "",
            table(["Strategi", "Overlap", "Kun i strategi", "Kun i balanced"], overlap_rows),
            "",
            "## 4. Spillere Kun Valgt I Én Strategi",
            "",
            table(["Strategi", "Spiller", "Hold", "Pos", "Strategy score"], unique_players),
            "",
            "## 5. Hold-/Landefordeling",
            "",
            table(["Strategi", "Fordeling"], team_rows),
            "",
            "## 6. Start Og Availability",
            "",
            table(["Strategi", "Avg start_prob", "Avg conditional_start_prob", "Availability risk"], availability_rows),
            "",
            "## 7. Fixture Attack Ind Ift. Balanced",
            "",
            table(["Spiller", "Hold", "Pos", "Pris", "EV", "Strategy score", "Start", "Conditional", "Risk"], pulled_in("fixture_attack")),
            "",
            "## 8. Safe Starters Ind Ift. Balanced",
            "",
            table(["Spiller", "Hold", "Pos", "Pris", "EV", "Strategy score", "Start", "Conditional", "Risk"], pulled_in("safe_starters")),
            "",
            "## 9. Long Run Value Ind Ift. Balanced",
            "",
            table(["Spiller", "Hold", "Pos", "Pris", "EV", "Strategy score", "Start", "Conditional", "Risk"], pulled_in("long_run_value")),
            "",
            "## 10. Kort Vurdering",
            "",
            f"- Strategierne giver forskellige hold, men ikke radikalt forskellige: fixture_attack overlapper {fixture_overlap}/11 med balanced, safe_starters overlapper {safe_overlap}/11.",
            "- safe_starters ser relevant ud som preset, fordi high_risk falder fra 4 til 1 og avg conditional_start_prob stiger.",
            "- fixture_attack bør vises med strategy score separat fra total_ev, fordi dens score indeholder fixture-boost og derfor ikke er direkte sammenlignelig med balanced total_ev.",
            "- clean_sheet_stack differentierer ikke nok lige nu." if clean_sheet_same_as_balanced else "- clean_sheet_stack differentierer noget, men bør stadig vurderes mod balanced overlap og defensiv sammensætning.",
            f"- Datagrundlag brugt til rapporten: {len(ev_rows)} EV-rækker og {len(player_pool)} player_pool-rækker.",
            "",
        ]
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Strategier: {len(strategies)}")
    print(f"Clean_sheet_stack samme spillere som balanced: {clean_sheet_same_as_balanced}")
    print(f"Fixture_attack overlap med balanced: {fixture_overlap}/11")
    print(f"Safe_starters overlap med balanced: {safe_overlap}/11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
