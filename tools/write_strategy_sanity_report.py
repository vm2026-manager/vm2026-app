from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

STRATEGIES_JSON_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA_DIR / "strategy_comparison_report.csv"
CONTEXT_PATH = DATA_DIR / "current_strategy_context.json"
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
            fmt(player.get("p_6_points_after_2"), 3),
            fmt(player.get("round3_rotation_factor"), 3),
        ]
        for player in squad
    ]


def main() -> int:
    strategies = json.loads(STRATEGIES_JSON_PATH.read_text(encoding="utf-8"))
    comparison_rows = list(csv.DictReader(COMPARISON_PATH.open(encoding="utf-8-sig", newline="")))
    context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8")) if CONTEXT_PATH.exists() else {}
    ev_rows = list(csv.DictReader(PLAYER_EV_PATH.open(encoding="utf-8-sig", newline="")))
    player_pool = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))

    summary_rows = []
    for row in comparison_rows:
        summary_rows.append(
            [
                row["display_name_da"],
                row["strategy"],
                row["formation"],
                f"{int(float(row['total_price'])):,}",
                fmt(row["total_score"], 3),
                fmt(row["total_ev"], 3),
                fmt(row["avg_conditional_start_prob"], 4),
                row["high_risk_players"],
                row["recommended_captain"],
                fmt(row["captain_expected_growth"], 3),
            ]
        )

    availability_rows = []
    team_rows = []
    captain_rows = []
    for row in comparison_rows:
        item = strategies[row["strategy"]]
        squad = item["best_squad"]
        availability_rows.append(
            [
                row["display_name_da"],
                fmt(row["avg_start_prob"], 4),
                fmt(row["avg_conditional_start_prob"], 4),
                risk_counts(squad),
            ]
        )
        team_rows.append([row["display_name_da"], team_counts(squad)])
        captain_rows.append(
            [
                row["display_name_da"],
                row["recommended_captain"],
                row["captain_round"],
                fmt(row["captain_expected_growth"], 3),
                row["captain_reason"],
            ]
        )

    lines = [
        "# Strategi Sanity Report",
        "",
        "Rapporten viser de fire brugerrettede strategier efter strategirydningen.",
        "",
        "## Aktuel Strategikontekst",
        "",
        f"- current_time_dk: {context.get('current_time_dk', '')}",
        f"- target_round: {context.get('target_round', '')}",
        f"- next_round_display_name: {context.get('next_round_display_name', '')}",
        f"- remaining_matches_in_target_round: {context.get('remaining_matches_in_target_round', '')}",
        "",
        "## Strategioversigt",
        "",
        table(
            ["Dansk navn", "Teknisk strategi", "Formation", "Pris", "Score", "EV", "Avg cond", "High risk", "Kaptajn", "Kaptajn vækst"],
            summary_rows,
        ),
        "",
        "## Kaptajn",
        "",
        table(["Strategi", "Kaptajn", "Runde", "Forventet vækst", "Årsag"], captain_rows),
        "",
        "## Spillerliste Pr. Strategi",
        "",
    ]

    for row in comparison_rows:
        strategy = row["strategy"]
        item = strategies[strategy]
        lines.extend(
            [
                f"### {row['display_name_da']}",
                "",
                table(
                    ["Spiller", "Hold", "Pos", "Pris", "EV", "Strategy score", "Start", "Conditional", "Risk", "P 6p efter 2", "R3 faktor"],
                    player_rows(item["best_squad"]),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Landefordeling",
            "",
            table(["Strategi", "Fordeling"], team_rows),
            "",
            "## Start Og Availability",
            "",
            table(["Strategi", "Avg start_prob", "Avg conditional_start_prob", "Availability risk"], availability_rows),
            "",
            "## Strateginoter",
            "",
            "- Næste runde bruger dynamisk target_round og scorer hårdest på den kommende runde.",
            "- 1. + 2. runde vægter de to første runder og straffer spillere, der kun topper i én kamp.",
            "- Gruppespil reducerer runde 3-bidrag via p_6_points_after_2 og round3_rotation_factor.",
            "- Lang sigt bruger team_market/team_long_run som proxy for turneringsvinderstyrke.",
            "- safe_starters er ikke længere en separat brugerrettet hovedstrategi; starterfokus er indbygget i alle fire strategier.",
            f"- Datagrundlag brugt til rapporten: {len(ev_rows)} EV-rækker og {len(player_pool)} player_pool-rækker.",
            "",
        ]
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Strategier: {len(strategies)}")
    print(f"Target round: {context.get('next_round_display_name', '')}")
    for row in comparison_rows:
        print(f"- {row['display_name_da']}: {row['formation']} | kaptajn={row['recommended_captain']} | EV={fmt(row['total_ev'], 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
