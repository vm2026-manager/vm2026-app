from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "strategy_squad_exports"
STANDARD_OUTPUT_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
OUT_MD = OUT_DIR / "mbappe_premium_long_run_report.md"
OUT_CSV = OUT_DIR / "mbappe_premium_long_run_report.csv"

sys.path.insert(0, str(TOOLS_DIR))
import optimize_squad_group_stage as optimizer  # noqa: E402


def solve_long_run_with_mbappe(
    players: pd.DataFrame,
    formation_name: str,
    formation: dict[str, int],
    mbappe_idx: int,
) -> pd.DataFrame:
    problem = pulp.LpProblem(f"mbappe_long_run_{formation_name.replace('-', '_')}", pulp.LpMaximize)
    variables = {
        idx: pulp.LpVariable(f"pick_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in players.index
    }
    score_expr = pulp.lpSum(
        float(players.loc[idx, "score_long_run"]) * variables[idx]
        for idx in players.index
    )
    total_price_expr = pulp.lpSum(
        float(players.loc[idx, "price_m"]) * variables[idx]
        for idx in players.index
    )
    underuse = pulp.LpVariable("long_run_budget_underuse", lowBound=0, cat="Continuous")

    problem += score_expr + 0.025 * total_price_expr - 0.18 * underuse
    problem += underuse >= 49.0 - total_price_expr
    problem += pulp.lpSum(variables.values()) == optimizer.SQUAD_SIZE
    problem += total_price_expr <= optimizer.BUDGET_M
    problem += variables[mbappe_idx] == 1

    problem += pulp.lpSum(
        variables[idx]
        for idx in players.index
        if optimizer.txt(players.loc[idx, "availability_risk"]) == "high_risk"
    ) <= 0
    problem += pulp.lpSum(
        float(players.loc[idx, "conditional_start_prob"]) * variables[idx]
        for idx in players.index
    ) >= 0.84 * optimizer.SQUAD_SIZE

    for idx in players.index[players["conditional_start_prob"] < 0.72].tolist():
        problem += variables[idx] == 0
    for idx in players.index[players["manual_avoid"]].tolist():
        problem += variables[idx] == 0

    tournament_strength = (
        0.75 * players["team_long_run_score"] + 0.25 * players["team_market_score"]
    ).clip(lower=0.0)
    strong_team_indices = players.index[tournament_strength >= 0.50].tolist()
    weak_team_indices = players.index[tournament_strength < 0.35].tolist()
    problem += pulp.lpSum(variables[idx] for idx in strong_team_indices) >= 7
    problem += pulp.lpSum(variables[idx] for idx in weak_team_indices) <= 2

    for position, count in formation.items():
        indices = players.index[players["position"] == position].tolist()
        problem += pulp.lpSum(variables[idx] for idx in indices) == count

    for _, team_players in players.groupby("team_id"):
        problem += pulp.lpSum(variables[idx] for idx in team_players.index.tolist()) <= optimizer.MAX_PER_TEAM

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[problem.status] != "Optimal":
        return pd.DataFrame()

    picked = [idx for idx, variable in variables.items() if variable.value() == 1]
    squad = players.loc[picked].copy()
    squad["strategy"] = "long_run"
    squad["selected_formation"] = formation_name
    squad["strategy_score"] = squad["score_long_run"]
    return squad.sort_values(
        ["position", "strategy_score", "optimizer_ev"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def standard_players(formation_entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(player.get("player_id")): player
        for player in formation_entry.get("squad", [])
    }


def player_names(players: dict[str, dict[str, Any]], ids: set[str]) -> str:
    return ", ".join(
        sorted(str(players[player_id].get("player_name") or player_id) for player_id in ids)
    )


def format_millions(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " mio."


def format_score(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    players = optimizer.load_players()
    mbappe_rows = players[
        players["player_name"].astype(str).str.casefold().str.contains("kylian mbapp")
    ]
    if len(mbappe_rows) != 1:
        raise RuntimeError(f"Forventede præcis én Kylian Mbappé, fandt {len(mbappe_rows)}")
    mbappe_idx = int(mbappe_rows.index[0])

    standard_data = json.loads(STANDARD_OUTPUT_PATH.read_text(encoding="utf-8-sig"))
    standard_formations = standard_data["long_run"]["squads_by_formation"]

    csv_rows: list[dict[str, Any]] = []
    md = [
        "# Lang sigt – Mbappé/premium audit",
        "",
        "Audit-only rapport. Standardstrategier og modeldata ændres ikke.",
        "Hvert hold bruger optimizerens normale long_run-score og constraints, men Kylian Mbappé er tvunget med.",
        "",
    ]

    for formation_name, formation in optimizer.FORMATIONS.items():
        standard_entry = standard_formations.get(formation_name, {})
        standard_squad = standard_players(standard_entry)
        standard_summary = standard_entry.get("summary", {})
        standard_score = float(standard_summary.get("total_score") or 0.0)
        standard_forwards = [
            str(player.get("player_name") or player_id)
            for player_id, player in standard_squad.items()
            if str(player.get("position") or "").upper() == "FWD"
        ]
        squad = solve_long_run_with_mbappe(players, formation_name, formation, mbappe_idx)

        md.extend([f"## {formation_name}", ""])
        if squad.empty:
            md.extend([
                "**Ingen gyldig løsning.** Mbappé kunne ikke inkluderes under de normale constraints.",
                "",
            ])
            csv_rows.append({
                "formation": formation_name,
                "status": "no_valid_solution",
                "budget_m": "",
                "long_run_score": "",
                "optimizer_ev": "",
                "standard_long_run_score": standard_score,
                "score_loss": "",
                "standard_forwards": ", ".join(standard_forwards),
                "forwards": "",
                "players_out": "",
                "players_in": "",
            })
            continue

        squad_by_id = {
            str(row["player_id"]): row
            for _, row in squad.iterrows()
        }
        squad_ids = set(squad_by_id)
        standard_ids = set(standard_squad)
        players_out_ids = standard_ids - squad_ids
        players_in_ids = squad_ids - standard_ids
        total_price = float(squad["price_m"].sum())
        total_score = float(squad["strategy_score"].sum())
        total_ev = float(squad["optimizer_ev"].sum())
        score_loss = standard_score - total_score
        forwards = squad[squad["position"] == "FWD"]["player_name"].astype(str).tolist()
        players_out = player_names(standard_squad, players_out_ids)
        players_in = ", ".join(
            sorted(str(squad_by_id[player_id]["player_name"]) for player_id in players_in_ids)
        )

        md.extend([
            f"- **Budget:** {format_millions(total_price)}",
            f"- **Long-run-score:** {format_score(total_score)}",
            f"- **Optimizer-EV:** {format_score(total_ev)}",
            f"- **Scoretab mod standard:** {format_score(score_loss)}",
            f"- **Normal angriberpakke:** {', '.join(standard_forwards)}",
            f"- **Mbappé-angriberpakke:** {', '.join(forwards)}",
            f"- **Ud:** {players_out or 'Ingen'}",
            f"- **Ind:** {players_in or 'Ingen'}",
            "",
            "| Pos. | Spiller | Land | Pris | Start | Long-run-score | Optimizer-EV |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for _, player in squad.iterrows():
            md.append(
                f"| {player['position']} | {player['player_name']} | {player['team_id']} | "
                f"{format_millions(float(player['price_m']))} | "
                f"{float(player['start_prob']) * 100:.0f}% | "
                f"{format_score(float(player['strategy_score']))} | "
                f"{format_score(float(player['optimizer_ev']))} |"
            )
        md.append("")

        csv_rows.append({
            "formation": formation_name,
            "status": "ok",
            "budget_m": round(total_price, 3),
            "long_run_score": round(total_score, 6),
            "optimizer_ev": round(total_ev, 6),
            "standard_long_run_score": round(standard_score, 6),
            "score_loss": round(score_loss, 6),
            "standard_forwards": ", ".join(standard_forwards),
            "forwards": ", ".join(forwards),
            "players_out": players_out,
            "players_in": players_in,
        })

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    valid = sum(row["status"] == "ok" for row in csv_rows)
    print(f"Gyldige Mbappé-hold: {valid}/{len(optimizer.FORMATIONS)}")
    print(f"Markdown: {OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"CSV: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
