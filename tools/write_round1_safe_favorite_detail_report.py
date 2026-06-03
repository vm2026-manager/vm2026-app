from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SQUADS_PATH = DATA / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA / "strategy_comparison_report.csv"
PLAYER_EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
MATCH_ODDS_PATH = DATA / "match_odds_probs.csv"
MULTIPLIERS_PATH = DATA / "fixture_strength_multipliers.csv"
CLEAN_SHEET_PATH = DATA / "clean_sheet_probs_bet365.csv"

OUT_MD = DATA / "round1_safe_favorite_detail_report.md"
OUT_CSV = DATA / "round1_safe_favorite_detail_report.csv"

OUT_FIELDS = [
    "player_name",
    "team_id",
    "position",
    "price",
    "EV",
    "strategy_score",
    "start_prob",
    "conditional_start_prob",
    "availability_risk",
    "round1_opponent",
    "round1_win_prob",
    "round1_goal_multiplier",
    "round1_assist_multiplier",
    "round1_clean_sheet_prob",
    "round1_clean_sheet_multiplier",
    "kort_note",
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


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_fixture_context() -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for row in read_csv(MULTIPLIERS_PATH):
        match_id = int(txt(row.get("match_id")) or "9999")
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        sides = [
            (
                home,
                away,
                to_float(row.get("home_win_prob_fair")),
                to_float(row.get("home_goal_multiplier"), 1.0),
                to_float(row.get("home_assist_multiplier"), 1.0),
                to_float(row.get("home_clean_sheet_prob_fair")),
                to_float(row.get("home_clean_sheet_multiplier"), 1.0),
            ),
            (
                away,
                home,
                to_float(row.get("away_win_prob_fair")),
                to_float(row.get("away_goal_multiplier"), 1.0),
                to_float(row.get("away_assist_multiplier"), 1.0),
                to_float(row.get("away_clean_sheet_prob_fair")),
                to_float(row.get("away_clean_sheet_multiplier"), 1.0),
            ),
        ]
        for team, opponent, win_prob, goal_mult, assist_mult, cs_prob, cs_mult in sides:
            current = contexts.get(team)
            if current is None or match_id < current["match_id"]:
                contexts[team] = {
                    "match_id": match_id,
                    "opponent": opponent,
                    "win_prob": win_prob,
                    "goal_multiplier": goal_mult,
                    "assist_multiplier": assist_mult,
                    "clean_sheet_prob": cs_prob,
                    "clean_sheet_multiplier": cs_mult,
                }
    return contexts


def team_clean_sheet_lookup() -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in read_csv(CLEAN_SHEET_PATH):
        lookup[(txt(row.get("match_id")), txt(row.get("team_id")).upper())] = to_float(row.get("clean_sheet_prob_fair"))
    return lookup


def note_for(player: dict[str, Any], context: dict[str, Any]) -> str:
    notes: list[str] = []
    position = txt(player.get("position")).upper()
    ev = to_float(player.get("optimizer_ev"))
    cond = to_float(player.get("conditional_start_prob"))
    risk = txt(player.get("availability_risk"))
    win_prob = to_float(context.get("win_prob"))
    cs_prob = to_float(context.get("clean_sheet_prob"))

    if risk == "high_risk":
        notes.append("manuel tjek: high_risk")
    if cond < 0.75:
        notes.append("manuel tjek: lav conditional start")
    elif cond >= 0.90:
        notes.append("stærk startsikkerhed")
    if win_prob >= 0.70:
        notes.append("klar runde 1-favorit")
    elif win_prob < 0.55:
        notes.append("ikke favorit i runde 1")
    if position in {"GK", "DEF"}:
        if cs_prob >= 0.50:
            notes.append("stærk clean sheet-profil")
        elif cs_prob < 0.30:
            notes.append("svag clean sheet-profil")
    if ev >= 4.5:
        notes.append("høj EV")
    elif ev < 3.0:
        notes.append("lav EV")

    return "; ".join(notes) if notes else "ok"


def build_rows() -> list[dict[str, str]]:
    data = read_json(SQUADS_PATH)
    squad = data["round1_safe_favorite"]["best_squad"]
    contexts = first_fixture_context()
    cs_lookup = team_clean_sheet_lookup()

    rows: list[dict[str, str]] = []
    for player in squad:
        team = txt(player.get("team_id")).upper()
        context = contexts.get(team, {})
        match_id = txt(context.get("match_id"))
        cs_prob = cs_lookup.get((match_id, team), context.get("clean_sheet_prob"))
        is_defensive = txt(player.get("position")).upper() in {"GK", "DEF"}

        rows.append(
            {
                "player_name": txt(player.get("player_name")),
                "team_id": team,
                "position": txt(player.get("position")),
                "price": txt(player.get("price")),
                "EV": fmt(to_float(player.get("optimizer_ev")), 6),
                "strategy_score": fmt(to_float(player.get("strategy_score")), 6),
                "start_prob": fmt(to_float(player.get("start_prob")), 4),
                "conditional_start_prob": fmt(to_float(player.get("conditional_start_prob")), 4),
                "availability_risk": txt(player.get("availability_risk")),
                "round1_opponent": txt(context.get("opponent")),
                "round1_win_prob": fmt(to_float(context.get("win_prob")), 4),
                "round1_goal_multiplier": fmt(to_float(context.get("goal_multiplier"), 1.0), 4),
                "round1_assist_multiplier": fmt(to_float(context.get("assist_multiplier"), 1.0), 4),
                "round1_clean_sheet_prob": fmt(to_float(cs_prob), 4) if is_defensive else "",
                "round1_clean_sheet_multiplier": fmt(to_float(context.get("clean_sheet_multiplier"), 1.0), 4) if is_defensive else "",
                "kort_note": note_for(player, {**context, "clean_sheet_prob": cs_prob}),
            }
        )
    return rows


def write_csv_report(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def comparison_summary() -> dict[str, str]:
    for row in read_csv(COMPARISON_PATH):
        if txt(row.get("strategy")) == "round1_safe_favorite":
            return row
    return {}


def write_md_report(rows: list[dict[str, str]]) -> None:
    summary = comparison_summary()
    manual = [
        row for row in rows
        if "manuel tjek" in row["kort_note"] or row["availability_risk"] == "high_risk"
    ]

    lines: list[str] = []
    lines.append("# Round1 Safe Favorite Detail Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Formation: {summary.get('formation', '')}")
    lines.append(f"- Total price: {int(to_float(summary.get('total_price'))) / 1_000_000:.1f} mio.")
    lines.append(f"- Total EV: {fmt(to_float(summary.get('total_ev')), 6)}")
    lines.append(f"- Total score: {fmt(to_float(summary.get('total_score')), 6)}")
    lines.append(f"- Avg conditional start: {fmt(to_float(summary.get('avg_conditional_start_prob')), 4)}")
    lines.append(f"- High risk players: {summary.get('high_risk_players', '')}")
    lines.append("")
    lines.append("## Spillere")
    lines.append("")
    lines.append("| Spiller | Land | Pos | Pris | EV | Score | Start | Cond start | Risk | R1 mod | R1 win | CS prob | CS mult | Note |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["player_name"],
                    row["team_id"],
                    row["position"],
                    row["price"],
                    row["EV"],
                    row["strategy_score"],
                    row["start_prob"],
                    row["conditional_start_prob"],
                    row["availability_risk"],
                    row["round1_opponent"],
                    row["round1_win_prob"],
                    row["round1_clean_sheet_prob"],
                    row["round1_clean_sheet_multiplier"],
                    row["kort_note"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Manuel Tjek")
    lines.append("")
    if manual:
        for row in manual:
            lines.append(f"- {row['player_name']} ({row['team_id']}, {row['position']}): {row['kort_note']}")
    else:
        lines.append("- Ingen oplagte manuelle tjek.")
    lines.append("")
    lines.append("## Datakilder")
    lines.append("")
    for path in [
        SQUADS_PATH,
        COMPARISON_PATH,
        PLAYER_EV_PATH,
        PLAYER_POOL_PATH,
        MATCH_ODDS_PATH,
        MULTIPLIERS_PATH,
        CLEAN_SHEET_PATH,
    ]:
        lines.append(f"- `{path.relative_to(ROOT)}`")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv_report(rows)
    write_md_report(rows)

    print(f"Skrevet: {OUT_MD.relative_to(ROOT)}")
    print(f"Skrevet: {OUT_CSV.relative_to(ROOT)}")
    print(f"Spillere: {len(rows)}")
    manual = [row for row in rows if "manuel tjek" in row["kort_note"] or row["availability_risk"] == "high_risk"]
    print(f"Manuelle tjek: {len(manual)}")
    for row in manual:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['kort_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
