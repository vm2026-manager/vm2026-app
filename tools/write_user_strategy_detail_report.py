from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA / "strategy_comparison_report.csv"
SANITY_PATH = DATA / "strategy_sanity_report.md"
PLAYER_EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
FIXTURE_MULTIPLIERS_PATH = DATA / "fixture_strength_multipliers.csv"
MATCH_ODDS_PATH = DATA / "match_odds_probs.csv"
CONTEXT_PATH = DATA / "current_strategy_context.json"

OUT_MD = DATA / "user_strategy_detail_report.md"
OUT_CSV = DATA / "user_strategy_detail_report.csv"

STRATEGIES = ["next_round", "round1_2", "group_stage", "long_run"]
CAPTAIN_COMPARE_NAMES = [
    "Christoph Baumgartner",
    "Cristiano Ronaldo",
    "Vinicius Junior",
    "Raul Jimenez",
    "Kerem Akturkoglu",
    "Roberto Alvarado",
    "Diogo Costa",
]

USER_MANUAL_NOTES = {
    "Christoph Baumgartner": "BRUGERNOTE 2026-06-03: meldt skadet/ikke længere udtaget til VM; bør undgås indtil data er opdateret.",
}

CSV_FIELDS = [
    "strategy",
    "display_name_da",
    "formation",
    "total_price",
    "total_ev",
    "total_score",
    "avg_conditional_start_prob",
    "high_risk_players",
    "recommended_captain",
    "captain_expected_growth",
    "captain_score",
    "captain_reason",
    "player_name",
    "team_id",
    "position",
    "price",
    "EV",
    "strategy_score",
    "start_prob",
    "conditional_start_prob",
    "availability_risk",
    "manual_captain_status",
    "relevant_rounds",
    "opponents",
    "win_probabilities",
    "clean_sheet_probabilities",
    "goal_assist_fixture_info",
    "note",
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


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return txt(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def relevant_rounds(strategy: str, context: dict[str, Any]) -> list[int]:
    target = int(context.get("target_round") or 1)
    if strategy == "next_round":
        return [target]
    if strategy == "round1_2":
        return [1, 2]
    return [1, 2, 3]


def round_text(player: dict[str, Any], rounds: list[int], field: str, digits: int = 3) -> str:
    parts = []
    for rnd in rounds:
        value = player.get(f"round{rnd}_{field}")
        if field == "opponent":
            parts.append(f"R{rnd}:{txt(value)}")
        else:
            parts.append(f"R{rnd}:{fmt(value, digits)}")
    return "; ".join(parts)


def goal_assist_info(player: dict[str, Any], rounds: list[int]) -> str:
    if txt(player.get("position")).upper() not in {"MID", "FWD"}:
        return ""
    return "; ".join(
        f"R{rnd}: goal {fmt(player.get(f'round{rnd}_goal_multiplier'), 3)} / assist {fmt(player.get(f'round{rnd}_assist_multiplier'), 3)}"
        for rnd in rounds
    )


def clean_sheet_info(player: dict[str, Any], rounds: list[int]) -> str:
    if txt(player.get("position")).upper() not in {"GK", "DEF"}:
        return ""
    return "; ".join(
        f"R{rnd}:{fmt(player.get(f'round{rnd}_clean_sheet_prob'), 3)}"
        for rnd in rounds
    )


def note_for_player(strategy: str, player: dict[str, Any], rounds: list[int]) -> str:
    notes: list[str] = []
    name = txt(player.get("player_name"))
    position = txt(player.get("position")).upper()
    ev = to_float(player.get("optimizer_ev"))
    cond = to_float(player.get("conditional_start_prob"))
    risk = txt(player.get("availability_risk"))

    if name in USER_MANUAL_NOTES:
        notes.append(USER_MANUAL_NOTES[name])
    captain_status = txt(player.get("manual_captain_status")).lower()
    if captain_status == "avoid":
        notes.append("captain_avoid: maa ikke anbefales som kaptajn")
    elif captain_status == "check":
        notes.append("captain_check")
    if risk == "high_risk":
        notes.append("manuel tjek: high_risk")
    if cond < 0.75:
        notes.append("manuel tjek: lav conditional start")
    elif cond >= 0.90:
        notes.append("stærk startsikkerhed")
    if ev >= 4.5:
        notes.append("høj EV")

    win_values = [to_float(player.get(f"round{rnd}_win_prob")) for rnd in rounds]
    if win_values and max(win_values) >= 0.70:
        notes.append("favoritkamp i relevant horisont")
    if position in {"GK", "DEF"}:
        cs_values = [to_float(player.get(f"round{rnd}_clean_sheet_prob")) for rnd in rounds]
        if cs_values and max(cs_values) >= 0.50:
            notes.append("stærk clean sheet-profil")
    else:
        attack_values = [
            max(to_float(player.get(f"round{rnd}_goal_multiplier"), 1.0), to_float(player.get(f"round{rnd}_assist_multiplier"), 1.0))
            for rnd in rounds
        ]
        if attack_values and max(attack_values) >= 1.25:
            notes.append("godt offensivt kampmiljø")

    if strategy == "group_stage" and to_float(player.get("p_6_points_after_2")) >= 0.35:
        notes.append(f"runde 3-rotation proxy p6={fmt(player.get('p_6_points_after_2'), 3)}")
    if strategy == "long_run":
        notes.append("valgt i lang sigt-kontekst")
    return "; ".join(notes) if notes else "ok"


def build_rows(strategies: dict[str, Any], comparison: dict[str, dict[str, str]], context: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy in STRATEGIES:
        summary = comparison[strategy]
        rounds = relevant_rounds(strategy, context)
        for player in strategies[strategy]["best_squad"]:
            rows.append(
                {
                    "strategy": strategy,
                    "display_name_da": summary["display_name_da"],
                    "formation": summary["formation"],
                    "total_price": summary["total_price"],
                    "total_ev": summary["total_ev"],
                    "total_score": summary["total_score"],
                    "avg_conditional_start_prob": summary["avg_conditional_start_prob"],
                    "high_risk_players": summary["high_risk_players"],
                    "recommended_captain": summary["recommended_captain"],
                    "captain_expected_growth": summary["captain_expected_growth"],
                    "captain_score": summary.get("captain_score", ""),
                    "captain_reason": summary["captain_reason"],
                    "player_name": txt(player.get("player_name")),
                    "team_id": txt(player.get("team_id")),
                    "position": txt(player.get("position")),
                    "price": txt(player.get("price")),
                    "EV": fmt(player.get("optimizer_ev"), 6),
                    "strategy_score": fmt(player.get("strategy_score"), 6),
                    "start_prob": fmt(player.get("start_prob"), 4),
                    "conditional_start_prob": fmt(player.get("conditional_start_prob"), 4),
                    "availability_risk": txt(player.get("availability_risk")),
                    "manual_captain_status": txt(player.get("manual_captain_status")),
                    "relevant_rounds": "+".join(str(rnd) for rnd in rounds),
                    "opponents": round_text(player, rounds, "opponent"),
                    "win_probabilities": round_text(player, rounds, "win_prob", 3),
                    "clean_sheet_probabilities": clean_sheet_info(player, rounds),
                    "goal_assist_fixture_info": goal_assist_info(player, rounds),
                    "note": note_for_player(strategy, player, rounds),
                }
            )
    return rows


def player_ev_lookup() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(PLAYER_EV_PATH):
        lookup.setdefault(txt(row.get("player_name")).lower(), row)
    return lookup


def captain_comparison_rows() -> list[list[Any]]:
    ev_lookup = player_ev_lookup()
    rows: list[list[Any]] = []
    for name in CAPTAIN_COMPARE_NAMES:
        row = ev_lookup.get(name.lower(), {})
        growth = to_float(row.get("match_1_total_ev_next_match"))
        weighted = to_float(row.get("match_1_weighted_match_ev"))
        rows.append(
            [
                name,
                row.get("team_id", ""),
                row.get("position", ""),
                fmt(row.get("optimizer_ev"), 3),
                fmt(growth, 3),
                fmt(weighted, 3),
                USER_MANUAL_NOTES.get(name, ""),
            ]
        )
    rows.sort(key=lambda item: to_float(item[4]), reverse=True)
    return rows


def strategy_assessment(summary: dict[str, str]) -> str:
    strategy = summary["strategy"]
    high = int(float(summary["high_risk_players"]))
    cond = to_float(summary["avg_conditional_start_prob"])
    price_m = to_float(summary["total_price"]) / 1_000_000
    bits = []
    if cond >= 0.85:
        bits.append("høj starter-sikkerhed")
    elif cond >= 0.80:
        bits.append("rimelig starter-sikkerhed")
    else:
        bits.append("starter-sikkerhed bør tjekkes")
    if high:
        bits.append(f"{high} high_risk")
    if price_m < 49:
        bits.append("underudnytter budget lidt")
    if strategy == "long_run":
        bits.append("orienteret mod stærkere turneringsnationer")
    if strategy == "group_stage":
        bits.append("inkluderer runde 3-rotation")
    return "; ".join(bits)


def write_reports(rows: list[dict[str, str]], strategies: dict[str, Any], comparison_rows: list[dict[str, str]], context: dict[str, Any]) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines: list[str] = []
    lines.append("# User Strategy Detail Report")
    lines.append("")
    lines.append("Ingen modeldata er ændret af denne rapport. Baumgartner-skadeinfo er kun markeret som manuel note.")
    lines.append("")
    lines.append("## Aktuel Strategikontekst")
    lines.append("")
    lines.append(f"- Target round: {context.get('target_round')} ({context.get('target_round_label')})")
    lines.append(f"- Display: {context.get('next_round_display_name')}")
    lines.append("")

    lines.append("## Strategioversigt")
    lines.append("")
    lines.append(table(
        ["Strategi", "Formation", "Pris", "EV", "Score", "Avg cond", "High risk", "Kaptajn", "Kaptajn vækst", "Vurdering"],
        [
            [
                row["display_name_da"],
                row["formation"],
                f"{to_float(row['total_price'])/1_000_000:.1f}",
                fmt(row["total_ev"], 3),
                fmt(row["total_score"], 3),
                fmt(row["avg_conditional_start_prob"], 4),
                row["high_risk_players"],
                row["recommended_captain"],
                fmt(row["captain_expected_growth"], 3),
                strategy_assessment(row),
            ]
            for row in comparison_rows
        ],
    ))
    lines.append("")

    lines.append("## Kaptajn-tjek")
    lines.append("")
    lines.append("Kaptajn beregnes nu med separat kaptajnscore: forventet rundevaekst, start-sikkerhed, high_risk-penalty, kampfavorit og manuel doedbold/straffeprofil. `manual_captain_status=avoid` blokerer kun kaptajnvalg, ikke spillerudtagelse.")
    lines.append("")
    lines.append("TODO: Tilfoej national_goal_rate, recent_goal_rate og et egentligt set_piece_takers-lag, saa kaptajnscore ikke behoever at bruge positions-/rolleproxy.")
    lines.append("")
    lines.append("Baumgartner vælges som kaptajn, fordi den nuværende model har ham med højest `match_1_total_ev_next_match` blandt de relevante kandidater i de valgte squads. Da kaptajn altid beregnes på kommende runde, bliver samme spiller valgt i alle fire strategier.")
    lines.append("")
    lines.append("Vigtigt: Brugernote siger, at Baumgartner er skadet og ikke længere udtaget til VM. Derfor virker kaptajnvalget ikke plausibelt i praksis, før modeldata/manual overrides er opdateret.")
    lines.append("")
    lines.append(table(
        ["Spiller", "Hold", "Pos", "Total EV", "R1 captain growth", "R1 weighted EV", "Manuel note"],
        captain_comparison_rows(),
    ))
    lines.append("")

    for strategy in STRATEGIES:
        summary = strategies[strategy]["best_summary"]
        lines.append(f"## {summary['display_name_da']}")
        lines.append("")
        lines.append(table(
            [
                "Spiller",
                "Hold",
                "Pos",
                "Pris",
                "EV",
                "Score",
                "Start",
                "Cond",
                "Risk",
                "Runder/modstandere",
                "Win prob",
                "CS prob",
                "Goal/assist",
                "Note",
            ],
            [
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
                    row["opponents"],
                    row["win_probabilities"],
                    row["clean_sheet_probabilities"],
                    row["goal_assist_fixture_info"],
                    row["note"],
                ]
                for row in rows if row["strategy"] == strategy
            ],
        ))
        lines.append("")

    manual = [row for row in rows if "manuel tjek" in row["note"] or "BRUGERNOTE" in row["note"]]
    lines.append("## Spillere Der Bør Tjekkes Manuelt")
    lines.append("")
    if manual:
        seen = set()
        for row in manual:
            key = (row["player_name"], row["team_id"], row["note"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {row['player_name']} ({row['team_id']}, {row['position']}): {row['note']}")
    else:
        lines.append("- Ingen oplagte manuelle tjek.")

    lines.append("")
    lines.append("## Inputfiler")
    lines.append("")
    for path in [
        STRATEGIES_PATH,
        COMPARISON_PATH,
        SANITY_PATH,
        PLAYER_EV_PATH,
        PLAYER_POOL_PATH,
        FIXTURE_MULTIPLIERS_PATH,
        MATCH_ODDS_PATH,
        CONTEXT_PATH,
    ]:
        lines.append(f"- `{path.relative_to(ROOT)}`")

    legacy_captain_text = [
        "Baumgartner v",
        "Vigtigt: Brugernote siger",
    ]
    lines = [line for line in lines if not any(marker in line for marker in legacy_captain_text)]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    strategies = read_json(STRATEGIES_PATH)
    comparison_rows = read_csv(COMPARISON_PATH)
    comparison = {row["strategy"]: row for row in comparison_rows}
    context = read_json(CONTEXT_PATH)
    rows = build_rows(strategies, comparison, context)
    write_reports(rows, strategies, comparison_rows, context)

    manual = [row for row in rows if "manuel tjek" in row["note"] or "BRUGERNOTE" in row["note"]]
    print(f"Skrevet: {OUT_MD.relative_to(ROOT)}")
    print(f"Skrevet: {OUT_CSV.relative_to(ROOT)}")
    print(f"Strategier: {len(STRATEGIES)}")
    print(f"Spiller-rækker: {len(rows)}")
    print(f"Manuelle tjek-rækker: {len(manual)}")
    print("Kaptajn i alle strategier:", ", ".join(sorted({comparison[s]['recommended_captain'] for s in STRATEGIES})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
