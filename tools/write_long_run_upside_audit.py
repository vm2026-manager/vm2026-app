from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from optimize_squad_group_stage import load_players


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
TEAM_MARKET_PATH = DATA / "team_market_odds_layer_v1.csv"
SET_PIECES_PATH = DATA / "set_piece_takers.csv"

OUT_MD = DATA / "long_run_upside_audit.md"
OUT_CSV = DATA / "long_run_upside_audit.csv"

FOCUS_PLAYERS = {"Declan Rice", "Rodrigo de Paul", "Manu Koné", "Manu Kone", "Aurelien Tchouameni"}

ATTACK_EV_COLS = [
    "match_1_goal_ev",
    "match_1_assist_ev",
    "match_1_shots_on_target_ev",
    "match_2_goal_ev",
    "match_2_assist_ev",
    "match_2_shots_on_target_ev",
    "match_3_goal_ev",
    "match_3_assist_ev",
    "match_3_shots_on_target_ev",
]

CSV_FIELDS = [
    "player_name",
    "team_id",
    "position",
    "price",
    "EV",
    "conditional_start_prob",
    "availability_risk",
    "winner_odds",
    "tournament_strength_score",
    "attacking_upside_index",
    "goal_share_norm",
    "assist_share_norm",
    "sot_share_norm",
    "role_upside_assessment",
    "best_offensive_alternative_1",
    "alternative_1_team",
    "alternative_1_position",
    "alternative_1_price",
    "alternative_1_ev",
    "alternative_1_conditional_start_prob",
    "alternative_1_availability_risk",
    "alternative_1_tournament_strength",
    "alternative_1_attacking_upside_index",
    "alternative_1_direct_swap_budget_feasible",
    "best_offensive_alternative_2",
    "best_offensive_alternative_3",
    "upside_label",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_team_market() -> dict[str, dict[str, Any]]:
    market: dict[str, dict[str, Any]] = {}
    for row in read_csv(TEAM_MARKET_PATH):
        team = txt(row.get("team_id")).upper()
        if not team:
            continue
        row["_tournament_strength_score"] = 0.75 * to_float(row.get("team_long_run_score")) + 0.25 * to_float(row.get("team_market_score"))
        market[team] = row
    return market


def add_upside_columns(players: pd.DataFrame) -> pd.DataFrame:
    work = players.copy()
    for col in ATTACK_EV_COLS + ["goal_share_norm", "assist_share_norm", "sot_share_norm"]:
        work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
    work["attacking_upside_index"] = work[ATTACK_EV_COLS].sum(axis=1)
    work["tournament_strength_score"] = (0.75 * work["team_long_run_score"] + 0.25 * work["team_market_score"]).clip(lower=0.0)
    return work


def winner_odds(player: dict[str, Any], market: dict[str, dict[str, Any]]) -> float:
    return to_float(market.get(txt(player.get("team_id")).upper(), {}).get("winner_odds"))


def role_assessment(player: dict[str, Any]) -> str:
    position = txt(player.get("position"))
    upside = to_float(player.get("attacking_upside_index"))
    ev = to_float(player.get("optimizer_ev"))
    goal_share = to_float(player.get("goal_share_norm"))
    assist_share = to_float(player.get("assist_share_norm"))
    sot_share = to_float(player.get("sot_share_norm"))

    if position == "FWD":
        if upside >= 1.00 or goal_share + sot_share >= 0.13:
            return "Offensiv starterprofil med reel mål-/skud-upside."
        return "Angriber fra stærk nation, men upside-proxy er ikke elite."
    if position == "MID":
        if upside >= 0.68 or goal_share + assist_share + sot_share >= 0.15:
            return "Midtbane med acceptabel offensiv involvering i modellen."
        if upside >= 0.55:
            return "Central/sikker midtbane med moderat offensiv upside."
        return "Sikker central/defensiv midtbaneprofil med lavere fantasy-ceiling."
    if position == "DEF":
        if ev >= 3.5 or assist_share >= 0.04:
            return "Defensivt long-run valg med clean-sheet/assist-proxy."
        return "Defensivt stærk-nation valg, men lav åben spil-upside."
    if position == "GK":
        return "Målmand: long-run styrke er primært clean sheet/holdvej, ikke offensiv upside."
    return "Rolle ikke tydeligt klassificeret."


def classify(player: dict[str, Any], alternatives: list[dict[str, Any]]) -> tuple[str, str]:
    position = txt(player.get("position"))
    upside = to_float(player.get("attacking_upside_index"))
    strength = to_float(player.get("tournament_strength_score"))
    cond = to_float(player.get("conditional_start_prob"))
    ev = to_float(player.get("optimizer_ev"))
    has_good_alt = any(to_float(alt.get("attacking_upside_index")) >= upside + 0.08 for alt in alternatives[:3])

    if position in {"FWD"} and upside >= 1.0 and strength >= 0.55:
        return "strong_long_run_pick", "Stærk nationsprofil og klar offensiv upside."
    if position == "MID" and upside < 0.52 and has_good_alt:
        return "should_review_for_upside", "Lav midtbane-upside og der findes offensive alternativer fra samme/stærkere nationer."
    if position == "MID" and upside < 0.52:
        return "safe_but_low_upside", "Sikker starter fra stærk nation, men lav fantasy-ceiling."
    if position == "MID" and upside < 0.62 and cond >= 0.85 and strength >= 0.55:
        return "safe_but_low_upside", "Robust long-run profil, men mere central/sikker end eksplosiv fantasy-profil."
    if position in {"GK", "DEF"} and ev < 2.0:
        return "safe_but_low_upside", "Stærk nations-/starterprofil, men lav direkte fantasy-upside."
    if position in {"GK", "DEF"}:
        return "acceptable_balance", "Defensivt valg passer til long-run, men bør balanceres af offensiv upside andre steder."
    return "acceptable_balance", "Balancen mellem nation, start og upside er acceptabel."


def offensive_alternatives(selected: dict[str, Any], players: list[dict[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
    position = txt(selected.get("position"))
    strength = to_float(selected.get("tournament_strength_score"))
    upside = to_float(selected.get("attacking_upside_index"))
    selected_price = to_float(selected.get("price_m"))
    alternatives: list[dict[str, Any]] = []
    for candidate in players:
        if txt(candidate.get("player_id")) in selected_ids:
            continue
        if txt(candidate.get("position")) != position:
            continue
        if to_float(candidate.get("tournament_strength_score")) + 1e-9 < strength:
            continue
        if to_float(candidate.get("attacking_upside_index")) <= upside + 0.02:
            continue
        if txt(candidate.get("availability_risk")) == "high_risk":
            continue
        if txt(candidate.get("manual_status")).lower() == "avoid" or txt(candidate.get("manual_start_status")).lower() == "avoid":
            continue
        alt = dict(candidate)
        alt["_direct_swap_budget_feasible"] = to_float(candidate.get("price_m")) <= selected_price + 1e-9
        alternatives.append(alt)
    return sorted(
        alternatives,
        key=lambda row: (
            not bool(row.get("_direct_swap_budget_feasible")),
            -to_float(row.get("attacking_upside_index")),
            -to_float(row.get("score_long_run")),
            -to_float(row.get("optimizer_ev")),
            txt(row.get("player_name")),
        ),
    )[:3]


def build_rows() -> list[dict[str, Any]]:
    strategies = read_json(STRATEGIES_PATH)
    selected = strategies["long_run"]["best_squad"]
    selected_ids = {txt(row.get("player_id")) for row in selected}
    selected_by_id = {txt(row.get("player_id")): row for row in selected}
    market = build_team_market()

    players_df = add_upside_columns(load_players())
    players = players_df.to_dict("records")
    players_by_id = {txt(row.get("player_id")): row for row in players}

    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        player = dict(players_by_id.get(txt(selected_row.get("player_id")), {}))
        player.update(selected_row)
        if "attacking_upside_index" not in player:
            player.update(players_by_id.get(txt(selected_row.get("player_id")), {}))
        alternatives = offensive_alternatives(player, players, selected_ids)
        label, note = classify(player, alternatives)
        alt1 = alternatives[0] if alternatives else {}
        rows.append(
            {
                "player_name": txt(player.get("player_name")),
                "team_id": txt(player.get("team_id")).upper(),
                "position": txt(player.get("position")),
                "price": int(round(to_float(player.get("price_m"), to_float(player.get("price")) / 1_000_000) * 1_000_000)),
                "EV": fmt(player.get("optimizer_ev"), 6),
                "conditional_start_prob": fmt(player.get("conditional_start_prob"), 4),
                "availability_risk": txt(player.get("availability_risk")),
                "winner_odds": fmt(winner_odds(player, market), 2),
                "tournament_strength_score": fmt(player.get("tournament_strength_score"), 6),
                "attacking_upside_index": fmt(player.get("attacking_upside_index"), 6),
                "goal_share_norm": fmt(player.get("goal_share_norm"), 4),
                "assist_share_norm": fmt(player.get("assist_share_norm"), 4),
                "sot_share_norm": fmt(player.get("sot_share_norm"), 4),
                "role_upside_assessment": role_assessment(player),
                "best_offensive_alternative_1": txt(alt1.get("player_name")),
                "alternative_1_team": txt(alt1.get("team_id")).upper(),
                "alternative_1_position": txt(alt1.get("position")),
                "alternative_1_price": int(round(to_float(alt1.get("price_m")) * 1_000_000)) if alt1 else "",
                "alternative_1_ev": fmt(alt1.get("optimizer_ev"), 6) if alt1 else "",
                "alternative_1_conditional_start_prob": fmt(alt1.get("conditional_start_prob"), 4) if alt1 else "",
                "alternative_1_availability_risk": txt(alt1.get("availability_risk")),
                "alternative_1_tournament_strength": fmt(alt1.get("tournament_strength_score"), 6) if alt1 else "",
                "alternative_1_attacking_upside_index": fmt(alt1.get("attacking_upside_index"), 6) if alt1 else "",
                "alternative_1_direct_swap_budget_feasible": "yes" if alt1.get("_direct_swap_budget_feasible") else ("no" if alt1 else ""),
                "best_offensive_alternative_2": txt(alternatives[1].get("player_name")) if len(alternatives) > 1 else "",
                "best_offensive_alternative_3": txt(alternatives[2].get("player_name")) if len(alternatives) > 2 else "",
                "upside_label": label,
                "note": note,
            }
        )
    return rows


def make_markdown(rows: list[dict[str, Any]]) -> str:
    labels = Counter(row["upside_label"] for row in rows)
    low_upside = [row for row in rows if row["upside_label"] == "safe_but_low_upside"]
    review = [row for row in rows if row["upside_label"] == "should_review_for_upside"]
    mids = [row for row in rows if row["position"] == "MID"]
    low_or_review_mids = [row for row in mids if row["upside_label"] in {"safe_but_low_upside", "should_review_for_upside"}]
    focus = [row for row in rows if row["player_name"] in FOCUS_PLAYERS or row["player_name"].replace("é", "e") in FOCUS_PLAYERS]

    if len(low_or_review_mids) >= 3:
        conclusion = "Long_run er blevet mere robust og stor-nationsorienteret, men midtbanen hælder lav-upside/central: flere valg er sikre startere frem for højt fantasy-ceiling."
    elif review:
        conclusion = "Long_run er ikke ekstremt defensivt, men enkelte spillere bør tjekkes for mere offensiv upside."
    else:
        conclusion = "Long_run ser ikke klart for defensivt ud i de nuværende proxydata, men det har stadig en sikkerhedsovervægt på midtbanen."

    lines = [
        "# Long run upside audit",
        "",
        "## Kort konklusion",
        "",
        conclusion,
        "",
        f"Labels: strong_long_run_pick={labels.get('strong_long_run_pick', 0)}, "
        f"acceptable_balance={labels.get('acceptable_balance', 0)}, "
        f"safe_but_low_upside={labels.get('safe_but_low_upside', 0)}, "
        f"should_review_for_upside={labels.get('should_review_for_upside', 0)}.",
        "",
        "## Dødboldsnote",
        "",
        "Dødbolde er ikke brugt i denne audit. data/set_piece_takers.csv findes, men dødbolde bør fortsat kun være et lille fremtidigt tie-breaker-signal og ikke en forklaring på long_run-valg.",
        "",
        "## Fokusspillere",
        "",
        table(
            ["Spiller", "Land", "Upside", "Label", "Bedste offensiv-alternativ", "Note"],
            [
                [
                    row["player_name"],
                    row["team_id"],
                    row["attacking_upside_index"],
                    row["upside_label"],
                    f"{row['best_offensive_alternative_1']} ({row['alternative_1_team']}, upside {row['alternative_1_attacking_upside_index']}, direct swap {row['alternative_1_direct_swap_budget_feasible']})" if row["best_offensive_alternative_1"] else "Ingen samme/stærkere nation med klarere offensiv proxy",
                    row["note"],
                ]
                for row in focus
            ],
        ),
        "",
        "## Safe But Low Upside",
        "",
        table(
            ["Spiller", "Land", "Pos", "EV", "Start", "Upside", "Alternativ"],
            [
                [
                    row["player_name"],
                    row["team_id"],
                    row["position"],
                    row["EV"],
                    row["conditional_start_prob"],
                    row["attacking_upside_index"],
                    row["best_offensive_alternative_1"] or "",
                ]
                for row in low_upside
            ],
        ),
        "",
        "## Alle Long Run-Spillere",
        "",
        table(
            ["Spiller", "Land", "Pos", "EV", "Winner odds", "Turnering", "Upside", "Label", "Rolle-/upside-vurdering"],
            [
                [
                    row["player_name"],
                    row["team_id"],
                    row["position"],
                    row["EV"],
                    row["winner_odds"],
                    row["tournament_strength_score"],
                    row["attacking_upside_index"],
                    row["upside_label"],
                    row["role_upside_assessment"],
                ]
                for row in rows
            ],
        ),
        "",
        "## Anbefaling",
        "",
    ]
    if review:
        lines.append("Overvej en lille fremtidig upside-balance i long_run, især for midtbanen, men behold den nye stor-nationskalibrering.")
    elif len(low_or_review_mids) >= 3:
        lines.append("Behold grundkalibreringen mod store nationer, men overvej et mildt offensivt ceiling-krav for MID-pladser, så holdet ikke bliver for centralt/defensivt.")
    else:
        lines.append("Behold long_run foreløbigt. Nuværende data viser ikke nok til at lave en ny modelændring uden ekstra rolledata.")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    write_csv(OUT_CSV, rows)
    OUT_MD.write_text(make_markdown(rows), encoding="utf-8")
    labels = Counter(row["upside_label"] for row in rows)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    print(f"Long-run picks: {len(rows)}")
    print(
        "Labels: "
        + ", ".join(
            f"{key}={labels.get(key, 0)}"
            for key in ["strong_long_run_pick", "acceptable_balance", "safe_but_low_upside", "should_review_for_upside"]
        )
    )
    low = [row["player_name"] for row in rows if row["upside_label"] == "safe_but_low_upside"]
    print("safe_but_low_upside: " + ("; ".join(low) if low else "ingen"))


if __name__ == "__main__":
    main()
