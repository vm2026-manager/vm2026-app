from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from optimize_squad_group_stage import BUDGET_M, MAX_PER_TEAM, load_players


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA / "strategy_comparison_report.csv"
USER_DETAIL_PATH = DATA / "user_strategy_detail_report.csv"
REPLACEMENT_PATH = DATA / "strategy_replacement_report.csv"
WORLD_CUP_OUTRIGHT_PATH = DATA / "worldcup_outright_odds.csv"
TEAM_MARKET_PATH = DATA / "team_market_odds_layer_v1.csv"
PLAYER_EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
MANUAL_OVERRIDES_PATH = DATA / "manual_player_overrides.csv"
DISPLAY_NAMES_PATH = DATA / "strategy_display_names.json"
SET_PIECES_PATH = DATA / "set_piece_takers.csv"

OUT_MD = DATA / "long_run_pick_audit.md"
OUT_CSV = DATA / "long_run_pick_audit.csv"

SPECIAL_PLAYERS = {"Patrick Agyemang", "Kerem Akturkoglu", "Roberto Alvarado"}

CSV_FIELDS = [
    "player_name",
    "team_id",
    "position",
    "price",
    "EV",
    "long_run_strategy_score",
    "conditional_start_prob",
    "availability_risk",
    "winner_odds",
    "tournament_strength_score",
    "score_ev_component",
    "score_tournament_component",
    "score_start_component",
    "weak_team_penalty",
    "why_selected",
    "best_stronger_replacement",
    "best_stronger_replacement_team",
    "best_stronger_replacement_price",
    "best_stronger_replacement_ev",
    "best_stronger_replacement_score",
    "best_stronger_replacement_winner_odds",
    "best_stronger_replacement_tournament_strength",
    "stronger_replacement_margin",
    "label",
    "recommendation",
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


def player_key(player: dict[str, Any]) -> str:
    return txt(player.get("player_id"))


def build_team_market() -> dict[str, dict[str, Any]]:
    rows = read_csv(WORLD_CUP_OUTRIGHT_PATH)
    if not rows:
        rows = read_csv(TEAM_MARKET_PATH)
    market: dict[str, dict[str, Any]] = {}
    for row in rows:
        team = txt(row.get("team_id")).upper()
        if not team:
            continue
        long_score = to_float(row.get("team_long_run_score"))
        market_score = to_float(row.get("team_market_score"))
        if long_score or market_score:
            strength = 0.75 * long_score + 0.25 * market_score
        else:
            winner_prob = to_float(row.get("winner_prob"))
            strength = winner_prob
        row["_tournament_strength_score"] = strength
        market[team] = row
    return market


def starter_component(player: dict[str, Any]) -> float:
    start = to_float(player.get("start_prob"))
    conditional = to_float(player.get("conditional_start_prob"))
    if start and conditional:
        return 0.55 * conditional + 0.45 * start
    return conditional or start


def tournament_strength(player: dict[str, Any], market: dict[str, dict[str, Any]]) -> float:
    direct = to_float(player.get("team_long_run_score")) * 0.75 + to_float(player.get("team_market_score")) * 0.25
    if direct:
        return direct
    team = txt(player.get("team_id")).upper()
    return to_float(market.get(team, {}).get("_tournament_strength_score"))


def winner_odds(player: dict[str, Any], market: dict[str, dict[str, Any]]) -> float:
    team = txt(player.get("team_id")).upper()
    return to_float(market.get(team, {}).get("winner_odds"))


def selected_team_counts(squad: list[dict[str, Any]]) -> Counter[str]:
    return Counter(txt(player.get("team_id")).upper() for player in squad)


def total_price_m(squad: list[dict[str, Any]]) -> float:
    return sum(to_float(player.get("price")) for player in squad) / 1_000_000


def merge_selected(selected: dict[str, Any], players_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = dict(players_by_id.get(player_key(selected), {}))
    merged.update(selected)
    return merged


def stronger_replacements(
    *,
    selected: dict[str, Any],
    all_players: list[dict[str, Any]],
    selected_ids: set[str],
    team_counts: Counter[str],
    squad_total_m: float,
    market: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_id = player_key(selected)
    selected_team = txt(selected.get("team_id")).upper()
    selected_price_m = to_float(selected.get("price")) / 1_000_000
    selected_strength = tournament_strength(selected, market)
    position = txt(selected.get("position"))
    rows: list[dict[str, Any]] = []

    for candidate in all_players:
        candidate_id = player_key(candidate)
        if not candidate_id or candidate_id in selected_ids or candidate_id == selected_id:
            continue
        if txt(candidate.get("position")) != position:
            continue
        if txt(candidate.get("manual_status")).lower() == "avoid":
            continue
        if txt(candidate.get("manual_start_status")).lower() == "avoid":
            continue

        candidate_strength = tournament_strength(candidate, market)
        if candidate_strength <= selected_strength + 1e-9:
            continue

        candidate_team = txt(candidate.get("team_id")).upper()
        candidate_price_m = to_float(candidate.get("price_m"), to_float(candidate.get("price")) / 1_000_000)
        next_total = squad_total_m - selected_price_m + candidate_price_m
        if next_total > BUDGET_M + 1e-9:
            continue

        next_team_count = team_counts[candidate_team] + (0 if candidate_team == selected_team else 1)
        if next_team_count > MAX_PER_TEAM:
            continue

        rows.append(candidate)

    return sorted(
        rows,
        key=lambda row: (
            -to_float(row.get("score_long_run")),
            -tournament_strength(row, market),
            -to_float(row.get("optimizer_ev")),
            to_float(row.get("price_m"), to_float(row.get("price")) / 1_000_000),
            txt(row.get("player_name")),
        ),
    )


def why_selected(player: dict[str, Any], strength: float, weak_penalty: float) -> str:
    reasons: list[str] = []
    ev = to_float(player.get("optimizer_ev"))
    start = starter_component(player)
    score = to_float(player.get("strategy_score"), to_float(player.get("score_long_run")))
    if ev >= 4.0:
        reasons.append("høj EV/value trækker kraftigt")
    elif ev >= 3.4:
        reasons.append("solid EV/value")
    else:
        reasons.append("moderat EV")
    if strength >= 0.55:
        reasons.append("stærk turneringsnation")
    elif strength >= 0.35:
        reasons.append("mellemstærk turneringsnation")
    else:
        reasons.append("svagere/mellemhold i vinderodds")
    if start >= 0.85:
        reasons.append("stærk starterprofil")
    elif start >= 0.75:
        reasons.append("acceptabel starterprofil")
    else:
        reasons.append("usikker starterprofil")
    if weak_penalty:
        reasons.append("rammes af svag-nation penalty, men penalty er lille")
    reasons.append(f"samlet long_run-score {fmt(score, 3)}")
    return "; ".join(reasons)


def classify(player: dict[str, Any], best: dict[str, Any] | None, strength: float, market: dict[str, dict[str, Any]]) -> tuple[str, str]:
    ev = to_float(player.get("optimizer_ev"))
    start = starter_component(player)
    risk = txt(player.get("availability_risk"))
    manual_status = txt(player.get("manual_status")).lower()
    manual_start = txt(player.get("manual_start_status")).lower()
    selected_score = to_float(player.get("strategy_score"), to_float(player.get("score_long_run")))
    best_score = to_float(best.get("score_long_run")) if best else 0.0
    best_strength = tournament_strength(best, market) if best else 0.0
    stronger_margin = selected_score - best_score if best else 999.0

    if strength >= 0.55 and start >= 0.78:
        return "true_long_run_pick", "Passer til Lang sigt: stærk nation plus rimelig starterprofil."
    if strength >= 0.35 and stronger_margin >= -0.25 and start >= 0.78:
        return "acceptable_tradeoff", "Kan forsvares som tradeoff mellem score, pris, start og turneringsstyrke."
    if strength < 0.35 and (manual_status == "check" or manual_start == "doubtful" or start < 0.78):
        return "questionable_long_run_pick", "Svagere turneringsprofil kombineret med manuel/start-risiko."
    if strength < 0.35 and best and best_strength >= 0.45 and stronger_margin < 0.75:
        return "questionable_long_run_pick", "Der findes et stærkere nationsalternativ tæt på eller over scoren."
    if strength < 0.35 and ev >= 3.8:
        return "value_filler", "Ligner primært value/budget-fill fra mellemhold, ikke en ren stor-nationslogik."
    if risk == "high_risk":
        return "questionable_long_run_pick", "High risk passer dårligt til en langsigtet strategi."
    return "acceptable_tradeoff", "Ikke et rent stor-nationsvalg, men kan forklares af samlet score og constraints."


def build_audit_rows() -> list[dict[str, Any]]:
    strategies = read_json(STRATEGIES_PATH)
    long_run = strategies["long_run"]
    selected_raw = long_run.get("best_squad", [])
    selected_ids = {player_key(player) for player in selected_raw}
    team_counts = selected_team_counts(selected_raw)
    squad_total_m = total_price_m(selected_raw)
    market = build_team_market()

    all_players = load_players().to_dict("records")
    players_by_id = {player_key(player): player for player in all_players if player_key(player)}

    rows: list[dict[str, Any]] = []
    for selected in selected_raw:
        player = merge_selected(selected, players_by_id)
        strength = tournament_strength(player, market)
        weak_penalty = 0.35 if strength < 0.18 else 0.0
        replacements = stronger_replacements(
            selected=player,
            all_players=all_players,
            selected_ids=selected_ids,
            team_counts=team_counts,
            squad_total_m=squad_total_m,
            market=market,
        )
        best = replacements[0] if replacements else None
        label, recommendation = classify(player, best, strength, market)
        selected_score = to_float(player.get("strategy_score"), to_float(player.get("score_long_run")))
        best_score = to_float(best.get("score_long_run")) if best else 0.0

        row = {
            "player_name": txt(player.get("player_name")),
            "team_id": txt(player.get("team_id")).upper(),
            "position": txt(player.get("position")),
            "price": int(round(to_float(player.get("price"), to_float(player.get("price_m")) * 1_000_000))),
            "EV": fmt(player.get("optimizer_ev"), 6),
            "long_run_strategy_score": fmt(selected_score, 6),
            "conditional_start_prob": fmt(player.get("conditional_start_prob"), 4),
            "availability_risk": txt(player.get("availability_risk")),
            "winner_odds": fmt(winner_odds(player, market), 2),
            "tournament_strength_score": fmt(strength, 6),
            "score_ev_component": fmt(1.20 * to_float(player.get("optimizer_ev")), 6),
            "score_tournament_component": fmt(1.35 * strength, 6),
            "score_start_component": fmt(0.80 * starter_component(player), 6),
            "weak_team_penalty": fmt(weak_penalty, 3),
            "why_selected": why_selected(player, strength, weak_penalty),
            "best_stronger_replacement": txt(best.get("player_name")) if best else "",
            "best_stronger_replacement_team": txt(best.get("team_id")).upper() if best else "",
            "best_stronger_replacement_price": int(round(to_float(best.get("price_m"), to_float(best.get("price")) / 1_000_000) * 1_000_000)) if best else "",
            "best_stronger_replacement_ev": fmt(best.get("optimizer_ev"), 6) if best else "",
            "best_stronger_replacement_score": fmt(best_score, 6) if best else "",
            "best_stronger_replacement_winner_odds": fmt(winner_odds(best, market), 2) if best else "",
            "best_stronger_replacement_tournament_strength": fmt(tournament_strength(best, market), 6) if best else "",
            "stronger_replacement_margin": fmt(selected_score - best_score, 6) if best else "",
            "label": label,
            "recommendation": recommendation,
        }
        rows.append(row)
    return rows


def make_markdown(rows: list[dict[str, Any]]) -> str:
    label_counts = Counter(row["label"] for row in rows)
    questionable = [row for row in rows if row["label"] in {"value_filler", "questionable_long_run_pick"}]
    special = [row for row in rows if row["player_name"] in SPECIAL_PLAYERS]
    true_core = [row for row in rows if row["label"] == "true_long_run_pick"]

    lines: list[str] = [
        "# Long run pick audit",
        "",
        "## Kort konklusion",
        "",
    ]
    if questionable:
        lines.append(
            "Auditten viser, at long_run stadig har flere value-/budgetvalg fra mellemhold. "
            "Det skyldes især, at scoreformlen stadig giver høj vægt til EV/value, mens turneringsstyrkeleddet og weak-team-penalty er relativt milde."
        )
    else:
        lines.append("Auditten finder ikke tydelige value-fillers; long_run-valgene passer overvejende til store nationer og starterprofil.")
    lines.extend(
        [
            "",
            f"Labels: true_long_run_pick={label_counts.get('true_long_run_pick', 0)}, "
            f"acceptable_tradeoff={label_counts.get('acceptable_tradeoff', 0)}, "
            f"value_filler={label_counts.get('value_filler', 0)}, "
            f"questionable_long_run_pick={label_counts.get('questionable_long_run_pick', 0)}.",
            "",
            "## Dødboldsnote",
            "",
            "data/set_piece_takers.csv findes, men dødbolde er ikke integreret i denne audit. "
            "Dødbolde bør kun være et lille tie-breaker-lag. De må ikke alene forklare eller retfærdiggøre long_run-valg, "
            "og en samlet set-piece-bonus bør være lavt capped, fx omkring 2-5 pct. af relevant strategi-score.",
            "",
            "## Særligt tjek",
            "",
            table(
                ["Spiller", "Land", "Label", "Hvorfor valgt", "Bedste stærkere alternativ", "Anbefaling"],
                [
                    [
                        row["player_name"],
                        row["team_id"],
                        row["label"],
                        row["why_selected"],
                        f"{row['best_stronger_replacement']} ({row['best_stronger_replacement_team']}, score {row['best_stronger_replacement_score']})" if row["best_stronger_replacement"] else "Ingen inden for constraints",
                        row["recommendation"],
                    ]
                    for row in special
                ],
            ),
            "",
            "## Robuste long_run-kernevalg",
            "",
            table(
                ["Spiller", "Land", "Pos", "Score", "Winner odds", "Turneringsstyrke"],
                [
                    [
                        row["player_name"],
                        row["team_id"],
                        row["position"],
                        row["long_run_strategy_score"],
                        row["winner_odds"],
                        row["tournament_strength_score"],
                    ]
                    for row in true_core
                ],
            ),
            "",
            "## Alle long_run-valg",
            "",
            table(
                ["Spiller", "Land", "Pos", "EV", "Score", "Start", "Winner odds", "Turnering", "Label", "Bedste stærkere alternativ"],
                [
                    [
                        row["player_name"],
                        row["team_id"],
                        row["position"],
                        row["EV"],
                        row["long_run_strategy_score"],
                        row["conditional_start_prob"],
                        row["winner_odds"],
                        row["tournament_strength_score"],
                        row["label"],
                        f"{row['best_stronger_replacement']} ({row['best_stronger_replacement_team']}, {row['best_stronger_replacement_score']})" if row["best_stronger_replacement"] else "",
                    ]
                    for row in rows
                ],
            ),
            "",
            "## Forslag til fremtidig kalibrering (ikke implementeret)",
            "",
            "- Gør weak_tournament_team_penalty hårdere for hold under en klar turneringsstyrkegrænse.",
            "- Øg tournament_strength_bonus, så store nationer fylder mere end billige value-spillere.",
            "- Tillad value-spillere fra mellemhold kun når EV er markant høj, conditional_start_prob er stærk, og der ikke findes et rimeligt stærkere nationsalternativ.",
            "- Undgå at budgetpresset alene tvinger fyldspillere ind i long_run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    # Touch/read optional inputs explicitly so the audit documents what exists.
    _ = read_csv(COMPARISON_PATH)
    _ = read_csv(USER_DETAIL_PATH)
    _ = read_csv(REPLACEMENT_PATH)
    _ = read_csv(PLAYER_EV_PATH)
    _ = read_csv(MANUAL_OVERRIDES_PATH)
    _ = read_json(DISPLAY_NAMES_PATH) if DISPLAY_NAMES_PATH.exists() else {}
    _ = read_json(PLAYER_POOL_PATH)
    _ = SET_PIECES_PATH.exists()

    rows = build_audit_rows()
    write_csv(OUT_CSV, rows)
    OUT_MD.write_text(make_markdown(rows), encoding="utf-8")

    counts = Counter(row["label"] for row in rows)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    print(f"Long-run picks: {len(rows)}")
    print(
        "Labels: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in ["true_long_run_pick", "acceptable_tradeoff", "value_filler", "questionable_long_run_pick"])
    )
    for name in sorted(SPECIAL_PLAYERS):
        row = next((item for item in rows if item["player_name"] == name), None)
        if row:
            print(f"{name}: {row['label']} | stronger replacement: {row['best_stronger_replacement']} ({row['best_stronger_replacement_team']})")


if __name__ == "__main__":
    main()
