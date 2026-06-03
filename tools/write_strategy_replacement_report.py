from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from optimize_squad_group_stage import BUDGET_M, MAX_PER_TEAM, load_players


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"
COMPARISON_PATH = DATA / "strategy_comparison_report.csv"
PLAYER_EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
MANUAL_OVERRIDES_PATH = DATA / "manual_player_overrides.csv"
DISPLAY_NAMES_PATH = DATA / "strategy_display_names.json"
CONTEXT_PATH = DATA / "current_strategy_context.json"

OUT_MD = DATA / "strategy_replacement_report.md"
OUT_CSV = DATA / "strategy_replacement_report.csv"

STRATEGY_ORDER = ["next_round", "round1_2", "group_stage", "long_run"]
LOW_CONDITIONAL_THRESHOLD = 0.75
FRAGILE_MARGIN_THRESHOLD = 0.35
ROBUST_MARGIN_THRESHOLD = 1.0
TOP_REPLACEMENTS = 5

CSV_FIELDS = [
    "strategy",
    "display_name_da",
    "selected_player",
    "selected_team",
    "selected_position",
    "selected_price",
    "selected_ev",
    "selected_strategy_score",
    "selected_conditional_start_prob",
    "selected_risk",
    "best_replacement_1",
    "replacement_1_team",
    "replacement_1_price",
    "replacement_1_ev",
    "replacement_1_strategy_score",
    "replacement_1_conditional_start_prob",
    "replacement_1_risk",
    "replacement_margin",
    "robustness_label",
    "manual_flag",
    "recommendation_note",
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
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def score_col(strategy: str) -> str:
    return f"score_{strategy}"


def player_key(player: dict[str, Any]) -> str:
    return txt(player.get("player_id"))


def manual_flag(player: dict[str, Any]) -> str:
    flags: list[str] = []
    risk = txt(player.get("availability_risk"))
    cond = to_float(player.get("conditional_start_prob"))
    if risk == "high_risk":
        flags.append("high_risk")
    if cond < LOW_CONDITIONAL_THRESHOLD:
        flags.append("low_conditional_start")
    for col in ["manual_status", "manual_start_status", "manual_captain_status", "manual_note", "manual_role_note", "manual_captain_note"]:
        value = txt(player.get(col))
        if value:
            flags.append(f"{col}:{value}")
    return "; ".join(flags)


def label_for(margin: float, flag: str) -> str:
    if flag:
        return "replace_candidate"
    if margin <= FRAGILE_MARGIN_THRESHOLD:
        return "fragile_pick"
    if margin >= ROBUST_MARGIN_THRESHOLD:
        return "robust_pick"
    return "watchlist"


def note_for(label: str, margin: float, flag: str, best: dict[str, Any] | None) -> str:
    if best is None:
        return "Ingen gyldig replacement fundet med samme position, budget og landeloft."
    best_name = txt(best.get("player_name"))
    best_score = fmt(best.get("_replacement_strategy_score"), 3)
    if label == "replace_candidate":
        if margin >= ROBUST_MARGIN_THRESHOLD:
            return f"Manuelt flag, men scoremargin {fmt(margin, 3)} er klar mod {best_name} ({best_score}); behold kun hvis lineup/startstatus bekræftes."
        return f"Manuelt flag + margin {fmt(margin, 3)} mod {best_name} ({best_score}); bør tjekkes før lås."
    if label == "fragile_pick":
        return f"Lille margin {fmt(margin, 3)} mod {best_name} ({best_score}); valg afhænger af små scoreforskelle."
    if label == "robust_pick":
        return f"Klar margin {fmt(margin, 3)} til bedste gyldige alternativ."
    return f"Mellemzone: margin {fmt(margin, 3)} mod {best_name} ({best_score})."


def display_name(strategy: str, display_names: dict[str, str], comparison: dict[str, dict[str, str]]) -> str:
    return txt(display_names.get(strategy)) or txt(comparison.get(strategy, {}).get("display_name_da")) or strategy


def squad_total_price_m(squad: list[dict[str, Any]]) -> float:
    return sum(to_float(player.get("price")) for player in squad) / 1_000_000


def selected_team_counts(squad: list[dict[str, Any]]) -> Counter[str]:
    return Counter(txt(player.get("team_id")).upper() for player in squad)


def feasible_replacements(
    *,
    players_by_id: dict[str, dict[str, Any]],
    all_players: list[dict[str, Any]],
    selected: dict[str, Any],
    selected_ids: set[str],
    team_counts: Counter[str],
    total_price_m: float,
    strategy: str,
) -> list[dict[str, Any]]:
    selected_team = txt(selected.get("team_id")).upper()
    selected_price_m = to_float(selected.get("price")) / 1_000_000
    position = txt(selected.get("position"))
    selected_id = player_key(selected)
    selected_score = to_float(selected.get("strategy_score"))

    replacements: list[dict[str, Any]] = []
    for candidate in all_players:
        candidate_id = player_key(candidate)
        if not candidate_id or candidate_id in selected_ids or candidate_id == selected_id:
            continue
        if txt(candidate.get("position")) != position:
            continue
        if txt(candidate.get("manual_status")).lower() == "avoid" or txt(candidate.get("manual_start_status")).lower() == "avoid":
            continue

        candidate_team = txt(candidate.get("team_id")).upper()
        candidate_price_m = to_float(candidate.get("price_m"), to_float(candidate.get("price")) / 1_000_000)
        next_total = total_price_m - selected_price_m + candidate_price_m
        if next_total > BUDGET_M + 1e-9:
            continue

        next_team_count = team_counts[candidate_team] + (0 if candidate_team == selected_team else 1)
        if next_team_count > MAX_PER_TEAM:
            continue

        candidate_copy = dict(candidate)
        candidate_score = to_float(candidate_copy.get(score_col(strategy)))
        candidate_copy["_replacement_strategy_score"] = candidate_score
        candidate_copy["_replacement_margin"] = selected_score - candidate_score
        replacements.append(candidate_copy)

    return sorted(
        replacements,
        key=lambda player: (
            -to_float(player.get("_replacement_strategy_score")),
            -to_float(player.get("optimizer_ev")),
            to_float(player.get("price_m")),
            txt(player.get("player_name")),
        ),
    )[:TOP_REPLACEMENTS]


def merge_selected_with_pool(selected: dict[str, Any], players_by_id: dict[str, dict[str, Any]], strategy: str) -> dict[str, Any]:
    merged = dict(players_by_id.get(player_key(selected), {}))
    merged.update(selected)
    if "strategy_score" not in merged or txt(merged.get("strategy_score")) == "":
        merged["strategy_score"] = merged.get(score_col(strategy), 0.0)
    return merged


def build_rows(
    strategies: dict[str, Any],
    comparison: dict[str, dict[str, str]],
    display_names: dict[str, str],
    all_players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    players_by_id = {player_key(player): player for player in all_players if player_key(player)}
    csv_rows: list[dict[str, Any]] = []
    replacement_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for strategy in STRATEGY_ORDER:
        squad = strategies[strategy]["best_squad"]
        selected_ids = {player_key(player) for player in squad}
        team_counts = selected_team_counts(squad)
        total_price_m = squad_total_price_m(squad)

        for selected_raw in squad:
            selected = merge_selected_with_pool(selected_raw, players_by_id, strategy)
            replacements = feasible_replacements(
                players_by_id=players_by_id,
                all_players=all_players,
                selected=selected,
                selected_ids=selected_ids,
                team_counts=team_counts,
                total_price_m=total_price_m,
                strategy=strategy,
            )
            replacement_lookup[(strategy, player_key(selected))] = replacements

            selected_score = to_float(selected.get("strategy_score"))
            best = replacements[0] if replacements else None
            best_score = to_float(best.get("_replacement_strategy_score")) if best else 0.0
            margin = selected_score - best_score if best else selected_score
            flag = manual_flag(selected)
            label = label_for(margin, flag)

            csv_rows.append(
                {
                    "strategy": strategy,
                    "display_name_da": display_name(strategy, display_names, comparison),
                    "selected_player": txt(selected.get("player_name")),
                    "selected_team": txt(selected.get("team_id")),
                    "selected_position": txt(selected.get("position")),
                    "selected_price": int(to_float(selected.get("price"))),
                    "selected_ev": fmt(selected.get("optimizer_ev"), 6),
                    "selected_strategy_score": fmt(selected_score, 6),
                    "selected_conditional_start_prob": fmt(selected.get("conditional_start_prob"), 4),
                    "selected_risk": txt(selected.get("availability_risk")),
                    "best_replacement_1": txt(best.get("player_name")) if best else "",
                    "replacement_1_team": txt(best.get("team_id")) if best else "",
                    "replacement_1_price": int(to_float(best.get("price"))) if best else "",
                    "replacement_1_ev": fmt(best.get("optimizer_ev"), 6) if best else "",
                    "replacement_1_strategy_score": fmt(best_score, 6) if best else "",
                    "replacement_1_conditional_start_prob": fmt(best.get("conditional_start_prob"), 4) if best else "",
                    "replacement_1_risk": txt(best.get("availability_risk")) if best else "",
                    "replacement_margin": fmt(margin, 6),
                    "robustness_label": label,
                    "manual_flag": flag,
                    "recommendation_note": note_for(label, margin, flag, best),
                }
            )

    return csv_rows, replacement_lookup


def core_players(strategies: dict[str, Any]) -> list[list[Any]]:
    appearances: dict[str, dict[str, Any]] = {}
    picked_in: dict[str, list[str]] = defaultdict(list)
    for strategy in STRATEGY_ORDER:
        for player in strategies[strategy]["best_squad"]:
            pid = player_key(player)
            appearances[pid] = player
            picked_in[pid].append(strategy)
    rows = []
    for pid, strategy_names in picked_in.items():
        if len(strategy_names) >= 3:
            player = appearances[pid]
            rows.append(
                [
                    txt(player.get("player_name")),
                    txt(player.get("team_id")),
                    txt(player.get("position")),
                    len(strategy_names),
                    "; ".join(strategy_names),
                ]
            )
    return sorted(rows, key=lambda row: (-int(row[3]), txt(row[2]), txt(row[0])))


def replacement_list_md(replacements: list[dict[str, Any]]) -> str:
    if not replacements:
        return "Ingen gyldige alternativer."
    return "; ".join(
        f"{idx}. {txt(player.get('player_name'))} {txt(player.get('team_id'))} "
        f"score {fmt(player.get('_replacement_strategy_score'), 3)} cond {fmt(player.get('conditional_start_prob'), 3)} "
        f"risk {txt(player.get('availability_risk'))}"
        for idx, player in enumerate(replacements, start=1)
    )


def captain_flag(player: dict[str, Any]) -> str:
    return manual_flag(player)


def captain_alternatives(squad: list[dict[str, Any]], captain_round: int) -> list[dict[str, Any]]:
    growth_col = f"round{captain_round}_captain_growth"
    return sorted(
        squad,
        key=lambda player: (to_float(player.get("captain_score")), to_float(player.get(growth_col))),
        reverse=True,
    )


def write_markdown(
    *,
    strategies: dict[str, Any],
    comparison: dict[str, dict[str, str]],
    rows: list[dict[str, Any]],
    replacements: dict[tuple[str, str], list[dict[str, Any]]],
    context: dict[str, Any],
) -> None:
    fragile = [row for row in rows if row["robustness_label"] in {"fragile_pick", "replace_candidate"}]
    manual = [row for row in rows if row["manual_flag"]]
    robust = [row for row in rows if row["robustness_label"] == "robust_pick"]

    flagged_focus = {"Roberto Alvarado", "Raul Jimenez", "Andreas Schjelderup", "Chris Richards"}
    focus_rows = [row for row in rows if row["selected_player"] in flagged_focus]
    captain_rows: list[list[Any]] = []
    captain_notes: list[str] = []

    for strategy in STRATEGY_ORDER:
        summary = strategies[strategy]["best_summary"]
        squad = strategies[strategy]["best_squad"]
        captain_name = txt(summary.get("recommended_captain"))
        captain_round = int(to_float(summary.get("captain_round"), to_float(context.get("target_round"), 1)))
        captain = next((player for player in squad if txt(player.get("player_name")) == captain_name), {})
        flag = captain_flag(captain) if captain else ""
        alternatives = captain_alternatives(squad, captain_round)[:5]
        captain_rows.append(
            [
                txt(summary.get("display_name_da")) or comparison.get(strategy, {}).get("display_name_da", strategy),
                captain_name,
                fmt(summary.get("captain_expected_growth"), 3),
                flag or "",
                "; ".join(
                    f"{txt(player.get('player_name'))} (score {fmt(player.get('captain_score'), 3)}, vaekst {fmt(player.get(f'round{captain_round}_captain_growth'), 3)})"
                    for player in alternatives
                ),
            ]
        )
        if flag:
            best_unflagged = next((player for player in alternatives if not manual_flag(player)), None)
            if best_unflagged:
                captain_notes.append(
                    f"{txt(summary.get('display_name_da'))}: kaptajn {captain_name} er flagged; bedste ikke-flagged alternativ blandt de 11 er "
                    f"{txt(best_unflagged.get('player_name'))} med kaptajnscore {fmt(best_unflagged.get('captain_score'), 3)} og rundevækst {fmt(best_unflagged.get(f'round{captain_round}_captain_growth'), 3)}."
                )
            else:
                captain_notes.append(f"{txt(summary.get('display_name_da'))}: kaptajnen er flagged, og topalternativerne har også flags eller mangler data.")

    fragile_only = [row for row in rows if row["robustness_label"] == "fragile_pick"]
    replace_candidates = [row for row in rows if row["robustness_label"] == "replace_candidate"]

    conclusion = [
        f"Fragile picks: {len(fragile_only)}. Replace-kandidater med manuelle flags: {len(replace_candidates)} af {len(rows)} strategi-valg.",
        f"Robuste valg med klar replacement-margin: {len(robust)}.",
        "Roberto Alvarado er fortsat valgt som spiller, men manual_captain_status=avoid blokerer ham som kaptajn.",
    ]

    def row_table(source: list[dict[str, Any]]) -> str:
        return table(
            ["Strategi", "Valgt", "Hold", "Pos", "Score", "Bedste alt.", "Alt score", "Margin", "Label", "Flag"],
            [
                [
                    row["display_name_da"],
                    row["selected_player"],
                    row["selected_team"],
                    row["selected_position"],
                    row["selected_strategy_score"],
                    row["best_replacement_1"],
                    row["replacement_1_strategy_score"],
                    row["replacement_margin"],
                    row["robustness_label"],
                    row["manual_flag"],
                ]
                for row in source
            ],
        )

    manual_detail_lines: list[str] = []
    for row in focus_rows:
        selected_id = next(
            player_key(player)
            for player in strategies[row["strategy"]]["best_squad"]
            if txt(player.get("player_name")) == row["selected_player"]
        )
        manual_detail_lines.extend(
            [
                f"### {row['display_name_da']} - {row['selected_player']}",
                "",
                f"- label: {row['robustness_label']}",
                f"- margin: {row['replacement_margin']}",
                f"- flag: {row['manual_flag'] or 'ingen'}",
                f"- note: {row['recommendation_note']}",
                f"- top 5 alternativer: {replacement_list_md(replacements.get((row['strategy'], selected_id), []))}",
                "",
            ]
        )

    lines = [
        "# Strategy Replacement Report",
        "",
        "## Kort Konklusion",
        "",
        *[f"- {item}" for item in conclusion],
        "",
        "## Robuste Kernevalg På Tværs Af Strategier",
        "",
        table(["Spiller", "Hold", "Pos", "Valgt i", "Strategier"], core_players(strategies)),
        "",
        "## Fragile Picks",
        "",
        row_table(fragile),
        "",
        "## Manuelle Tjek Med Bedste Alternativer",
        "",
        row_table(manual),
        "",
        *manual_detail_lines,
        "## Kaptajn",
        "",
        table(["Strategi", "Kaptajn", "Vækst", "Flag", "Top kaptajnalternativer blandt 11"], captain_rows),
        "",
        *[f"- {note}" for note in captain_notes],
        "",
        "## Alle Valgte Spillere Mod Bedste Replacement",
        "",
        row_table(rows),
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for path in [STRATEGIES_PATH, COMPARISON_PATH, PLAYER_EV_PATH, PLAYER_POOL_PATH, MANUAL_OVERRIDES_PATH, DISPLAY_NAMES_PATH, CONTEXT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Mangler input: {path.relative_to(ROOT)}")

    strategies = read_json(STRATEGIES_PATH)
    comparison = {row["strategy"]: row for row in read_csv(COMPARISON_PATH)}
    display_names = read_json(DISPLAY_NAMES_PATH)
    context = read_json(CONTEXT_PATH)

    all_players = load_players().to_dict(orient="records")
    rows, replacements = build_rows(strategies, comparison, display_names, all_players)

    write_csv(OUT_CSV, rows)
    write_markdown(strategies=strategies, comparison=comparison, rows=rows, replacements=replacements, context=context)

    fragile_count = sum(1 for row in rows if row["robustness_label"] == "fragile_pick")
    replace_count = sum(1 for row in rows if row["robustness_label"] == "replace_candidate")
    print(f"Skrevet: {OUT_CSV.relative_to(ROOT)}")
    print(f"Skrevet: {OUT_MD.relative_to(ROOT)}")
    print(f"Fragile picks: {fragile_count}")
    print(f"Replace-kandidater: {replace_count}")
    for name in ["Roberto Alvarado", "Raul Jimenez", "Andreas Schjelderup", "Chris Richards"]:
        focus = [row for row in rows if row["selected_player"] == name]
        if not focus:
            print(f"{name}: ikke valgt i de fire strategier")
            continue
        alts = "; ".join(
            f"{row['display_name_da']} -> {row['best_replacement_1']} ({row['replacement_1_team']}, margin {row['replacement_margin']}, {row['robustness_label']})"
            for row in focus
        )
        print(f"{name}: {alts}")
    captain_flagged = any(
        row["selected_player"] == comparison[strategy]["recommended_captain"] and row["manual_flag"]
        for strategy in STRATEGY_ORDER
        for row in rows
        if row["strategy"] == strategy
    )
    print(f"Kaptajn flagged: {'ja' if captain_flagged else 'nej'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
