from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import optimize_squad_group_stage as optimizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

OPTIMAL_SQUADS_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
POSITIONAL_AUDIT_PATH = DATA_DIR / "positional_budget_value_audit.csv"
REMAINING_FLAGS_PATH = DATA_DIR / "bubble_player_remaining_flags_report.csv"

OUT_COMPONENT_CSV = DATA_DIR / "offensive_ceiling_component_audit.csv"
OUT_COMPONENT_MD = DATA_DIR / "offensive_ceiling_component_audit.md"
OUT_ROUND_CSV = DATA_DIR / "round_context_quality_audit.csv"
OUT_ROUND_MD = DATA_DIR / "round_context_quality_audit.md"

STRATEGIES = ["next_round", "round1_2", "group_stage", "long_run"]
FORMATIONS = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
STRATEGY_SCORE_COL = {
    "next_round": "score_next_round",
    "round1_2": "score_round1_2",
    "group_stage": "score_group_stage",
    "long_run": "score_long_run",
}

PREMIUM_CEILING_NAMES = {
    "erling haaland",
    "kylian mbappe",
    "harry kane",
    "luis diaz",
    "michael olise",
    "jamal musiala",
}
LOW_UPSIDE_NAMES = {
    "konrad laimer",
    "scott mctominay",
    "manu kone",
    "aurelien tchouameni",
    "rodrigo de paul",
    "declan rice",
    "joshua kimmich",
}
ROUND_CONTEXT_NAMES = {
    "erling haaland",
    "harry kane",
    "jurrien timber",
    "wesley franca",
    "raphinha",
    "mahmoud trezeguet",
    "deniz undav",
}
EXCLUDE_CALIBRATION_NAMES = {"jules kounde", "manuel neuer"}

COMPONENT_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "position",
    "price",
    "strategy",
    "formation",
    "target_round",
    "start_prob",
    "appearance_prob",
    "availability_risk",
    "goal_component",
    "assist_component",
    "shot_component",
    "match_winner_component",
    "hattrick_component",
    "player_of_match_component",
    "penalty_component",
    "team_result_component",
    "clean_sheet_component",
    "appearance_component",
    "start_security_effect",
    "fixture_strength_component",
    "round_context_component",
    "long_run_effect",
    "price_value_component",
    "final_ev",
    "strategy_score",
    "suspected_issue",
]

ROUND_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "position",
    "has_real_fixture_specific_ev",
    "has_distributed_round_context",
    "round_1_ev",
    "round_2_ev",
    "round_3_ev",
    "fixture_strength_round_1",
    "fixture_strength_round_2",
    "fixture_strength_round_3",
    "round_context_quality",
    "suspected_issue",
    "recommended_next_action",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    return str(round(to_float(value), digits))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def round_fixture_bonus_row(row: pd.Series, rnd: int) -> float:
    position = txt(row.get("position"))
    offensive = 1.0 if position in {"MID", "FWD"} else 0.0
    defensive = 1.0 if position in {"GK", "DEF"} else 0.0
    win_prob = to_float(row.get(f"round{rnd}_win_prob"))
    favorite = (
        0.45 * float(win_prob >= 0.75)
        + 0.27 * float(0.65 <= win_prob < 0.75)
        + 0.10 * float(0.60 <= win_prob < 0.65)
        - 0.10 * float(win_prob < 0.50)
    )
    attack = (
        max(to_float(row.get(f"round{rnd}_goal_multiplier")) - 1.0, 0.0) * 0.90
        + max(to_float(row.get(f"round{rnd}_assist_multiplier")) - 1.0, 0.0) * 0.55
    ) * offensive
    clean = max(to_float(row.get(f"round{rnd}_clean_sheet_multiplier")) - 1.0, 0.0) * 0.95 * defensive
    return favorite + attack + clean


def starter_component_row(row: pd.Series) -> float:
    cond = to_float(row.get("conditional_start_prob"))
    risk = txt(row.get("availability_risk"))
    return (
        0.45 * float(cond >= 0.85)
        + 0.18 * float(0.75 <= cond < 0.85)
        - 0.65 * float(cond < 0.70)
        - 0.28 * float(0.70 <= cond < 0.75)
        - 0.85 * float(risk == "high_risk")
        - 0.08 * float(risk == "medium_risk")
    )


def sum_match_component(row: pd.Series, suffix: str) -> float:
    return sum(to_float(row.get(f"match_{idx}_{suffix}")) for idx in [1, 2, 3])


def strategy_round_context_component(row: pd.Series, strategy: str, target_round: int) -> float:
    if strategy == "next_round":
        return 2.20 * to_float(row.get(f"round{target_round}_ev"))
    if strategy == "round1_2":
        return 1.85 * (1.05 * to_float(row.get("round1_ev")) + 1.00 * to_float(row.get("round2_ev")))
    if strategy == "group_stage":
        group_value = (
            to_float(row.get("round1_ev"))
            + to_float(row.get("round2_ev"))
            + to_float(row.get("round3_ev")) * to_float(row.get("round3_rotation_factor"), 1.0)
        )
        return 1.75 * group_value
    return 0.0


def strategy_fixture_component(row: pd.Series, strategy: str, target_round: int) -> float:
    if strategy == "next_round":
        return round_fixture_bonus_row(row, target_round)
    if strategy == "round1_2":
        return 0.55 * (round_fixture_bonus_row(row, 1) + round_fixture_bonus_row(row, 2))
    if strategy == "group_stage":
        return 0.35 * (round_fixture_bonus_row(row, 1) + round_fixture_bonus_row(row, 2) + round_fixture_bonus_row(row, 3))
    return 0.0


def long_run_component(row: pd.Series) -> float:
    tournament_strength = max(0.75 * to_float(row.get("team_long_run_score")) + 0.25 * to_float(row.get("team_market_score")), 0.0)
    weak_team_penalty = (
        2.20 * float(tournament_strength < 0.18)
        + 1.25 * float(0.18 <= tournament_strength < 0.28)
        + 0.45 * float(0.28 <= tournament_strength < 0.40)
    )
    manual_penalty = (
        0.95 * float(txt(row.get("manual_status")).lower() == "check")
        + 0.95 * float(txt(row.get("manual_start_status")).lower() == "doubtful")
        + 0.30 * float(txt(row.get("manual_captain_status")).lower() == "avoid")
    )
    mid_team_penalty = 0.65 * float(tournament_strength < 0.35) * float(
        to_float(row.get("optimizer_ev")) < 4.60
        or to_float(row.get("conditional_start_prob")) < 0.88
        or txt(row.get("availability_risk")) == "high_risk"
    )
    return 3.40 * tournament_strength - weak_team_penalty - manual_penalty - mid_team_penalty


def price_value_component(row: pd.Series) -> float:
    before = to_float(row.get("optimizer_ev_before_price_quality"))
    after = to_float(row.get("optimizer_ev"))
    if before:
        return after - before
    return to_float(row.get("price_quality_ev"))


def load_selected_context() -> dict[tuple[str, str], set[str]]:
    if not OPTIMAL_SQUADS_PATH.exists():
        return {}
    data = json.loads(OPTIMAL_SQUADS_PATH.read_text(encoding="utf-8-sig"))
    selected: dict[tuple[str, str], set[str]] = {}
    for strategy, strategy_data in data.items():
        for formation, payload in (strategy_data.get("squads_by_formation") or {}).items():
            selected[(strategy, formation)] = {txt(player.get("player_id")) for player in payload.get("squad") or []}
    return selected


def target_player_ids(players: pd.DataFrame, selected: dict[tuple[str, str], set[str]]) -> set[str]:
    names = PREMIUM_CEILING_NAMES | LOW_UPSIDE_NAMES | ROUND_CONTEXT_NAMES | EXCLUDE_CALIBRATION_NAMES
    ids = set(players.loc[players["player_name"].map(norm).isin(names), "player_id"].astype(str))
    for row in load_csv(REMAINING_FLAGS_PATH):
        ids.add(txt(row.get("player_id")))
    for row in load_csv(POSITIONAL_AUDIT_PATH):
        for field in ["expensive_player_id", "cheap_player_id"]:
            if txt(row.get(field)):
                ids.add(txt(row.get(field)))
    for (strategy, formation), squad_ids in selected.items():
        if formation in {"4-5-1", "5-4-1"}:
            ids.update(squad_ids)
    return ids


def component_issue(row: pd.Series, strategy: str) -> str:
    name_key = norm(row.get("player_name"))
    if name_key in EXCLUDE_CALIBRATION_NAMES and to_float(row.get("optimizer_ev")) <= 0.05:
        return "missing_ev_source_not_model_calibration_evidence"
    if name_key in PREMIUM_CEILING_NAMES:
        missing = []
        if not any(f"match_{idx}_goal_ev" in row.index for idx in [1, 2, 3]):
            missing.append("goal_ev")
        for component in ["match_winner_goal_ev", "hattrick_ev", "player_of_the_match_ev", "penalty_ev"]:
            if component not in row.index:
                missing.append(component)
        if missing:
            return "premium_ceiling_components_missing_or_not_explicit: " + ",".join(missing)
        if strategy != "long_run" and strategy_round_context_component(row, strategy, 1) < to_float(row.get("optimizer_ev")):
            return "absolute_ev_exceeds_round_context_component"
    if name_key in LOW_UPSIDE_NAMES and price_value_component(row) > 0.50:
        return "low_upside_player_helped_by_price_value_layer"
    if txt(row.get("position")) == "FWD" and to_float(row.get("price")) <= 5_000_000 and price_value_component(row) > 0.75:
        return "cheap_fwd_boosted_by_price_value_layer"
    return "no_component_issue_from_available_outputs"


def build_component_rows(players: pd.DataFrame, selected: dict[tuple[str, str], set[str]]) -> list[dict[str, str]]:
    target_ids = target_player_ids(players, selected)
    by_id = {txt(row.get("player_id")): row for _, row in players.iterrows()}
    target_round = int(optimizer.get_current_target_round().get("target_round") or 1)
    rows: list[dict[str, str]] = []

    contexts: list[tuple[str, str, set[str]]] = []
    for strategy in STRATEGIES:
        for formation in FORMATIONS:
            contexts.append((strategy, formation, selected.get((strategy, formation), set())))

    for player_id in sorted(target_ids):
        row = by_id.get(player_id)
        if row is None:
            continue
        for strategy, formation, squad_ids in contexts:
            if player_id not in squad_ids and norm(row.get("player_name")) not in (PREMIUM_CEILING_NAMES | LOW_UPSIDE_NAMES | ROUND_CONTEXT_NAMES | EXCLUDE_CALIBRATION_NAMES):
                continue
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": txt(row.get("player_name")),
                    "team": txt(row.get("team_id")),
                    "position": txt(row.get("position")),
                    "price": str(int(to_float(row.get("price")))),
                    "strategy": strategy,
                    "formation": formation,
                    "target_round": str(target_round),
                    "start_prob": fmt(row.get("start_prob")),
                    "appearance_prob": fmt(row.get("appearance_prob") or row.get("availability_prob")),
                    "availability_risk": txt(row.get("availability_risk")),
                    "goal_component": fmt(sum_match_component(row, "goal_ev")),
                    "assist_component": fmt(sum_match_component(row, "assist_ev")),
                    "shot_component": fmt(sum_match_component(row, "shots_on_target_ev")),
                    "match_winner_component": "",
                    "hattrick_component": "",
                    "player_of_match_component": "",
                    "penalty_component": "",
                    "team_result_component": fmt(sum_match_component(row, "result_ev") + sum_match_component(row, "team_scores_ev") + sum_match_component(row, "opponent_scores_ev")),
                    "clean_sheet_component": fmt(sum_match_component(row, "clean_sheet_ev")),
                    "appearance_component": fmt(sum_match_component(row, "on_pitch_ev") + sum_match_component(row, "start_minutes_ev")),
                    "start_security_effect": fmt(starter_component_row(row)),
                    "fixture_strength_component": fmt(strategy_fixture_component(row, strategy, target_round)),
                    "round_context_component": fmt(strategy_round_context_component(row, strategy, target_round)),
                    "long_run_effect": fmt(long_run_component(row) if strategy == "long_run" else 0.0),
                    "price_value_component": fmt(price_value_component(row)),
                    "final_ev": fmt(row.get("optimizer_ev")),
                    "strategy_score": fmt(row.get(STRATEGY_SCORE_COL[strategy])),
                    "suspected_issue": component_issue(row, strategy),
                }
            )
    return rows


def fixture_strength_label(row: pd.Series, rnd: int) -> float:
    goal = to_float(row.get(f"round{rnd}_goal_multiplier"))
    assist = to_float(row.get(f"round{rnd}_assist_multiplier"))
    clean = to_float(row.get(f"round{rnd}_clean_sheet_multiplier"))
    win = to_float(row.get(f"round{rnd}_win_prob"))
    return round((goal - 1.0) + (assist - 1.0) + (clean - 1.0) + (win - 0.5), 4)


def build_round_rows(players: pd.DataFrame) -> list[dict[str, str]]:
    names = ROUND_CONTEXT_NAMES | EXCLUDE_CALIBRATION_NAMES
    remaining_ids = {txt(row.get("player_id")) for row in load_csv(REMAINING_FLAGS_PATH) if row.get("primary_flag_category") in {"missing_or_weak_round_context", "missing_ev_source"}}
    rows: list[dict[str, str]] = []
    for _, row in players.iterrows():
        if norm(row.get("player_name")) not in names and txt(row.get("player_id")) not in remaining_ids:
            continue
        round_evs = [to_float(row.get(f"round{rnd}_ev")) for rnd in [1, 2, 3]]
        match_weighted = [to_float(row.get(f"match_{rnd}_weighted_match_ev")) for rnd in [1, 2, 3]]
        has_real = any(txt(row.get(f"match_{rnd}_opponent_team")) for rnd in [1, 2, 3]) and sum(match_weighted) > 0.05
        optimizer_ev = to_float(row.get("optimizer_ev"))
        has_distributed = has_real and optimizer_ev > 0.05 and max(round_evs) > 0 and abs(sum(round_evs) - optimizer_ev) / max(optimizer_ev, 0.01) < 0.55
        if optimizer_ev <= 0.05 and max(round_evs) <= 0.05:
            quality = "missing_ev_source"
            issue = "No usable optimizer_ev or round EV."
            action = "Fix EV source/fixture mapping; do not use as calibration evidence."
        elif not has_real:
            quality = "weak_or_missing_fixture_specific_ev"
            issue = "No clear fixture-specific weighted match EV in exported columns."
            action = "Audit EV export/match mapping before changing weights."
        elif optimizer_ev > 0.05 and max(round_evs) <= 0.10:
            quality = "aggregate_ev_without_round_support"
            issue = "Aggregate EV exists but round EV is near zero."
            action = "Check round export/distribution."
        elif norm(row.get("player_name")) in {"erling haaland", "harry kane"} and max(round_evs) < optimizer_ev * 0.30:
            quality = "weak_round_context_for_premium_fwd"
            issue = "Premium FWD has real EV, but specific round values are low relative to aggregate EV."
            action = "Review fixture-specific offensive ceiling and round weighting."
        elif has_distributed:
            quality = "distributed_but_plausible"
            issue = "Round EV appears distributed from fixture-specific match output."
            action = "No data repair required; use model calibration audit if player still looks wrong."
        else:
            quality = "real_fixture_specific_context"
            issue = "Fixture-specific round context exists."
            action = "Treat as model/role question, not missing data."
        rows.append(
            {
                "player_id": txt(row.get("player_id")),
                "player_name": txt(row.get("player_name")),
                "team": txt(row.get("team_id")),
                "position": txt(row.get("position")),
                "has_real_fixture_specific_ev": str(bool(has_real)),
                "has_distributed_round_context": str(bool(has_distributed)),
                "round_1_ev": fmt(row.get("round1_ev")),
                "round_2_ev": fmt(row.get("round2_ev")),
                "round_3_ev": fmt(row.get("round3_ev")),
                "fixture_strength_round_1": fmt(fixture_strength_label(row, 1)),
                "fixture_strength_round_2": fmt(fixture_strength_label(row, 2)),
                "fixture_strength_round_3": fmt(fixture_strength_label(row, 3)),
                "round_context_quality": quality,
                "suspected_issue": issue,
                "recommended_next_action": action,
            }
        )
    return rows


def md_table(rows: list[dict[str, str]], cols: list[str], limit: int = 20) -> list[str]:
    if not rows:
        return ["Ingen rækker."]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(txt(row.get(col)).replace("|", "/") for col in cols) + " |")
    return out


def write_component_md(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    premium = df[df["player_name"].map(norm).isin(PREMIUM_CEILING_NAMES)] if not df.empty else pd.DataFrame()
    cheap_fwd = df[(df["position"] == "FWD") & (pd.to_numeric(df["price"], errors="coerce") <= 5_000_000)] if not df.empty else pd.DataFrame()
    low_upside = df[df["player_name"].map(norm).isin(LOW_UPSIDE_NAMES)] if not df.empty else pd.DataFrame()

    def avg(frame: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(frame.get(col), errors="coerce").fillna(0).mean()) if not frame.empty else 0.0

    issue_counts = df["suspected_issue"].value_counts().to_dict() if not df.empty else {}
    lines = [
        "# Offensive Ceiling Component Audit",
        "",
        "Ren audit af eksisterende modeloutput. Ingen optimizer, strategioutput, EV-modelkalibrering eller frontend er koert.",
        "",
        "## Komponenter i outputtet",
        "",
        "- Eksisterer eksplicit: goal_ev, assist_ev, shots_on_target_ev, clean_sheet_ev, result/team-score/on-pitch/start-minutes, fixture multipliers, round EV, price_quality og strategy_score.",
        "- Findes ikke eksplicit i nuvaerende output: match_winner_goal_ev, hattrick_ev, player_of_the_match_ev og penalty_ev. De er derfor blanke i CSV'en og maa ikke tolkes som nul-effekt.",
        "",
        "## Kort konklusion",
        "",
        f"- Premium/ceiling-spillere har i gennemsnit goal_component {avg(premium, 'goal_component'):.3f}, price_value_component {avg(premium, 'price_value_component'):.3f}.",
        f"- Billige FWD-rækker har i gennemsnit price_value_component {avg(cheap_fwd, 'price_value_component'):.3f}, hvilket viser at price/value-laget ofte er positivt for value-spillere.",
        f"- Lav-upside MID-rækker har i gennemsnit strategy_score {avg(low_upside, 'strategy_score'):.3f}; de får især hjælp af start_security_effect og fixture/round-kontekst.",
        "- Det stærkeste audit-signal er kombinationen af manglende eksplicit ceiling-bonusser og et tydeligt price/value-lag, ikke én isoleret datakolonne.",
        "",
        "## Suspected issue distribution",
        "",
        "| suspected_issue | rows |",
        "|---|---:|",
    ]
    for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {issue} | {count} |")

    lines += [
        "",
        "## Premium og ceiling-spillere",
        "",
        *md_table(
            premium.drop_duplicates(["player_id", "strategy"])[
                [
                    "player_name",
                    "strategy",
                    "goal_component",
                    "assist_component",
                    "shot_component",
                    "round_context_component",
                    "price_value_component",
                    "strategy_score",
                    "suspected_issue",
                ]
            ].to_dict("records") if not premium.empty else [],
            [
                "player_name",
                "strategy",
                "goal_component",
                "assist_component",
                "shot_component",
                "round_context_component",
                "price_value_component",
                "strategy_score",
                "suspected_issue",
            ],
            30,
        ),
        "",
        "## Lav-upside MID/DEF reference",
        "",
        *md_table(
            low_upside.drop_duplicates(["player_id", "strategy"])[
                [
                    "player_name",
                    "strategy",
                    "start_security_effect",
                    "round_context_component",
                    "price_value_component",
                    "strategy_score",
                    "suspected_issue",
                ]
            ].to_dict("records") if not low_upside.empty else [],
            ["player_name", "strategy", "start_security_effect", "round_context_component", "price_value_component", "strategy_score", "suspected_issue"],
            30,
        ),
        "",
        "## Svar paa price/value-spoergsmaal",
        "",
        "- Ja, en billig spiller kan ranke hoejt delvist fordi price/value-laget er positivt, især hvis han samtidig har god start/fixture-kontekst.",
        "- Auditoutputtet viser, at price/value er et stærkt forklaringslag for billige FWDs, men det er ikke alene: round_context og start_security driver også.",
        "- 4-5-1 og 5-4-1 er udsatte, fordi de kun har én FWD-slot; hvis den slot går til value fremfor ceiling, forsvinder meget offensiv upside.",
        "- Kounde og Neuer er fortsat missing EV-source og bruges ikke som kalibreringsbevis.",
        "",
        "## Mulige fremtidige modeltests",
        "",
        "1. Eksporter/estimer eksplicit offensive ceiling: multi-goal, matchwinner, player-of-match og penalty EV.",
        "2. Cap eller positionsjuster price/value-effekten for FWD, så billig value ikke automatisk slår høj absolut ceiling.",
        "3. Tilføj formation-aware ceiling floor for 4-5-1/5-4-1, fx krav om høj FWD absolut EV eller captain-growth.",
        "4. Test om lav-upside MID/DEF skal have ceiling- eller role-penalty i offensive strategier.",
    ]
    OUT_COMPONENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_round_md(rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["round_context_quality"]] = counts.get(row["round_context_quality"], 0) + 1
    focus = [row for row in rows if norm(row["player_name"]) in ROUND_CONTEXT_NAMES | EXCLUDE_CALIBRATION_NAMES]
    lines = [
        "# Round Context Quality Audit",
        "",
        "Audit af eksisterende round-/fixture-kontekst. Ingen EV eller strategioutput er genberegnet.",
        "",
        "## Fordeling",
        "",
        "| round_context_quality | rows |",
        "|---|---:|",
    ]
    for quality, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {quality} | {count} |")
    lines += [
        "",
        "## Fokusspillere",
        "",
        *md_table(
            focus,
            [
                "player_name",
                "team",
                "position",
                "round_1_ev",
                "round_2_ev",
                "round_3_ev",
                "round_context_quality",
                "suspected_issue",
                "recommended_next_action",
            ],
            40,
        ),
        "",
        "## Konklusion",
        "",
        "- Mange remaining flags skyldes ikke manglende spiller-match, men svag eller utilstrækkeligt forklarende round context.",
        "- Haaland og Kane har reel model-EV, men deres specifikke runde-EV er relativt svag ift. premium-forventning; det peger på round/fixture-ceiling audit snarere end simpel value-fejl.",
        "- Kounde og Neuer er rene missing EV-source cases og bør ikke bruges som modelkalibreringsbevis.",
        "- Raphinha, Wesley Franca, Trezeguet og Timber har round context, men auditten viser fortsat, at rundeværdien skal forklares bedre, før de bruges som argument for vægtændring.",
    ]
    OUT_ROUND_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    players = optimizer.load_players()
    selected = load_selected_context()
    component_rows = build_component_rows(players, selected)
    round_rows = build_round_rows(players)

    write_csv(OUT_COMPONENT_CSV, COMPONENT_COLUMNS, component_rows)
    write_component_md(component_rows)
    write_csv(OUT_ROUND_CSV, ROUND_COLUMNS, round_rows)
    write_round_md(round_rows)

    print("Offensive ceiling component audit")
    print("---------------------------------")
    print(f"Component rows: {len(component_rows)}")
    print(f"Round context rows: {len(round_rows)}")
    print(f"Wrote: {OUT_COMPONENT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_COMPONENT_MD.relative_to(ROOT)}")
    print(f"Wrote: {OUT_ROUND_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_ROUND_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
