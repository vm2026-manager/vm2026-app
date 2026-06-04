from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
MATCH_ODDS_PATH = ROOT / "match_odds_probs.csv"
FIXTURE_STRENGTH_PATH = DATA / "fixture_strength_multipliers.csv"
OUT_CSV = DATA / "player_ev_component_dependency_audit.csv"
OUT_MD = DATA / "player_ev_component_dependency_audit.md"
TEAM_AUDIT_CSV = DATA / "team_match_component_consistency_audit.csv"
TEAM_AUDIT_MD = DATA / "team_match_component_consistency_audit.md"
TEAM_REPAIR_CSV = DATA / "team_component_repair_report.csv"
TEAM_REPAIR_MD = DATA / "team_component_repair_report.md"
CLEAN_SHEET_AUDIT_CSV = DATA / "clean_sheet_component_repair_audit.csv"
CLEAN_SHEET_AUDIT_MD = DATA / "clean_sheet_component_repair_audit.md"

ROUND_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.90}
GOAL_POINTS = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
ASSIST_POINTS = 3.0
SHOT_ON_TARGET_POINTS = 1.0
CLEAN_SHEET_POINTS = {"GK": 2.8, "DEF": 2.2, "MID": 0.0, "FWD": 0.0}
YELLOW_CARD_POINTS = -1.0
ON_PITCH_POINTS = 7.0
NOT_ON_PITCH_POINTS = -5.0
WIN_POINTS = 25.0
DRAW_POINTS = 5.0
LOSS_POINTS = -8.0
TEAM_SCORES_POINTS = 10.0
OPPONENT_SCORES_POINTS = -8.0
COMPONENT_POINT_SCALE = 100.0

START_DEPENDENT_COMPONENTS = [
    "goal_ev",
    "assist_ev",
    "shots_on_target_ev",
    "card_ev",
]

TEAM_CONTEXT_COMPONENTS = [
    "result_ev",
    "team_scores_ev",
    "opponent_scores_ev",
    "on_pitch_ev",
]

AUDIT_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "start_prob",
    "minute_share",
    "match_1_start_minutes_ev",
    "match_1_goal_ev",
    "match_1_assist_ev",
    "match_1_shots_on_target_ev",
    "match_1_weighted_match_ev",
    "weighted_group_stage_ev",
    "component_source",
    "component_recomputed",
    "suspected_issue",
]

TEAM_AUDIT_FIELDS = [
    "match_no",
    "team_id",
    "opponent",
    "component",
    "players",
    "min",
    "max",
    "spread",
    "negative_count",
    "high_start_players",
    "high_start_min",
    "high_start_max",
    "high_start_spread",
    "high_start_negative_count",
]

TEAM_REPAIR_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "match_no",
    "opponent",
    "start_prob",
    "appearance_prob_used",
    "old_on_pitch_ev",
    "new_on_pitch_ev",
    "old_result_ev",
    "new_result_ev",
    "old_team_scores_ev",
    "new_team_scores_ev",
    "old_opponent_scores_ev",
    "new_opponent_scores_ev",
    "old_weighted_match_ev",
    "new_weighted_match_ev",
    "repair_reason",
]

CLEAN_SHEET_AUDIT_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "match_no",
    "opponent",
    "start_prob",
    "clean_sheet_prob_used",
    "old_clean_sheet_ev",
    "new_clean_sheet_ev",
    "old_weighted_match_ev",
    "new_weighted_match_ev",
    "repair_status",
]

SANITY_NAMES = {
    "Erling Haaland",
    "Harry Kane",
    "Raphinha",
    "Jules Kounde",
    "Jules Koundé",
    "Manuel Neuer",
    "Martin Ødegaard",
    "Martin Odegaard",
    "Antonio Nusa",
    "Alexander Sørloth",
    "Alexander Sorloth",
    "Alexander Schlager",
    "Patrick Pentz",
    "Stefan Posch",
    "Philipp Lienhart",
    "Kevin Danso",
    "Maximilian WÃ¶ber",
    "Maximilian Wöber",
    "Mike Maignan",
    "Gregor Kobel",
}


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


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def match_weighted_sum(row: dict[str, Any]) -> float:
    return sum(to_float(row.get(f"match_{idx}_weighted_match_ev")) for idx in [1, 2, 3])


def match_total_sum(row: dict[str, Any]) -> float:
    return sum(to_float(row.get(f"match_{idx}_total_ev_next_match")) for idx in [1, 2, 3])


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def has_any_base_component(row: dict[str, Any], match_no: int) -> bool:
    return any(txt(row.get(f"match_{match_no}_{component}")) for component in START_DEPENDENT_COMPONENTS + TEAM_CONTEXT_COMPONENTS)


def has_any_match_component(row: dict[str, Any]) -> bool:
    return any(has_any_base_component(row, idx) for idx in [1, 2, 3])


def has_any_round_ev(row: dict[str, Any]) -> bool:
    return any(abs(to_float(row.get(f"match_{idx}_weighted_match_ev"))) > 1e-9 for idx in [1, 2, 3])


def implied_component_start(row: dict[str, Any], match_no: int) -> float:
    minutes = to_float(row.get(f"match_{match_no}_minutes_if_start"))
    start_minutes = to_float(row.get(f"match_{match_no}_start_minutes_ev"))
    if minutes <= 0 or start_minutes <= 0:
        return 0.0
    return start_minutes / minutes


def stale_matches(row: dict[str, Any], threshold: float = 0.05) -> list[int]:
    start = to_float(row.get("start_prob"))
    out = []
    for idx in [1, 2, 3]:
        implied = implied_component_start(row, idx)
        if implied > 0 and abs(start - implied) >= threshold:
            out.append(idx)
    return out


def recompute_match_total(row: dict[str, Any], match_no: int) -> float:
    position = txt(row.get("position")).upper()
    goal_ev = to_float(row.get(f"match_{match_no}_goal_ev"))
    assist_ev = to_float(row.get(f"match_{match_no}_assist_ev"))
    sot_ev = to_float(row.get(f"match_{match_no}_shots_on_target_ev"))
    clean_sheet_ev = to_float(row.get(f"match_{match_no}_clean_sheet_ev"))
    card_ev = to_float(row.get(f"match_{match_no}_card_ev"))
    result_ev = to_float(row.get(f"match_{match_no}_result_ev"))
    team_scores_ev = to_float(row.get(f"match_{match_no}_team_scores_ev"))
    opponent_scores_ev = to_float(row.get(f"match_{match_no}_opponent_scores_ev"))
    on_pitch_ev = to_float(row.get(f"match_{match_no}_on_pitch_ev"))

    return (
        goal_ev * GOAL_POINTS.get(position, 5.0)
        + assist_ev * ASSIST_POINTS
        + sot_ev * SHOT_ON_TARGET_POINTS
        + clean_sheet_ev * CLEAN_SHEET_POINTS.get(position, 0.0)
        + card_ev * YELLOW_CARD_POINTS
        + result_ev
        + team_scores_ev
        + opponent_scores_ev
        + on_pitch_ev
    )


def load_match_odds_lookup() -> dict[tuple[str, str, str], dict[str, float]]:
    _, rows = read_csv(MATCH_ODDS_PATH)
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        kickoff = txt(row.get("kickoff_dk"))
        home_win = to_float(row.get("home_win_prob_fair"))
        draw = to_float(row.get("draw_prob_fair"))
        away_win = to_float(row.get("away_win_prob_fair"))
        if not home or not away or not kickoff:
            continue
        lookup[(home, away, kickoff)] = {"win": home_win, "draw": draw, "loss": away_win}
        lookup[(away, home, kickoff)] = {"win": away_win, "draw": draw, "loss": home_win}
    return lookup


def load_clean_sheet_lookup() -> dict[tuple[str, str, str], dict[str, float]]:
    _, rows = read_csv(FIXTURE_STRENGTH_PATH)
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        kickoff = txt(row.get("kickoff_dk"))
        if not home or not away or not kickoff:
            continue
        lookup[(home, away, kickoff)] = {
            "prob": to_float(row.get("home_clean_sheet_prob_fair"), -1.0),
            "multiplier": to_float(row.get("home_clean_sheet_multiplier"), 1.0),
        }
        lookup[(away, home, kickoff)] = {
            "prob": to_float(row.get("away_clean_sheet_prob_fair"), -1.0),
            "multiplier": to_float(row.get("away_clean_sheet_multiplier"), 1.0),
        }
    return lookup


def match_key(row: dict[str, Any], match_no: int) -> tuple[str, str, str]:
    return (
        txt(row.get("team_id")).upper(),
        txt(row.get(f"match_{match_no}_opponent_team")).upper(),
        txt(row.get(f"match_{match_no}_kickoff")),
    )


def old_on_pitch_appearance_signal(row: dict[str, Any], match_no: int) -> float:
    old = to_float(row.get(f"match_{match_no}_on_pitch_ev"), 999.0)
    if old == 999.0:
        return 0.0
    return clamp(((old * COMPONENT_POINT_SCALE) - NOT_ON_PITCH_POINTS) / (ON_PITCH_POINTS - NOT_ON_PITCH_POINTS))


def appearance_prob(row: dict[str, Any], match_no: int) -> float:
    explicit = to_float(row.get("appearance_prob"), -1.0)
    if explicit >= 0:
        return clamp(explicit)
    existing = old_on_pitch_appearance_signal(row, match_no)
    return clamp(max(to_float(row.get("start_prob")), existing))


def clean_sheet_eligibility_prob(row: dict[str, Any], match_no: int) -> float:
    minutes_if_start = to_float(row.get(f"match_{match_no}_minutes_if_start"))
    start = clamp(to_float(row.get("start_prob")))
    if minutes_if_start <= 0:
        return start
    # Clean sheet points require enough minutes. The model already stores expected
    # minutes when starting, so use it as a conservative 60-minute eligibility proxy.
    return start * clamp(minutes_if_start / 60.0)


def clean_sheet_component_value(row: dict[str, Any], match_no: int, clean_sheet_prob: float) -> float:
    position = txt(row.get("position")).upper()
    if position not in {"GK", "DEF"} or clean_sheet_prob < 0:
        return 0.0
    return clean_sheet_prob * clean_sheet_eligibility_prob(row, match_no)


def result_base_from_odds(odds: dict[str, float] | None) -> float:
    if not odds:
        return 0.0
    expected = odds["win"] * WIN_POINTS + odds["draw"] * DRAW_POINTS + odds["loss"] * LOSS_POINTS
    return expected / COMPONENT_POINT_SCALE


def build_team_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    odds_lookup = load_match_odds_lookup()
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], int]]] = {}
    for row in rows:
        for match_no in [1, 2, 3]:
            key = match_key(row, match_no)
            if key[0] and key[1]:
                grouped.setdefault(key, []).append((row, match_no))

    baselines: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, entries in grouped.items():
        team_score_probs: list[float] = []
        opponent_score_probs: list[float] = []
        result_bases: list[float] = []
        for row, match_no in entries:
            app = max(appearance_prob(row, match_no), 0.01)
            team_scores = to_float(row.get(f"match_{match_no}_team_scores_ev"))
            opponent_scores = to_float(row.get(f"match_{match_no}_opponent_scores_ev"))
            result_ev = to_float(row.get(f"match_{match_no}_result_ev"))
            if team_scores > 0:
                team_score_probs.append(clamp((team_scores * COMPONENT_POINT_SCALE) / (TEAM_SCORES_POINTS * app)))
            if opponent_scores < 0:
                opponent_score_probs.append(clamp((abs(opponent_scores) * COMPONENT_POINT_SCALE) / (abs(OPPONENT_SCORES_POINTS) * app)))
            if result_ev > 0:
                result_bases.append(result_ev / app)

        odds_base = result_base_from_odds(odds_lookup.get(key))
        baselines[key] = {
            "result_base": odds_base or (max(result_bases) if result_bases else 0.0),
            "team_scores_prob": max(team_score_probs) if team_score_probs else 0.0,
            "opponent_scores_prob": max(opponent_score_probs) if opponent_score_probs else 0.0,
        }
    return baselines


def repair_team_context_components(
    row: dict[str, Any],
    match_no: int,
    baselines: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[bool, dict[str, Any] | None]:
    key = match_key(row, match_no)
    baseline = baselines.get(key)
    if not baseline:
        return False, None

    app = appearance_prob(row, match_no)
    old_weighted = to_float(row.get(f"match_{match_no}_weighted_match_ev"))
    old_on_pitch = txt(row.get(f"match_{match_no}_on_pitch_ev"))
    old_result = txt(row.get(f"match_{match_no}_result_ev"))
    old_team_scores = txt(row.get(f"match_{match_no}_team_scores_ev"))
    old_opponent_scores = txt(row.get(f"match_{match_no}_opponent_scores_ev"))

    new_on_pitch = ((app * ON_PITCH_POINTS) + ((1.0 - app) * NOT_ON_PITCH_POINTS)) / COMPONENT_POINT_SCALE
    new_result = baseline["result_base"] * app
    new_team_scores = (baseline["team_scores_prob"] * TEAM_SCORES_POINTS * app) / COMPONENT_POINT_SCALE
    new_opponent_scores = (baseline["opponent_scores_prob"] * OPPONENT_SCORES_POINTS * app) / COMPONENT_POINT_SCALE

    changed = any(
        abs(to_float(old) - new) > 0.000001
        for old, new in [
            (old_on_pitch, new_on_pitch),
            (old_result, new_result),
            (old_team_scores, new_team_scores),
            (old_opponent_scores, new_opponent_scores),
        ]
    )
    if not changed:
        return False, None

    row[f"match_{match_no}_on_pitch_ev"] = fmt(new_on_pitch)
    row[f"match_{match_no}_result_ev"] = fmt(new_result)
    row[f"match_{match_no}_team_scores_ev"] = fmt(new_team_scores)
    row[f"match_{match_no}_opponent_scores_ev"] = fmt(new_opponent_scores)

    report = {
        "player_id": txt(row.get("player_id")),
        "player_name": txt(row.get("player_name")),
        "team_id": txt(row.get("team_id")),
        "position": txt(row.get("position")),
        "match_no": match_no,
        "opponent": txt(row.get(f"match_{match_no}_opponent_team")),
        "start_prob": txt(row.get("start_prob")),
        "appearance_prob_used": fmt(app),
        "old_on_pitch_ev": old_on_pitch,
        "new_on_pitch_ev": fmt(new_on_pitch),
        "old_result_ev": old_result,
        "new_result_ev": fmt(new_result),
        "old_team_scores_ev": old_team_scores,
        "new_team_scores_ev": fmt(new_team_scores),
        "old_opponent_scores_ev": old_opponent_scores,
        "new_opponent_scores_ev": fmt(new_opponent_scores),
        "old_weighted_match_ev": fmt(old_weighted),
        "new_weighted_match_ev": "",
        "repair_reason": "rebuilt_from_match_context_and_appearance_prob",
    }
    return True, report


def repair_clean_sheet_component(
    row: dict[str, Any],
    match_no: int,
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[bool, dict[str, Any] | None]:
    position = txt(row.get("position")).upper()
    key = match_key(row, match_no)
    context = clean_sheet_lookup.get(key)
    if not context or context["prob"] < 0:
        return False, None

    old_weighted = to_float(row.get(f"match_{match_no}_weighted_match_ev"))
    old_clean = txt(row.get(f"match_{match_no}_clean_sheet_ev"))
    new_clean = clean_sheet_component_value(row, match_no, context["prob"])
    if position not in {"GK", "DEF"}:
        new_clean = 0.0

    changed = abs(to_float(old_clean) - new_clean) > 0.000001
    row[f"match_{match_no}_clean_sheet_prob"] = fmt(context["prob"])
    row[f"match_{match_no}_clean_sheet_multiplier"] = fmt(context["multiplier"])
    if not changed:
        return False, {
            "player_id": txt(row.get("player_id")),
            "player_name": txt(row.get("player_name")),
            "team_id": txt(row.get("team_id")),
            "position": position,
            "match_no": match_no,
            "opponent": txt(row.get(f"match_{match_no}_opponent_team")),
            "start_prob": txt(row.get("start_prob")),
            "clean_sheet_prob_used": fmt(context["prob"]),
            "old_clean_sheet_ev": old_clean,
            "new_clean_sheet_ev": fmt(new_clean),
            "old_weighted_match_ev": fmt(old_weighted),
            "new_weighted_match_ev": txt(row.get(f"match_{match_no}_weighted_match_ev")),
            "repair_status": "unchanged_already_consistent",
        }

    row[f"match_{match_no}_clean_sheet_ev"] = fmt(new_clean)
    return True, {
        "player_id": txt(row.get("player_id")),
        "player_name": txt(row.get("player_name")),
        "team_id": txt(row.get("team_id")),
        "position": position,
        "match_no": match_no,
        "opponent": txt(row.get(f"match_{match_no}_opponent_team")),
        "start_prob": txt(row.get("start_prob")),
        "clean_sheet_prob_used": fmt(context["prob"]),
        "old_clean_sheet_ev": old_clean,
        "new_clean_sheet_ev": fmt(new_clean),
        "old_weighted_match_ev": fmt(old_weighted),
        "new_weighted_match_ev": "",
        "repair_status": "rebuilt_from_fixture_clean_sheet_prob",
    }


def classify_row(row: dict[str, Any]) -> tuple[str, str]:
    weighted_ev = max(to_float(row.get("weighted_group_stage_ev")), to_float(row.get("optimizer_ev")))
    stale = stale_matches(row)
    has_components = has_any_match_component(row)
    has_round_ev = has_any_round_ev(row)

    if stale and has_components:
        return "existing_components", "stale_start_dependent_components"
    if weighted_ev > 0 and has_round_ev and not has_components:
        return "aggregate_round_ev_only", "aggregate_ev_but_missing_base_components"
    if weighted_ev <= 0 and not has_round_ev:
        return "missing_ev_source", "no_player_ev_source"
    if has_components:
        return "existing_components", "ok"
    return "unknown", "needs_manual_pipeline_review"


def audit_row(row: dict[str, Any], recomputed: str) -> dict[str, Any]:
    source, issue = classify_row(row)
    return {
        "player_id": txt(row.get("player_id")),
        "player_name": txt(row.get("player_name")),
        "team_id": txt(row.get("team_id")),
        "position": txt(row.get("position")),
        "start_prob": txt(row.get("start_prob")),
        "minute_share": txt(row.get("minute_share")),
        "match_1_start_minutes_ev": txt(row.get("match_1_start_minutes_ev")),
        "match_1_goal_ev": txt(row.get("match_1_goal_ev")),
        "match_1_assist_ev": txt(row.get("match_1_assist_ev")),
        "match_1_shots_on_target_ev": txt(row.get("match_1_shots_on_target_ev")),
        "match_1_weighted_match_ev": txt(row.get("match_1_weighted_match_ev")),
        "weighted_group_stage_ev": txt(row.get("weighted_group_stage_ev")),
        "component_source": source,
        "component_recomputed": recomputed,
        "suspected_issue": issue,
    }


def repair_row(
    row: dict[str, Any],
    baselines: dict[tuple[str, str, str], dict[str, float]],
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    start = to_float(row.get("start_prob"))
    if start <= 0:
        return False, [], []

    changed = False
    reports: list[dict[str, Any]] = []
    clean_sheet_reports: list[dict[str, Any]] = []
    for idx in [1, 2, 3]:
        implied = implied_component_start(row, idx)
        minutes = to_float(row.get(f"match_{idx}_minutes_if_start"))
        if implied > 0 and minutes > 0:
            ratio = start / implied if implied > 0 else 1.0
            if abs(ratio - 1.0) >= 0.01:
                row[f"match_{idx}_start_minutes_ev"] = fmt(start * minutes)
                for component in START_DEPENDENT_COMPONENTS:
                    key = f"match_{idx}_{component}"
                    if txt(row.get(key)):
                        row[key] = fmt(to_float(row.get(key)) * ratio)
                changed = True

        clean_changed, clean_report = repair_clean_sheet_component(row, idx, clean_sheet_lookup)
        if clean_changed:
            changed = True
        if clean_report:
            clean_sheet_reports.append(clean_report)

        team_changed, report = repair_team_context_components(row, idx, baselines)
        if team_changed:
            changed = True
            if report:
                reports.append(report)

        if team_changed or clean_changed or (implied > 0 and minutes > 0):
            total = recompute_match_total(row, idx)
            row[f"match_{idx}_total_ev_next_match"] = fmt(total)
            row[f"match_{idx}_weighted_match_ev"] = fmt(total * ROUND_WEIGHTS[idx])
            if reports and to_float(reports[-1].get("match_no")) == idx:
                reports[-1]["new_weighted_match_ev"] = txt(row.get(f"match_{idx}_weighted_match_ev"))
            if clean_sheet_reports and to_float(clean_sheet_reports[-1].get("match_no")) == idx:
                clean_sheet_reports[-1]["new_weighted_match_ev"] = txt(row.get(f"match_{idx}_weighted_match_ev"))

    if changed:
        new_weighted_sum = match_weighted_sum(row)
        new_total_sum = match_total_sum(row)
        row["weighted_group_stage_ev_before_price_quality"] = fmt(max(0.0, new_weighted_sum))
        row["optimizer_ev_before_price_quality"] = fmt(max(0.0, new_weighted_sum))
        row["model_ev_before_price_quality"] = fmt(max(0.0, new_weighted_sum))
        row["weighted_group_stage_ev"] = fmt(max(0.0, new_weighted_sum))
        row["optimizer_ev"] = fmt(max(0.0, new_weighted_sum))
        row["total_ev_group_stage"] = fmt(max(0.0, new_total_sum))

    return changed, reports, clean_sheet_reports


def team_component_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match_no in [1, 2, 3]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            team = txt(row.get("team_id")).upper()
            opponent = txt(row.get(f"match_{match_no}_opponent_team")).upper()
            if team and opponent:
                grouped.setdefault((team, opponent), []).append(row)
        for (team, opponent), group_rows in grouped.items():
            for suffix in TEAM_CONTEXT_COMPONENTS:
                key = f"match_{match_no}_{suffix}"
                values = [to_float(row.get(key)) for row in group_rows if txt(row.get(key))]
                high_values = [to_float(row.get(key)) for row in group_rows if txt(row.get(key)) and to_float(row.get("start_prob")) >= 0.70]
                if not values:
                    continue
                row = {
                    "match_no": match_no,
                    "team_id": team,
                    "opponent": opponent,
                    "component": key,
                    "players": len(values),
                    "min": fmt(min(values)),
                    "max": fmt(max(values)),
                    "spread": fmt(max(values) - min(values)),
                    "negative_count": sum(1 for value in values if value < 0),
                    "high_start_players": len(high_values),
                    "high_start_min": fmt(min(high_values)) if high_values else "",
                    "high_start_max": fmt(max(high_values)) if high_values else "",
                    "high_start_spread": fmt(max(high_values) - min(high_values)) if high_values else "",
                    "high_start_negative_count": sum(1 for value in high_values if value < 0),
                }
                out.append(row)
    out.sort(key=lambda row: (to_float(row.get("high_start_spread")), to_float(row.get("spread"))), reverse=True)
    return out


def negative_high_start_on_pitch(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if to_float(row.get("start_prob")) < 0.70:
            continue
        for idx in [1, 2, 3]:
            if txt(row.get(f"match_{idx}_on_pitch_ev")) and to_float(row.get(f"match_{idx}_on_pitch_ev")) < 0:
                count += 1
    return count


def count_high_start_spreads(audit: list[dict[str, Any]], threshold: float = 0.05) -> int:
    return sum(
        1
        for row in audit
        if row["component"].endswith("on_pitch_ev") and to_float(row.get("high_start_spread")) > threshold
    )


def max_high_start_on_pitch_spread(audit: list[dict[str, Any]]) -> float:
    values = [
        to_float(row.get("high_start_spread"))
        for row in audit
        if row["component"].endswith("on_pitch_ev") and txt(row.get("high_start_spread"))
    ]
    return max(values) if values else 0.0


def nor_irq_sanity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"Oscar Bobb", "Erling Haaland", "Antonio Nusa", "Alexander SÃ¸rloth", "Alexander Sørloth", "Martin Ã˜degaard", "Martin Ødegaard"}
    out = []
    for row in rows:
        if txt(row.get("team_id")) != "NOR" or txt(row.get("match_1_opponent_team")) != "IRQ":
            continue
        if txt(row.get("player_name")) not in wanted:
            continue
        out.append(
            {
                "player_name": txt(row.get("player_name")),
                "start_prob": txt(row.get("start_prob")),
                "match_1_result_ev": txt(row.get("match_1_result_ev")),
                "match_1_team_scores_ev": txt(row.get("match_1_team_scores_ev")),
                "match_1_opponent_scores_ev": txt(row.get("match_1_opponent_scores_ev")),
                "match_1_on_pitch_ev": txt(row.get("match_1_on_pitch_ev")),
            }
        )
    return sorted(out, key=lambda row: to_float(row.get("start_prob")), reverse=True)


def clean_sheet_prob_for_row(
    row: dict[str, Any],
    match_no: int,
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> float:
    context = clean_sheet_lookup.get(match_key(row, match_no))
    if context and context["prob"] >= 0:
        return context["prob"]
    return to_float(row.get(f"match_{match_no}_clean_sheet_prob"), -1.0)


def clean_sheet_zero_high_start_count(
    rows: list[dict[str, Any]],
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> int:
    count = 0
    for row in rows:
        if txt(row.get("position")).upper() not in {"GK", "DEF"} or to_float(row.get("start_prob")) < 0.70:
            continue
        for idx in [1, 2, 3]:
            prob = clean_sheet_prob_for_row(row, idx, clean_sheet_lookup)
            if prob > 0 and abs(to_float(row.get(f"match_{idx}_clean_sheet_ev"))) <= 1e-12:
                count += 1
    return count


def clean_sheet_nan_count(
    rows: list[dict[str, Any]],
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> int:
    count = 0
    for row in rows:
        if txt(row.get("position")).upper() not in {"GK", "DEF"}:
            continue
        for idx in [1, 2, 3]:
            prob = clean_sheet_prob_for_row(row, idx, clean_sheet_lookup)
            value = txt(row.get(f"match_{idx}_clean_sheet_ev")).casefold()
            if prob > 0 and value in {"nan", "na", "null"}:
                count += 1
    return count


def clean_sheet_sanity_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {
        "Alexander Schlager",
        "Patrick Pentz",
        "Stefan Posch",
        "Philipp Lienhart",
        "Kevin Danso",
        "Maximilian WÃ¶ber",
        "Maximilian Wöber",
        "Mike Maignan",
        "Gregor Kobel",
        "Jules Kounde",
        "Jules KoundÃ©",
        "Manuel Neuer",
    }
    before_by_id = {txt(row.get("player_id")): row for row in before_rows}
    out = []
    for row in after_rows:
        if txt(row.get("player_name")) not in names:
            continue
        old = before_by_id.get(txt(row.get("player_id")), {})
        out.append(
            {
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "position": txt(row.get("position")),
                "start_prob": txt(row.get("start_prob")),
                "match_1_opponent": txt(row.get("match_1_opponent_team")),
                "match_1_clean_sheet_prob": txt(row.get("match_1_clean_sheet_prob")),
                "match_1_clean_sheet_ev_before": txt(old.get("match_1_clean_sheet_ev")),
                "match_1_clean_sheet_ev_after": txt(row.get("match_1_clean_sheet_ev")),
                "match_1_weighted_before": txt(old.get("match_1_weighted_match_ev")),
                "match_1_weighted_after": txt(row.get("match_1_weighted_match_ev")),
            }
        )
    return out


def issue_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["suspected_issue"] for row in rows)


def sanity_rows(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id_before = {txt(row.get("player_id")): row for row in before}
    out = []
    for row in after:
        if txt(row.get("player_name")) not in SANITY_NAMES:
            continue
        old = by_id_before.get(txt(row.get("player_id")), {})
        out.append(
            {
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "start_prob_before": txt(old.get("start_prob")),
                "start_prob_after": txt(row.get("start_prob")),
                "minute_share_before": txt(old.get("minute_share")),
                "minute_share_after": txt(row.get("minute_share")),
                "match_1_goal_ev_before": txt(old.get("match_1_goal_ev")),
                "match_1_goal_ev_after": txt(row.get("match_1_goal_ev")),
                "match_1_start_minutes_ev_before": txt(old.get("match_1_start_minutes_ev")),
                "match_1_start_minutes_ev_after": txt(row.get("match_1_start_minutes_ev")),
                "match_1_weighted_match_ev_before": txt(old.get("match_1_weighted_match_ev")),
                "match_1_weighted_match_ev_after": txt(row.get("match_1_weighted_match_ev")),
                "weighted_group_stage_ev_before": txt(old.get("weighted_group_stage_ev")),
                "weighted_group_stage_ev_after": txt(row.get("weighted_group_stage_ev")),
                "issue_after": classify_row(row)[1],
            }
        )
    return out


def md_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> list[str]:
    subset = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in subset:
        lines.append("| " + " | ".join(txt(row.get(field)) for field in fields) + " |")
    return lines


def write_md(
    before_audit: list[dict[str, Any]],
    after_audit: list[dict[str, Any]],
    before_rows: list[dict[str, str]],
    after_rows: list[dict[str, str]],
    changed_ids: set[str],
    backup_path: Path,
    before_team_audit: list[dict[str, Any]],
    after_team_audit: list[dict[str, Any]],
) -> None:
    before_counts = issue_counts(before_audit)
    after_counts = issue_counts(after_audit)
    sanity = sanity_rows(before_rows, after_rows)
    aggregate_missing = [row for row in after_audit if row["suspected_issue"] == "aggregate_ev_but_missing_base_components"]
    no_source = [row for row in after_audit if row["suspected_issue"] == "no_player_ev_source"]

    lines = [
        "# Player EV Component Dependency Audit",
        "",
        "Audit og maalrettet rebuild efter start_prob-repair. Optimizer, strategi-output og frontend er ikke genkoert.",
        "",
        "## Rodarsag",
        "",
        "`start_prob` og `minute_share` blev repareret fra player pool, men eksisterende kampkomponenter blev liggende fra den gamle startbasis. Den gamle basis kan ses direkte som `match_n_start_minutes_ev / match_n_minutes_if_start`. For Haaland var basis ca. 0.163, mens ny dokumenteret `start_prob` er 0.8883.",
        "",
        "## Rebuild-regel",
        "",
        "- Kun spillere med eksisterende per-kamp-komponenter og udledelig gammel startbasis blev genberegnet.",
        "- Startafhaengige komponenter blev skaleret med `ny_start_prob / gammel_komponent_startbasis`: goal, assist, shots_on_target, clean_sheet, card og on_pitch.",
        "- `match_n_start_minutes_ev` blev sat til `start_prob * match_n_minutes_if_start`.",
        "- `match_n_total_ev_next_match` og `match_n_weighted_match_ev` blev genberegnet med eksisterende pointformel fra `build_player_ev_group_stage.py`.",
        "- `weighted_group_stage_ev` og `optimizer_ev` bevarer den eksisterende aggregerings-/price-quality-skala som multiplikator; price/value-laget er ikke kalibreret om.",
        "- Spillere uden basekomponenter fik ikke opfundet maal/assist/SOT-komponenter.",
        "",
        "## Counts",
        "",
        f"- Stale komponenter foer: {before_counts.get('stale_start_dependent_components', 0)}",
        f"- Stale komponenter efter: {after_counts.get('stale_start_dependent_components', 0)}",
        f"- Rækker genberegnet: {len(changed_ids)}",
        f"- Team/match/on_pitch high-start spreads > 0.05 foer: {count_high_start_spreads(before_team_audit)}",
        f"- Team/match/on_pitch high-start spreads > 0.05 efter: {count_high_start_spreads(after_team_audit)}",
        f"- Negative on_pitch_ev for start_prob >= 0.70 foer: {negative_high_start_on_pitch(before_rows)}",
        f"- Negative on_pitch_ev for start_prob >= 0.70 efter: {negative_high_start_on_pitch(after_rows)}",
        f"- Stoerste high-start on_pitch spread foer: {max_high_start_on_pitch_spread(before_team_audit):.6f}",
        f"- Stoerste high-start on_pitch spread efter: {max_high_start_on_pitch_spread(after_team_audit):.6f}",
        f"- Samlet EV men manglende basekomponenter efter: {after_counts.get('aggregate_ev_but_missing_base_components', 0)}",
        f"- Uden EV-kilde efter: {after_counts.get('no_player_ev_source', 0)}",
        f"- Backup: `{backup_path.relative_to(ROOT)}`",
        "",
        "## Sanity-spillere",
        "",
        *md_table(
            sanity,
            [
                "player_name",
                "team_id",
                "start_prob_before",
                "start_prob_after",
                "match_1_goal_ev_before",
                "match_1_goal_ev_after",
                "match_1_start_minutes_ev_before",
                "match_1_start_minutes_ev_after",
                "match_1_weighted_match_ev_before",
                "match_1_weighted_match_ev_after",
                "weighted_group_stage_ev_before",
                "weighted_group_stage_ev_after",
                "issue_after",
            ],
        ),
        "",
        "## NOR vs IRQ efter",
        "",
        *md_table(
            nor_irq_sanity(after_rows),
            [
                "player_name",
                "start_prob",
                "match_1_result_ev",
                "match_1_team_scores_ev",
                "match_1_opponent_scores_ev",
                "match_1_on_pitch_ev",
            ],
        ),
        "",
        "## Saerlige rodarsager",
        "",
        "- Erling Haaland: havde korrekte nye startfelter, men komponenterne var stadig baseret paa gammel `team_minute_rank`-basis ca. 0.163. Genberegnet fra eksisterende komponenter.",
        "- Harry Kane: komponentbasis var gammel `name+team`-basis ca. 0.456. Genberegnet fra eksisterende komponenter.",
        "- Raphinha: har samlet/fordelt runde-EV, men mangler basekomponenter som goal/assist/SOT/start_minutes. Ikke genberegnet, fordi det ville kraeve at opfinde komponentfordeling.",
        "- Jules Kounde: mangler fortsat EV-kilde og komponenter. Ikke genberegnet.",
        "- Manuel Neuer: mangler fortsat EV-kilde og komponenter. Ikke genberegnet.",
        "",
        "## Eksempler paa samlet EV men manglende komponenter",
        "",
        *md_table(aggregate_missing, ["player_name", "team_id", "position", "weighted_group_stage_ev", "suspected_issue"], 12),
        "",
        "## Eksempler paa spillere uden EV-kilde",
        "",
        *md_table(no_source, ["player_name", "team_id", "position", "weighted_group_stage_ev", "suspected_issue"], 12),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_team_audit_md(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    before_audit: list[dict[str, Any]],
    after_audit: list[dict[str, Any]],
) -> None:
    lines = [
        "# Team Match Component Consistency Audit",
        "",
        "High-start betyder `start_prob >= 0.70`. Spreads er beregnet pr. hold/kamp/komponent.",
        "",
        "## Foer/efter",
        "",
        f"- Team/match/on_pitch high-start spreads > 0.05 foer: {count_high_start_spreads(before_audit)}",
        f"- Team/match/on_pitch high-start spreads > 0.05 efter: {count_high_start_spreads(after_audit)}",
        f"- Negative on_pitch_ev for start_prob >= 0.70 foer: {negative_high_start_on_pitch(before_rows)}",
        f"- Negative on_pitch_ev for start_prob >= 0.70 efter: {negative_high_start_on_pitch(after_rows)}",
        f"- Stoerste high-start on_pitch spread foer: {max_high_start_on_pitch_spread(before_audit):.6f}",
        f"- Stoerste high-start on_pitch spread efter: {max_high_start_on_pitch_spread(after_audit):.6f}",
        "",
        "## Stoerste resterende high-start on_pitch spreads",
        "",
        *md_table(
            [row for row in after_audit if row["component"].endswith("on_pitch_ev")],
            [
                "match_no",
                "team_id",
                "opponent",
                "high_start_players",
                "high_start_min",
                "high_start_max",
                "high_start_spread",
                "high_start_negative_count",
            ],
            20,
        ),
        "",
        "## NOR vs IRQ sanity",
        "",
        *md_table(
            nor_irq_sanity(after_rows),
            [
                "player_name",
                "start_prob",
                "match_1_result_ev",
                "match_1_team_scores_ev",
                "match_1_opponent_scores_ev",
                "match_1_on_pitch_ev",
            ],
        ),
    ]
    TEAM_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_team_repair_md(reports: list[dict[str, Any]], backup_path: Path) -> None:
    sanity_names = {
        "Erling Haaland",
        "Antonio Nusa",
        "Alexander SÃ¸rloth",
        "Alexander Sørloth",
        "Martin Ã˜degaard",
        "Martin Ødegaard",
        "Oscar Bobb",
        "Marko Arnautovic",
        "Jonathan David",
        "Kerem Akturkoglu",
        "Kylian Mbappe",
        "Harry Kane",
    }
    sanity = [
        row
        for row in reports
        if txt(row.get("player_name")) in sanity_names and to_float(row.get("match_no")) == 1
    ]
    lines = [
        "# Team Component Repair Report",
        "",
        f"- Repair rows: {len(reports)}",
        f"- Backup: `{backup_path.relative_to(ROOT)}`",
        "",
        "## Sanity-spillere match 1",
        "",
        *md_table(
            sanity,
            [
                "player_name",
                "team_id",
                "opponent",
                "start_prob",
                "appearance_prob_used",
                "old_on_pitch_ev",
                "new_on_pitch_ev",
                "old_result_ev",
                "new_result_ev",
                "old_team_scores_ev",
                "new_team_scores_ev",
                "old_opponent_scores_ev",
                "new_opponent_scores_ev",
                "old_weighted_match_ev",
                "new_weighted_match_ev",
            ],
        ),
    ]
    TEAM_REPAIR_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_sheet_audit_md(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    clean_sheet_lookup: dict[tuple[str, str, str], dict[str, float]],
    backup_path: Path,
) -> None:
    changed = [row for row in reports if row["repair_status"] == "rebuilt_from_fixture_clean_sheet_prob"]
    lines = [
        "# Clean Sheet Component Repair Audit",
        "",
        "Clean sheet-komponenten er genberegnet for GK/DEF fra `fixture_strength_multipliers.csv` clean sheet-probability og en 60-minutters eligibility-proxy: `start_prob * min(match_n_minutes_if_start / 60, 1)`.",
        "",
        "MID/FWD holdes paa 0, fordi den eksplicitte Holdet clean sheet-regel her kun bruges for GK/DEF.",
        "",
        "## Foer/efter",
        "",
        f"- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 foer: {clean_sheet_zero_high_start_count(before_rows, clean_sheet_lookup)}",
        f"- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 efter: {clean_sheet_zero_high_start_count(after_rows, clean_sheet_lookup)}",
        f"- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob foer: {clean_sheet_nan_count(before_rows, clean_sheet_lookup)}",
        f"- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob efter: {clean_sheet_nan_count(after_rows, clean_sheet_lookup)}",
        f"- Clean sheet repair rows: {len(changed)}",
        f"- Backup: `{backup_path.relative_to(ROOT)}`",
        "",
        "## Sanity-spillere",
        "",
        *md_table(
            clean_sheet_sanity_rows(before_rows, after_rows),
            [
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "match_1_opponent",
                "match_1_clean_sheet_prob",
                "match_1_clean_sheet_ev_before",
                "match_1_clean_sheet_ev_after",
                "match_1_weighted_before",
                "match_1_weighted_after",
            ],
        ),
        "",
        "## Schlager vs Pentz vs Posch vs Lienhart",
        "",
        *md_table(
            [
                row
                for row in clean_sheet_sanity_rows(before_rows, after_rows)
                if row["player_name"] in {"Alexander Schlager", "Patrick Pentz", "Stefan Posch", "Philipp Lienhart"}
            ],
            [
                "player_name",
                "position",
                "start_prob",
                "match_1_clean_sheet_prob",
                "match_1_clean_sheet_ev_before",
                "match_1_clean_sheet_ev_after",
            ],
        ),
        "",
        "## Maignan vs Schlager",
        "",
        *md_table(
            [
                row
                for row in clean_sheet_sanity_rows(before_rows, after_rows)
                if row["player_name"] in {"Mike Maignan", "Alexander Schlager"}
            ],
            [
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "match_1_opponent",
                "match_1_clean_sheet_prob",
                "match_1_clean_sheet_ev_before",
                "match_1_clean_sheet_ev_after",
            ],
        ),
    ]
    CLEAN_SHEET_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    fields, rows = read_csv(EV_PATH)
    before_rows = [dict(row) for row in rows]
    before_audit = [audit_row(row, "no") for row in before_rows]
    before_team_audit = team_component_audit(before_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_component_rebuild_{timestamp}.csv")
    shutil.copy2(EV_PATH, backup_path)

    baselines = build_team_baselines(before_rows)
    clean_sheet_lookup = load_clean_sheet_lookup()
    changed_ids: set[str] = set()
    repair_reports: list[dict[str, Any]] = []
    clean_sheet_reports: list[dict[str, Any]] = []
    for row in rows:
        changed, reports, cs_reports = repair_row(row, baselines, clean_sheet_lookup)
        if changed:
            changed_ids.add(txt(row.get("player_id")))
            repair_reports.extend(reports)
        clean_sheet_reports.extend(cs_reports)

    rows.sort(key=lambda row: (to_float(row.get("weighted_group_stage_ev")), to_float(row.get("total_ev_group_stage"))), reverse=True)
    write_csv(EV_PATH, fields, rows)

    after_audit = [audit_row(row, "yes" if txt(row.get("player_id")) in changed_ids else "no") for row in rows]
    after_team_audit = team_component_audit(rows)
    write_csv(OUT_CSV, AUDIT_FIELDS, after_audit)
    write_csv(TEAM_AUDIT_CSV, TEAM_AUDIT_FIELDS, after_team_audit)
    write_csv(TEAM_REPAIR_CSV, TEAM_REPAIR_FIELDS, repair_reports)
    write_csv(CLEAN_SHEET_AUDIT_CSV, CLEAN_SHEET_AUDIT_FIELDS, clean_sheet_reports)
    write_md(before_audit, after_audit, before_rows, rows, changed_ids, backup_path, before_team_audit, after_team_audit)
    write_team_audit_md(before_rows, rows, before_team_audit, after_team_audit)
    write_team_repair_md(repair_reports, backup_path)
    write_clean_sheet_audit_md(before_rows, rows, clean_sheet_reports, clean_sheet_lookup, backup_path)

    before_counts = issue_counts(before_audit)
    after_counts = issue_counts(after_audit)
    print("EV component dependency repair")
    print("------------------------------")
    print(f"Stale components before: {before_counts.get('stale_start_dependent_components', 0)}")
    print(f"Stale components after: {after_counts.get('stale_start_dependent_components', 0)}")
    print(f"Rows recomputed: {len(changed_ids)}")
    print(f"Team repair rows: {len(repair_reports)}")
    print(f"Clean sheet audit rows: {len(clean_sheet_reports)}")
    print(f"Clean sheet high-start zero before: {clean_sheet_zero_high_start_count(before_rows, clean_sheet_lookup)}")
    print(f"Clean sheet high-start zero after: {clean_sheet_zero_high_start_count(rows, clean_sheet_lookup)}")
    print(f"Clean sheet NaN before: {clean_sheet_nan_count(before_rows, clean_sheet_lookup)}")
    print(f"Clean sheet NaN after: {clean_sheet_nan_count(rows, clean_sheet_lookup)}")
    print(f"On-pitch high-start spreads > 0.05 before: {count_high_start_spreads(before_team_audit)}")
    print(f"On-pitch high-start spreads > 0.05 after: {count_high_start_spreads(after_team_audit)}")
    print(f"Negative on_pitch_ev for start_prob >= 0.70 before: {negative_high_start_on_pitch(before_rows)}")
    print(f"Negative on_pitch_ev for start_prob >= 0.70 after: {negative_high_start_on_pitch(rows)}")
    print(f"Max high-start on_pitch spread before: {max_high_start_on_pitch_spread(before_team_audit):.6f}")
    print(f"Max high-start on_pitch spread after: {max_high_start_on_pitch_spread(after_team_audit):.6f}")
    print(f"Aggregate EV missing components after: {after_counts.get('aggregate_ev_but_missing_base_components', 0)}")
    print(f"No EV source after: {after_counts.get('no_player_ev_source', 0)}")
    print(f"Backup: {backup_path.relative_to(ROOT)}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote: {TEAM_AUDIT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {TEAM_AUDIT_MD.relative_to(ROOT)}")
    print(f"Wrote: {TEAM_REPAIR_CSV.relative_to(ROOT)}")
    print(f"Wrote: {TEAM_REPAIR_MD.relative_to(ROOT)}")
    print(f"Wrote: {CLEAN_SHEET_AUDIT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {CLEAN_SHEET_AUDIT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
