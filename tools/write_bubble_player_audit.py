from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from diff_holdet_new_players import TEAM_ALIASES as HOLDET_TEAM_ALIASES
import optimize_squad_group_stage as optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
START_SECURITY_PATH = DATA_DIR / "player_start_security_nt.csv"
OPTIMAL_SQUADS_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
OUT_CSV = DATA_DIR / "bubble_player_audit.csv"
OUT_MD = DATA_DIR / "bubble_player_audit.md"
REMAINING_FLAGS_CSV = DATA_DIR / "bubble_player_remaining_flags_report.csv"
REMAINING_FLAGS_MD = DATA_DIR / "bubble_player_remaining_flags_report.md"

STRATEGIES = ["next_round", "round1_2", "group_stage", "long_run"]
CSV_COLUMNS = [
    "player_name",
    "matched_player_id",
    "team_id",
    "position",
    "price",
    "specific_round",
    "start_prob",
    "conditional_start_prob",
    "appearance_prob",
    "availability_risk",
    "manual_status",
    "next_round_opponent",
    "next_round_win_prob",
    "next_round_score",
    "round1_2_score",
    "group_stage_score",
    "long_run_score",
    "total_ev",
    "round_1_ev",
    "round_2_ev",
    "round_3_ev",
    "selected_in_strategies",
    "model_error_flag",
    "model_error_reason",
    "recommended_manual_action",
    "note",
]


@dataclass(frozen=True)
class BubblePlayer:
    name: str
    team_id: str = ""
    position: str = ""
    price: int | None = None
    specific_round: str = ""
    note: str = ""
    sanity_control: bool = False


BUBBLE_PLAYERS = [
    BubblePlayer("Alexander Schlager", "AUT", "GK", 3500000),
    BubblePlayer("Phillipp Mwene", "AUT", "DEF", 2500000),
    BubblePlayer("Cristian Romero", "ARG", "DEF", 4000000),
    BubblePlayer("Jules Kounde", "FRA", "DEF", 3500000),
    BubblePlayer("Julian Ryerson", "NOR", "DEF", 3500000, "Runde 1"),
    BubblePlayer("Christoph Baumgartner", "AUT", "MID", 3500000, "", "Ude af VM - ikke vaelg"),
    BubblePlayer("Antonio Nusa", "NOR", "MID", 3500000, "Runde 1"),
    BubblePlayer("Konrad Laimer", "AUT", "MID", 3000000, "Runde 1"),
    BubblePlayer("Kylian Mbappe", "FRA", "FWD", 10000000),
    BubblePlayer("Alexander Sorloth", "NOR", "FWD", 4500000, "Runde 1"),
    BubblePlayer("Erling Haaland", "NOR", "FWD", 8500000, "Runde 1"),
    BubblePlayer("Theo Hernandez", "FRA", "DEF", 4500000),
    BubblePlayer("Nicolas Tagliafico", "ARG", "DEF"),
    BubblePlayer("Nahuel Molina", "ARG", "DEF"),
    BubblePlayer("Ibrahim Maza", "ALG", "", None, "Runde 2"),
    BubblePlayer("Mohamed Amoura", "ALG", "", None, "Runde 2"),
    BubblePlayer("Cesar Montes", "MEX", "DEF", None, "Runde 1"),
    BubblePlayer("Ladislav Krejci", "CZE"),
    BubblePlayer("Vladimir Coufal", "CZE", "DEF"),
    BubblePlayer("Scott McTominay", "SCO", "MID", None, "Runde 1"),
    BubblePlayer("Brian Gutierrez"),
    BubblePlayer("Enner Valencia", "ECU", "FWD", 4000000, "Runde 2"),
    BubblePlayer("Arthur Theate", "BEL", "DEF", 3000000),
    BubblePlayer("Maxim De Cuyper", "BEL", "DEF", 4000000),
    BubblePlayer("Mahmoud Trezeguet", "EGY", "Offensiv", 3000000, "Runde 2"),
    BubblePlayer("Kevin De Bruyne", "BEL", "MID", 7000000),
    BubblePlayer("Mehdi Taremi", "IRN", "FWD", 3500000, "Runde 1"),
    BubblePlayer("Jurrien Timber", "NED", "DEF", 2500000, "Runde 2-3"),
    BubblePlayer("Nico Schlotterbeck", "GER", "DEF", 4000000, "Runde 1"),
    BubblePlayer("Wesley Franca", "BRA", "DEF", 3500000, "Runde 2-3"),
    BubblePlayer("Ismael Saibari", "MAR", "Offensiv", 3500000, "Runde 3"),
    BubblePlayer("Raphinha", "BRA", "Offensiv", 6500000, "Runde 2-3"),
    BubblePlayer("Gregor Kobel", "SUI", "GK", 4000000, "Runde 1-3"),
    BubblePlayer("Silvan Widmer", "SUI", "DEF", 3000000, "Runde 1-3"),
    BubblePlayer("Ismaila Sarr", "SEN", "Offensiv", 3500000, "Runde 3"),
    BubblePlayer("David Raum", "GER", "DEF", 4500000, "Runde 1"),
    BubblePlayer("Florian Wirtz", "GER", "MID", 7500000, "Runde 1"),
    BubblePlayer("Fabian Rieder", "SUI", "MID", 3000000),
    BubblePlayer("Aleksandar Pavlovic", "GER", "MID", 3500000, "Runde 1"),
    BubblePlayer("Patrick Wimmer", "AUT", "MID", 3000000),
    BubblePlayer("Harry Kane", "ENG", "FWD", 9500000, "Runde 1"),
    BubblePlayer("Luis Diaz", "COL", "FWD", 5500000),
    BubblePlayer("Deniz Undav", "GER", "FWD", 3500000, "Runde 1"),
    BubblePlayer("Michael Olise", "FRA", "Offensiv", 7000000),
    BubblePlayer("Jamal Musiala", "GER", "MID", 6500000, "Runde 1"),
    BubblePlayer("Manuel Neuer", "GER", "GK", 5000000, "Runde 1"),
    BubblePlayer("Mike Maignan", "FRA", "GK", 5000000, "", "Sanity control: must not sit on unexplained fallback 0.48.", True),
    BubblePlayer("Andreas Schjelderup", "NOR", "MID", 3500000, "", "Sanity control: start probability must not be driven by sub appearances.", True),
    BubblePlayer("Manu Kone", "FRA", "MID", 3500000, "", "Sanity control: latest injury/absence must be reflected.", True),
]

PREMIUM_OFFENSIVE = {
    "kylian mbappe",
    "erling haaland",
    "harry kane",
    "florian wirtz",
    "jamal musiala",
    "kevin de bruyne",
    "michael olise",
    "raphinha",
    "luis diaz",
}
UNCERTAIN_START_NAMES = {
    "deniz undav",
    "jurrien timber",
    "ismael saibari",
    "ismaila sarr",
    "patrick wimmer",
    "andreas schjelderup",
    "antonio nusa",
    "raphinha",
}
CENTRAL_LOW_UPSIDE_MIDS = {
    "konrad laimer",
    "scott mctominay",
    "aleksandar pavlovic",
    "fabian rieder",
    "manu kone",
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        parsed = float(raw)
        return parsed if isfinite(parsed) else default
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(to_float(value, float(default))))
    except ValueError:
        return default


def norm(value: Any) -> str:
    text = txt(value)
    text = (
        text.replace("Æ", "Ae")
        .replace("æ", "ae")
        .replace("Ø", "O")
        .replace("ø", "o")
        .replace("Å", "A")
        .replace("å", "a")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def canonical_team(value: Any) -> str:
    team = txt(value).upper()
    return HOLDET_TEAM_ALIASES.get(team, team)


def round_value(value: Any, digits: int = 4) -> str:
    if value is None or txt(value) == "":
        return ""
    return str(round(to_float(value), digits))


def load_start_security() -> dict[str, list[dict[str, str]]]:
    if not START_SECURITY_PATH.exists():
        return {}
    with START_SECURITY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_name.setdefault(norm(row.get("player_name")), []).append(row)
    return by_name


def load_player_pool_by_id() -> dict[str, dict[str, Any]]:
    if not PLAYER_POOL_PATH.exists():
        return {}
    with PLAYER_POOL_PATH.open("r", encoding="utf-8-sig") as f:
        rows = json.load(f)
    return {txt(row.get("player_id")): row for row in rows}


def add_audit_only_player_pool_fields(players: pd.DataFrame) -> None:
    pool_by_id = load_player_pool_by_id()
    audit_fields = [
        "appearance_prob",
        "availability_prob",
        "availability_risk",
        "round_specific_rotation_risk",
        "start_signal_context_note",
    ]
    for field in audit_fields:
        if field in players.columns:
            players[field] = players[field].astype("object")
    for idx, player in players.iterrows():
        pool_row = pool_by_id.get(txt(player.get("player_id")))
        if not pool_row:
            continue
        for field in audit_fields:
            if pool_row.get(field) not in (None, ""):
                if field not in players.columns:
                    players[field] = ""
                players.at[idx, field] = txt(pool_row.get(field))


def selected_strategy_map() -> dict[str, set[str]]:
    if not OPTIMAL_SQUADS_PATH.exists():
        return {}
    data = json.loads(OPTIMAL_SQUADS_PATH.read_text(encoding="utf-8-sig"))
    selected: dict[str, set[str]] = {}
    for strategy, strategy_data in data.items():
        formation_data = strategy_data.get("squads_by_formation") or {}
        if formation_data:
            for formation, payload in formation_data.items():
                for player in payload.get("squad") or payload.get("players") or []:
                    selected.setdefault(txt(player.get("player_id")), set()).add(f"{strategy}:{formation}")
        for player in strategy_data.get("best_squad") or []:
            selected.setdefault(txt(player.get("player_id")), set()).add(f"{strategy}:best")
    return selected


def match_player(players: pd.DataFrame, target: BubblePlayer) -> pd.Series | None:
    name_key = norm(target.name)
    expected_team = target.team_id.upper()
    matches = players[players["player_name"].map(norm).eq(name_key)]
    if expected_team:
        team_matches = matches[matches["team_id"].map(canonical_team).eq(expected_team)]
        if not team_matches.empty:
            return team_matches.iloc[0]
    if not matches.empty:
        return matches.iloc[0]

    name_tokens = set(name_key.split())
    if not name_tokens:
        return None
    candidates: list[tuple[float, int]] = []
    for idx, row in players.iterrows():
        player_tokens = set(norm(row.get("player_name")).split())
        if not player_tokens:
            continue
        overlap = len(name_tokens & player_tokens) / len(name_tokens | player_tokens)
        if expected_team and canonical_team(row.get("team_id")) == expected_team:
            overlap += 0.25
        if overlap >= 0.55:
            candidates.append((overlap, int(idx)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return players.loc[candidates[0][1]]


def best_usage_row(start_rows: dict[str, list[dict[str, str]]], player_name: str, team_id: str) -> dict[str, str] | None:
    rows = start_rows.get(norm(player_name), [])
    if not rows:
        return None
    team = team_id.upper()
    code_matches = [r for r in rows if txt(r.get("team_id")).upper() == team]
    if code_matches:
        return code_matches[0]
    return max(rows, key=lambda r: to_float(r.get("start_probability_pct")))


def flag_row(target: BubblePlayer, row: pd.Series | None, selected: dict[str, set[str]], usage: dict[str, str] | None) -> dict[str, str]:
    reasons: list[str] = []
    notes: list[str] = []
    actions: list[str] = []
    labels: list[str] = []
    severity = 0

    if row is None:
        reasons.append("missing_player_match")
        actions.append("Review name/team/position match before any model decision.")
        return {
            "player_name": target.name,
            "matched_player_id": "",
            "team_id": target.team_id,
            "position": target.position,
            "price": "" if target.price is None else str(target.price),
            "specific_round": target.specific_round,
            "start_prob": "",
            "conditional_start_prob": "",
            "appearance_prob": "",
            "availability_risk": "",
            "manual_status": "",
            "next_round_opponent": "",
            "next_round_win_prob": "",
            "next_round_score": "",
            "round1_2_score": "",
            "group_stage_score": "",
            "long_run_score": "",
            "total_ev": "",
            "round_1_ev": "",
            "round_2_ev": "",
            "round_3_ev": "",
            "selected_in_strategies": "",
            "model_error_flag": "yes",
            "model_error_reason": "missing_player_match",
            "recommended_manual_action": "Review missing player/name/team mapping.",
            "note": target.note,
            "_severity": "10",
            "_labels": "missing_data",
        }

    player_name = txt(row.get("player_name"))
    player_key = norm(player_name)
    raw_team_id = txt(row.get("team_id")).upper()
    team_id = canonical_team(raw_team_id)
    position = txt(row.get("position")).upper()
    price = to_int(row.get("price"))
    start_prob = to_float(row.get("start_prob"))
    conditional = to_float(row.get("conditional_start_prob"))
    appearance_source = "appearance_prob" if txt(row.get("appearance_prob")) else "availability_prob_proxy"
    appearance = to_float(row.get("appearance_prob") or row.get("availability_prob"))
    risk = txt(row.get("availability_risk")) or "unknown"
    manual_status = txt(row.get("manual_status"))
    manual_start_status = txt(row.get("manual_start_status"))
    manual_note = txt(row.get("manual_note"))
    total_ev_raw = to_float(row.get("total_ev_group_stage"))
    weighted_ev = to_float(row.get("weighted_group_stage_ev"))
    total_ev = total_ev_raw if total_ev_raw > 0 else weighted_ev
    round_evs = [to_float(row.get(f"round{rnd}_ev")) for rnd in [1, 2, 3]]
    next_score = to_float(row.get("score_next_round"))
    round12_score = to_float(row.get("score_round1_2"))
    group_score = to_float(row.get("score_group_stage"))
    long_score = to_float(row.get("score_long_run"))

    if raw_team_id != team_id:
        notes.append(f"canonical_team_alias_applied={raw_team_id}->{team_id}")
    if target.team_id and team_id != target.team_id.upper():
        reasons.append(f"team_id_mismatch_input={target.team_id}_matched={team_id}")
        actions.append("Review Holdet/team mapping.")
        labels.append("missing_or_bad_mapping")
        severity = max(severity, 9)
    if target.position and target.position.upper() not in {"OFFENSIV", position}:
        if target.position.upper() == "OFFENSIV" and position in {"MID", "FWD"}:
            pass
        else:
            reasons.append(f"position_mismatch_input={target.position}_matched={position}")
            actions.append("Review position mapping.")
            labels.append("missing_or_bad_mapping")
            severity = max(severity, 8)
    if target.price and price and abs(price - target.price) >= 500000:
        reasons.append(f"price_mismatch_input={target.price}_matched={price}")
        actions.append("Review price source.")
        labels.append("missing_or_bad_mapping")
        severity = max(severity, 6)

    if manual_status.lower() == "avoid" or manual_start_status.lower() == "avoid":
        if player_key == "christoph baumgartner":
            notes.append("Control OK: Baumgartner is marked avoid.")
            actions.append("Keep avoid override.")
        else:
            reasons.append("manual_avoid_player")
            actions.append("Do not select unless override is removed.")
            labels.append("manual_check")
            severity = max(severity, 6)

    usage_start = to_float(usage.get("start_probability_pct"), -1) / 100 if usage else -1
    usage_starts = to_float(usage.get("starts_def_used"), -1) if usage else -1
    if usage_start >= 0.90 and start_prob < 0.60:
        reasons.append(f"usage_start_high_but_model_start_low_usage={usage_start:.2f}")
        actions.append("Check start-probability merge/fallback layer.")
        labels.append("start_model_error")
        severity = max(severity, 9)
    if conditional >= 0.82 and 0 <= usage_starts <= 2:
        reasons.append(f"high_conditional_with_few_starts_starts={usage_starts:g}")
        actions.append("Check whether sub appearances are inflating start probability.")
        labels.append("start_model_error")
        severity = max(severity, 8)
    if start_prob <= 0.30 and conditional >= 0.70:
        reasons.append("appearance_or_availability_may_be_mixed_with_start_prob")
        actions.append("Split appearance/availability from true starting probability.")
        labels.append("start_model_error")
        severity = max(severity, 7)

    if player_key in UNCERTAIN_START_NAMES and conditional < 0.80:
        reasons.append("requested_uncertain_start_check")
        actions.append("Manual lineup/start check before selection.")
        labels.append("manual_start_check")
        severity = max(severity, 6)
    if player_key == "manu kone" and risk != "high_risk":
        reasons.append("injury_absence_sanity_check_not_high_risk")
        actions.append("Review latest Manu Kone injury/absence data.")
        labels.append("availability_check")
        severity = max(severity, 8)

    max_round_ev = max(round_evs)
    optimizer_ev = to_float(row.get("optimizer_ev"))
    if total_ev <= 0.05 and optimizer_ev <= 0.05 and max_round_ev <= 0.05:
        reasons.append("ev_and_fixture_values_missing_in_model_data")
        actions.append("Review player EV source and fixture mapping before strategy use.")
        labels.append("missing_ev")
        severity = max(severity, 9)
    elif max_round_ev <= 0.05 and optimizer_ev > 0.05:
        if any(txt(row.get(f"round{rnd}_opponent")) for rnd in [1, 2, 3]) or any(txt(row.get(f"match_{rnd}_opponent_team")) for rnd in [1, 2, 3]):
            reasons.append("round_ev_much_lower_than_optimizer_ev")
            actions.append("Review whether aggregate EV should be distributed to round EV for this player.")
            labels.append("round_value_mismatch")
            severity = max(severity, 7)
        else:
            reasons.append("round_fixture_context_missing_but_optimizer_ev_present")
            actions.append("Document as model-data gap; do not infer round-specific upside from generic optimizer_ev.")
            labels.append("missing_ev")
            severity = max(severity, 8)
    elif max_round_ev <= 0.05 and target.specific_round:
        reasons.append("specific_round_value_not_supported_by_round_ev")
        actions.append("Review whether user round note is external/manual rather than model round EV.")
        labels.append("round_value_mismatch")
        severity = max(severity, 6)

    if player_key in PREMIUM_OFFENSIVE:
        if next_score < 6.0 and to_float(row.get("round1_win_prob")) >= 0.60:
            reasons.append("premium_offensive_low_next_round_score_in_good_fixture")
            actions.append("Review premium attacker goal/upside weighting.")
            labels.append("undervalued")
            severity = max(severity, 7)
        elif next_score >= 7.0:
            notes.append("Premium/offensive player has plausible strong next-round score.")

    if player_key in CENTRAL_LOW_UPSIDE_MIDS and max(next_score, group_score) >= 8.0:
        reasons.append("central_or_defensive_mid_may_be_overvalued")
        actions.append("Review role/upside weighting for central MID.")
        labels.append("overvalued")
        severity = max(severity, 6)

    if price <= 3500000 and max(next_score, round12_score, group_score) >= 9.0 and total_ev < 3.2:
        reasons.append("high_score_may_be_value_price_driven")
        actions.append("Check whether cheap price/value proxy is too strong.")
        labels.append("overvalued")
        severity = max(severity, 6)

    if "1" in target.specific_round and next_score < 5.5:
        reasons.append("specific_round_1_but_low_next_round_score")
        actions.append("Check round 1 fixture/upside assumptions.")
        labels.append("round_value_mismatch")
        severity = max(severity, 6)
    if "2" in target.specific_round and round_evs[1] < 1.0:
        reasons.append("specific_round_2_but_low_round_2_ev")
        actions.append("Check round 2 fixture/upside assumptions.")
        labels.append("round_value_mismatch")
        severity = max(severity, 6)
    if "3" in target.specific_round and round_evs[2] < 0.8:
        reasons.append("specific_round_3_but_low_round_3_ev")
        actions.append("Check round 3 fixture/upside/rotation assumptions.")
        labels.append("round_value_mismatch")
        severity = max(severity, 6)

    if (
        appearance_source == "availability_prob_proxy"
        and appearance
        and start_prob
        and abs(appearance - start_prob) < 0.015
        and risk != "low_risk"
    ):
        notes.append("appearance_prob column uses availability_prob proxy; no separate appearance_prob source found.")

    if target.sanity_control:
        notes.append("Sanity-control row requested by user.")
    if target.note:
        notes.append(target.note)
    if manual_note:
        notes.append(f"manual_note={manual_note}")

    selected_str = "; ".join(sorted(selected.get(txt(row.get("player_id")), set())))
    if not selected_str:
        notes.append("Not selected in exported strategy/formation squads.")

    if not actions:
        actions.append("No immediate manual action from audit rules.")

    return {
        "player_name": player_name,
        "matched_player_id": txt(row.get("player_id")),
        "team_id": team_id,
        "position": position,
        "price": str(price),
        "specific_round": target.specific_round,
        "start_prob": round_value(start_prob),
        "conditional_start_prob": round_value(conditional),
        "appearance_prob": round_value(appearance),
        "availability_risk": risk,
        "manual_status": manual_status,
        "next_round_opponent": txt(row.get("round1_opponent")),
        "next_round_win_prob": round_value(row.get("round1_win_prob")),
        "next_round_score": round_value(next_score),
        "round1_2_score": round_value(round12_score),
        "group_stage_score": round_value(group_score),
        "long_run_score": round_value(long_score),
        "total_ev": round_value(total_ev),
        "round_1_ev": round_value(round_evs[0]),
        "round_2_ev": round_value(round_evs[1]),
        "round_3_ev": round_value(round_evs[2]),
        "selected_in_strategies": selected_str,
        "model_error_flag": "yes" if reasons else "no",
        "model_error_reason": "; ".join(reasons),
        "recommended_manual_action": " ".join(dict.fromkeys(actions)),
        "note": " ".join(notes),
        "_severity": str(severity),
        "_labels": ";".join(sorted(set(labels))),
        "_optimizer_ev": round_value(optimizer_ev),
    }


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def md_table(rows: list[dict[str, str]], columns: list[str], limit: int = 10) -> list[str]:
    if not rows:
        return ["Ingen."]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(txt(row.get(col)).replace("|", "/") for col in columns) + " |")
    return out


def write_md(rows: list[dict[str, str]]) -> None:
    flagged = [r for r in rows if r["model_error_flag"] == "yes"]
    top_errors = sorted(flagged, key=lambda r: int(r.get("_severity", "0")), reverse=True)
    undervalued = [r for r in rows if "undervalued" in r.get("_labels", "")]
    overvalued = [r for r in rows if "overvalued" in r.get("_labels", "")]
    manual_start = [r for r in rows if any(label in r.get("_labels", "") for label in ["manual_start_check", "start_model_error", "availability_check"])]
    missing = [r for r in rows if any(label in r.get("_labels", "") for label in ["missing_data", "missing_ev", "missing_or_bad_mapping"])]

    lines = [
        "# Bubble Player Audit",
        "",
        "Ren audit baseret paa eksisterende data. Audit-scriptet aendrer ikke optimizer, EV eller spillerpool.",
        "",
        "## Summary",
        "",
        f"- Audit rows: {len(rows)}",
        f"- Matched players: {sum(1 for r in rows if r['matched_player_id'])}",
        f"- Model error flags: {len(flagged)}",
        "- appearance_prob uses the explicit player_pool column when available; otherwise it falls back to availability_prob as a proxy.",
        "",
        "## 1. Biggest likely model errors",
        "",
        *md_table(top_errors, ["player_name", "team_id", "position", "model_error_reason", "recommended_manual_action"], 10),
        "",
        "## 2. Players that look undervalued",
        "",
        *md_table(undervalued, ["player_name", "team_id", "next_round_score", "round_1_ev", "model_error_reason"], 10),
        "",
        "## 3. Players that look overvalued",
        "",
        *md_table(overvalued, ["player_name", "team_id", "price", "next_round_score", "model_error_reason"], 10),
        "",
        "## 4. Manual start/availability checks",
        "",
        *md_table(manual_start, ["player_name", "team_id", "start_prob", "conditional_start_prob", "availability_risk", "model_error_reason"], 15),
        "",
        "## 5. Missing player/position/price/EV data",
        "",
        *md_table(missing, ["player_name", "team_id", "position", "price", "model_error_reason"], 15),
        "",
        "## 6. Recommended model tracks to fix first",
        "",
        "1. Fix player/team/fixture mappings that produce zero EV or placeholder teams.",
        "2. Split true start probability, appearance probability, and availability probability explicitly.",
        "3. Review premium attacker next-round scoring against low-upside MID/DEF upgrades.",
        "4. Review central MID role/upside weighting so safe starters do not dominate purely on security/value.",
        "5. Keep manual avoid controls active; Baumgartner is a control row and remains avoid.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


REMAINING_FLAG_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "position",
    "price",
    "model_error_flag",
    "primary_flag_category",
    "start_prob",
    "conditional_start_prob",
    "appearance_prob",
    "availability_risk",
    "optimizer_ev",
    "round_1_ev",
    "round_2_ev",
    "round_3_ev",
    "suspected_issue",
    "recommended_next_action",
]


def classify_remaining_flag(row: dict[str, str]) -> tuple[str, str, str]:
    reason = txt(row.get("model_error_reason"))
    labels = txt(row.get("_labels"))
    player_key = norm(row.get("player_name"))

    if "ev_and_fixture_values_missing_in_model_data" in reason:
        return (
            "missing_ev_source",
            "Reel datamangel: spilleren har ingen brugbar EV-/fixturevaerdi i modeloutputtet.",
            "Ret EV-/fixturekilden foer spilleren bruges i strategi eller replacement-vurdering.",
        )
    if "round_fixture_context_missing" in reason or "round_ev_much_lower" in reason or "specific_round_" in reason:
        return (
            "missing_or_weak_round_context",
            "Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala.",
            "Audit round EV og fixturefordeling foer eventuel modelkalibrering.",
        )
    if "team_id_mismatch" in reason or "position_mismatch" in reason or "price_mismatch" in reason:
        return (
            "price_or_position_review",
            "Mulig data-/mappingreview: pris, position eller holdmatch boer verificeres.",
            "Ret mapping eller prisgrundlag hvis den autoritative Holdet-kilde afviger.",
        )
    if "missing_player_match" in reason:
        return (
            "data_issue_other",
            "Reel datamangel: spilleren kunne ikke matches stabilt.",
            "Tilfoej alias/mapping foer modelbrug.",
        )
    if "injury" in reason or "availability_check" in labels or txt(row.get("availability_risk")) == "high_risk":
        return (
            "availability_risk",
            "Reel spillerusikkerhed: availability/injury-risk er selve problemet, ikke primaert scoremodellen.",
            "Afvent lineup/skadenyt eller hold spilleren paa manuel tjekliste.",
        )
    if "requested_uncertain_start_check" in reason or "start_model_error" in labels:
        return (
            "uncertain_start",
            "Reel spillerusikkerhed: startchance er ikke sikker nok til at behandle spilleren som fast starter.",
            "Lav manuel starttjek; aendr ikke modelvaegte alene pga. denne spiller.",
        )
    if "premium_offensive_low_next_round_score" in reason or player_key in {"erling haaland", "harry kane", "luis diaz", "michael olise", "jamal musiala"}:
        return (
            "likely_underweighted_by_model",
            "Mulig modelvaegtning: premium/offensiv upside kan vaere for lavt vaegtet ift. maal, flere maal og captain-ceiling.",
            "Brug positional budget-auditten foer eventuel senere modelaendring.",
        )
    if "central_or_defensive_mid_may_be_overvalued" in reason or "high_score_may_be_value_price_driven" in reason:
        return (
            "likely_overweighted_by_model",
            "Mulig modelvaegtning: sikkerhed/value eller central MID/DEF-score kan fylde for meget.",
            "Sammenlign marginal budgetbrug mod premium FWD-upside foer kalibrering.",
        )
    if "manual_avoid_player" in reason:
        return (
            "plausible_but_needs_manual_review",
            "Spilleren er plausibel som kontrolflag, men manuel avoid/status er den afgoerende faktor.",
            "Bevar manuel override indtil bruger bevidst fjerner den.",
        )
    return (
        "plausible_but_needs_manual_review",
        "Flagget ser ikke ud som en ren datafejl; det boer bruges som manuel review-markoer.",
        "Ingen modelaendring uden separat kalibreringsaudit.",
    )


def write_remaining_flags_report(rows: list[dict[str, str]]) -> None:
    flagged = [row for row in rows if row.get("model_error_flag") == "yes"]
    out_rows: list[dict[str, str]] = []
    for row in flagged:
        category, issue, action = classify_remaining_flag(row)
        out_rows.append(
            {
                "player_id": txt(row.get("matched_player_id")),
                "player_name": txt(row.get("player_name")),
                "team": txt(row.get("team_id")),
                "position": txt(row.get("position")),
                "price": txt(row.get("price")),
                "model_error_flag": txt(row.get("model_error_flag")),
                "primary_flag_category": category,
                "start_prob": txt(row.get("start_prob")),
                "conditional_start_prob": txt(row.get("conditional_start_prob")),
                "appearance_prob": txt(row.get("appearance_prob")),
                "availability_risk": txt(row.get("availability_risk")),
                "optimizer_ev": txt(row.get("_optimizer_ev") or row.get("total_ev")),
                "round_1_ev": txt(row.get("round_1_ev")),
                "round_2_ev": txt(row.get("round_2_ev")),
                "round_3_ev": txt(row.get("round_3_ev")),
                "suspected_issue": issue,
                "recommended_next_action": action,
            }
        )

    with REMAINING_FLAGS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REMAINING_FLAG_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["primary_flag_category"]] = counts.get(row["primary_flag_category"], 0) + 1

    focus_names = {
        "erling haaland",
        "luis diaz",
        "michael olise",
        "jamal musiala",
        "konrad laimer",
        "scott mctominay",
        "harry kane",
        "jules kounde",
        "manuel neuer",
        "jurrien timber",
        "deniz undav",
        "wesley franca",
        "raphinha",
        "mahmoud trezeguet",
    }
    focus = [row for row in out_rows if norm(row["player_name"]) in focus_names]

    lines = [
        "# Remaining Bubble Flags Report",
        "",
        "Ren klassifikation af de resterende bubble-flags. Ingen optimizer-, EV- eller frontend-output er genberegnet.",
        "",
        "## Fordeling",
        "",
        "| primary_flag_category | count |",
        "|---|---:|",
    ]
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")

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
                "primary_flag_category",
                "start_prob",
                "optimizer_ev",
                "recommended_next_action",
            ],
            25,
        ),
        "",
        "## Alle flags",
        "",
        *md_table(
            out_rows,
            [
                "player_name",
                "team",
                "position",
                "primary_flag_category",
                "suspected_issue",
            ],
            40,
        ),
        "",
    ]
    REMAINING_FLAGS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(PLAYER_POOL_PATH)

    players = optimizer.load_players()
    add_audit_only_player_pool_fields(players)
    selected = selected_strategy_map()
    start_rows = load_start_security()

    rows: list[dict[str, str]] = []
    for target in BUBBLE_PLAYERS:
        match = match_player(players, target)
        usage = best_usage_row(
            start_rows,
            txt(match.get("player_name")) if match is not None else target.name,
            canonical_team(match.get("team_id")) if match is not None else target.team_id,
        )
        rows.append(flag_row(target, match, selected, usage))

    write_csv(rows)
    write_md(rows)
    write_remaining_flags_report(rows)

    flagged = sum(1 for r in rows if r["model_error_flag"] == "yes")
    matched = sum(1 for r in rows if r["matched_player_id"])
    print(f"Skrevet: {OUT_CSV.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
    print(f"Skrevet: {OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {REMAINING_FLAGS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {REMAINING_FLAGS_MD.relative_to(PROJECT_ROOT)}")
    print(f"Matched players: {matched}")
    print(f"Model error flags: {flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
