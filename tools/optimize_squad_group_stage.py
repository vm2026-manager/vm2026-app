from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
FIXTURES_PATH = DATA_DIR / "fixtures_group.csv"
FIXTURE_MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
MATCH_ODDS_PATH = DATA_DIR / "match_odds_probs.csv"
TEAM_MARKET_CSV_PATH = DATA_DIR / "team_market_odds_layer_v1.csv"
TEAM_MARKET_JSON_PATH = DATA_DIR / "team_market_odds_layer_v1.json"

OUT_STRATEGIES_JSON = DATA_DIR / "optimal_squads_by_strategy.json"
OUT_COMPARISON_CSV = DATA_DIR / "strategy_comparison_report.csv"
OUT_FORMATION_COMPARISON_CSV = DATA_DIR / "strategy_formation_comparison_report.csv"
OUT_CONTEXT_JSON = DATA_DIR / "current_strategy_context.json"
OUT_DISPLAY_NAMES_JSON = DATA_DIR / "strategy_display_names.json"
OUT_CLEANUP_REPORT = DATA_DIR / "strategy_cleanup_report.md"
CONFIRMED_LINEUPS_PATH = DATA_DIR / "confirmed_lineups.csv"
CURRENT_SQUAD_PATH = DATA_DIR / "current_squad.csv"
MANUAL_OVERRIDES_PATH = DATA_DIR / "manual_player_overrides.csv"

BUDGET_M = 50.0
SQUAD_SIZE = 11
MAX_PER_TEAM = 4
LOW_CONDITIONAL_THRESHOLD = 0.75
TRANSFER_FEE_RATE = 0.01
DK_TZ = ZoneInfo("Europe/Copenhagen")

FORMATIONS: dict[str, dict[str, int]] = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

STRATEGIES = ["next_round", "round1_2", "group_stage", "practical_start", "long_run"]

DISPLAY_NAMES_DA = {
    "next_round": "Næste runde",
    "round1_2": "1. + 2. runde",
    "group_stage": "Gruppespil",
    "practical_start": "1. + 2. runde",
    "long_run": "Lang sigt",
}

POSITION_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
    "KEEPER": "GK",
    "GOALKEEPER": "GK",
    "DEFENDER": "DEF",
    "MIDFIELDER": "MID",
    "ATTACKER": "FWD",
    "FORWARD": "FWD",
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


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def standardize_position(value: Any) -> str:
    return POSITION_MAP.get(txt(value).upper(), txt(value).upper())


def normalize_price_to_millions(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return values
    return values / 1_000_000 if float(non_null.median()) > 1000 else values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def round_for_match_id(match_id: Any) -> int:
    mid = int(txt(match_id) or "0")
    if 1 <= mid <= 24:
        return 1
    if 25 <= mid <= 48:
        return 2
    if 49 <= mid <= 72:
        return 3
    return 0


def parse_kickoff_dk(value: Any) -> datetime | None:
    raw = txt(value)
    if not raw:
        return None
    for fmt_str in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(raw, fmt_str).replace(tzinfo=DK_TZ)
        except ValueError:
            continue
    return None


def get_current_target_round(now: datetime | None = None) -> dict[str, Any]:
    fixtures = read_csv(FIXTURES_PATH)
    current = now.astimezone(DK_TZ) if now else datetime.now(DK_TZ)
    by_round: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for row in fixtures:
        rnd = int(row.get("round") or round_for_match_id(row.get("match_id")))
        kickoff = parse_kickoff_dk(row.get("kickoff_dk"))
        if rnd in by_round and kickoff is not None:
            by_round[rnd].append({"row": row, "kickoff": kickoff})

    target_round: int | None = None
    remaining: list[dict[str, Any]] = []
    for rnd in [1, 2, 3]:
        future = [item for item in by_round[rnd] if item["kickoff"] > current]
        if future:
            target_round = rnd
            remaining = future
            break

    if target_round is None:
        return {
            "generated_at": datetime.now(DK_TZ).isoformat(timespec="seconds"),
            "current_time_dk": current.isoformat(timespec="seconds"),
            "target_round": "",
            "target_round_label": "knockout_next",
            "next_round_display_name": "Næste runde",
            "remaining_matches_in_target_round": 0,
            "first_kickoff_in_target_round": "",
            "last_kickoff_in_target_round": "",
        }

    kickoffs = sorted(item["kickoff"] for item in remaining)
    return {
        "generated_at": datetime.now(DK_TZ).isoformat(timespec="seconds"),
        "current_time_dk": current.isoformat(timespec="seconds"),
        "target_round": target_round,
        "target_round_label": f"runde {target_round}",
        "next_round_display_name": f"Næste runde (runde {target_round})",
        "remaining_matches_in_target_round": len(remaining),
        "first_kickoff_in_target_round": kickoffs[0].isoformat(timespec="minutes"),
        "last_kickoff_in_target_round": kickoffs[-1].isoformat(timespec="minutes"),
    }


def load_player_pool_layer() -> pd.DataFrame:
    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    pool = pd.DataFrame(data)
    if pool.empty:
        return pd.DataFrame(columns=["player_id"])
    keep = [
        "player_id",
        "holdet_is_out",
        "is_out",
        "conditional_start_prob",
        "availability_prob",
        "availability_risk",
        "availability_status",
        "start_status",
        "start_prob",
    ]
    existing = [col for col in keep if col in pool.columns]
    return pool[existing].drop_duplicates(subset=["player_id"], keep="first")


def load_team_market_scores() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if TEAM_MARKET_CSV_PATH.exists():
        rows = read_csv(TEAM_MARKET_CSV_PATH)
    elif TEAM_MARKET_JSON_PATH.exists():
        rows = json.loads(TEAM_MARKET_JSON_PATH.read_text(encoding="utf-8-sig"))
    if not rows:
        return pd.DataFrame(columns=["team_id", "team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"])
    market = pd.DataFrame(rows)
    market["team_id"] = market["team_id"].astype(str).str.strip().str.upper()
    for col in ["team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]:
        if col not in market.columns:
            market[col] = 0.0
        market[col] = pd.to_numeric(market[col], errors="coerce").fillna(0.0)
    return market[["team_id", "team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]]


def load_manual_overrides() -> pd.DataFrame:
    columns = [
        "player_name",
        "team_id",
        "manual_status",
        "manual_start_status",
        "manual_captain_status",
        "manual_role_note",
        "manual_set_piece_role",
        "manual_captain_note",
        "manual_note",
    ]
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=columns)
    overrides = pd.DataFrame(read_csv(MANUAL_OVERRIDES_PATH))
    if overrides.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in overrides.columns:
            overrides[col] = ""
    overrides["player_name"] = overrides["player_name"].astype(str).str.strip()
    overrides["team_id"] = overrides["team_id"].astype(str).str.strip().str.upper()
    for col in ["manual_status", "manual_start_status", "manual_captain_status"]:
        overrides[col] = overrides[col].astype(str).str.strip().str.lower()
    return overrides[columns].drop_duplicates(subset=["player_name", "team_id"], keep="last")


def load_current_squad() -> pd.DataFrame:
    columns = ["player_id", "player_name", "team_id", "position", "current_value", "owned_since_round"]
    if not CURRENT_SQUAD_PATH.exists():
        return pd.DataFrame(columns=columns)
    squad = pd.DataFrame(read_csv(CURRENT_SQUAD_PATH))
    if squad.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in squad.columns:
            squad[col] = ""
    squad["player_id"] = squad["player_id"].astype(str).str.strip()
    squad["player_name"] = squad["player_name"].astype(str).str.strip()
    squad["team_id"] = squad["team_id"].astype(str).str.strip().str.upper()
    squad["position"] = squad["position"].map(standardize_position)
    squad["current_value"] = pd.to_numeric(squad["current_value"], errors="coerce")
    squad["owned_since_round"] = pd.to_numeric(squad["owned_since_round"], errors="coerce")
    squad = squad[(squad["player_id"] != "") | ((squad["player_name"] != "") & (squad["team_id"] != ""))].copy()
    return squad[columns].drop_duplicates(subset=["player_id", "player_name", "team_id"], keep="last")


def attach_current_squad_layer(players: pd.DataFrame) -> pd.DataFrame:
    current = load_current_squad()
    work = players.copy()
    work["current_squad_current_value"] = pd.NA
    work["owned_since_round"] = pd.NA
    work["is_current_squad_player"] = False
    if current.empty:
        return work

    with_ids = current[current["player_id"] != ""].copy()
    if not with_ids.empty:
        id_layer = with_ids[["player_id", "current_value", "owned_since_round"]].rename(
            columns={
                "current_value": "current_squad_current_value_by_id",
                "owned_since_round": "owned_since_round_by_id",
            }
        )
        work = work.merge(id_layer, on="player_id", how="left")
        matched = work["current_squad_current_value_by_id"].notna() | work["owned_since_round_by_id"].notna()
        work.loc[matched, "current_squad_current_value"] = work.loc[matched, "current_squad_current_value_by_id"]
        work.loc[matched, "owned_since_round"] = work.loc[matched, "owned_since_round_by_id"]
        work.loc[matched, "is_current_squad_player"] = True
        work = work.drop(columns=["current_squad_current_value_by_id", "owned_since_round_by_id"])

    by_name = current[(current["player_name"] != "") & (current["team_id"] != "")].copy()
    if not by_name.empty:
        by_name["current_squad_key"] = by_name["player_name"].str.casefold() + "|" + by_name["team_id"]
        work["current_squad_key"] = work["player_name"].astype(str).str.casefold() + "|" + work["team_id"].astype(str)
        name_layer = by_name[["current_squad_key", "current_value", "owned_since_round"]].rename(
            columns={
                "current_value": "current_squad_current_value_by_name",
                "owned_since_round": "owned_since_round_by_name",
            }
        )
        work = work.merge(name_layer, on="current_squad_key", how="left")
        unmatched = ~work["is_current_squad_player"].astype(bool)
        matched = unmatched & (
            work["current_squad_current_value_by_name"].notna()
            | work["owned_since_round_by_name"].notna()
        )
        work.loc[matched, "current_squad_current_value"] = work.loc[matched, "current_squad_current_value_by_name"]
        work.loc[matched, "owned_since_round"] = work.loc[matched, "owned_since_round_by_name"]
        work.loc[matched, "is_current_squad_player"] = True
        work = work.drop(columns=["current_squad_key", "current_squad_current_value_by_name", "owned_since_round_by_name"])

    return work


def load_fixture_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_csv(FIXTURE_MULTIPLIERS_PATH):
        match_id = txt(row.get("match_id"))
        rnd = round_for_match_id(match_id)
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        kickoff = txt(row.get("kickoff_dk"))
        lookup[(home, away, kickoff)] = {
            "match_id": match_id,
            "round": rnd,
            "opponent": away,
            "win_prob": to_float(row.get("home_win_prob_fair")),
            "goal_multiplier": to_float(row.get("home_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("home_assist_multiplier"), 1.0),
            "clean_sheet_prob": to_float(row.get("home_clean_sheet_prob_fair")),
            "clean_sheet_multiplier": to_float(row.get("home_clean_sheet_multiplier"), 1.0),
        }
        lookup[(away, home, kickoff)] = {
            "match_id": match_id,
            "round": rnd,
            "opponent": home,
            "win_prob": to_float(row.get("away_win_prob_fair")),
            "goal_multiplier": to_float(row.get("away_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("away_assist_multiplier"), 1.0),
            "clean_sheet_prob": to_float(row.get("away_clean_sheet_prob_fair")),
            "clean_sheet_multiplier": to_float(row.get("away_clean_sheet_multiplier"), 1.0),
        }
    return lookup


def team_round_win_probs() -> dict[str, dict[int, float]]:
    wins: dict[str, dict[int, float]] = {}
    for row in read_csv(MATCH_ODDS_PATH):
        match_id = txt(row.get("match_id"))
        rnd = round_for_match_id(match_id)
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        wins.setdefault(home, {})[rnd] = to_float(row.get("home_win_prob_fair"))
        wins.setdefault(away, {})[rnd] = to_float(row.get("away_win_prob_fair"))
    return wins


def add_player_round_context(players: pd.DataFrame) -> pd.DataFrame:
    lookup = load_fixture_lookup()
    win_probs = team_round_win_probs()
    work = players.copy()
    for rnd in [1, 2, 3]:
        for col in [
            "ev",
            "captain_growth",
            "win_prob",
            "goal_multiplier",
            "assist_multiplier",
            "clean_sheet_prob",
            "clean_sheet_multiplier",
        ]:
            work[f"round{rnd}_{col}"] = 0.0
        work[f"round{rnd}_opponent"] = ""
        work[f"round{rnd}_match_id"] = ""

    for idx, row in work.iterrows():
        team = txt(row.get("team_id")).upper()
        for match_no in [1, 2, 3]:
            opponent = txt(row.get(f"match_{match_no}_opponent_team")).upper()
            kickoff = txt(row.get(f"match_{match_no}_kickoff"))
            context = lookup.get((team, opponent, kickoff))
            if not context:
                continue
            rnd = int(context["round"])
            if rnd not in [1, 2, 3]:
                continue
            work.at[idx, f"round{rnd}_ev"] += to_float(row.get(f"match_{match_no}_weighted_match_ev"))
            work.at[idx, f"round{rnd}_captain_growth"] += to_float(row.get(f"match_{match_no}_total_ev_next_match"))
            work.at[idx, f"round{rnd}_opponent"] = txt(context["opponent"])
            work.at[idx, f"round{rnd}_match_id"] = txt(context["match_id"])
            work.at[idx, f"round{rnd}_win_prob"] = float(context["win_prob"])
            work.at[idx, f"round{rnd}_goal_multiplier"] = float(context["goal_multiplier"])
            work.at[idx, f"round{rnd}_assist_multiplier"] = float(context["assist_multiplier"])
            work.at[idx, f"round{rnd}_clean_sheet_prob"] = float(context["clean_sheet_prob"])
            work.at[idx, f"round{rnd}_clean_sheet_multiplier"] = float(context["clean_sheet_multiplier"])

        p6 = win_probs.get(team, {}).get(1, 0.0) * win_probs.get(team, {}).get(2, 0.0)
        work.at[idx, "p_6_points_after_2"] = p6
        if p6 >= 0.55:
            rotation_factor = 0.62
        elif p6 >= 0.40:
            rotation_factor = 0.74
        elif p6 >= 0.25:
            rotation_factor = 0.86
        else:
            rotation_factor = 1.0
        work.at[idx, "round3_rotation_factor"] = rotation_factor
    return work


def load_players() -> pd.DataFrame:
    ev = pd.read_csv(PLAYER_EV_PATH)
    required = ["player_id", "player_name", "team_id", "position", "price", "optimizer_ev", "start_prob"]
    missing = [col for col in required if col not in ev.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i {PLAYER_EV_PATH}: {missing}")

    players = ev.copy()
    players["team_id"] = players["team_id"].astype(str).str.strip().str.upper()
    players["position"] = players["position"].map(standardize_position)
    players["price_m"] = normalize_price_to_millions(players["price"])
    players["optimizer_ev"] = pd.to_numeric(players["optimizer_ev"], errors="coerce").fillna(0.0)
    players["weighted_group_stage_ev"] = pd.to_numeric(players.get("weighted_group_stage_ev", players["optimizer_ev"]), errors="coerce").fillna(players["optimizer_ev"])
    players["start_prob"] = pd.to_numeric(players["start_prob"], errors="coerce").fillna(0.48).clip(0.0, 1.0)

    players = players.merge(load_player_pool_layer(), on="player_id", how="left", suffixes=("", "_pool"))
    if "start_prob_pool" in players.columns:
        players["start_prob"] = pd.to_numeric(players["start_prob_pool"], errors="coerce").fillna(players["start_prob"]).clip(0.0, 1.0)
    players["conditional_start_prob"] = pd.to_numeric(players.get("conditional_start_prob"), errors="coerce").fillna(players["start_prob"]).clip(0.0, 1.0)
    players["availability_prob"] = pd.to_numeric(players.get("availability_prob"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    players["availability_risk"] = players.get("availability_risk", "unknown").fillna("unknown").astype(str)
    out_flag = pd.Series(False, index=players.index)
    for field in ["holdet_is_out", "is_out", "holdet_is_out_pool", "is_out_pool"]:
        if field in players.columns:
            out_flag |= players[field].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "ja"})
    players = players[~out_flag].copy()

    players = players.merge(load_team_market_scores(), on="team_id", how="left")
    for col in ["team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]:
        players[col] = pd.to_numeric(players.get(col), errors="coerce").fillna(0.0)

    players = players.merge(load_manual_overrides(), on=["player_name", "team_id"], how="left")
    for col in [
        "manual_status",
        "manual_start_status",
        "manual_captain_status",
        "manual_role_note",
        "manual_set_piece_role",
        "manual_captain_note",
        "manual_note",
    ]:
        players[col] = players.get(col, "").fillna("").astype(str)
    players["manual_avoid"] = (
        players["manual_status"].str.lower().eq("avoid")
        | players["manual_start_status"].str.lower().eq("avoid")
    )

    players = players.dropna(subset=["player_id", "player_name", "team_id", "position", "price_m"]).copy()
    players = players[players["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()
    players = players[players["price_m"] > 0].copy()
    players = attach_current_squad_layer(players)
    players = add_player_round_context(players)
    return add_strategy_scores(players.reset_index(drop=True))


def starter_component(work: pd.DataFrame) -> pd.Series:
    return (
        0.45 * (work["conditional_start_prob"] >= 0.85).astype(float)
        + 0.18 * ((work["conditional_start_prob"] >= 0.75) & (work["conditional_start_prob"] < 0.85)).astype(float)
        - 0.65 * (work["conditional_start_prob"] < 0.70).astype(float)
        - 0.28 * ((work["conditional_start_prob"] >= 0.70) & (work["conditional_start_prob"] < 0.75)).astype(float)
        - 0.85 * work["availability_risk"].eq("high_risk").astype(float)
        - 0.08 * work["availability_risk"].eq("medium_risk").astype(float)
    )


def round_fixture_bonus(work: pd.DataFrame, rnd: int) -> pd.Series:
    offensive = work["position"].isin(["MID", "FWD"]).astype(float)
    defensive = work["position"].isin(["GK", "DEF"]).astype(float)
    favorite = (
        0.45 * (work[f"round{rnd}_win_prob"] >= 0.75).astype(float)
        + 0.27 * ((work[f"round{rnd}_win_prob"] >= 0.65) & (work[f"round{rnd}_win_prob"] < 0.75)).astype(float)
        + 0.10 * ((work[f"round{rnd}_win_prob"] >= 0.60) & (work[f"round{rnd}_win_prob"] < 0.65)).astype(float)
        - 0.10 * (work[f"round{rnd}_win_prob"] < 0.50).astype(float)
    )
    attack = (
        (work[f"round{rnd}_goal_multiplier"] - 1.0).clip(lower=0.0) * 0.90
        + (work[f"round{rnd}_assist_multiplier"] - 1.0).clip(lower=0.0) * 0.55
    ) * offensive
    clean = (work[f"round{rnd}_clean_sheet_multiplier"] - 1.0).clip(lower=0.0) * 0.95 * defensive
    return favorite + attack + clean


def add_strategy_scores(players: pd.DataFrame) -> pd.DataFrame:
    work = players.copy()
    start = starter_component(work)
    target_round = int(get_current_target_round().get("target_round") or 1)
    current_value = pd.to_numeric(work.get("current_squad_current_value"), errors="coerce").fillna(work["price_m"] * 1_000_000)
    owned = work.get("is_current_squad_player", False).fillna(False).astype(bool)
    transfer_fee = pd.Series(0.0, index=work.index)
    if target_round > 1:
        transfer_fee = current_value.fillna(work["price_m"] * 1_000_000) * TRANSFER_FEE_RATE
        transfer_fee.loc[owned] = 0.0
    work["transfer_fee"] = transfer_fee.round(0).astype(int)
    work["transfer_fee_m"] = transfer_fee / 1_000_000.0

    # Runde 1 kampmilj?-bonus:
    # L?fter kun sikre/relevante startere fra de st?rste favoritkampe.
    # Skal f? N?ste runde til at l?ne sig mere mod GER/ESP/NOR/SUI/AUT/POR
    # uden at tvinge svage picks ind.
    round1_ev_for_favorite_bonus = pd.to_numeric(
        work.get("round1_ev", work.get("match_1_weighted_match_ev", 0.0)),
        errors="coerce",
    ).fillna(0.0)

    next_round_favorite_tier = work["team_id"].fillna("").astype(str).map({
        "GER": 1.05,
        "ESP": 1.05,
        "NOR": 0.85,
        "SUI": 0.85,
        "AUT": 0.85,
        "POR": 0.85,
        "ARG": 0.25,
        "FRA": 0.25,
        "BRA": 0.25,
        "MEX": 0.20,
        "CAN": 0.20,
        "BEL": 0.20,
    }).fillna(0.0)

    next_round_favorite_bonus = (
        next_round_favorite_tier
        * (start >= 0.75).astype(float)
        * (round1_ev_for_favorite_bonus >= 1.00).astype(float)
    )

    work["score_next_round"] = (
        next_round_favorite_bonus
        +         2.20 * work[f"round{target_round}_ev"]
        + 0.82 * work["optimizer_ev"]
        + round_fixture_bonus(work, target_round)
        + start
        - work["transfer_fee_m"]
    ).clip(lower=0.0)

    round12_value = 1.05 * work["round1_ev"] + 1.00 * work["round2_ev"]
    one_round_spike_penalty = (work[["round1_ev", "round2_ev"]].max(axis=1) - work[["round1_ev", "round2_ev"]].min(axis=1)).clip(lower=0.0) * 0.22
    work["score_round1_2"] = (
        1.85 * round12_value
        + 0.60 * work["optimizer_ev"]
        + 0.55 * (round_fixture_bonus(work, 1) + round_fixture_bonus(work, 2))
        + start
        - one_round_spike_penalty
    ).clip(lower=0.0)

    group_value = work["round1_ev"] + work["round2_ev"] + work["round3_ev"] * work["round3_rotation_factor"]
    rotation_note_penalty = (1.0 - work["round3_rotation_factor"]) * work["round3_ev"] * 0.30
    work["score_group_stage"] = (
        1.75 * group_value
        + 0.70 * work["optimizer_ev"]
        + 0.35 * (round_fixture_bonus(work, 1) + round_fixture_bonus(work, 2) + round_fixture_bonus(work, 3))
        + start
        - rotation_note_penalty
    ).clip(lower=0.0)

    # Praktisk startstrategi:
    # Kig prim?rt 1-2 kampe frem, nedton runde 3, prioriter offensive profiler
    # og billige sikre GK/DEF-startere.
    practical_value = (
        1.25 * work["round1_ev"]
        + 0.85 * work["round2_ev"]
        + 0.20 * work["round3_ev"] * work["round3_rotation_factor"]
    )

    position = work["position"].fillna("").astype(str).str.upper()
    price_m = pd.to_numeric(work["price_m"], errors="coerce").fillna(0.0)

    offensive_profile_bonus = (
        0.70 * position.eq("FWD").astype(float)
        + 0.34 * ((position.eq("MID")) & (work["optimizer_ev"] >= 3.0)).astype(float)
        + 0.25 * ((position.eq("MID")) & (work["team_attack_score"] >= 0.45)).astype(float)
        + 0.18 * ((position.isin(["MID", "FWD"])) & (price_m >= 5.5)).astype(float)
    )

    cheap_defensive_starter_bonus = (
        0.46
        * position.isin(["GK", "DEF"]).astype(float)
        * (price_m <= 4.0).astype(float)
        * (work["conditional_start_prob"] >= 0.82).astype(float)
    )

    expensive_defensive_penalty = (
        0.24
        * position.isin(["GK", "DEF"]).astype(float)
        * (price_m >= 4.5).astype(float)
        * (work["optimizer_ev"] < 4.0).astype(float)
    )

    practical_market_bonus = (
        0.32 * work["team_group_stage_score"]
        + 0.22 * work["team_attack_score"]
        + 0.12 * work["team_long_run_score"]
    )

    work["score_practical_start"] = (
        1.95 * practical_value
        + 0.42 * work["optimizer_ev"]
        + 0.70 * (round_fixture_bonus(work, 1) + 0.65 * round_fixture_bonus(work, 2))
        + 1.05 * start
        + practical_market_bonus
        + offensive_profile_bonus
        + cheap_defensive_starter_bonus
        - expensive_defensive_penalty
        - 0.55 * work["transfer_fee_m"]
    ).clip(lower=0.0)

    tournament_strength = (0.75 * work["team_long_run_score"] + 0.25 * work["team_market_score"]).clip(lower=0.0)
    weak_team_penalty = (
        2.20 * (tournament_strength < 0.18).astype(float)
        + 1.25 * ((tournament_strength >= 0.18) & (tournament_strength < 0.28)).astype(float)
        + 0.45 * ((tournament_strength >= 0.28) & (tournament_strength < 0.40)).astype(float)
    )
    manual_long_run_penalty = (
        0.95 * work["manual_status"].str.lower().eq("check").astype(float)
        + 0.95 * work["manual_start_status"].str.lower().eq("doubtful").astype(float)
        + 0.30 * work["manual_captain_status"].str.lower().eq("avoid").astype(float)
    )
    mid_team_value_penalty = (
        0.65
        * (tournament_strength < 0.35).astype(float)
        * (
            (work["optimizer_ev"] < 4.60)
            | (work["conditional_start_prob"] < 0.88)
            | work["availability_risk"].eq("high_risk")
        ).astype(float)
    )
    work["score_long_run"] = (
        0.95 * work["optimizer_ev"]
        + 3.40 * tournament_strength
        + 1.15 * start
        - weak_team_penalty
        - manual_long_run_penalty
        - mid_team_value_penalty
    ).clip(lower=0.0)

    return work


def strategy_score_column(strategy: str) -> str:
    return f"score_{strategy}"


def solve_formation(players: pd.DataFrame, strategy: str, formation_name: str, formation: dict[str, int]) -> pd.DataFrame:
    score_col = strategy_score_column(strategy)
    problem = pulp.LpProblem(f"{strategy}_{formation_name.replace('-', '_')}", pulp.LpMaximize)
    variables = {idx: pulp.LpVariable(f"pick_{idx}", lowBound=0, upBound=1, cat="Binary") for idx in players.index}
    score_expr = pulp.lpSum(float(players.loc[idx, score_col]) * variables[idx] for idx in players.index)
    total_price_expr = pulp.lpSum(float(players.loc[idx, "price_m"]) * variables[idx] for idx in players.index)
    underuse = pulp.LpVariable(f"{strategy}_budget_underuse", lowBound=0, cat="Continuous")

    floor = 49.5 if strategy in {"next_round", "practical_start"} else 49.0
    penalty = 1.10 if strategy == "next_round" else (0.75 if strategy == "practical_start" else (0.18 if strategy == "long_run" else 0.55))
    problem += score_expr + 0.025 * total_price_expr - penalty * underuse
    problem += underuse >= floor - total_price_expr
    problem += pulp.lpSum(variables[idx] for idx in players.index) == SQUAD_SIZE
    problem += total_price_expr <= BUDGET_M
    high_risk_limit = 0 if strategy == "long_run" else (1 if strategy in {"next_round", "practical_start"} else 2)
    avg_cond_floor = 0.84 if strategy in {"next_round", "long_run", "practical_start"} else 0.80
    min_cond = 0.72 if strategy == "long_run" else (0.70 if strategy in {"next_round", "practical_start"} else 0.65)

    problem += pulp.lpSum(
        variables[idx] for idx in players.index if txt(players.loc[idx, "availability_risk"]) == "high_risk"
    ) <= high_risk_limit
    problem += pulp.lpSum(float(players.loc[idx, "conditional_start_prob"]) * variables[idx] for idx in players.index) >= avg_cond_floor * SQUAD_SIZE
    for idx in players.index[players["conditional_start_prob"] < min_cond].tolist():
        problem += variables[idx] == 0
    for idx in players.index[players["manual_avoid"]].tolist():
        problem += variables[idx] == 0

    if strategy == "long_run":
        tournament_strength = (0.75 * players["team_long_run_score"] + 0.25 * players["team_market_score"]).clip(lower=0.0)
        strong_team_indices = players.index[tournament_strength >= 0.50].tolist()
        weak_team_indices = players.index[tournament_strength < 0.35].tolist()
        problem += pulp.lpSum(variables[idx] for idx in strong_team_indices) >= 7
        problem += pulp.lpSum(variables[idx] for idx in weak_team_indices) <= 2

    for pos, count in formation.items():
        indices = players.index[players["position"] == pos].tolist()
        problem += pulp.lpSum(variables[idx] for idx in indices) == count

    for team_id, sub in players.groupby("team_id"):
        problem += pulp.lpSum(variables[idx] for idx in sub.index.tolist()) <= MAX_PER_TEAM

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[problem.status] != "Optimal":
        return pd.DataFrame()

    picked = [idx for idx, variable in variables.items() if variable.value() == 1]
    squad = players.loc[picked].copy()
    squad["strategy"] = strategy
    squad["selected_formation"] = formation_name
    squad["strategy_score"] = squad[score_col]
    return squad.sort_values(["position", "strategy_score", "optimizer_ev"], ascending=[True, False, False]).reset_index(drop=True)


def captain_score_reason_v2(player: pd.Series, target_round: int, growth_col: str) -> str:
    bits = [f"kaptajnscore baseret paa runde {target_round}-vaekst {fmt(to_float(player.get(growth_col)), 3)}"]
    cond = to_float(player.get("conditional_start_prob"))
    risk = txt(player.get("availability_risk"))
    role = txt(player.get("manual_set_piece_role")).lower()
    position = txt(player.get("position")).upper()
    if "penalty" in role or "straffe" in role:
        bits.append("penalty/manual doedbold bonus")
    elif role:
        bits.append("manual doedbold/rolle bonus")
    else:
        bits.append("ingen registreret straffe-/doedboldsrolle")
    if cond >= 0.90:
        bits.append("hoej startsikkerhed")
    elif cond < LOW_CONDITIONAL_THRESHOLD:
        bits.append("straf for lav conditional start")
    elif cond < 0.85:
        bits.append("mild straf for usikker start")
    if risk == "high_risk":
        bits.append("straf for high_risk")
    elif risk == "medium_risk":
        bits.append("mild risk-straf")
    if position == "FWD":
        bits.append("maalprofil-proxy: angriber")
    elif position == "MID":
        bits.append("maalprofil-proxy: midtbane; TODO national_goal_rate/recent_goal_rate")
    else:
        bits.append("lavere maalprofil-proxy for position")
    return "; ".join(bits)


def add_captain_scores_v2(squad: pd.DataFrame, target_round: int) -> pd.DataFrame:
    work = squad.copy()
    growth_col = f"round{target_round}_captain_growth"
    if growth_col not in work.columns:
        growth_col = f"round{target_round}_ev"
    growth = pd.to_numeric(work.get(growth_col, 0.0), errors="coerce").fillna(0.0)
    cond = pd.to_numeric(work.get("conditional_start_prob", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    win = pd.to_numeric(work.get(f"round{target_round}_win_prob", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    risk = work.get("availability_risk", "").fillna("").astype(str)
    role = work.get("manual_set_piece_role", "").fillna("").astype(str).str.lower()
    position = work.get("position", "").fillna("").astype(str).str.upper()
    start_penalty = (
        1.55 * (cond < 0.70).astype(float)
        + 0.95 * ((cond >= 0.70) & (cond < 0.75)).astype(float)
        + 0.45 * ((cond >= 0.75) & (cond < 0.85)).astype(float)
    )
    risk_penalty = 1.15 * risk.eq("high_risk").astype(float) + 0.22 * risk.eq("medium_risk").astype(float)
    set_piece_bonus = (
        0.75 * role.str.contains("penalty|straffe", regex=True).astype(float)
        + 0.28 * role.str.contains("direct_fk|corner|indirect|fk|free", regex=True).astype(float)
    )
    scorer_proxy = (
        0.32 * position.eq("FWD").astype(float)
        + 0.10 * position.eq("MID").astype(float)
        - 0.20 * position.isin(["GK", "DEF"]).astype(float)
    )
    favorite_bonus = (
        0.28 * (win >= 0.70).astype(float)
        + 0.14 * ((win >= 0.60) & (win < 0.70)).astype(float)
        - 0.12 * (win < 0.50).astype(float)
    )
    captain_blocked = (
        work.get("manual_avoid", False).astype(bool)
        | work.get("manual_status", "").fillna("").astype(str).str.lower().eq("avoid")
        | work.get("manual_start_status", "").fillna("").astype(str).str.lower().eq("avoid")
        | work.get("manual_captain_status", "").fillna("").astype(str).str.lower().eq("avoid")
    )
    work["captain_eligible"] = ~captain_blocked
    position_for_captain = work["position"].fillna("").astype(str).str.upper() if "position" in work.columns else pd.Series("", index=work.index)
    price_m_for_captain = pd.to_numeric(work.get("price", 0.0), errors="coerce").fillna(0.0) / 1_000_000
    role_for_captain = work.get("manual_set_piece_role", "").fillna("").astype(str).str.lower()

    offensive_mid_for_captain = (
        position_for_captain.eq("MID")
        & (
            (price_m_for_captain >= 6.0)
            | role_for_captain.str.contains("penalty|straffe|wing|kant|forward|angriber|10|offensive", regex=True)
        )
    )

    captain_position_multiplier = pd.Series(0.65, index=work.index, dtype=float)
    captain_position_multiplier.loc[position_for_captain.eq("FWD")] = 1.25
    captain_position_multiplier.loc[offensive_mid_for_captain] = 1.05
    captain_position_multiplier.loc[position_for_captain.eq("DEF")] = 0.35
    captain_position_multiplier.loc[position_for_captain.eq("GK")] = 0.25

    central_mid_extra_penalty = (
        position_for_captain.eq("MID")
        & ~offensive_mid_for_captain
        & (price_m_for_captain < 6.0)
    ).astype(float) * 0.18

    captain_base_score = growth + set_piece_bonus + scorer_proxy + favorite_bonus
    work["captain_position_multiplier"] = captain_position_multiplier
    work["captain_score"] = (
        captain_base_score * captain_position_multiplier
        - start_penalty
        - risk_penalty
        - central_mid_extra_penalty
    )
    work.loc[~work["captain_eligible"], "captain_score"] = -9999.0
    work["captain_score_reason"] = work.apply(lambda row: captain_score_reason_v2(row, target_round, growth_col), axis=1)
    return work


def captain_for_squad_v2(squad: pd.DataFrame, target_round: int) -> tuple[dict[str, Any], pd.DataFrame]:
    if squad.empty:
        return {"recommended_captain": "", "captain_expected_growth": 0.0, "captain_round": target_round, "captain_reason": ""}, squad
    squad = add_captain_scores_v2(squad, target_round)
    candidates = squad[squad["captain_eligible"]].copy()
    if candidates.empty:
        return {"recommended_captain": "", "captain_expected_growth": 0.0, "captain_round": target_round, "captain_reason": "Ingen kaptajn efter manual captain-filter"}, squad
    col = f"round{target_round}_captain_growth"
    if col not in squad.columns:
        col = f"round{target_round}_ev"
    player = candidates.sort_values(["captain_score", col, "optimizer_ev"], ascending=[False, False, False]).iloc[0]
    return {
        "recommended_captain": txt(player.get("player_name")),
        "captain_expected_growth": round(float(player.get(col, 0.0)), 6),
        "captain_round": target_round,
        "captain_reason": txt(player.get("captain_score_reason")),
        "captain_score": round(float(player.get("captain_score", 0.0)), 6),
    }, squad


def captain_for_squad(squad: pd.DataFrame, target_round: int) -> dict[str, Any]:
    if squad.empty:
        return {"recommended_captain": "", "captain_expected_growth": 0.0, "captain_round": target_round, "captain_reason": ""}
    squad = squad[~squad.get("manual_avoid", False).astype(bool)].copy()
    if squad.empty:
        return {"recommended_captain": "", "captain_expected_growth": 0.0, "captain_round": target_round, "captain_reason": "Ingen kaptajn efter manual avoid-filter"}
    col = f"round{target_round}_captain_growth"
    if col not in squad.columns:
        col = f"round{target_round}_ev"
    player = squad.sort_values([col, "optimizer_ev"], ascending=[False, False]).iloc[0]
    return {
        "recommended_captain": txt(player.get("player_name")),
        "captain_expected_growth": round(float(player.get(col, 0.0)), 6),
        "captain_round": target_round,
        "captain_reason": f"Højeste forventede vækst i runde {target_round}",
    }


def squad_summary(strategy: str, squad: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    teams = squad["team_id"].value_counts().sort_index()
    target_round = int(context.get("target_round") or 1)
    captain, scored_squad = captain_for_squad_v2(squad, target_round)
    for col in ["captain_eligible", "captain_score", "captain_score_reason"]:
        if col in scored_squad.columns:
            squad[col] = scored_squad[col]
    display_name = context["next_round_display_name"] if strategy == "next_round" else DISPLAY_NAMES_DA[strategy]
    return {
        "strategy": strategy,
        "display_name_da": display_name,
        "formation": txt(squad["selected_formation"].iloc[0]) if not squad.empty else "",
        "total_score": round(float(squad["strategy_score"].sum()), 6) if not squad.empty else 0.0,
        "total_ev": round(float(squad["optimizer_ev"].sum()), 6) if not squad.empty else 0.0,
        "total_price": int(round(float(squad["price_m"].sum()) * 1_000_000)) if not squad.empty else 0,
        "avg_start_prob": round(float(squad["start_prob"].mean()), 4) if not squad.empty else 0.0,
        "avg_conditional_start_prob": round(float(squad["conditional_start_prob"].mean()), 4) if not squad.empty else 0.0,
        "high_risk_players": int((squad["availability_risk"] == "high_risk").sum()) if not squad.empty else 0,
        "teams_summary": "; ".join(f"{team}:{count}" for team, count in teams.items()),
        "player_names": "; ".join(squad["player_name"].astype(str).tolist()),
        **captain,
    }


def squad_records(squad: pd.DataFrame) -> list[dict[str, Any]]:
    keep = [
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "price",
        "price_m",
        "start_prob",
        "conditional_start_prob",
        "availability_prob",
        "availability_risk",
        "manual_status",
        "manual_start_status",
        "manual_captain_status",
        "manual_role_note",
        "manual_set_piece_role",
        "manual_captain_note",
        "manual_note",
        "manual_avoid",
        "is_current_squad_player",
        "current_squad_current_value",
        "owned_since_round",
        "transfer_fee",
        "transfer_fee_m",
        "captain_eligible",
        "captain_score",
        "captain_score_reason",
        "optimizer_ev",
        "weighted_group_stage_ev",
        "strategy_score",
        "selected_formation",
        "p_6_points_after_2",
        "round3_rotation_factor",
    ]
    for rnd in [1, 2, 3]:
        keep.extend([
            f"round{rnd}_ev",
            f"round{rnd}_captain_growth",
            f"round{rnd}_opponent",
            f"round{rnd}_win_prob",
            f"round{rnd}_goal_multiplier",
            f"round{rnd}_assist_multiplier",
            f"round{rnd}_clean_sheet_prob",
            f"round{rnd}_clean_sheet_multiplier",
        ])
    existing = [col for col in keep if col in squad.columns]
    return json.loads(squad[existing].to_json(orient="records", force_ascii=False))


def ensure_templates() -> None:
    templates = [
        (CONFIRMED_LINEUPS_PATH, ["round", "match_id", "team_id", "player_name", "lineup_status", "source", "note"]),
        (CURRENT_SQUAD_PATH, ["player_id", "player_name", "team_id", "position", "current_value", "owned_since_round"]),
        (MANUAL_OVERRIDES_PATH, ["player_name", "team_id", "manual_status", "manual_start_status", "manual_captain_status", "manual_role_note", "manual_set_piece_role", "manual_captain_note", "manual_note"]),
    ]
    for path, fields in templates:
        if not path.exists():
            write_csv(path, fields, [])


def write_strategy_metadata(context: dict[str, Any]) -> None:
    display = {
        "next_round": context["next_round_display_name"],
        "round1_2": DISPLAY_NAMES_DA["round1_2"],
        "group_stage": DISPLAY_NAMES_DA["group_stage"],
        "practical_start": DISPLAY_NAMES_DA["practical_start"],
        "long_run": DISPLAY_NAMES_DA["long_run"],
    }
    OUT_CONTEXT_JSON.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DISPLAY_NAMES_JSON.write_text(json.dumps(display, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Strategy Cleanup Report",
        "",
        "## Brugerrettede Strategier",
        "",
        f"- next_round: {display['next_round']}",
        f"- round1_2: {display['round1_2']}",
        f"- group_stage: {display['group_stage']}",
        f"- practical_start: {display['practical_start']}",
        f"- long_run: {display['long_run']}",
        "",
        "## Mapping",
        "",
        "- round1_safe_favorite er erstattet af next_round.",
        "- safe_starters er ikke længere en brugerrettet hovedstrategi; starter-sikkerhed indgår i alle strategier.",
        "- fixture_attack og clean_sheet_stack indgår som komponenter via kamp-multipliers og clean sheet-data.",
        "- balanced/debug-output er ikke længere primært strategi-output.",
        "",
        "## Dynamisk Næste Runde",
        "",
        f"- target_round: {context.get('target_round')}",
        f"- display: {context.get('next_round_display_name')}",
        "- target_round beregnes som laveste grupperunde med mindst én kamp, der endnu ikke er startet.",
        "",
        "## Kaptajn",
        "",
        "- Kaptajn vælges pr. strategi som spilleren med højeste forventede vækst i target_round.",
        "- Kaptajn-output skrives i strategy_comparison_report.csv og optimal_squads_by_strategy.json.",
        "",
        "## Forberedte Inputlag",
        "",
        "- data/confirmed_lineups.csv er oprettet som struktur til bekræftede lineups.",
        "- data/current_squad.csv er oprettet som struktur til transfergebyr efter runde 1.",
        "- data/manual_player_overrides.csv er oprettet som struktur til manuelle locks/check/avoid.",
        "",
        "## TODO",
        "",
        "- UI-visning af strategiknapper og kaptajnmarkering er ikke ændret i denne opgave.",
        "- Transfergebyr efter runde 1 indgår i next_round, når data/current_squad.csv indeholder brugerens nuværende hold.",
        "- confirmed_lineups påvirker endnu ikke optimizer direkte; strukturen er klar til integration.",
    ]
    OUT_CLEANUP_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    context = get_current_target_round()
    ensure_templates()
    write_strategy_metadata(context)
    players = load_players()
    all_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    formation_comparison_rows: list[dict[str, Any]] = []
    missing_formations: list[tuple[str, str]] = []

    print(f"Optimizer-pool spillere: {len(players)}")
    print(f"Budget: {BUDGET_M:.1f} mio. | Maks pr. land: {MAX_PER_TEAM}")
    print(f"Next round: {context['next_round_display_name']}")

    for strategy in STRATEGIES:
        best_squad = pd.DataFrame()
        best_summary: dict[str, Any] | None = None
        formation_records: dict[str, list[dict[str, Any]]] = {}
        squads_by_formation: dict[str, dict[str, Any]] = {}
        for formation_name, formation in FORMATIONS.items():
            squad = solve_formation(players, strategy, formation_name, formation)
            if squad.empty:
                formation_records[formation_name] = []
                squads_by_formation[formation_name] = {
                    "summary": {
                        "strategy": strategy,
                        "display_name_da": context["next_round_display_name"] if strategy == "next_round" else DISPLAY_NAMES_DA.get(strategy, strategy),
                        "formation": formation_name,
                        "total_score": 0.0,
                        "total_ev": 0.0,
                        "total_price": 0,
                        "avg_start_prob": 0.0,
                        "avg_conditional_start_prob": 0.0,
                        "high_risk_players": 0,
                        "teams_summary": "",
                        "player_names": "",
                        "recommended_captain": "",
                        "captain_expected_growth": 0.0,
                        "captain_round": int(context.get("target_round") or 1),
                        "captain_score": 0.0,
                        "captain_reason": "Ingen gyldig optimizer-løsning for formationen",
                    },
                    "squad": [],
                    "status": "no_valid_solution",
                }
                missing_formations.append((strategy, formation_name))
                continue
            summary = squad_summary(strategy, squad, context)
            records = squad_records(squad)
            formation_records[formation_name] = records
            squads_by_formation[formation_name] = {
                "summary": summary,
                "squad": records,
                "status": "ok",
            }
            formation_comparison_rows.append(summary)
            if best_summary is None or float(summary["total_score"]) > float(best_summary["total_score"]):
                best_summary = summary
                best_squad = squad.copy()

        if best_summary is None:
            best_summary = {
                "strategy": strategy,
                "display_name_da": DISPLAY_NAMES_DA.get(strategy, strategy),
                "formation": "",
                "total_score": 0.0,
                "total_ev": 0.0,
                "total_price": 0,
                "avg_start_prob": 0.0,
                "avg_conditional_start_prob": 0.0,
                "high_risk_players": 0,
                "teams_summary": "",
                "player_names": "",
                "recommended_captain": "",
                "captain_expected_growth": 0.0,
                "captain_round": int(context.get("target_round") or 1),
                "captain_score": 0.0,
                "captain_reason": "",
            }

        comparison_rows.append(best_summary)
        all_results[strategy] = {
            "best_summary": best_summary,
            "best_squad": squad_records(best_squad) if not best_squad.empty else [],
            "squads_by_formation": squads_by_formation,
            "formations": formation_records,
        }
        print(
            f"{strategy}: {best_summary['display_name_da']} | formation={best_summary['formation']}, "
            f"pris={best_summary['total_price']:,}, score={best_summary['total_score']:.3f}, "
            f"EV={best_summary['total_ev']:.3f}, high_risk={best_summary['high_risk_players']}, "
            f"kaptajn={best_summary['recommended_captain']}"
        )

    OUT_STRATEGIES_JSON.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "strategy",
        "display_name_da",
        "formation",
        "total_score",
        "total_ev",
        "total_price",
        "avg_start_prob",
        "avg_conditional_start_prob",
        "high_risk_players",
        "teams_summary",
        "player_names",
        "recommended_captain",
        "captain_expected_growth",
        "captain_round",
        "captain_score",
        "captain_reason",
    ]
    write_csv(OUT_COMPARISON_CSV, fieldnames, comparison_rows)
    formation_fieldnames = [
        "strategy",
        "display_name_da",
        "formation",
        "total_price",
        "total_ev",
        "total_score",
        "avg_conditional_start_prob",
        "high_risk_players",
        "recommended_captain",
        "player_names",
    ]
    write_csv(
        OUT_FORMATION_COMPARISON_CSV,
        formation_fieldnames,
        [{key: row.get(key, "") for key in formation_fieldnames} for row in formation_comparison_rows],
    )
    print(f"Skrevet: {OUT_STRATEGIES_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {OUT_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {OUT_FORMATION_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {OUT_CONTEXT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {OUT_CLEANUP_REPORT.relative_to(PROJECT_ROOT)}")
    if missing_formations:
        missing_text = "; ".join(f"{strategy}/{formation}" for strategy, formation in missing_formations)
        print(f"Manglende gyldige formationer: {missing_text}")
    else:
        print(f"Strategi x formation genereret: {len(STRATEGIES)} x {len(FORMATIONS)} = {len(formation_comparison_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
