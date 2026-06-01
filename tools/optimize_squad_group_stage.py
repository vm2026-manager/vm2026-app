from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
FIXTURE_MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
EV_IMPACT_PATH = DATA_DIR / "player_ev_fixture_strength_impact_report.csv"
TEAM_MARKET_CSV_PATH = DATA_DIR / "team_market_odds_layer_v1.csv"
TEAM_MARKET_JSON_PATH = DATA_DIR / "team_market_odds_layer_v1.json"

OUT_STRATEGIES_JSON = DATA_DIR / "optimal_squads_by_strategy.json"
OUT_COMPARISON_CSV = DATA_DIR / "strategy_comparison_report.csv"

BUDGET_M = 50.0
SQUAD_SIZE = 11
MAX_PER_TEAM = 4

FORMATIONS: dict[str, dict[str, int]] = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

STRATEGIES = [
    "balanced",
    "safe_starters",
    "fixture_attack",
    "clean_sheet_stack",
    "long_run_value",
]

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
    text = txt(value).replace(",", ".")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def normalize_price_to_millions(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return values
    if float(non_null.median()) > 1000:
        return values / 1_000_000
    return values


def standardize_position(value: Any) -> str:
    raw = txt(value).upper()
    return POSITION_MAP.get(raw, raw)


def slug(value: Any) -> str:
    text = txt(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def load_player_pool_layer() -> pd.DataFrame:
    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    pool = pd.DataFrame(data)
    if pool.empty:
        return pd.DataFrame(columns=["player_id"])

    keep_cols = [
        "player_id",
        "conditional_start_prob",
        "availability_prob",
        "availability_risk",
        "availability_status",
        "start_status",
        "start_prob",
    ]
    existing = [col for col in keep_cols if col in pool.columns]
    out = pool[existing].copy()
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return out


def load_ev_impact() -> pd.DataFrame:
    if not EV_IMPACT_PATH.exists():
        return pd.DataFrame(columns=["player_id", "ev_diff", "ev_diff_pct"])
    impact = pd.read_csv(EV_IMPACT_PATH)
    impact["ev_diff"] = pd.to_numeric(impact.get("ev_diff", 0.0), errors="coerce").fillna(0.0)
    impact["ev_diff_pct"] = pd.to_numeric(impact.get("ev_diff_pct", 0.0), errors="coerce").fillna(0.0)
    return impact[["player_id", "ev_diff", "ev_diff_pct"]].drop_duplicates(subset=["player_id"], keep="first")


def load_team_market_scores() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if TEAM_MARKET_CSV_PATH.exists():
        with TEAM_MARKET_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif TEAM_MARKET_JSON_PATH.exists():
        with TEAM_MARKET_JSON_PATH.open(encoding="utf-8-sig") as f:
            rows = json.load(f)

    if not rows:
        return pd.DataFrame(columns=["team_id", "team_long_run_score", "team_market_score", "team_attack_score"])

    market = pd.DataFrame(rows)
    market["team_id"] = market["team_id"].astype(str).str.strip().str.upper()
    for col in ["team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]:
        if col not in market.columns:
            market[col] = 0.0
        market[col] = pd.to_numeric(market[col], errors="coerce").fillna(0.0)
    return market[["team_id", "team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]]


def load_clean_sheet_team_scores() -> pd.DataFrame:
    if not FIXTURE_MULTIPLIERS_PATH.exists():
        return pd.DataFrame(columns=["team_id", "avg_clean_sheet_multiplier", "max_clean_sheet_multiplier"])

    rows = []
    with FIXTURE_MULTIPLIERS_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"team_id": txt(row.get("home")).upper(), "cs_mult": to_float(row.get("home_clean_sheet_multiplier"), 1.0)})
            rows.append({"team_id": txt(row.get("away")).upper(), "cs_mult": to_float(row.get("away_clean_sheet_multiplier"), 1.0)})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["team_id", "avg_clean_sheet_multiplier", "max_clean_sheet_multiplier"])
    grouped = df.groupby("team_id")["cs_mult"].agg(["mean", "max"]).reset_index()
    grouped = grouped.rename(columns={"mean": "avg_clean_sheet_multiplier", "max": "max_clean_sheet_multiplier"})
    return grouped


def apply_light_optimizer_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["base_ev"] = pd.to_numeric(work["optimizer_ev"], errors="coerce").fillna(0.0)
    work["start_prob"] = pd.to_numeric(work["start_prob"], errors="coerce").fillna(0.48).clip(0.0, 1.0)
    work["minute_share"] = pd.to_numeric(work.get("minute_share", 0.0), errors="coerce").fillna(0.0)

    reliability = 0.96 + 0.025 * work["start_prob"] + 0.015 * (work["minute_share"] / 0.09).clip(0.0, 1.0)
    work["balanced_score"] = (work["base_ev"] * reliability).clip(lower=0.0)

    mid_mask = (work["position"] == "MID") & (work["balanced_score"] >= 1.0) & (work["start_prob"] >= 0.45)
    work.loc[mid_mask, "balanced_score"] = work.loc[mid_mask, "balanced_score"] + 0.08

    team_total = work.groupby("team_id")["balanced_score"].transform("sum")
    team_share = (work["balanced_score"] / team_total).fillna(0.0)
    penalty = (1.0 - 0.8 * (team_share - 0.14).clip(lower=0.0)).clip(lower=0.88, upper=1.0)
    work["balanced_score"] = work["balanced_score"] * penalty
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

    players = players.merge(load_ev_impact(), on="player_id", how="left")
    players["ev_diff"] = pd.to_numeric(players.get("ev_diff"), errors="coerce").fillna(0.0)
    players["ev_diff_pct"] = pd.to_numeric(players.get("ev_diff_pct"), errors="coerce").fillna(0.0)

    players = players.merge(load_clean_sheet_team_scores(), on="team_id", how="left")
    players["avg_clean_sheet_multiplier"] = pd.to_numeric(players.get("avg_clean_sheet_multiplier"), errors="coerce").fillna(1.0)
    players["max_clean_sheet_multiplier"] = pd.to_numeric(players.get("max_clean_sheet_multiplier"), errors="coerce").fillna(1.0)

    players = players.merge(load_team_market_scores(), on="team_id", how="left")
    for col in ["team_long_run_score", "team_market_score", "team_attack_score", "team_group_stage_score"]:
        players[col] = pd.to_numeric(players.get(col), errors="coerce").fillna(0.0)

    players = players.dropna(subset=["player_id", "player_name", "team_id", "position", "price_m"]).copy()
    players = players[players["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()
    players = players[players["price_m"] > 0].copy()
    players = apply_light_optimizer_adjustments(players)
    return players.reset_index(drop=True)


def add_strategy_scores(players: pd.DataFrame) -> pd.DataFrame:
    work = players.copy()
    base = work["balanced_score"]

    safe_bonus = 0.18 * ((work["conditional_start_prob"] >= 0.85) & work["availability_risk"].isin(["low_risk", "medium_risk"])).astype(float)
    safe_penalty = 0.24 * (work["availability_risk"].eq("high_risk")).astype(float)
    safe_penalty += 0.18 * (work["conditional_start_prob"] < 0.65).astype(float)
    cheap_reserve_penalty = 0.08 * ((work["price_m"] <= 2.5) & (work["balanced_score"] < 1.0)).astype(float)
    work["score_safe_starters"] = (base + safe_bonus - safe_penalty - cheap_reserve_penalty).clip(lower=0.0)

    offensive = work["position"].isin(["MID", "FWD"]).astype(float)
    attack_bonus = work["ev_diff"].clip(lower=0.0) * (0.70 + 0.45 * offensive)
    attack_bonus += 0.12 * work["team_attack_score"] * offensive
    attack_penalty = (-work["ev_diff"].clip(upper=0.0)) * (0.20 + 0.25 * offensive)
    work["score_fixture_attack"] = (base + attack_bonus - attack_penalty).clip(lower=0.0)

    defensive = work["position"].isin(["GK", "DEF"]).astype(float)
    defensive_clean_sheet_bonus = (
        (work["avg_clean_sheet_multiplier"] - 1.0).clip(lower=0.0) * 2.20
        + (work["max_clean_sheet_multiplier"] - 1.0).clip(lower=0.0) * 0.70
    )
    cs_start_bonus = 0.10 * work["conditional_start_prob"]
    cs_risk_penalty = (
        0.18 * work["availability_risk"].eq("high_risk").astype(float)
        + 0.10 * (work["conditional_start_prob"] < 0.65).astype(float)
    )
    weak_cs_penalty = (1.0 - work["avg_clean_sheet_multiplier"]).clip(lower=0.0) * 0.60
    work["score_clean_sheet_stack"] = (
        base
        + defensive * (0.35 * defensive_clean_sheet_bonus + cs_start_bonus - cs_risk_penalty - weak_cs_penalty)
    ).clip(lower=0.0)

    long_bonus = 0.45 * work["team_long_run_score"] + 0.18 * work["team_market_score"]
    risk_penalty = 0.16 * work["availability_risk"].eq("high_risk").astype(float)
    work["score_long_run_value"] = (base + long_bonus - risk_penalty).clip(lower=0.0)

    work["score_balanced"] = base
    return work


def strategy_score_column(strategy: str) -> str:
    return f"score_{strategy}"


def solve_formation(players: pd.DataFrame, strategy: str, formation_name: str, formation: dict[str, int]) -> pd.DataFrame:
    score_col = strategy_score_column(strategy)
    problem = pulp.LpProblem(f"{strategy}_{formation_name.replace('-', '_')}", pulp.LpMaximize)
    variables = {idx: pulp.LpVariable(f"pick_{idx}", lowBound=0, upBound=1, cat="Binary") for idx in players.index}

    problem += pulp.lpSum(float(players.loc[idx, score_col]) * variables[idx] for idx in players.index)
    problem += pulp.lpSum(variables[idx] for idx in players.index) == SQUAD_SIZE
    problem += pulp.lpSum(float(players.loc[idx, "price_m"]) * variables[idx] for idx in players.index) <= BUDGET_M

    for pos, count in formation.items():
        indices = players.index[players["position"] == pos].tolist()
        problem += pulp.lpSum(variables[idx] for idx in indices) == count

    for team_id, sub in players.groupby("team_id"):
        problem += pulp.lpSum(variables[idx] for idx in sub.index.tolist()) <= MAX_PER_TEAM

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    if pulp.LpStatus[problem.status] != "Optimal":
        return pd.DataFrame()

    picked = [idx for idx, variable in variables.items() if variable.value() == 1]
    squad = players.loc[picked].copy()
    squad["strategy"] = strategy
    squad["selected_formation"] = formation_name
    squad["strategy_score"] = squad[score_col]
    squad = squad.sort_values(["position", "strategy_score", "optimizer_ev"], ascending=[True, False, False]).reset_index(drop=True)
    return squad


def squad_summary(strategy: str, squad: pd.DataFrame) -> dict[str, Any]:
    teams = squad["team_id"].value_counts().sort_index()
    return {
        "strategy": strategy,
        "formation": txt(squad["selected_formation"].iloc[0]) if not squad.empty else "",
        "total_score": round(float(squad["strategy_score"].sum()), 6) if not squad.empty else 0.0,
        "total_ev": round(float(squad["optimizer_ev"].sum()), 6) if not squad.empty else 0.0,
        "total_price": int(round(float(squad["price_m"].sum()) * 1_000_000)) if not squad.empty else 0,
        "avg_start_prob": round(float(squad["start_prob"].mean()), 4) if not squad.empty else 0.0,
        "avg_conditional_start_prob": round(float(squad["conditional_start_prob"].mean()), 4) if not squad.empty else 0.0,
        "high_risk_players": int((squad["availability_risk"] == "high_risk").sum()) if not squad.empty else 0,
        "teams_summary": "; ".join(f"{team}:{count}" for team, count in teams.items()),
        "player_names": "; ".join(squad["player_name"].astype(str).tolist()),
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
        "optimizer_ev",
        "weighted_group_stage_ev",
        "ev_diff",
        "avg_clean_sheet_multiplier",
        "team_long_run_score",
        "strategy_score",
        "selected_formation",
    ]
    existing = [col for col in keep if col in squad.columns]
    records = squad[existing].copy()
    return json.loads(records.to_json(orient="records", force_ascii=False))


def main() -> int:
    players = add_strategy_scores(load_players())
    all_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []

    print(f"Optimizer-pool spillere: {len(players)}")
    print(f"Budget: {BUDGET_M:.1f} mio. | Maks pr. land: {MAX_PER_TEAM}")

    for strategy in STRATEGIES:
        best_squad = pd.DataFrame()
        best_summary: dict[str, Any] | None = None
        formation_records: dict[str, list[dict[str, Any]]] = {}

        for formation_name, formation in FORMATIONS.items():
            squad = solve_formation(players, strategy, formation_name, formation)
            formation_records[formation_name] = squad_records(squad) if not squad.empty else []
            if squad.empty:
                continue
            summary = squad_summary(strategy, squad)
            if best_summary is None or float(summary["total_score"]) > float(best_summary["total_score"]):
                best_summary = summary
                best_squad = squad.copy()

        if best_summary is None:
            best_summary = {
                "strategy": strategy,
                "formation": "",
                "total_score": 0.0,
                "total_ev": 0.0,
                "total_price": 0,
                "avg_start_prob": 0.0,
                "avg_conditional_start_prob": 0.0,
                "high_risk_players": 0,
                "teams_summary": "",
                "player_names": "",
            }

        comparison_rows.append(best_summary)
        all_results[strategy] = {
            "best_summary": best_summary,
            "best_squad": squad_records(best_squad) if not best_squad.empty else [],
            "formations": formation_records,
        }

        print(
            f"{strategy}: formation={best_summary['formation']}, "
            f"pris={best_summary['total_price']:,}, score={best_summary['total_score']:.3f}, "
            f"EV={best_summary['total_ev']:.3f}, high_risk={best_summary['high_risk_players']}"
        )

    with OUT_STRATEGIES_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with OUT_COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "strategy",
            "formation",
            "total_score",
            "total_ev",
            "total_price",
            "avg_start_prob",
            "avg_conditional_start_prob",
            "high_risk_players",
            "teams_summary",
            "player_names",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Skrevet: {OUT_STRATEGIES_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {OUT_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
