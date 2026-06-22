from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from json_file_safety import write_json_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
TEAM_MARKET_PATH = DATA_DIR / "team_market_odds_layer_v1.csv"
OUT_PATH = DATA_DIR / "optimal_squads_by_formation.json"
DIAG_PATH = DATA_DIR / "clean_market_optimizer_diagnostics.csv"

BUDGET_M = 50.0
MIN_BUDGET_M = 49.0
MAX_PER_TEAM = 4

# Realistisk managerlogik:
# - ingen automatisk spiller under 3,0 mio.
# - maks 1 spiller under 3,5 mio.
# - maks 1 spiller fra hold uden for stÃ¦rk/langsigtet topgruppe
MIN_PLAYER_PRICE_M = 2.5
CHEAP_PLAYER_LIMIT_M = 3.5
MAX_CHEAP_PLAYERS = 11

LOW_MARKET_TEAM_THRESHOLD = 0.25
MAX_LOW_MARKET_TEAM_PLAYERS = 11

BEAM_WIDTH = 80
CANDIDATES_PER_POSITION = 45

FORMATIONS = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

# Sikrer at 5-back ikke vinder, bare fordi billige forsvarere fra store nationer scorer hÃ¸jt.
FORMATION_PENALTY = {
    "3-4-3": 0.0,
    "3-5-2": 0.2,
    "4-3-3": 0.0,
    "4-4-2": 0.0,
    "4-5-1": 3.5,
    "5-3-2": 5.5,
    "5-4-1": 8.0,
}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def clean_value(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def normalize_position(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "MÃ…LMAND": "GK",
        "MALMAND": "GK",
        "KEEPER": "GK",
        "GOALKEEPER": "GK",
        "FORSVAR": "DEF",
        "DEFENDER": "DEF",
        "DEFENSE": "DEF",
        "MIDTBANE": "MID",
        "MIDFIELDER": "MID",
        "MIDFIELD": "MID",
        "ANGRIBER": "FWD",
        "FORWARD": "FWD",
        "STRIKER": "FWD",
    }
    return aliases.get(text, text)


def get_price_m(row: pd.Series | dict[str, Any]) -> float:
    for col in ["price_m", "price_estimate_m", "price_mio"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return to_float(value)

    for col in ["price", "price_estimate", "holdet_price"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            num = to_float(value)
            return num / 1_000_000 if num > 1000 else num

    return 0.0


def get_model_ev(row: pd.Series | dict[str, Any]) -> float:
    # Brug helst base-EV fÃ¸r tidligere price/odds-lag.
    for col in [
        "model_ev_before_price_quality",
        "weighted_group_stage_ev_before_price_quality",
        "weighted_group_stage_ev",
        "optimizer_ev_base",
        "optimizer_ev",
    ]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return to_float(value)
    return 0.0


def get_start_prob(row: pd.Series | dict[str, Any]) -> float:
    for col in ["start_prob", "start_probability", "start_probability_pct"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            num = to_float(value)
            return num / 100 if num > 1 else num
    return 0.25


def load_players() -> pd.DataFrame:
    if not PLAYER_EV_PATH.exists():
        raise FileNotFoundError(f"Mangler {PLAYER_EV_PATH}")
    if not TEAM_MARKET_PATH.exists():
        raise FileNotFoundError(f"Mangler {TEAM_MARKET_PATH}. KÃ¸r build_market_odds_layer.py fÃ¸rst.")

    players = pd.read_csv(PLAYER_EV_PATH)
    market = pd.read_csv(TEAM_MARKET_PATH)

    players = players.copy()
    market = market.copy()

    players["team_id"] = players["team_id"].astype(str).str.upper().str.strip()
    players["position"] = players["position"].map(normalize_position)
    players["price_m"] = players.apply(get_price_m, axis=1)
    players["model_ev"] = players.apply(get_model_ev, axis=1)
    players["start_prob_display"] = players.apply(get_start_prob, axis=1)

    keep_market_cols = [
        "team_id",
        "team_market_score",
        "team_long_run_score",
        "team_group_stage_score",
        "team_attack_score",
        "winner_odds",
        "reach_qf_odds",
        "reach_sf_odds",
        "reach_final_odds",
        "group_win_odds",
        "highest_scoring_team_odds",
        "lowest_scoring_team_odds",
    ]
    keep_market_cols = [c for c in keep_market_cols if c in market.columns]
    market["team_id"] = market["team_id"].astype(str).str.upper().str.strip()

    df = players.merge(market[keep_market_cols], on="team_id", how="left")

    for col in ["team_market_score", "team_long_run_score", "team_group_stage_score", "team_attack_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["price_m"] = pd.to_numeric(df["price_m"], errors="coerce").fillna(0.0)
    df["model_ev"] = pd.to_numeric(df["model_ev"], errors="coerce").fillna(0.0)

    df = df.loc[df["price_m"] >= MIN_PLAYER_PRICE_M].copy()

    # Angrebs-/forsvarsprofil. NÃ¥r vi senere fylder highest scoring og lowest scoring,
    # slÃ¥r de igennem her uden ny kode.
    df["position_market_score"] = df["team_market_score"]
    df.loc[df["position"].isin(["FWD", "MID"]), "position_market_score"] = (
        0.65 * df["team_market_score"] + 0.35 * df["team_attack_score"]
    )
    df.loc[df["position"].isin(["GK", "DEF"]), "position_market_score"] = (
        0.75 * df["team_market_score"] + 0.25 * df["team_long_run_score"]
    )

    df["is_cheap_player"] = df["price_m"] < CHEAP_PLAYER_LIMIT_M
    df["is_low_market_team"] = df["team_market_score"] < LOW_MARKET_TEAM_THRESHOLD

    # Hovedlogik:
    # Pris og team-odds er baseline.
    # Model-EV er tiebreaker.
    # Start indgår som blød straf/bonus.
    #
    # Formål:
    # - Lav startchance skal straffes.
    # - Modellen må ikke blindt skifte til billige startere fra svagere hold.
    # - Budgetforbrug og stærke markedsmiljøer skal stadig dominere.
    df["start_prob_model"] = pd.to_numeric(df["start_prob_display"], errors="coerce").fillna(0.25)
    df.loc[df["start_prob_model"] > 1.0, "start_prob_model"] = df.loc[df["start_prob_model"] > 1.0, "start_prob_model"] / 100.0
    df["start_prob_model"] = df["start_prob_model"].clip(lower=0.0, upper=1.0)

    df["start_risk_penalty"] = (
        (0.70 - df["start_prob_model"]).clip(lower=0.0) * 3.00
        + (0.45 - df["start_prob_model"]).clip(lower=0.0) * 4.00
    )

    df["start_security_bonus"] = (
        (df["start_prob_model"] - 0.70).clip(lower=0.0) * 1.00
    )

    df["selection_score"] = (
        1.85 * df["price_m"]
        + 7.25 * df["position_market_score"]
        + 0.55 * df["model_ev"]
        - df["start_risk_penalty"]
        + df["start_security_bonus"]
    )

    # Billige spillere fra topnationer skal ikke automatisk blive value-picks.
    df["cheap_penalty"] = (CHEAP_PLAYER_LIMIT_M - df["price_m"]).clip(lower=0.0) * 6.0
    df["selection_score"] -= df["cheap_penalty"]

    return df


def formation_slots(formation: str) -> list[str]:
    counts = FORMATIONS[formation]
    slots: list[str] = []
    slots.extend(["GK"] * counts["GK"])
    slots.extend(["FWD"] * counts["FWD"])
    slots.extend(["MID"] * counts["MID"])
    slots.extend(["DEF"] * counts["DEF"])
    return slots


def min_remaining_price(slots_left: list[str], candidates_by_pos: dict[str, pd.DataFrame]) -> float:
    total = 0.0
    for pos in set(slots_left):
        need = slots_left.count(pos)
        prices = candidates_by_pos[pos]["price_m"].sort_values().head(need)
        if len(prices) < need:
            return math.inf
        total += float(prices.sum())
    return total


def row_to_player(row: pd.Series, formation: str) -> dict[str, Any]:
    item = {col: clean_value(row[col]) for col in row.index}

    price_m = float(row["price_m"])
    model_ev = float(row["model_ev"])
    score = float(row["selection_score"])

    item["player_id"] = str(row["player_id"])
    item["player_name"] = row["player_name"]
    item["team_id"] = row["team_id"]
    item["position"] = row["position"]
    item["price_m"] = price_m
    item["price"] = int(round(price_m * 1_000_000))
    item["price_estimate"] = int(round(price_m * 1_000_000))
    item["optimizer_ev"] = model_ev
    item["optimizer_ev_adj"] = score
    item["selected_formation"] = formation
    item["budget_m"] = BUDGET_M
    item["max_per_team"] = MAX_PER_TEAM
    item["solver_quality_profile"] = "clean_market_optimizer_v1"
    return item


def build_candidates_by_pos(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        part = df.loc[df["position"] == pos].copy()
        part = part.sort_values(["selection_score", "price_m"], ascending=[False, False])
        out[pos] = part.head(CANDIDATES_PER_POSITION).reset_index(drop=True)
    return out


def build_squad(formation: str, df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slots = formation_slots(formation)
    candidates_by_pos = build_candidates_by_pos(df)

    beam = [
        {
            "rows": [],
            "used_ids": set(),
            "team_counts": {},
            "price": 0.0,
            "score": 0.0,
            "cheap_count": 0,
            "low_market_count": 0,
        }
    ]

    for idx, pos in enumerate(slots):
        slots_left = slots[idx + 1:]
        min_left = min_remaining_price(slots_left, candidates_by_pos)
        new_beam = []

        for state in beam:
            pool = candidates_by_pos[pos]

            for _, row in pool.iterrows():
                player_id = str(row["player_id"])
                if player_id in state["used_ids"]:
                    continue

                team = str(row["team_id"])
                team_count = state["team_counts"].get(team, 0)
                if team_count >= MAX_PER_TEAM:
                    continue

                price = float(row["price_m"])
                new_price = state["price"] + price
                if new_price > BUDGET_M + 1e-9:
                    continue
                if new_price + min_left > BUDGET_M + 1e-9:
                    continue

                cheap_count = state["cheap_count"] + (1 if bool(row["is_cheap_player"]) else 0)
                if cheap_count > MAX_CHEAP_PLAYERS:
                    continue

                low_market_count = state["low_market_count"] + (1 if bool(row["is_low_market_team"]) else 0)
                if low_market_count > MAX_LOW_MARKET_TEAM_PLAYERS:
                    continue

                team_counts = dict(state["team_counts"])
                team_counts[team] = team_count + 1

                used_ids = set(state["used_ids"])
                used_ids.add(player_id)

                # Lille bonus for at nÃ¦rme sig fuldt budget.
                budget_bonus = 0.05 * new_price

                new_beam.append(
                    {
                        "rows": state["rows"] + [row],
                        "used_ids": used_ids,
                        "team_counts": team_counts,
                        "price": new_price,
                        "score": state["score"] + float(row["selection_score"]) + budget_bonus,
                        "cheap_count": cheap_count,
                        "low_market_count": low_market_count,
                    }
                )

        if not new_beam:
            raise RuntimeError(f"Ingen gyldig lÃ¸sning for {formation} ved slot {idx + 1}/{len(slots)} ({pos}).")

        new_beam.sort(
            key=lambda s: (
                s["price"] >= MIN_BUDGET_M,
                s["score"],
                s["price"],
                -s["cheap_count"],
                -s["low_market_count"],
            ),
            reverse=True,
        )

        beam = new_beam[:BEAM_WIDTH]

    finals = [s for s in beam if MIN_BUDGET_M <= s["price"] <= BUDGET_M]
    if not finals:
        finals = beam

    formation_penalty = FORMATION_PENALTY.get(formation, 0.0)

    finals.sort(
        key=lambda s: (
            s["price"] >= MIN_BUDGET_M,
            s["score"] - formation_penalty,
            s["price"],
            -s["cheap_count"],
            -s["low_market_count"],
        ),
        reverse=True,
    )

    best = finals[0]
    players = [row_to_player(row, formation) for row in best["rows"]]

    total_price = round(sum(float(p["price_m"]) for p in players), 3)
    raw_ev = round(sum(float(p["optimizer_ev"]) for p in players), 6)
    score_before_penalty = round(sum(float(p["optimizer_ev_adj"]) for p in players), 6)
    adj_score = round(score_before_penalty - formation_penalty, 6)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    players.sort(key=lambda p: (pos_order.get(p["position"], 9), p["player_name"]))

    for p in players:
        p["squad_total_price_m"] = total_price
        p["squad_total_ev"] = raw_ev
        p["squad_total_raw_ev"] = raw_ev
        p["squad_total_adj_ev_before_formation_penalty"] = score_before_penalty
        p["formation_score_penalty"] = formation_penalty
        p["squad_total_adj_ev"] = adj_score
        p["squad_cheap_player_count"] = best["cheap_count"]
        p["squad_low_market_team_count"] = best["low_market_count"]

    summary = {
        "formation": formation,
        "price_m": total_price,
        "raw_ev": raw_ev,
        "score_before_penalty": score_before_penalty,
        "formation_penalty": formation_penalty,
        "adj_score": adj_score,
        "cheap_count": best["cheap_count"],
        "low_market_count": best["low_market_count"],
        "teams": dict(sorted(best["team_counts"].items())),
    }

    return players, summary


def write_json(path: Path, data: Any) -> None:
    write_json_strict(path, data)


def main() -> None:
    df = load_players()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if OUT_PATH.exists():
        backup_path = OUT_PATH.with_name(f"optimal_squads_by_formation.backup_before_clean_market_{timestamp}.json")
        shutil.copy2(OUT_PATH, backup_path)
    else:
        backup_path = None

    output: dict[str, Any] = {}
    summaries = []
    diag_rows = []

    for formation in FORMATIONS:
        players, summary = build_squad(formation, df)
        output[formation] = players
        summaries.append(summary)

        for p in players:
            diag_rows.append(
                {
                    "formation": formation,
                    "player_name": p.get("player_name"),
                    "team_id": p.get("team_id"),
                    "position": p.get("position"),
                    "price_m": p.get("price_m"),
                    "optimizer_ev": p.get("optimizer_ev"),
                    "optimizer_ev_adj": p.get("optimizer_ev_adj"),
                    "team_market_score": p.get("team_market_score"),
                    "team_long_run_score": p.get("team_long_run_score"),
                    "team_attack_score": p.get("team_attack_score"),
                    "winner_odds": p.get("winner_odds"),
                    "is_cheap_player": p.get("is_cheap_player"),
                    "is_low_market_team": p.get("is_low_market_team"),
                }
            )

    write_json(OUT_PATH, output)
    pd.DataFrame(diag_rows).to_csv(DIAG_PATH, index=False, encoding="utf-8-sig")

    summaries = sorted(summaries, key=lambda x: x["adj_score"], reverse=True)

    print("")
    print("CLEAN MARKET OPTIMIZER")
    print(f"Budget: {BUDGET_M:.1f} mio.")
    print(f"Min budget: {MIN_BUDGET_M:.1f} mio.")
    print(f"Min spillerpris: {MIN_PLAYER_PRICE_M:.1f} mio.")
    print(f"Maks billige < {CHEAP_PLAYER_LIMIT_M:.1f} mio.: {MAX_CHEAP_PLAYERS}")
    print(f"Maks lav market-score-hold: {MAX_LOW_MARKET_TEAM_PLAYERS}")
    print("")

    for s in summaries:
        print(
            f"{s['formation']}: pris={s['price_m']:.1f}, "
            f"score={s['adj_score']:.3f}, rawEV={s['raw_ev']:.3f}, "
            f"cheap={s['cheap_count']}, lowMarket={s['low_market_count']}, "
            f"teams={s['teams']}"
        )

    best = summaries[0]
    print("")
    print("Bedste formation:", best["formation"])
    print("Bedste hold:")
    for p in output[best["formation"]]:
        print(
            f"- {p['player_name']} | {p['team_id']} | {p['position']} "
            f"| pris={p['price_m']} | odds={p.get('winner_odds')} "
            f"| market={to_float(p.get('team_market_score')):.3f}"
        )

    print("")
    if backup_path:
        print("Backup:")
        print(backup_path)
    print("")
    print("Skrev:")
    print(OUT_PATH)
    print(DIAG_PATH)


if __name__ == "__main__":
    main()



