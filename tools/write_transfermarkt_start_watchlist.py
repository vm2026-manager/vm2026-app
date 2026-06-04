from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TM_DIR = DATA_DIR / "transfermarkt_national_team"

OPTIMAL_PATH = DATA_DIR / "optimal_squads_by_formation.json"
TM_SUMMARY_PATH = TM_DIR / "player_national_team_usage_transfermarkt.csv"
TM_COMP_PATH = TM_DIR / "player_national_team_usage_by_competition.csv"
TM_COMPETITIVE_SUMMARY_PATH = TM_DIR / "player_national_team_usage_competitive_summary.csv"

OUT_CSV = TM_DIR / "selected_squad_transfermarkt_watchlist.csv"
OUT_TXT = TM_DIR / "selected_squad_transfermarkt_watchlist.txt"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        for key in ["players", "squad", "rows"]:
            if isinstance(value.get(key), list):
                return value[key]

    return []


def get_start_prob(row: dict[str, Any]) -> float:
    for key in [
        "start_prob_model",
        "start_prob_display",
        "start_prob",
        "start_probability",
        "start_probability_pct",
    ]:
        if key in row and row.get(key) not in [None, ""]:
            value = to_float(row.get(key), 0.0)
            if value > 1:
                value = value / 100
            return max(0.0, min(1.0, value))

    return 0.0


def get_best_squad() -> tuple[str, list[dict[str, Any]], float]:
    data = load_json(OPTIMAL_PATH)

    best_formation = ""
    best_rows: list[dict[str, Any]] = []
    best_score = -10**9

    for formation, value in data.items():
        rows = get_rows(value)
        if not rows:
            continue

        score = max(to_float(r.get("squad_total_adj_ev"), 0.0) for r in rows)
        if score > best_score:
            best_score = score
            best_formation = formation
            best_rows = rows

    return best_formation, best_rows, best_score


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)

    return df


def first_row_by_player(df: pd.DataFrame, player_id: str) -> dict[str, Any]:
    if df.empty or "player_id" not in df.columns:
        return {}

    rows = df.loc[df["player_id"].astype(str) == str(player_id)]

    if rows.empty:
        return {}

    return rows.iloc[0].to_dict()


def get_comp_value(
    comp: pd.DataFrame,
    player_id: str,
    category_label_contains: str,
    value_col: str,
    default: float = 0.0,
) -> float:
    if comp.empty:
        return default

    if "player_id" not in comp.columns:
        return default

    if "competition_category_label" not in comp.columns:
        return default

    if value_col not in comp.columns:
        return default

    rows = comp.loc[
        (comp["player_id"].astype(str) == str(player_id))
        & comp["competition_category_label"].astype(str).str.contains(
            category_label_contains,
            case=False,
            na=False,
        )
    ]

    if rows.empty:
        return default

    return to_float(rows[value_col].max(), default)


def get_comp_sum(
    comp: pd.DataFrame,
    player_id: str,
    value_col: str,
    default: float = 0.0,
) -> float:
    if comp.empty:
        return default

    if "player_id" not in comp.columns or value_col not in comp.columns:
        return default

    rows = comp.loc[comp["player_id"].astype(str) == str(player_id)]

    if rows.empty:
        return default

    return to_float(rows[value_col].sum(), default)


def assess_watch_reason(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []

    model_start = to_float(row.get("model_start_share"))
    usage = to_float(row.get("tm_usage_score"))
    recent_20 = to_float(row.get("tm_recent_20_start_share"))
    recency_comp = to_float(row.get("tm_recency_weighted_competitive_start_score"))

    finals_start = to_float(row.get("tm_finals_start_share"))
    qual_start = to_float(row.get("tm_qualification_start_share"))

    finals_recency = to_float(row.get("tm_finals_recency_start_share"))
    qual_recency = to_float(row.get("tm_qualification_recency_start_share"))

    caps = to_float(row.get("tm_caps"))
    not_in_squad = to_float(row.get("tm_recent_20_not_in_squad"))
    bench = to_float(row.get("tm_recent_20_on_bench"))
    unknown_abs = to_float(row.get("tm_not_selected_or_unknown_absences"))

    strong_recency = (
        recency_comp >= 0.70
        or finals_recency >= 0.80
        or qual_recency >= 0.80
    )

    strong_usage = (
        usage >= 0.75
        or recent_20 >= 0.80
        or strong_recency
        or finals_start >= 0.85
        or qual_start >= 0.85
    )

    weak_recent_signal = (
        usage < 0.55
        and recent_20 < 0.55
        and recency_comp < 0.65
    )

    if model_start >= 0.70 and usage < 0.45 and not strong_recency:
        reasons.append("Model-start høj, men samlet Transfermarkt-usage lav")

    if model_start >= 0.70 and recent_20 < 0.50 and recency_comp < 0.70:
        reasons.append("Model-start høj, men nyere landsholdsbrug er usikker/lav")

    if model_start >= 0.70 and recency_comp < 0.50:
        reasons.append("Model-start høj, men recency-vægtet konkurrencesignal er lavt")

    if finals_recency > 0 and finals_recency < 0.50 and recency_comp < 0.70:
        reasons.append("Lav recency-vægtet slutrunde-startandel")

    if qual_recency > 0 and qual_recency < 0.50 and recency_comp < 0.70:
        reasons.append("Lav recency-vægtet kval-startandel")

    if finals_start > 0 and finals_start < 0.50 and not strong_recency:
        reasons.append("Lav historisk slutrunde-startandel")

    if qual_start > 0 and qual_start < 0.50 and not strong_recency:
        reasons.append("Lav historisk kval-startandel")

    if not_in_squad >= 4 and not strong_usage:
        reasons.append("Ofte ikke i trup seneste 20")

    if bench >= 4 and not strong_usage:
        reasons.append("Ofte på bænken seneste 20")

    if unknown_abs >= 3 and weak_recent_signal:
        reasons.append("Flere fravær uden skade-/karantæneforklaring og svagt nyere usage-signal")

    # Få landskampe skal kun flagges, hvis recency-signalet ikke allerede er stærkt.
    if caps < 15 and model_start >= 0.65 and not strong_recency:
        reasons.append("Relativt få landskampe ift. høj model-start")

    flag = "JA" if reasons else ""

    return flag, "; ".join(reasons)


def main() -> None:
    if not OPTIMAL_PATH.exists():
        raise FileNotFoundError(f"Mangler {OPTIMAL_PATH}")

    formation, squad, score = get_best_squad()

    tm = load_csv(TM_SUMMARY_PATH)
    comp = load_csv(TM_COMP_PATH)
    comp_player = load_csv(TM_COMPETITIVE_SUMMARY_PATH)

    rows = []

    for player in squad:
        player_id = str(player.get("player_id", ""))

        tm_row = first_row_by_player(tm, player_id)
        comp_player_row = first_row_by_player(comp_player, player_id)

        model_start_share = get_start_prob(player)

        out = {
            "formation": formation,
            "player_id": player_id,
            "player_name": player.get("player_name"),
            "team_id": player.get("team_id"),
            "position": player.get("position"),
            "price_m": player.get("price_m"),
            "model_start_share": round(model_start_share, 3),
            "model_start_pct": round(model_start_share * 100, 1),
            "team_market_score": player.get("team_market_score"),
            "winner_odds": player.get("winner_odds"),
            "optimizer_ev_adj": player.get("optimizer_ev_adj"),
            "tm_caps": tm_row.get("tm_caps", ""),
            "tm_goals": tm_row.get("tm_goals", ""),
            "tm_recent_20_start_share": tm_row.get("recent_20_start_share", ""),
            "tm_recent_10_start_share": tm_row.get("recent_10_start_share", ""),
            "tm_recent_20_on_bench": tm_row.get("recent_20_on_bench", ""),
            "tm_recent_20_not_in_squad": tm_row.get("recent_20_not_in_squad", ""),
            "tm_usage_score": tm_row.get("national_team_usage_score", ""),
            "tm_last_national_row_date": tm_row.get("last_national_row_date", ""),
            "tm_url": tm_row.get("transfermarkt_national_url", ""),
            "tm_weighted_competitive_start_score": comp_player_row.get("tm_weighted_competitive_start_score", ""),
            "tm_recency_weighted_competitive_start_score": comp_player_row.get("tm_recency_weighted_competitive_start_score", ""),
            "tm_finals_start_share": get_comp_value(comp, player_id, "Slutrunde", "start_share", 0.0),
            "tm_qualification_start_share": get_comp_value(comp, player_id, "Kvalifikation", "start_share", 0.0),
            "tm_nations_league_start_share": get_comp_value(comp, player_id, "Nations League", "start_share", 0.0),
            "tm_friendly_start_share": get_comp_value(comp, player_id, "Venskabskamp", "start_share", 0.0),
            "tm_finals_recency_start_share": get_comp_value(comp, player_id, "Slutrunde", "recency_weighted_start_share", 0.0),
            "tm_qualification_recency_start_share": get_comp_value(comp, player_id, "Kvalifikation", "recency_weighted_start_share", 0.0),
            "tm_nations_league_recency_start_share": get_comp_value(comp, player_id, "Nations League", "recency_weighted_start_share", 0.0),
            "tm_friendly_recency_start_share": get_comp_value(comp, player_id, "Venskabskamp", "recency_weighted_start_share", 0.0),
            "tm_injury_or_fitness_absences": get_comp_sum(comp, player_id, "injury_or_fitness_absences", 0.0),
            "tm_suspension_absences": get_comp_sum(comp, player_id, "suspension_absences", 0.0),
            "tm_not_selected_or_unknown_absences": get_comp_sum(comp, player_id, "not_selected_or_unknown_absences", 0.0),
            "manual_note": "",
        }

        flag, reason = assess_watch_reason(out)
        out["watch_flag"] = flag
        out["watch_reason"] = reason

        rows.append(out)

    df = pd.DataFrame(rows)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["_pos_order"] = df["position"].map(pos_order).fillna(9)
    df = df.sort_values(["_pos_order", "team_id", "player_name"]).drop(columns=["_pos_order"])

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("SELECTED SQUAD TRANSFERMARKT WATCHLIST")
    lines.append("=" * 90)
    lines.append(f"Formation: {formation}")
    lines.append(f"Optimizer score: {score:.3f}")
    lines.append("")

    watch = df.loc[df["watch_flag"] == "JA"].copy()

    if watch.empty:
        lines.append("Ingen klare Transfermarkt-røde flag i nuværende bedste hold.")
    else:
        lines.append("Spillere til manuel vurdering:")
        lines.append("")

        for _, r in watch.iterrows():
            lines.append(
                f"- {r['player_name']} | {r['team_id']} | {r['position']} | "
                f"model-start={r['model_start_pct']}% | "
                f"TM usage={r['tm_usage_score']} | "
                f"recent20_start={r['tm_recent_20_start_share']} | "
                f"recency_comp={r['tm_recency_weighted_competitive_start_score']} | "
                f"slutrunde_recency={r['tm_finals_recency_start_share']} | "
                f"kval_recency={r['tm_qualification_recency_start_share']} | "
                f"{r['watch_reason']}"
            )

    lines.append("")
    lines.append("Note:")
    lines.append("- Watchlisten ændrer ikke modellen.")
    lines.append("- Brug den som beslutningsstøtte indtil trupper og sidste testkampe er kendt.")
    lines.append("- Recency vægter nyere landsholdskampe højere end ældre kampe.")
    lines.append("- Slutrunde/kvalkampe vægter fagligt højere end venskabskampe.")
    lines.append("- Fravær pga. skade/karantæne bør tolkes anderledes end ukendt fravær/vragning.")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Skrev: {OUT_CSV}")
    print(f"Skrev: {OUT_TXT}")


if __name__ == "__main__":
    main()