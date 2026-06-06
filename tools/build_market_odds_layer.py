from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OUTRIGHT_PATH = DATA_DIR / "worldcup_outright_odds.csv"

TEAM_MARKET_MANUAL_PATH = DATA_DIR / "team_market_odds_manual.csv"
TEAM_LAYER_CSV_PATH = DATA_DIR / "team_market_odds_layer_v1.csv"
TEAM_LAYER_JSON_PATH = DATA_DIR / "team_market_odds_layer_v1.json"


TEAM_MARKET_COLUMNS = [
    "team_id",
    "team_name",
    "winner_odds",
    "reach_qf_odds",
    "reach_sf_odds",
    "reach_final_odds",
    "group_win_odds",
    "highest_scoring_team_odds",
    "lowest_scoring_team_odds",
    "source",
    "snapshot_date",
]


def to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = str(value).replace(",", ".").strip()
        if not text:
            return None
        num = float(text)
        if num <= 0:
            return None
        return num
    except Exception:
        return None


def implied_prob_from_odds(value: Any) -> float | None:
    odds = to_float(value)
    if not odds:
        return None
    return 1.0 / odds


def normalize_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    max_value = s.max(skipna=True)

    if pd.isna(max_value) or max_value <= 0:
        return pd.Series([0.0] * len(series), index=series.index)

    return (s / max_value).fillna(0.0)


def create_manual_team_market_file_if_missing() -> None:
    if TEAM_MARKET_MANUAL_PATH.exists():
        print(f"Findes allerede: {TEAM_MARKET_MANUAL_PATH}")
        return

    if not OUTRIGHT_PATH.exists():
        raise FileNotFoundError(f"Mangler {OUTRIGHT_PATH}")

    outright = pd.read_csv(OUTRIGHT_PATH)

    required = {"team_id", "team_name", "unibet_win_odds"}
    missing = required - set(outright.columns)
    if missing:
        raise ValueError(f"{OUTRIGHT_PATH} mangler kolonner: {sorted(missing)}")

    out = pd.DataFrame()
    out["team_id"] = outright["team_id"]
    out["team_name"] = outright["team_name"]
    out["winner_odds"] = outright["unibet_win_odds"]

    for col in TEAM_MARKET_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["source"] = "unibet/bet365/manual"
    out["snapshot_date"] = "2026-05-16"

    out = out[TEAM_MARKET_COLUMNS]
    out.to_csv(TEAM_MARKET_MANUAL_PATH, index=False, encoding="utf-8-sig")

    print(f"Oprettede: {TEAM_MARKET_MANUAL_PATH}")


def build_team_market_layer() -> pd.DataFrame:
    df = pd.read_csv(TEAM_MARKET_MANUAL_PATH)
    df = df.copy()

    for col in TEAM_MARKET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    odds_cols = [
        "winner_odds",
        "reach_qf_odds",
        "reach_sf_odds",
        "reach_final_odds",
        "group_win_odds",
        "highest_scoring_team_odds",
        "lowest_scoring_team_odds",
    ]

    for col in odds_cols:
        prob_col = col.replace("_odds", "_prob")
        score_col = col.replace("_odds", "_score")

        df[prob_col] = df[col].map(implied_prob_from_odds)
        df[score_col] = normalize_series(df[prob_col])

    # Langsigtet turneringsscore:
    # To-reach-markederne er bedre end kun vinderodds, men winner bruges som fallback.
    long_run_components = {
        # Fantasy long-run should care more about reaching useful future rounds
        # than pure tournament winner odds.
        "winner_score": 0.15,
        "reach_qf_score": 0.35,
        "reach_sf_score": 0.25,
        "reach_final_score": 0.20,
        "group_win_score": 0.05,
    }

    group_stage_components = {
        # Group-stage value should be driven mainly by group strength/path,
        # with reach-QF as a better practical signal than pure winner odds.
        "group_win_score": 0.55,
        "reach_qf_score": 0.25,
        "highest_scoring_team_score": 0.15,
        "winner_score": 0.05,
    }

    attack_components = {
        # Attacking environment should be based on scoring market first,
        # then group path and broad team strength.
        "highest_scoring_team_score": 0.50,
        "group_win_score": 0.25,
        "reach_qf_score": 0.15,
        "winner_score": 0.10,
    }

    def weighted_available(row: pd.Series, components: dict[str, float]) -> float:
        total_weight = 0.0
        total_value = 0.0

        for col, weight in components.items():
            value = float(row.get(col, 0.0) or 0.0)
            if value > 0:
                total_value += value * weight
                total_weight += weight

        if total_weight <= 0:
            return 0.0

        return total_value / total_weight

    df["team_long_run_score"] = df.apply(
        lambda row: weighted_available(row, long_run_components),
        axis=1,
    )

    df["team_group_stage_score"] = df.apply(
        lambda row: weighted_available(row, group_stage_components),
        axis=1,
    )

    df["team_attack_score"] = df.apply(
        lambda row: weighted_available(row, attack_components),
        axis=1,
    )

    # Lav odds for lowest scoring team = dårligt angreb.
    # Jo lavere odds på at score færrest, desto større angrebsstraf.
    df["team_low_scoring_risk"] = df["lowest_scoring_team_score"]

    # Samlet praktisk score til første modelversion.
    df["team_market_score"] = (
        0.50 * df["team_long_run_score"]
        + 0.30 * df["team_group_stage_score"]
        + 0.20 * df["team_attack_score"]
        - 0.15 * df["team_low_scoring_risk"]
    ).clip(lower=0.0)

    df = df.sort_values("team_market_score", ascending=False).reset_index(drop=True)

    return df


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    create_manual_team_market_file_if_missing()

    layer = build_team_market_layer()

    layer.to_csv(TEAM_LAYER_CSV_PATH, index=False, encoding="utf-8-sig")
    write_json(TEAM_LAYER_JSON_PATH, layer.where(pd.notna(layer), None).to_dict(orient="records"))

    print("")
    print("TEAM MARKET ODDS LAYER")
    print(f"Hold: {len(layer)}")
    print("")
    print("Top 20:")
    cols = [
        "team_id",
        "team_name",
        "winner_odds",
        "reach_qf_odds",
        "reach_sf_odds",
        "reach_final_odds",
        "group_win_odds",
        "highest_scoring_team_odds",
        "lowest_scoring_team_odds",
        "team_long_run_score",
        "team_group_stage_score",
        "team_attack_score",
        "team_low_scoring_risk",
        "team_market_score",
    ]
    print(layer[cols].head(20).to_string(index=False))

    print("")
    print("Skrev:")
    print(TEAM_MARKET_MANUAL_PATH)
    print(TEAM_LAYER_CSV_PATH)
    print(TEAM_LAYER_JSON_PATH)


if __name__ == "__main__":
    main()