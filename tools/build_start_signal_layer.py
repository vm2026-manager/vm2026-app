from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_CSV = DATA_DIR / "player_ev_group_stage_v1.csv"
OUTPUT_CSV = DATA_DIR / "player_start_signal_layer_v1.csv"
OUTPUT_JSON = DATA_DIR / "player_start_signal_layer_v1.json"

POSITION_MAP = {
    "Målmand": "GK",
    "Forsvar": "DEF",
    "Midtbane": "MID",
    "Angriber": "FWD",
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

POSITION_SLOT_TARGETS = {
    "GK": 1,
    "DEF": 4,
    "MID": 4,
    "FWD": 2,
}

POSITION_WEIGHT = {
    "GK": 1.00,
    "DEF": 1.00,
    "MID": 1.05,
    "FWD": 1.05,
}


def standardize_positions(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().map(POSITION_MAP)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=a.index, dtype=float)
    mask = b.fillna(0.0) != 0
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out.fillna(0.0)


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Mangler inputfil: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV).copy()
    df["team_id"] = df["team_id"].astype(str).str.strip()
    df["position"] = standardize_positions(df["position"])

    numeric_cols = [
        "start_prob",
        "minute_share",
        "weighted_group_stage_ev",
        "total_ev_group_stage",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def build_base_signals(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["start_prob_clipped"] = work["start_prob"].clip(0.0, 1.0)
    work["minute_share_clipped"] = (work["minute_share"] / 0.09).clip(0.0, 1.0)

    team_ev_total = work.groupby("team_id")["weighted_group_stage_ev"].transform("sum")
    work["team_ev_share"] = safe_divide(work["weighted_group_stage_ev"], team_ev_total).clip(0.0, 1.0)

    pos_ev_total = work.groupby(["team_id", "position"])["weighted_group_stage_ev"].transform("sum")
    work["team_position_ev_share"] = safe_divide(work["weighted_group_stage_ev"], pos_ev_total).clip(0.0, 1.0)

    work["start_signal_raw"] = (
        0.58 * work["start_prob_clipped"]
        + 0.27 * work["minute_share_clipped"]
        + 0.10 * work["team_position_ev_share"]
        + 0.05 * work["team_ev_share"]
    )

    work["start_signal_raw"] = (
        work["start_signal_raw"]
        * work["position"].map(POSITION_WEIGHT).fillna(1.0)
    ).clip(0.0, 1.0)

    return work


def add_team_position_rankings(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work = work.sort_values(
        ["team_id", "position", "start_signal_raw", "start_prob_clipped", "minute_share_clipped", "weighted_group_stage_ev"],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)

    work["team_position_rank"] = (
        work.groupby(["team_id", "position"]).cumcount() + 1
    )

    work["position_slot_target"] = work["position"].map(POSITION_SLOT_TARGETS).fillna(99)

    work["inside_expected_slot_band"] = (
        work["team_position_rank"] <= work["position_slot_target"]
    ).astype(int)

    work["start_signal_rank_adjusted"] = work["start_signal_raw"]
    work.loc[work["inside_expected_slot_band"] == 1, "start_signal_rank_adjusted"] += 0.05

    outside_mask = work["team_position_rank"] >= (work["position_slot_target"] + 2)
    work.loc[outside_mask, "start_signal_rank_adjusted"] -= 0.08

    work["start_signal_rank_adjusted"] = work["start_signal_rank_adjusted"].clip(0.0, 1.0)

    return work


def assign_start_signal_tiers(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["start_signal_tier"] = pd.cut(
        work["start_signal_rank_adjusted"],
        bins=[-0.0001, 0.36, 0.52, 0.68, 0.82, 1.0001],
        labels=[
            "Langt fra start",
            "Bænk/rotation",
            "Tvivlsom starter",
            "Sandsynlig starter",
            "Klar starter",
        ],
    ).astype(str)

    return work


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_id",
        "player_name",
        "team_id",
        "position",
        "start_prob",
        "minute_share",
        "weighted_group_stage_ev",
        "total_ev_group_stage",
        "team_ev_share",
        "team_position_ev_share",
        "team_position_rank",
        "position_slot_target",
        "inside_expected_slot_band",
        "start_signal_raw",
        "start_signal_rank_adjusted",
        "start_signal_tier",
    ]
    out = df[cols].copy()
    out = out.sort_values(
        ["team_id", "position", "start_signal_rank_adjusted", "weighted_group_stage_ev"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return out


def print_summary(out: pd.DataFrame) -> None:
    print("Top 25 spillere efter start_signal_rank_adjusted:")
    print(
        out[
            [
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "minute_share",
                "team_position_rank",
                "start_signal_rank_adjusted",
                "start_signal_tier",
                "weighted_group_stage_ev",
            ]
        ]
        .sort_values(["start_signal_rank_adjusted", "weighted_group_stage_ev"], ascending=[False, False])
        .head(25)
        .to_string(index=False)
    )

    print("\nFordeling på tiers:")
    print(out["start_signal_tier"].value_counts().to_string())

    print("\nEksempel: top 3 pr. hold/position efter startsignal")
    sample = (
        out.sort_values(["team_id", "position", "start_signal_rank_adjusted"], ascending=[True, True, False])
           .groupby(["team_id", "position"], as_index=False)
           .head(3)
           .copy()
    )
    print(
        sample[
            [
                "team_id",
                "position",
                "team_position_rank",
                "player_name",
                "start_signal_rank_adjusted",
                "start_signal_tier",
            ]
        ]
        .head(60)
        .to_string(index=False)
    )


def write_json(out: pd.DataFrame) -> None:
    records = out.to_dict(orient="records")
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = load_data()
    df = build_base_signals(df)
    df = add_team_position_rankings(df)
    df = assign_start_signal_tiers(df)
    out = build_output(df)

    out.to_csv(OUTPUT_CSV, index=False)
    write_json(out)

    print(f"Skrev: {OUTPUT_CSV}")
    print(f"Skrev: {OUTPUT_JSON}")
    print("")
    print_summary(out)


if __name__ == "__main__":
    main()