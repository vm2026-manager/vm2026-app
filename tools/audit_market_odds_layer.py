from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

MANUAL_PATH = DATA_DIR / "team_market_odds_manual.csv"
LAYER_PATH = DATA_DIR / "team_market_odds_layer_v1.csv"


ODDS_COLS = [
    "winner_odds",
    "reach_qf_odds",
    "reach_sf_odds",
    "reach_final_odds",
    "group_win_odds",
    "highest_scoring_team_odds",
    "lowest_scoring_team_odds",
]


SCORE_COLS = [
    "team_long_run_score",
    "team_group_stage_score",
    "team_attack_score",
    "team_low_scoring_risk",
    "team_market_score",
]


def main() -> None:
    if not MANUAL_PATH.exists():
        raise FileNotFoundError(f"Mangler {MANUAL_PATH}")

    manual = pd.read_csv(MANUAL_PATH)

    print("MARKET ODDS MANUAL COVERAGE")
    print("=" * 60)
    print(f"Hold i manualfil: {len(manual)}")
    print("")

    for col in ODDS_COLS:
        if col not in manual.columns:
            print(f"MANGLER KOLONNE: {col}")
            continue

        values = pd.to_numeric(manual[col], errors="coerce")
        filled = int(values.notna().sum())
        missing = int(values.isna().sum())
        print(f"{col:30s} udfyldt={filled:2d}  mangler={missing:2d}")

    print("")

    if LAYER_PATH.exists():
        layer = pd.read_csv(LAYER_PATH)

        print("TEAM MARKET SCORE TOP 25")
        print("=" * 60)

        cols = [
            "team_id",
            "team_name",
            "winner_odds",
            "reach_qf_odds",
            "group_win_odds",
            "highest_scoring_team_odds",
            "team_long_run_score",
            "team_group_stage_score",
            "team_attack_score",
            "team_market_score",
        ]
        cols = [c for c in cols if c in layer.columns]

        print(layer[cols].head(25).to_string(index=False))

        print("")
        print("SCORE COLUMNS")
        print("=" * 60)

        for col in SCORE_COLS:
            if col in layer.columns:
                print(
                    f"{col:30s} "
                    f"min={layer[col].min():.3f} "
                    f"mean={layer[col].mean():.3f} "
                    f"max={layer[col].max():.3f}"
                )

    print("")
    print("NÆSTE ANBEFALEDE KOLONNE:")
    print("- group_win_odds")
    print("")
    print("Hvorfor:")
    print("- winner_odds = langsigtet styrke")
    print("- reach_qf_odds = sandsynlighed for flere kampe")
    print("- highest_scoring_team_odds = angrebsmiljø")
    print("- group_win_odds = gruppespilsstyrke og første tre runder")


if __name__ == "__main__":
    main()