from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from repair_ev_price_quality_consistency import (
    PRICE_QUALITY_SPREAD_MULTIPLIER,
    PRICE_QUALITY_WEIGHT,
    apply_price_quality_consistency,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
OUT_DIAG_PATH = DATA_DIR / "price_quality_ev_diagnostics.csv"


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def add_price_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "weighted_group_stage_ev" not in df.columns:
        raise ValueError("Mangler kolonnen weighted_group_stage_ev i EV-filen.")
    if "position" not in df.columns:
        raise ValueError("Mangler kolonnen position i EV-filen.")

    rows = df.where(pd.notna(df), "").to_dict(orient="records")
    return pd.DataFrame(apply_price_quality_consistency(rows))


def print_summary(after: pd.DataFrame) -> None:
    base = after["model_ev_before_price_quality"].map(lambda x: safe_float(x, 0.0))
    final = after["weighted_group_stage_ev"].map(lambda x: safe_float(x, 0.0))

    print("\nPRICE QUALITY BLEND")
    print(f"Raekker: {len(after)}")
    print(f"Prisvaegt: {PRICE_QUALITY_WEIGHT:.2f}")
    print(f"Spread multiplier: {PRICE_QUALITY_SPREAD_MULTIPLIER:.2f}")
    print("")
    print("Samlet EV:")
    print(f"Base sum:   {base.sum():.3f}")
    print(f"Efter sum:  {final.sum():.3f}")
    print(f"Base mean:  {base.mean():.4f}")
    print(f"Efter mean: {final.mean():.4f}")
    print("")

    cols = [
        "player_name",
        "team_id",
        "position",
        "price_m",
        "model_ev_before_price_quality",
        "price_quality_ev",
        "weighted_group_stage_ev",
        "price_rank_pct_position",
        "component_source",
        "base_ev_source",
    ]
    existing_cols = [col for col in cols if col in after.columns]
    print("Top 25 efter ny EV:")
    print(after.sort_values("weighted_group_stage_ev", ascending=False).head(25)[existing_cols].to_string(index=False))


def main() -> None:
    if not EV_PATH.exists():
        raise FileNotFoundError(f"Mangler {EV_PATH}")

    before = pd.read_csv(EV_PATH)
    after = add_price_quality(before)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_price_quality_v2_{timestamp}.csv")
    shutil.copy2(EV_PATH, backup_path)

    after.to_csv(EV_PATH, index=False, encoding="utf-8-sig")

    diag_cols = [
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "price_m",
        "price_rank_pct_position",
        "model_ev_before_price_quality",
        "price_quality_ev",
        "weighted_group_stage_ev",
        "price_quality_weight",
        "price_quality_spread_multiplier",
        "price_quality_applied",
        "component_source",
        "base_ev_source",
        "repair_status",
    ]
    existing_diag_cols = [col for col in diag_cols if col in after.columns]
    after[existing_diag_cols].to_csv(OUT_DIAG_PATH, index=False, encoding="utf-8-sig")

    print_summary(after)
    print("\nBackup:")
    print(backup_path)
    print("\nSkrev:")
    print(EV_PATH)
    print(OUT_DIAG_PATH)


if __name__ == "__main__":
    main()
