from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
OUT_DIAG_PATH = DATA_DIR / "price_quality_ev_diagnostics.csv"


# Stærkere end første forsøg.
# 0.45 betyder: 55 pct. eksisterende model-EV + 45 pct. prisbaseret kvalitet.
PRICE_QUALITY_WEIGHT = 0.45

# Giver dyre spillere et tydeligere løft uden helt at lade pris overtage modellen.
PRICE_QUALITY_SPREAD_MULTIPLIER = 1.35


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_price_m(row: pd.Series) -> float:
    for col in ["price_m", "price_estimate_m", "price_mio"]:
        if col in row and pd.notna(row[col]):
            return safe_float(row[col])

    for col in ["price", "price_estimate", "holdet_price"]:
        if col in row and pd.notna(row[col]):
            value = safe_float(row[col])
            if value > 1000:
                return value / 1_000_000
            return value

    return 0.0


def get_base_model_ev(df: pd.DataFrame) -> pd.Series:
    """
    Gør scriptet idempotent.

    Hvis vi allerede tidligere har lagt price quality på, bruger vi den oprindelige
    model-EV-kolonne som base. Ellers bruger vi weighted_group_stage_ev.
    """
    for col in [
        "weighted_group_stage_ev_before_price_quality",
        "model_ev_original",
        "model_ev_before_price_quality",
    ]:
        if col in df.columns:
            return df[col].map(lambda x: safe_float(x, 0.0))

    return df["weighted_group_stage_ev"].map(lambda x: safe_float(x, 0.0))


def add_price_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "weighted_group_stage_ev" not in df.columns:
        raise ValueError("Mangler kolonnen weighted_group_stage_ev i EV-filen.")

    if "position" not in df.columns:
        raise ValueError("Mangler kolonnen position i EV-filen.")

    df["price_m"] = df.apply(get_price_m, axis=1)

    base_ev = get_base_model_ev(df)
    df["model_ev_before_price_quality"] = base_ev
    df["weighted_group_stage_ev_before_price_quality"] = base_ev

    # Prisrangering inden for position, så GK ikke sammenlignes direkte med angribere.
    df["price_rank_pct_position"] = (
        df.groupby("position")["price_m"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )

    price_quality_parts = []

    for position, part in df.groupby("position", dropna=False):
        part = part.copy()

        ev_values = part["model_ev_before_price_quality"].astype(float)

        # Robust skala, men lidt bredere end første forsøg.
        ev_p20 = float(ev_values.quantile(0.20))
        ev_p90 = float(ev_values.quantile(0.90))

        if ev_p90 <= ev_p20:
            ev_p20 = float(ev_values.median())
            ev_p90 = float(ev_values.max())

        spread = max(ev_p90 - ev_p20, 0.85) * PRICE_QUALITY_SPREAD_MULTIPLIER

        # Dyreste i positionen får cirka øvre EV-niveau.
        # Billigste får stadig et lavt, men ikke negativt, kvalitetssignal.
        part["price_quality_ev"] = ev_p20 + part["price_rank_pct_position"] * spread

        price_quality_parts.append(part)

    df = pd.concat(price_quality_parts, ignore_index=True)

    df["weighted_group_stage_ev"] = (
        (1.0 - PRICE_QUALITY_WEIGHT) * df["model_ev_before_price_quality"]
        + PRICE_QUALITY_WEIGHT * df["price_quality_ev"]
    ).round(6)

    df["optimizer_ev_before_price_quality"] = df.get(
        "optimizer_ev",
        df["model_ev_before_price_quality"],
    )

    df["optimizer_ev"] = df["weighted_group_stage_ev"]

    df["price_quality_weight"] = PRICE_QUALITY_WEIGHT
    df["price_quality_spread_multiplier"] = PRICE_QUALITY_SPREAD_MULTIPLIER
    df["price_quality_applied"] = True

    return df


def print_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print("\nPRICE QUALITY BLEND")
    print(f"Rækker: {len(after)}")
    print(f"Prisvægt: {PRICE_QUALITY_WEIGHT:.2f}")
    print(f"Spread multiplier: {PRICE_QUALITY_SPREAD_MULTIPLIER:.2f}")
    print("")

    before_base = get_base_model_ev(before)

    print("Samlet EV før/efter:")
    print(f"Base sum:   {before_base.sum():.3f}")
    print(f"Efter sum:  {after['weighted_group_stage_ev'].astype(float).sum():.3f}")
    print(f"Base mean:  {before_base.mean():.4f}")
    print(f"Efter mean: {after['weighted_group_stage_ev'].astype(float).mean():.4f}")
    print("")

    print("Prisfordeling:")
    print(f"Min:    {after['price_m'].min():.2f}")
    print(f"Median: {after['price_m'].median():.2f}")
    print(f"Max:    {after['price_m'].max():.2f}")
    print("")

    print("Top 25 efter ny EV:")
    cols = [
        "player_name",
        "team_id",
        "position",
        "price_m",
        "model_ev_before_price_quality",
        "price_quality_ev",
        "weighted_group_stage_ev",
        "price_rank_pct_position",
    ]
    existing_cols = [c for c in cols if c in after.columns]
    print(
        after.sort_values("weighted_group_stage_ev", ascending=False)
        .head(25)[existing_cols]
        .to_string(index=False)
    )

    print("\nStørste løft fra pris:")
    after = after.copy()
    after["price_quality_delta"] = (
        after["weighted_group_stage_ev"]
        - after["model_ev_before_price_quality"]
    )
    print(
        after.sort_values("price_quality_delta", ascending=False)
        .head(25)[existing_cols + ["price_quality_delta"]]
        .to_string(index=False)
    )


def main() -> None:
    if not EV_PATH.exists():
        raise FileNotFoundError(f"Mangler {EV_PATH}")

    before = pd.read_csv(EV_PATH)
    after = add_price_quality(before)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = EV_PATH.with_name(
        f"player_ev_group_stage_v1.backup_before_price_quality_v2_{timestamp}.csv"
    )
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
    ]
    existing_diag_cols = [c for c in diag_cols if c in after.columns]
    after[existing_diag_cols].to_csv(OUT_DIAG_PATH, index=False, encoding="utf-8-sig")

    print_summary(before, after)

    print("\nBackup:")
    print(backup_path)
    print("\nSkrev:")
    print(EV_PATH)
    print(OUT_DIAG_PATH)


if __name__ == "__main__":
    main()