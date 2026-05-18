from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OPTIMAL_PATH = DATA_DIR / "optimal_squads_by_formation.json"
OUT_CSV = DATA_DIR / "selected_squad_watchlist.csv"
OUT_TXT = DATA_DIR / "selected_squad_watchlist.txt"


WATCH_START_THRESHOLD = 0.75


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number
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
                value = value / 100.0
            return max(0.0, min(1.0, value))
    return 0.0


def main() -> None:
    if not OPTIMAL_PATH.exists():
        raise FileNotFoundError(f"Mangler {OPTIMAL_PATH}")

    data = load_json(OPTIMAL_PATH)

    best_formation = None
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

    out_rows = []

    for row in best_rows:
        start_prob = get_start_prob(row)
        watch_flag = "JA" if start_prob < WATCH_START_THRESHOLD else ""

        reason_parts = []
        if start_prob < 0.55:
            reason_parts.append("Lav startsandsynlighed")
        elif start_prob < WATCH_START_THRESHOLD:
            reason_parts.append("Middel/usikker startsandsynlighed")

        if to_float(row.get("price_m")) <= 3.5:
            reason_parts.append("Billig spiller kan være proxy/value-pick")

        out_rows.append(
            {
                "formation": best_formation,
                "player_name": row.get("player_name"),
                "team_id": row.get("team_id"),
                "position": row.get("position"),
                "price_m": row.get("price_m"),
                "start_probability_pct": round(start_prob * 100, 1),
                "team_market_score": row.get("team_market_score"),
                "winner_odds": row.get("winner_odds"),
                "optimizer_ev_adj": row.get("optimizer_ev_adj"),
                "watch_flag": watch_flag,
                "watch_reason": "; ".join(reason_parts),
                "manual_note": "",
            }
        )

    df = pd.DataFrame(out_rows)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["_pos_order"] = df["position"].map(pos_order).fillna(9)
    df = df.sort_values(["_pos_order", "team_id", "player_name"]).drop(columns=["_pos_order"])

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("SELECTED SQUAD WATCHLIST")
    lines.append("=" * 80)
    lines.append(f"Formation: {best_formation}")
    lines.append(f"Score: {best_score:.3f}")
    lines.append("")
    lines.append("Spillere til manuel vurdering:")
    lines.append("")

    watch = df.loc[df["watch_flag"] == "JA"].copy()

    if watch.empty:
        lines.append("Ingen spillere under watch-threshold.")
    else:
        for _, r in watch.iterrows():
            lines.append(
                f"- {r['player_name']} | {r['team_id']} | {r['position']} | "
                f"start={r['start_probability_pct']}% | pris={r['price_m']} | "
                f"{r['watch_reason']}"
            )

    lines.append("")
    lines.append("Note:")
    lines.append("- Denne fil ændrer ikke modellen.")
    lines.append("- Brug den kun som observationsliste frem mod de sidste testkampe før VM.")
    lines.append("- Startdata bør først justeres hårdt, når trupper og startellevere er mere sikre.")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Skrev: {OUT_CSV}")
    print(f"Skrev: {OUT_TXT}")


if __name__ == "__main__":
    main()