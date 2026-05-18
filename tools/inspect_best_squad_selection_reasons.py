from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMAL_PATH = PROJECT_ROOT / "data" / "optimal_squads_by_formation.json"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
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


def main() -> None:
    if not OPTIMAL_PATH.exists():
        raise FileNotFoundError(f"Mangler {OPTIMAL_PATH}")

    data = load_json(OPTIMAL_PATH)

    best_formation = None
    best_rows = []
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

    print("BEDSTE HOLD – SELECTION REASONS")
    print("=" * 90)
    print(f"Formation: {best_formation}")
    print(f"Score: {best_score:.3f}")
    print(f"Pris: {sum(to_float(r.get('price_m')) for r in best_rows):.1f} mio.")
    print("")

    headers = [
        "pos",
        "player",
        "team",
        "price",
        "start",
        "start_pen",
        "market",
        "model_ev",
        "sel_score",
        "odds",
    ]
    print(
        f"{headers[0]:<4} {headers[1]:<28} {headers[2]:<5} "
        f"{headers[3]:>5} {headers[4]:>7} {headers[5]:>9} "
        f"{headers[6]:>7} {headers[7]:>8} {headers[8]:>9} {headers[9]:>7}"
    )
    print("-" * 100)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    best_rows = sorted(
        best_rows,
        key=lambda r: (pos_order.get(str(r.get("position")), 9), str(r.get("player_name"))),
    )

    for r in best_rows:
        start_prob = to_float(r.get("start_prob_model", r.get("start_prob_display", r.get("start_probability_pct", 0))))
        if start_prob > 1:
            start_prob = start_prob / 100

        print(
            f"{str(r.get('position','')):<4} "
            f"{str(r.get('player_name',''))[:28]:<28} "
            f"{str(r.get('team_id','')):<5} "
            f"{to_float(r.get('price_m')):>5.1f} "
            f"{start_prob:>6.0%} "
            f"{to_float(r.get('start_risk_penalty')):>9.3f} "
            f"{to_float(r.get('team_market_score')):>7.3f} "
            f"{to_float(r.get('model_ev', r.get('optimizer_ev'))):>8.3f} "
            f"{to_float(r.get('selection_score', r.get('optimizer_ev_adj'))):>9.3f} "
            f"{to_float(r.get('winner_odds')):>7.1f}"
        )

    print("")
    print("Lav start + høj startstraf = kandidat til manuel/modelmæssig justering.")
    print("Hvis en spiller stadig virker forkert, skal vi justere startvægten eller selve startfilen.")


if __name__ == "__main__":
    main()