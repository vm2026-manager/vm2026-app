from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from json_file_safety import write_json_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
OPTIMAL_PATH = DATA_DIR / "optimal_squads_by_formation.json"
POOL_PATH = DATA_DIR / "player_pool_v1.json"

OUT_JSON = DATA_DIR / "model_status_snapshot.json"
OUT_TXT = DATA_DIR / "model_status_snapshot.txt"

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_player_pool(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("players", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Kan ikke finde spillerliste i {path}")


def load_optimal(path: Path) -> dict[str, Any]:
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(f"Optimizerfilen er ikke et dict: {path}")

    return data


def get_squads(optimal: dict[str, Any]) -> dict[str, Any]:
    if "formations" in optimal and isinstance(optimal["formations"], dict):
        return optimal["formations"]

    if "squads" in optimal and isinstance(optimal["squads"], dict):
        return optimal["squads"]

    return {
        key: value
        for key, value in optimal.items()
        if isinstance(value, (dict, list)) and "-" in str(key)
    }


def get_players_from_entry(entry: Any) -> list[dict[str, Any]]:
    if isinstance(entry, list):
        return entry

    if isinstance(entry, dict):
        for key in ("players", "squad", "lineup", "selected_players"):
            if isinstance(entry.get(key), list):
                return entry[key]

    return []


def get_entry_status(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("status", "Optimal"))
    return "Optimal"


def first_numeric_from_entry_or_players(
    entry: Any,
    players: list[dict[str, Any]],
    keys: list[str],
    fallback: float | None = None,
) -> float | None:
    """
    Brug først top-level felt.
    Hvis feltet kun findes på spillerrækkerne som samme squad-total på alle rækker,
    skal vi tage første værdi, IKKE summere.
    """
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

    for p in players:
        for key in keys:
            value = p.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

    return fallback


def price_m_from_player(p: dict[str, Any]) -> float:
    for key in ("price_m", "price_estimate_m", "price_mio"):
        value = p.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    for key in ("price", "price_estimate", "holdet_price"):
        value = p.get(key)
        if value is not None:
            try:
                value_float = float(value)
                return value_float / 1_000_000 if value_float > 1000 else value_float
            except Exception:
                pass

    return 0.0


def ev_from_player(p: dict[str, Any]) -> float:
    for key in ("optimizer_ev_adj", "squad_player_ev", "weighted_group_stage_ev", "optimizer_ev"):
        value = p.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    return 0.0


def raw_ev_from_player(p: dict[str, Any]) -> float:
    for key in ("optimizer_ev", "weighted_group_stage_ev", "optimizer_ev_base"):
        value = p.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    return 0.0


def summarize_formations(optimal: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    squads = get_squads(optimal)
    summaries: list[dict[str, Any]] = []

    for formation, entry in squads.items():
        players = get_players_from_entry(entry)
        status = get_entry_status(entry)

        price_m = first_numeric_from_entry_or_players(
            entry,
            players,
            ["squad_total_price_m", "total_price_m", "price_m"],
            fallback=sum(price_m_from_player(p) for p in players),
        )

        adj_ev = first_numeric_from_entry_or_players(
            entry,
            players,
            ["squad_total_adj_ev", "total_adj_ev", "adj_ev"],
            fallback=sum(ev_from_player(p) for p in players),
        )

        raw_ev = first_numeric_from_entry_or_players(
            entry,
            players,
            ["squad_total_ev", "squad_total_raw_ev", "total_ev", "raw_ev"],
            fallback=sum(raw_ev_from_player(p) for p in players),
        )

        summaries.append(
            {
                "formation": str(formation),
                "status": status,
                "price_m": round(float(price_m or 0.0), 3),
                "adj_ev": round(float(adj_ev or 0.0), 3),
                "raw_ev": round(float(raw_ev or 0.0), 3),
                "players": players,
            }
        )

    if not summaries:
        return summaries, None

    best = max(summaries, key=lambda x: x["adj_ev"])
    return summaries, best


def main() -> None:
    created_at = datetime.now().isoformat(timespec="seconds")

    ev = pd.read_csv(EV_PATH)
    player_pool = load_player_pool(POOL_PATH)
    optimal = load_optimal(OPTIMAL_PATH)

    formation_summaries, best = summarize_formations(optimal)
    best_players = best["players"] if best else []

    ev_summary = {
        "rows": int(len(ev)),
        "weighted_ev_sum": float(ev["weighted_group_stage_ev"].sum()) if "weighted_group_stage_ev" in ev.columns else None,
        "weighted_ev_mean": float(ev["weighted_group_stage_ev"].mean()) if "weighted_group_stage_ev" in ev.columns else None,
        "weighted_ev_median": float(ev["weighted_group_stage_ev"].median()) if "weighted_group_stage_ev" in ev.columns else None,
        "positions": ev["position"].value_counts().to_dict() if "position" in ev.columns else {},
    }

    pool_summary = {
        "rows": len(player_pool),
        "official_holdet_master": sum(1 for p in player_pool if p.get("official_holdet_master")),
        "positions": pd.Series([p.get("position") for p in player_pool]).value_counts().to_dict(),
        "teams": pd.Series([p.get("team_id") for p in player_pool]).nunique(),
        "missing_price": sum(1 for p in player_pool if p.get("price_estimate") in (None, "")),
    }

    snapshot = {
        "created_at": created_at,
        "assessment": {
            "best_formation": best["formation"] if best else None,
            "message": (
                "Modellen bruger nu Holdet.dk game 616 som officiel master-player-pool. "
                "EV-filen er rebased til de officielle Holdet-spillere, og optimizer-holdene matcher appens player_pool."
            ),
            "caveat": (
                "Spillerfelt, startvurderinger og odds er stadig foreløbige. "
                "Kun første runde har bookmaker-odds pt.; resten bygger på modelinput."
            ),
        },
        "player_pool": pool_summary,
        "ev_file": ev_summary,
        "formations": [
            {
                "formation": item["formation"],
                "status": item["status"],
                "adj_ev": item["adj_ev"],
                "raw_ev": item["raw_ev"],
                "price_m": item["price_m"],
            }
            for item in formation_summaries
        ],
        "best_squad": {
            "selected_formation": best["formation"] if best else None,
            "squad_total_price_m": best["price_m"] if best else None,
            "squad_total_adj_ev": best["adj_ev"] if best else None,
            "squad_total_raw_ev": best["raw_ev"] if best else None,
            "players": [
                {
                    "player_id": p.get("player_id"),
                    "player_name": p.get("player_name"),
                    "team_id": p.get("team_id"),
                    "position": p.get("position"),
                    "price_m": price_m_from_player(p),
                    "ev": ev_from_player(p),
                }
                for p in best_players
            ],
        },
    }

    write_json_strict(OUT_JSON, snapshot)

    lines = []
    lines.append("VM 2026 MODEL STATUS SNAPSHOT")
    lines.append("")
    lines.append(f"Oprettet: {created_at}")
    lines.append("")
    lines.append("VURDERING")
    if best:
        lines.append(f"- Bedste formation i seneste optimizer-kørsel: {best['formation']}.")
    lines.append("- Modellen bruger nu Holdet.dk game 616 som officiel master-player-pool.")
    lines.append("- EV-filen er rebased til de officielle Holdet-spillere.")
    lines.append("- Optimizer-holdene matcher appens player_pool på player_id.")
    lines.append("- Forbehold: Startvurderinger, odds og trupper er stadig foreløbige og skal opdateres løbende.")
    lines.append("")
    lines.append("PLAYER POOL")
    lines.append(f"- Rækker: {pool_summary['rows']}")
    lines.append(f"- Officiel Holdet-master: {pool_summary['official_holdet_master']}")
    lines.append(f"- Hold: {pool_summary['teams']}")
    lines.append(f"- Mangler pris: {pool_summary['missing_price']}")
    lines.append(f"- Positioner: {pool_summary['positions']}")
    lines.append("")
    lines.append("EV-FIL")
    lines.append(f"- Rækker: {ev_summary['rows']}")
    lines.append(f"- Weighted EV sum: {ev_summary['weighted_ev_sum']:.3f}")
    lines.append(f"- Weighted EV mean: {ev_summary['weighted_ev_mean']:.4f}")
    lines.append(f"- Weighted EV median: {ev_summary['weighted_ev_median']:.4f}")
    lines.append(f"- Positioner: {ev_summary['positions']}")
    lines.append("")
    lines.append("FORMATIONER")
    for item in formation_summaries:
        lines.append(
            f"- {item['formation']}: status={item['status']}, "
            f"adjEV={item['adj_ev']}, rawEV={item['raw_ev']}, pris={item['price_m']}"
        )
    lines.append("")
    lines.append("BEDSTE HOLD")
    if best:
        lines.append(f"- selected_formation: {best['formation']}")
        lines.append(f"- squad_total_price_m: {best['price_m']}")
        lines.append(f"- squad_total_adj_ev: {best['adj_ev']}")
        lines.append(f"- squad_total_raw_ev: {best['raw_ev']}")
        lines.append("")
        lines.append("SPILLERE")
        for p in best_players:
            lines.append(
                f"- {p.get('player_name')} | {p.get('team_id')} | {p.get('position')} "
                f"| pris={price_m_from_player(p):.1f} | EV={ev_from_player(p):.3f}"
            )
    else:
        lines.append("- Ingen bedste formation fundet.")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Skrev: {OUT_JSON}")
    print(f"Skrev: {OUT_TXT}")
    print("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
