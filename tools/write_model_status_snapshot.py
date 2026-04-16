from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_CSV = DATA_DIR / "player_ev_group_stage_v1.csv"
OPTIMAL_SQUAD_CSV = DATA_DIR / "optimal_squad_group_stage.csv"
FORMATION_SUMMARY_CSV = DATA_DIR / "optimal_formations_summary.csv"
OPTIMAL_BY_FORMATION_JSON = DATA_DIR / "optimal_squads_by_formation.json"

OUTPUT_JSON = DATA_DIR / "model_status_snapshot.json"
OUTPUT_TXT = DATA_DIR / "model_status_snapshot.txt"


def read_if_exists_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_snapshot() -> dict:
    ev_df = read_if_exists_csv(PLAYER_EV_CSV)
    best_df = read_if_exists_csv(OPTIMAL_SQUAD_CSV)
    formation_df = read_if_exists_csv(FORMATION_SUMMARY_CSV)

    snapshot: dict = {
        "snapshot_created_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "player_ev_csv": str(PLAYER_EV_CSV),
            "optimal_squad_csv": str(OPTIMAL_SQUAD_CSV),
            "formation_summary_csv": str(FORMATION_SUMMARY_CSV),
            "optimal_by_formation_json": str(OPTIMAL_BY_FORMATION_JSON),
        },
        "current_assessment": {
            "formation_conclusion": (
                "5-4-1 vinder stadig i den nuværende model, men det vurderes ikke som afgørende endnu, "
                "fordi spillerfelt og priser stadig er foreløbige/proxy-baserede."
            ),
            "main_takeaway": (
                "Det vigtigste lige nu er en nogenlunde sund EV-motor og en meningsfuld optimizer. "
                "Formationsfinjustering kan vente eller evt. håndteres med en midlertidig balanced-toggle senere."
            ),
            "next_recommended_focus": (
                "Flyt fokus væk fra mere formation-tuning og over på datagrundlag, "
                "workflow, rigtige priser og bedre spiller-/starterinput."
            ),
        },
    }

    if not ev_df.empty:
        pos_counts = ev_df["position"].astype(str).value_counts().to_dict() if "position" in ev_df.columns else {}
        snapshot["ev_file"] = {
            "rows": int(len(ev_df)),
            "columns": list(ev_df.columns),
            "position_counts": pos_counts,
            "weighted_ev_sum": float(ev_df["weighted_group_stage_ev"].sum()) if "weighted_group_stage_ev" in ev_df.columns else None,
            "weighted_ev_mean": float(ev_df["weighted_group_stage_ev"].mean()) if "weighted_group_stage_ev" in ev_df.columns else None,
            "weighted_ev_median": float(ev_df["weighted_group_stage_ev"].median()) if "weighted_group_stage_ev" in ev_df.columns else None,
        }

    if not formation_df.empty:
        snapshot["formations"] = formation_df.to_dict(orient="records")

    if not best_df.empty:
        best_meta = {}
        for col in ["selected_formation", "solver_quality_profile", "squad_total_price_m", "squad_total_ev", "squad_total_adj_ev", "squad_total_raw_ev"]:
            if col in best_df.columns:
                value = best_df[col].iloc[0]
                if pd.notna(value):
                    if hasattr(value, "item"):
                        value = value.item()
                    best_meta[col] = value

        keep_cols = [c for c in ["player_name", "team_id", "position", "price_m", "optimizer_ev", "optimizer_ev_adj", "start_prob"] if c in best_df.columns]

        snapshot["best_squad"] = {
            "meta": best_meta,
            "players": best_df[keep_cols].to_dict(orient="records"),
        }

    if OPTIMAL_BY_FORMATION_JSON.exists():
        try:
            with OPTIMAL_BY_FORMATION_JSON.open("r", encoding="utf-8") as f:
                by_formation = json.load(f)
            snapshot["formations_available"] = list(by_formation.keys())
        except Exception as e:
            snapshot["formations_available_error"] = str(e)

    return snapshot


def write_text_summary(snapshot: dict) -> str:
    lines: list[str] = []

    lines.append("VM 2026 MODEL STATUS SNAPSHOT")
    lines.append("")
    lines.append(f"Oprettet: {snapshot.get('snapshot_created_at', '')}")
    lines.append("")

    assessment = snapshot.get("current_assessment", {})
    lines.append("VURDERING")
    lines.append(f"- Formation: {assessment.get('formation_conclusion', '')}")
    lines.append(f"- Hovedpointe: {assessment.get('main_takeaway', '')}")
    lines.append(f"- Næste fokus: {assessment.get('next_recommended_focus', '')}")
    lines.append("")

    ev_info = snapshot.get("ev_file", {})
    if ev_info:
        lines.append("EV-FIL")
        lines.append(f"- Rækker: {ev_info.get('rows')}")
        lines.append(f"- Weighted EV sum: {round(ev_info.get('weighted_ev_sum', 0.0), 3) if ev_info.get('weighted_ev_sum') is not None else 'n/a'}")
        lines.append(f"- Weighted EV mean: {round(ev_info.get('weighted_ev_mean', 0.0), 4) if ev_info.get('weighted_ev_mean') is not None else 'n/a'}")
        lines.append(f"- Weighted EV median: {round(ev_info.get('weighted_ev_median', 0.0), 4) if ev_info.get('weighted_ev_median') is not None else 'n/a'}")
        pos_counts = ev_info.get("position_counts", {})
        if pos_counts:
            lines.append(f"- Positioner: {pos_counts}")
        lines.append("")

    formations = snapshot.get("formations", [])
    if formations:
        lines.append("FORMATIONER")
        for row in formations:
            formation = row.get("formation", "")
            status = row.get("status", "")
            ev = row.get("squad_total_adj_ev", row.get("squad_total_ev"))
            price = row.get("squad_total_price_m")
            lines.append(f"- {formation}: status={status}, EV={ev}, pris={price}")
        lines.append("")

    best_squad = snapshot.get("best_squad", {})
    if best_squad:
        lines.append("BEDSTE HOLD")
        meta = best_squad.get("meta", {})
        for key, value in meta.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("SPILLERE")
        for player in best_squad.get("players", []):
            name = player.get("player_name", "")
            team = player.get("team_id", "")
            pos = player.get("position", "")
            price = player.get("price_m", "")
            ev = player.get("optimizer_ev_adj", player.get("optimizer_ev", ""))
            lines.append(f"- {name} | {team} | {pos} | pris={price} | EV={ev}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    snapshot = load_snapshot()

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    text_summary = write_text_summary(snapshot)
    OUTPUT_TXT.write_text(text_summary, encoding="utf-8")

    print(f"Skrev: {OUTPUT_JSON}")
    print(f"Skrev: {OUTPUT_TXT}")
    print("")
    print(text_summary)


if __name__ == "__main__":
    main()