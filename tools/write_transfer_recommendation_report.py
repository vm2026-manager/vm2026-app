from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CURRENT_SQUAD_PATH = DATA_DIR / "current_squad.csv"
PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
CONTEXT_PATH = DATA_DIR / "current_strategy_context.json"
MANUAL_OVERRIDES_PATH = DATA_DIR / "manual_player_overrides.csv"
OUT_CSV = DATA_DIR / "transfer_recommendation_report.csv"
OUT_MD = DATA_DIR / "transfer_recommendation_report.md"

BUDGET = 50_000_000
MAX_PER_TEAM = 4
TRANSFER_FEE_RATE = 0.01
FIELDNAMES = [
    "outgoing_player",
    "incoming_player",
    "incoming_price",
    "transfer_fee",
    "outgoing_expected_growth",
    "incoming_expected_growth",
    "net_gain",
    "recommendation",
    "strategy",
    "target_round",
    "net_gain_is_proxy",
    "proxy_note",
]

PROXY_NOTE = "Midlertidig score-til-kr.-proxy; ikke præcis forventet nettogevinst i kroner."
TODO_NOTE = "TODO: Erstat score-til-kr.-proxy med egentlig forventet rundevækst i kroner pr. spiller og runde."


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    return int(round(to_float(value, float(default))))


def normalize_name(value: Any) -> str:
    return txt(value).casefold()


def target_round_from_context() -> int:
    if not CONTEXT_PATH.exists():
        return 1
    context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8-sig"))
    return int(context.get("target_round") or 1)


def player_key(row: dict[str, Any] | pd.Series) -> tuple[str, str]:
    return (normalize_name(row.get("player_name")), txt(row.get("team_id")).upper())


def load_manual_avoid_keys() -> set[tuple[str, str]]:
    rows = read_csv(MANUAL_OVERRIDES_PATH)
    avoid = set()
    for row in rows:
        manual_status = txt(row.get("manual_status")).lower()
        manual_start_status = txt(row.get("manual_start_status")).lower()
        if manual_status == "avoid" or manual_start_status == "avoid":
            avoid.add((normalize_name(row.get("player_name")), txt(row.get("team_id")).upper()))
    return avoid


def load_current_squad() -> pd.DataFrame:
    rows = read_csv(CURRENT_SQUAD_PATH)
    columns = ["player_id", "player_name", "team_id", "position", "current_value", "owned_since_round"]
    if not rows:
        return pd.DataFrame(columns=columns)
    squad = pd.DataFrame(rows)
    for col in columns:
        if col not in squad.columns:
            squad[col] = ""
    squad["team_id"] = squad["team_id"].astype(str).str.strip().str.upper()
    squad["position"] = squad["position"].astype(str).str.strip().str.upper()
    squad["current_value"] = pd.to_numeric(squad["current_value"], errors="coerce")
    squad["owned_since_round"] = pd.to_numeric(squad["owned_since_round"], errors="coerce")
    squad = squad[(squad["player_id"].astype(str).str.strip() != "") | (squad["player_name"].astype(str).str.strip() != "")]
    return squad[columns].copy()


def growth_for_player(row: pd.Series, target_round: int) -> float:
    candidates = [
        f"match_{target_round}_weighted_match_ev",
        f"match_{target_round}_total_ev_next_match",
        "optimizer_ev",
        "weighted_group_stage_ev",
    ]
    for col in candidates:
        if col in row:
            value = to_float(row.get(col))
            if value > 0:
                return value
    return 0.0


def find_player(pool: pd.DataFrame, current_row: pd.Series) -> pd.Series | None:
    player_id = txt(current_row.get("player_id"))
    if player_id:
        match = pool[pool["player_id"].astype(str) == player_id]
        if not match.empty:
            return match.iloc[0]
    name, team = player_key(current_row)
    match = pool[
        (pool["player_name"].astype(str).map(normalize_name) == name)
        & (pool["team_id"].astype(str).str.upper() == team)
    ]
    if not match.empty:
        return match.iloc[0]
    return None


def build_report(target_round: int) -> list[dict[str, Any]]:
    current = load_current_squad()
    if current.empty:
        return []

    pool = pd.read_csv(PLAYER_EV_PATH)
    pool["team_id"] = pool["team_id"].astype(str).str.strip().str.upper()
    pool["position"] = pool["position"].astype(str).str.strip().str.upper()
    pool["price"] = pd.to_numeric(pool["price"], errors="coerce").fillna(pd.to_numeric(pool.get("price_estimate"), errors="coerce")).fillna(0).astype(int)
    avoid_keys = load_manual_avoid_keys()
    owned_ids = set(current["player_id"].astype(str).str.strip())
    owned_keys = {player_key(row) for _, row in current.iterrows()}
    current_team_counts = current["team_id"].value_counts().to_dict()
    current_values: dict[int, int] = {}
    for idx, row in current.iterrows():
        pool_row = find_player(pool, row)
        fallback_price = int(pool_row.get("price", 0)) if pool_row is not None else 0
        current_values[int(idx)] = to_int(row.get("current_value"), fallback_price)
    current_total_value = sum(current_values.values())

    rows: list[dict[str, Any]] = []
    for idx, outgoing in current.iterrows():
        outgoing_pool = find_player(pool, outgoing)
        if outgoing_pool is None:
            continue
        outgoing_growth = growth_for_player(outgoing_pool, target_round)
        outgoing_value = current_values.get(int(idx), int(outgoing_pool.get("price", 0)))
        outgoing_team = txt(outgoing.get("team_id")).upper()
        position = txt(outgoing.get("position") or outgoing_pool.get("position")).upper()

        candidates = pool[pool["position"] == position].copy()
        for _, incoming in candidates.iterrows():
            incoming_id = txt(incoming.get("player_id"))
            incoming_key = player_key(incoming)
            if incoming_id in owned_ids or incoming_key in owned_keys:
                continue
            if incoming_key in avoid_keys:
                continue

            incoming_price = int(incoming.get("price") or 0)
            if current_total_value - outgoing_value + incoming_price > BUDGET:
                continue
            team_counts = dict(current_team_counts)
            team_counts[outgoing_team] = max(0, int(team_counts.get(outgoing_team, 0)) - 1)
            incoming_team = txt(incoming.get("team_id")).upper()
            if int(team_counts.get(incoming_team, 0)) + 1 > MAX_PER_TEAM:
                continue

            incoming_growth = growth_for_player(incoming, target_round)
            transfer_fee = 0 if target_round <= 1 else int(round(incoming_price * TRANSFER_FEE_RATE))
            net_gain = int(round((incoming_growth - outgoing_growth) * 1_000_000 - transfer_fee))
            rows.append({
                "outgoing_player": txt(outgoing.get("player_name") or outgoing_pool.get("player_name")),
                "incoming_player": txt(incoming.get("player_name")),
                "incoming_price": incoming_price,
                "transfer_fee": transfer_fee,
                "outgoing_expected_growth": round(outgoing_growth, 6),
                "incoming_expected_growth": round(incoming_growth, 6),
                "net_gain": net_gain,
                "recommendation": "recommend_transfer" if net_gain > 0 else "hold_current",
                "strategy": "next_round",
                "target_round": target_round,
                "net_gain_is_proxy": "yes",
                "proxy_note": PROXY_NOTE,
            })

    return sorted(rows, key=lambda row: row["net_gain"], reverse=True)


def write_md(rows: list[dict[str, Any]], target_round: int) -> None:
    lines = [
        "# Transfer Recommendation Report",
        "",
        f"- target_round: {target_round}",
        f"- transfer_fee: {'0' if target_round <= 1 else '1 pct. af incoming current value'}",
        f"- current_squad rows: {len(load_current_squad())}",
        f"- net_gain: {PROXY_NOTE}",
        f"- {TODO_NOTE}",
        "",
    ]
    if not rows:
        lines.extend([
            "Ingen anbefalinger skrevet.",
            "",
            "Årsag: data/current_squad.csv er tom eller matcher ingen spillere i player_ev_group_stage_v1.csv.",
            "Der gættes ikke på brugerens aktuelle hold.",
        ])
    else:
        lines.extend(["## Top Nettogevinster", ""])
        for row in rows[:20]:
            lines.append(
                f"- {row['outgoing_player']} -> {row['incoming_player']}: "
                f"estimeret net_gain proxy {row['net_gain']:,} kr., gebyr {row['transfer_fee']:,} kr. "
                f"({row['recommendation']})".replace(",", ".")
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-round", type=int, default=None)
    args = parser.parse_args()
    target_round = args.target_round or target_round_from_context()
    rows = build_report(target_round)
    write_csv(OUT_CSV, rows)
    write_md(rows, target_round)
    print(f"Skrevet: {OUT_CSV.relative_to(PROJECT_ROOT)} ({len(rows)} rækker)")
    print(f"Skrevet: {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
