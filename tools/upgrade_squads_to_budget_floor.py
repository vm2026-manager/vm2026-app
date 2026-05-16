from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OPTIMAL_PATH = DATA_DIR / "optimal_squads_by_formation.json"
EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
OUT_DIAG_PATH = DATA_DIR / "budget_floor_squad_upgrades.csv"

BUDGET_M = 50.0
TARGET_MIN_PRICE_M = 49.0
MAX_PER_TEAM = 4

# Høj værdi af at bruge budget, men ikke uendelig.
# Billige spillere kan stadig overleve, hvis deres EV-forspring er stort.
BUDGET_FILL_VALUE_PER_M = 0.55

# Hvis der stadig mangler budget, tillader vi et kontrolleret EV-tab pr. swap.
MAX_ACCEPTED_EV_LOSS_PER_SWAP = 1.35

# Undgå at bytte ind i helt døde spillere, medmindre data mangler.
MIN_REASONABLE_START_PROB = 0.20


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_price_m_from_row(row: dict[str, Any] | pd.Series) -> float:
    for col in ["price_m", "price_estimate_m", "price_mio"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return to_float(value, 0.0)

    for col in ["price", "price_estimate", "holdet_price"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            value_float = to_float(value, 0.0)
            return value_float / 1_000_000 if value_float > 1000 else value_float

    return 0.0


def get_start_prob(row: dict[str, Any] | pd.Series) -> float:
    for col in ["start_prob", "start_probability", "start_probability_pct"]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            num = to_float(value, 0.0)
            if num > 1:
                return num / 100
            return num
    return 0.25


def get_model_ev(row: dict[str, Any] | pd.Series) -> float:
    for col in [
        "optimizer_ev_adj",
        "weighted_group_stage_ev",
        "optimizer_ev",
        "model_ev_before_price_quality",
        "optimizer_ev_base",
    ]:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return to_float(value, 0.0)
    return 0.0


def get_effective_ev(row: dict[str, Any] | pd.Series) -> float:
    base = get_model_ev(row)
    start_prob = get_start_prob(row)

    # Meget lav startsandsynlighed skal koste lidt, men ikke dominere alt.
    low_start_penalty = max(0.0, MIN_REASONABLE_START_PROB - start_prob) * 1.25

    return base - low_start_penalty


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_formation_container(data: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(data, dict):
        raise ValueError("optimal_squads_by_formation.json skal være et JSON-objekt.")

    if isinstance(data.get("formations"), dict):
        return data["formations"], "formations"

    if isinstance(data.get("squads"), dict):
        return data["squads"], "squads"

    # Nuværende app kan læse top-level formationer.
    return data, "top_level"


def get_players_from_entry(entry: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(entry, list):
        return entry, "list"

    if isinstance(entry, dict):
        for key in ["players", "squad", "lineup", "selected_players"]:
            if isinstance(entry.get(key), list):
                return entry[key], key

    return [], ""


def set_players_on_entry(entry: Any, players: list[dict[str, Any]], players_key: str) -> Any:
    if players_key == "list":
        return players

    if isinstance(entry, dict):
        entry[players_key] = players
        return entry

    return players


def normalize_position(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "MÅLMAND": "GK",
        "MALMAND": "GK",
        "KEEPER": "GK",
        "GOALKEEPER": "GK",
        "FORSVAR": "DEF",
        "DEFENDER": "DEF",
        "DEFENSE": "DEF",
        "MIDTBANE": "MID",
        "MIDFIELDER": "MID",
        "MIDFIELD": "MID",
        "ANGRIBER": "FWD",
        "FORWARD": "FWD",
        "STRIKER": "FWD",
    }
    return aliases.get(text, text)


def load_candidate_pool() -> pd.DataFrame:
    if not EV_PATH.exists():
        raise FileNotFoundError(f"Mangler {EV_PATH}")

    df = pd.read_csv(EV_PATH)

    if "player_id" not in df.columns:
        raise ValueError("EV-filen mangler player_id.")

    if "position" not in df.columns:
        raise ValueError("EV-filen mangler position.")

    if "team_id" not in df.columns:
        raise ValueError("EV-filen mangler team_id.")

    if "player_name" not in df.columns:
        raise ValueError("EV-filen mangler player_name.")

    df = df.copy()
    df["position"] = df["position"].map(normalize_position)
    df["price_m"] = df.apply(get_price_m_from_row, axis=1)
    df["model_ev_for_budget_upgrade"] = df.apply(get_model_ev, axis=1)
    df["effective_ev_for_budget_upgrade"] = df.apply(get_effective_ev, axis=1)
    df["start_prob_for_budget_upgrade"] = df.apply(get_start_prob, axis=1)

    # Kandidater uden pris kan ikke bruges.
    df = df.loc[df["price_m"] > 0].copy()

    return df


def row_to_player_dict(row: pd.Series, formation: str) -> dict[str, Any]:
    item = {col: clean_value(row[col]) for col in row.index}

    price_m = get_price_m_from_row(row)
    model_ev = get_model_ev(row)
    effective_ev = get_effective_ev(row)

    item["player_id"] = str(item.get("player_id"))
    item["player_name"] = item.get("player_name")
    item["team_id"] = str(item.get("team_id"))
    item["position"] = normalize_position(item.get("position"))
    item["price_m"] = price_m
    item["price"] = int(round(price_m * 1_000_000))
    item["price_estimate"] = int(round(price_m * 1_000_000))
    item["optimizer_ev"] = model_ev
    item["optimizer_ev_adj"] = effective_ev
    item["selected_formation"] = formation
    item["budget_m"] = BUDGET_M
    item["max_per_team"] = MAX_PER_TEAM
    item["solver_quality_profile"] = "budget_floor_v1_price_as_baseline"

    return item


def team_counts(players: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in players:
        team = str(player.get("team_id") or player.get("team_name") or "")
        if not team:
            continue
        counts[team] = counts.get(team, 0) + 1
    return counts


def squad_price(players: list[dict[str, Any]]) -> float:
    return sum(get_price_m_from_row(player) for player in players)


def squad_raw_ev(players: list[dict[str, Any]]) -> float:
    return sum(get_model_ev(player) for player in players)


def squad_adj_ev(players: list[dict[str, Any]]) -> float:
    return sum(get_effective_ev(player) for player in players)


def is_valid_team_swap(
    current_players: list[dict[str, Any]],
    old_player: dict[str, Any],
    candidate: pd.Series,
) -> bool:
    counts = team_counts(current_players)

    old_team = str(old_player.get("team_id") or old_player.get("team_name") or "")
    new_team = str(candidate.get("team_id") or candidate.get("team_name") or "")

    if old_team:
        counts[old_team] = counts.get(old_team, 0) - 1
        if counts[old_team] <= 0:
            del counts[old_team]

    if new_team:
        counts[new_team] = counts.get(new_team, 0) + 1

    return all(count <= MAX_PER_TEAM for count in counts.values())


def build_selected_id_set(players: list[dict[str, Any]]) -> set[str]:
    return {str(player.get("player_id")) for player in players if player.get("player_id") is not None}


def find_best_budget_upgrade(
    formation: str,
    players: list[dict[str, Any]],
    candidates: pd.DataFrame,
) -> dict[str, Any] | None:
    current_price = squad_price(players)
    selected_ids = build_selected_id_set(players)

    best: dict[str, Any] | None = None

    for old_idx, old_player in enumerate(players):
        old_position = normalize_position(old_player.get("position"))
        old_price = get_price_m_from_row(old_player)
        old_ev = get_effective_ev(old_player)

        same_position_candidates = candidates.loc[
            (candidates["position"] == old_position)
            & (~candidates["player_id"].astype(str).isin(selected_ids))
            & (candidates["price_m"] > old_price)
        ].copy()

        if same_position_candidates.empty:
            continue

        for _, candidate in same_position_candidates.iterrows():
            new_price = float(candidate["price_m"])
            price_delta = new_price - old_price
            new_total_price = current_price + price_delta

            if new_total_price > BUDGET_M + 1e-9:
                continue

            if not is_valid_team_swap(players, old_player, candidate):
                continue

            candidate_start = float(candidate.get("start_prob_for_budget_upgrade", 0.25))
            if candidate_start < MIN_REASONABLE_START_PROB:
                continue

            new_ev = float(candidate["effective_ev_for_budget_upgrade"])
            ev_delta = new_ev - old_ev

            below_target = current_price < TARGET_MIN_PRICE_M
            budget_bonus = BUDGET_FILL_VALUE_PER_M * price_delta if below_target else 0.08 * price_delta

            utility = ev_delta + budget_bonus

            # Hvis vi stadig er under budgetgulvet, accepterer vi kontrolleret EV-tab.
            acceptable = utility > 0
            if below_target and ev_delta >= -MAX_ACCEPTED_EV_LOSS_PER_SWAP:
                acceptable = True

            if not acceptable:
                continue

            record = {
                "formation": formation,
                "old_idx": old_idx,
                "old_player_id": old_player.get("player_id"),
                "old_player_name": old_player.get("player_name"),
                "old_team_id": old_player.get("team_id"),
                "old_position": old_position,
                "old_price_m": old_price,
                "old_ev": old_ev,
                "new_player_id": str(candidate["player_id"]),
                "new_player_name": candidate["player_name"],
                "new_team_id": candidate["team_id"],
                "new_position": old_position,
                "new_price_m": new_price,
                "new_ev": new_ev,
                "price_delta_m": price_delta,
                "ev_delta": ev_delta,
                "utility": utility,
                "new_total_price_m": new_total_price,
                "candidate_row": candidate,
            }

            if best is None:
                best = record
                continue

            # Primært: størst utility.
            # Sekundært: brug mest budget.
            if record["utility"] > best["utility"] + 1e-9:
                best = record
            elif abs(record["utility"] - best["utility"]) <= 1e-9 and record["new_total_price_m"] > best["new_total_price_m"]:
                best = record

    return best


def apply_budget_floor_to_formation(
    formation: str,
    players: list[dict[str, Any]],
    candidates: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    upgraded = [dict(player) for player in players]
    changes: list[dict[str, Any]] = []

    # Normalisér eksisterende spillere.
    for player in upgraded:
        player["position"] = normalize_position(player.get("position"))
        player["price_m"] = get_price_m_from_row(player)
        player["optimizer_ev"] = get_model_ev(player)
        player["optimizer_ev_adj"] = get_effective_ev(player)
        player["selected_formation"] = formation

    iterations = 0
    while squad_price(upgraded) < TARGET_MIN_PRICE_M - 1e-9 and iterations < 25:
        iterations += 1

        best_swap = find_best_budget_upgrade(
            formation=formation,
            players=upgraded,
            candidates=candidates,
        )

        if best_swap is None:
            break

        candidate_row = best_swap.pop("candidate_row")
        new_player = row_to_player_dict(candidate_row, formation)

        upgraded[best_swap["old_idx"]] = new_player
        changes.append(best_swap)

    # Når vi har ramt budgetgulvet, prøv én ekstra lille opgradering, hvis den næsten er gratis i EV.
    extra_swap = find_best_budget_upgrade(
        formation=formation,
        players=upgraded,
        candidates=candidates,
    )

    if extra_swap is not None:
        if extra_swap["new_total_price_m"] <= BUDGET_M and extra_swap["ev_delta"] >= -0.25 and extra_swap["price_delta_m"] <= 1.5:
            candidate_row = extra_swap.pop("candidate_row")
            new_player = row_to_player_dict(candidate_row, formation)
            upgraded[extra_swap["old_idx"]] = new_player
            changes.append(extra_swap)

    total_price = round(squad_price(upgraded), 3)
    total_raw_ev = round(squad_raw_ev(upgraded), 6)
    total_adj_ev = round(squad_adj_ev(upgraded), 6)

    for player in upgraded:
        player["selected_formation"] = formation
        player["squad_total_price_m"] = total_price
        player["squad_total_ev"] = total_raw_ev
        player["squad_total_raw_ev"] = total_raw_ev
        player["squad_total_adj_ev"] = total_adj_ev
        player["budget_m"] = BUDGET_M
        player["max_per_team"] = MAX_PER_TEAM
        player["solver_quality_profile"] = "budget_floor_v1_price_as_baseline"

    return upgraded, changes


def print_summary(diag: pd.DataFrame, formation_summaries: list[dict[str, Any]]) -> None:
    print("\nBUDGET FLOOR UPGRADE")
    print(f"Budget: {BUDGET_M:.1f} mio.")
    print(f"Mål: mindst {TARGET_MIN_PRICE_M:.1f} mio. pr. formation")
    print("")

    for item in formation_summaries:
        print(
            f"{item['formation']}: "
            f"{item['price_before_m']:.1f} -> {item['price_after_m']:.1f} mio. "
            f"| swaps={item['swaps']} "
            f"| adjEV {item['adj_ev_before']:.3f} -> {item['adj_ev_after']:.3f}"
        )

    print("")
    if diag.empty:
        print("Ingen swaps lavet.")
    else:
        print("Swaps:")
        cols = [
            "formation",
            "old_player_name",
            "old_team_id",
            "old_price_m",
            "new_player_name",
            "new_team_id",
            "new_price_m",
            "price_delta_m",
            "ev_delta",
            "new_total_price_m",
        ]
        print(diag[cols].to_string(index=False))


def main() -> None:
    if not OPTIMAL_PATH.exists():
        raise FileNotFoundError(f"Mangler {OPTIMAL_PATH}")

    raw = load_json(OPTIMAL_PATH)
    formations_obj, container_key = get_formation_container(raw)
    candidates = load_candidate_pool()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = OPTIMAL_PATH.with_name(
        f"optimal_squads_by_formation.backup_before_budget_floor_{timestamp}.json"
    )
    shutil.copy2(OPTIMAL_PATH, backup_path)

    all_changes: list[dict[str, Any]] = []
    formation_summaries: list[dict[str, Any]] = []

    for formation, entry in list(formations_obj.items()):
        if "-" not in str(formation):
            continue

        players, players_key = get_players_from_entry(entry)
        if not players:
            continue

        price_before = squad_price(players)
        adj_before = squad_adj_ev(players)

        upgraded_players, changes = apply_budget_floor_to_formation(
            formation=str(formation),
            players=players,
            candidates=candidates,
        )

        price_after = squad_price(upgraded_players)
        adj_after = squad_adj_ev(upgraded_players)

        for change in changes:
            all_changes.append(change)

        formation_summaries.append(
            {
                "formation": str(formation),
                "price_before_m": price_before,
                "price_after_m": price_after,
                "adj_ev_before": adj_before,
                "adj_ev_after": adj_after,
                "swaps": len(changes),
            }
        )

        formations_obj[formation] = set_players_on_entry(entry, upgraded_players, players_key)

    dump_json(OPTIMAL_PATH, raw)

    diag = pd.DataFrame(all_changes)
    diag.to_csv(OUT_DIAG_PATH, index=False, encoding="utf-8-sig")

    print_summary(diag, formation_summaries)

    print("\nBackup:")
    print(backup_path)
    print("\nSkrev:")
    print(OPTIMAL_PATH)
    print(OUT_DIAG_PATH)


if __name__ == "__main__":
    main()