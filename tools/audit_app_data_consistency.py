from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
OPTIMAL_SQUADS_PATH = DATA_DIR / "optimal_squads_by_formation.json"
FIXTURES_PATH = DATA_DIR / "fixtures_group.csv"
START_SECURITY_PATH = DATA_DIR / "player_start_security_nt.csv"
MATCH_ODDS_PROBS_PATH = DATA_DIR / "match_odds_probs.csv"

OUT_TXT = DATA_DIR / "audit_app_data_consistency.txt"
OUT_JSON = DATA_DIR / "audit_app_data_consistency.json"

BUDGET_TOTAL = 50_000_000
MAX_PER_TEAM = 4

FORMATIONS = {
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
}

POSITION_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
    "MÃ…LMAND": "GK",
    "MALMAND": "GK",
    "FORSVAR": "DEF",
    "MIDTBANE": "MID",
    "ANGRIBER": "FWD",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_position(value: Any) -> str:
    text = str(value or "").strip().upper()
    return POSITION_MAP.get(text, text)


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = (
        text.replace("Ã¡", "a")
        .replace("Ã ", "a")
        .replace("Ã¤", "a")
        .replace("Ã¢", "a")
        .replace("Ã£", "a")
        .replace("Ã¥", "a")
        .replace("Ã©", "e")
        .replace("Ã¨", "e")
        .replace("Ã«", "e")
        .replace("Ãª", "e")
        .replace("Ã­", "i")
        .replace("Ã¬", "i")
        .replace("Ã¯", "i")
        .replace("Ã®", "i")
        .replace("Ã³", "o")
        .replace("Ã²", "o")
        .replace("Ã¶", "o")
        .replace("Ã´", "o")
        .replace("Ãµ", "o")
        .replace("Ãº", "u")
        .replace("Ã¹", "u")
        .replace("Ã¼", "u")
        .replace("Ã»", "u")
        .replace("Ã½", "y")
        .replace("Ã¿", "y")
        .replace("Ã±", "n")
        .replace("Ã§", "c")
        .replace("Å™", "r")
        .replace("Ä", "c")
        .replace("Å¡", "s")
        .replace("Å¾", "z")
        .replace("Ä‡", "c")
        .replace("Ä‘", "d")
        .replace("Å‚", "l")
        .replace("Ã¸", "o")
        .replace("Ã¦", "ae")
        .replace("Å“", "oe")
        .replace("ÃŸ", "ss")
        .replace("â€™", "")
        .replace("'", "")
    )
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def as_int_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def get_pool_price_dkk(player: dict[str, Any]) -> float:
    return to_float(player.get("price_estimate"), 0.0)


def get_optimizer_price_dkk(row: dict[str, Any]) -> float:
    if "price_m" in row:
        return to_float(row.get("price_m"), 0.0) * 1_000_000
    if "price_estimate" in row:
        return to_float(row.get("price_estimate"), 0.0)
    if "price" in row:
        price = to_float(row.get("price"), 0.0)
        if price < 100:
            return price * 1_000_000
        return price
    return 0.0


def make_merge_key(name: Any, team: Any, position: Any) -> str:
    return f"{normalize_name(name)}__{str(team or '').strip().lower()}__{normalize_position(position).lower()}"


def flatten_optimizer(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            if not isinstance(row, dict):
                continue
            formation = row.get("selected_formation") or row.get("formation") or "UNKNOWN"
            grouped[str(formation)].append(row)
        return dict(grouped)

    if isinstance(raw, dict):
        grouped = {}
        for formation, rows in raw.items():
            if not isinstance(rows, list):
                continue
            fixed_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fixed = dict(row)
                fixed["selected_formation"] = fixed.get("selected_formation") or fixed.get("formation") or formation
                fixed_rows.append(fixed)
            grouped[str(formation)] = fixed_rows
        return grouped

    return {}


def build_player_indexes(players: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    by_merge_key: dict[str, dict[str, Any]] = {}
    by_name_team_pos: dict[str, dict[str, Any]] = {}

    for player in players:
        player_id = as_int_str(player.get("player_id") or player.get("id"))
        if player_id:
            by_id[player_id] = player

        team = player.get("team_id") or player.get("team_name")
        pos = player.get("position")
        name = player.get("player_name") or player.get("name")

        explicit_merge_key = str(player.get("merge_key") or "").strip().lower()
        if explicit_merge_key:
            by_merge_key[explicit_merge_key] = player

        generated_key = make_merge_key(name, team, pos)
        by_name_team_pos[generated_key] = player

    return {
        "by_id": by_id,
        "by_merge_key": by_merge_key,
        "by_name_team_pos": by_name_team_pos,
    }


def find_player_for_optimizer_row(row: dict[str, Any], indexes: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    row_id = as_int_str(row.get("player_id") or row.get("id"))
    if row_id and row_id in indexes["by_id"]:
        return indexes["by_id"][row_id], "player_id"

    row_merge_key = str(row.get("merge_key") or "").strip().lower()
    if row_merge_key and row_merge_key in indexes["by_merge_key"]:
        return indexes["by_merge_key"][row_merge_key], "merge_key"

    generated_key = make_merge_key(
        row.get("player_name") or row.get("name"),
        row.get("team_id") or row.get("team_name"),
        row.get("position"),
    )
    if generated_key in indexes["by_name_team_pos"]:
        return indexes["by_name_team_pos"][generated_key], "name_team_position"

    return None, "not_found"


def audit_required_files() -> dict[str, Any]:
    required = {
        "player_pool_v1.json": PLAYER_POOL_PATH,
        "optimal_squads_by_formation.json": OPTIMAL_SQUADS_PATH,
        "fixtures_group.csv": FIXTURES_PATH,
        "player_start_security_nt.csv": START_SECURITY_PATH,
        "match_odds_probs.csv": MATCH_ODDS_PROBS_PATH,
    }

    result = {}
    for name, path in required.items():
        result[name] = {
            "exists": path.exists(),
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return result


def audit_match_odds_probs() -> dict[str, Any]:
    if not MATCH_ODDS_PROBS_PATH.exists():
        return {
            "exists": False,
            "row_count": 0,
            "covered_match_ids": [],
            "message": "match_odds_probs.csv mangler.",
        }

    rows = load_csv(MATCH_ODDS_PROBS_PATH)
    match_ids = [row.get("match_id", "") for row in rows if row.get("match_id", "")]
    teams = sorted(
        {
            str(row.get("home", "")).strip()
            for row in rows
            if row.get("home")
        }
        | {
            str(row.get("away", "")).strip()
            for row in rows
            if row.get("away")
        }
    )

    return {
        "exists": True,
        "row_count": len(rows),
        "first_match_id": match_ids[0] if match_ids else None,
        "last_match_id": match_ids[-1] if match_ids else None,
        "team_count": len(teams),
        "teams": teams,
    }


def audit_player_pool(players: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {}
    duplicate_ids = []
    position_counts = Counter()
    team_counts = Counter()
    missing_price = []
    missing_position = []
    missing_team = []

    for player in players:
        player_id = as_int_str(player.get("player_id") or player.get("id"))
        if not player_id:
            continue

        if player_id in by_id:
            duplicate_ids.append(player_id)
        by_id[player_id] = player

        position = normalize_position(player.get("position"))
        team_name = str(player.get("team_name") or player.get("team_id") or "").strip()

        position_counts[position] += 1
        team_counts[team_name] += 1

        if get_pool_price_dkk(player) <= 0:
            missing_price.append(player_id)
        if not position:
            missing_position.append(player_id)
        if not team_name:
            missing_team.append(player_id)

    return {
        "player_count": len(players),
        "unique_player_ids": len(by_id),
        "duplicate_ids": duplicate_ids[:50],
        "position_counts": dict(position_counts),
        "team_count": len(team_counts),
        "missing_price_count": len(missing_price),
        "missing_position_count": len(missing_position),
        "missing_team_count": len(missing_team),
        "sample_missing_price_ids": missing_price[:20],
    }


def audit_optimizer(
    optimizer_by_formation: dict[str, list[dict[str, Any]]],
    indexes: dict[str, Any],
) -> dict[str, Any]:
    result = {}

    for formation, expected_counts in FORMATIONS.items():
        rows = optimizer_by_formation.get(formation, [])
        issues = []
        warnings = []

        if not rows:
            result[formation] = {
                "row_count": 0,
                "complete": False,
                "valid": False,
                "issues": [f"Ingen optimizer-rÃ¦kker fundet for {formation}."],
                "warnings": [],
            }
            continue

        row_count = len(rows)
        if row_count != 11:
            issues.append(f"Forkert antal rÃ¦kker: {row_count}/11.")

        optimizer_position_counts = Counter(normalize_position(row.get("position")) for row in rows)
        for pos, expected_count in expected_counts.items():
            actual_count = optimizer_position_counts.get(pos, 0)
            if actual_count != expected_count:
                issues.append(f"Forkert antal {pos}: {actual_count}/{expected_count}.")

        used_pool_ids = []
        missing_players = []
        position_mismatches = []
        price_mismatches = []
        pool_total_price = 0.0
        optimizer_total_price = 0.0
        teams = []
        match_methods = Counter()

        for row in rows:
            player, method = find_player_for_optimizer_row(row, indexes)
            match_methods[method] += 1

            optimizer_pos = normalize_position(row.get("position"))
            optimizer_total_price += get_optimizer_price_dkk(row)

            if not player:
                missing_players.append(
                    {
                        "player_id": row.get("player_id"),
                        "player_name": row.get("player_name"),
                        "team_id": row.get("team_id"),
                        "position": row.get("position"),
                        "merge_key": row.get("merge_key"),
                    }
                )
                continue

            pool_id = as_int_str(player.get("player_id") or player.get("id"))
            used_pool_ids.append(pool_id)

            pool_pos = normalize_position(player.get("position"))
            if pool_pos != optimizer_pos:
                position_mismatches.append(
                    {
                        "player_id": pool_id,
                        "player_name": player.get("player_name"),
                        "optimizer_position": optimizer_pos,
                        "player_pool_position": pool_pos,
                    }
                )

            pool_price = get_pool_price_dkk(player)
            pool_total_price += pool_price

            opt_price = get_optimizer_price_dkk(row)
            if opt_price > 0 and pool_price > 0:
                diff = abs(pool_price - opt_price)
                if diff > 1_000:
                    price_mismatches.append(
                        {
                            "player_id": pool_id,
                            "player_name": player.get("player_name"),
                            "optimizer_price": round(opt_price),
                            "player_pool_price": round(pool_price),
                            "difference": round(diff),
                        }
                    )

            team_name = str(player.get("team_name") or player.get("team_id") or row.get("team_id") or "").strip()
            teams.append(team_name)

        duplicate_ids = [player_id for player_id, count in Counter(used_pool_ids).items() if player_id and count > 1]
        if duplicate_ids:
            issues.append(f"Duplikatspillere i optimizer-hold: {', '.join(duplicate_ids[:10])}.")

        if missing_players:
            issues.append(f"{len(missing_players)} optimizer-spillere findes ikke i player_pool_v1.json.")

        if position_mismatches:
            issues.append(f"{len(position_mismatches)} position mismatch mellem optimizer og player_pool.")

        if pool_total_price > BUDGET_TOTAL:
            issues.append(
                f"Player-pool-pris overskrider budget: {pool_total_price:,.0f} > {BUDGET_TOTAL:,.0f}."
            )

        team_counter = Counter(teams)
        teams_over_limit = {team: count for team, count in team_counter.items() if team and count > MAX_PER_TEAM}
        if teams_over_limit:
            issues.append(f"Maks {MAX_PER_TEAM} fra land overskredet: {teams_over_limit}.")

        if price_mismatches:
            warnings.append(
                f"{len(price_mismatches)} prisforskelle mellem optimizer og player_pool over 1.000 kr."
            )

        result[formation] = {
            "row_count": row_count,
            "complete": row_count == 11,
            "valid": len(issues) == 0,
            "optimizer_position_counts": dict(optimizer_position_counts),
            "pool_total_price": round(pool_total_price),
            "optimizer_total_price": round(optimizer_total_price),
            "budget_total": BUDGET_TOTAL,
            "team_counts": dict(team_counter),
            "match_methods": dict(match_methods),
            "issues": issues,
            "warnings": warnings,
            "missing_players": missing_players[:30],
            "position_mismatches": position_mismatches[:30],
            "price_mismatches_sample": price_mismatches[:20],
        }

    return result


def format_money_like(value: Any) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", ".")
    except Exception:
        return str(value)


def format_report(audit: dict[str, Any]) -> str:
    lines = []
    lines.append("VM 2026 APP DATA AUDIT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("1. PÃ¥krÃ¦vede app-filer")
    lines.append("-" * 60)
    for filename, info in audit["required_files"].items():
        status = "OK" if info["exists"] else "MANGLER"
        size = info["size_bytes"] if info["size_bytes"] is not None else "-"
        lines.append(f"{status:8} {filename:35} {size}")
    lines.append("")

    lines.append("2. Player pool")
    lines.append("-" * 60)
    pool = audit["player_pool"]
    lines.append(f"Spillere: {pool['player_count']}")
    lines.append(f"Unikke player_id: {pool['unique_player_ids']}")
    lines.append(f"Positioner: {pool['position_counts']}")
    lines.append(f"Lande/hold: {pool['team_count']}")
    lines.append(f"Mangler pris: {pool['missing_price_count']}")
    lines.append(f"Mangler position: {pool['missing_position_count']}")
    lines.append(f"Mangler land/hold: {pool['missing_team_count']}")
    if pool["duplicate_ids"]:
        lines.append(f"Dubletter: {pool['duplicate_ids']}")
    lines.append("")

    lines.append("3. Match odds probs")
    lines.append("-" * 60)
    odds = audit["match_odds_probs"]
    if odds["exists"]:
        lines.append(f"RÃ¦kker: {odds['row_count']}")
        lines.append(f"FÃ¸rste match_id: {odds['first_match_id']}")
        lines.append(f"Sidste match_id: {odds['last_match_id']}")
        lines.append(f"Antal hold dÃ¦kket: {odds['team_count']}")
    else:
        lines.append("match_odds_probs.csv mangler.")
    lines.append("")

    lines.append("4. Optimizer vs app-data")
    lines.append("-" * 60)
    optimizer = audit["optimizer"]
    for formation, info in optimizer.items():
        status = "OK" if info["valid"] else "FEJL"
        lines.append(
            f"{status:5} {formation:5} rÃ¦kker={info['row_count']} "
            f"pris={format_money_like(info.get('pool_total_price', 0))} "
            f"pos={info.get('optimizer_position_counts', {})} "
            f"match={info.get('match_methods', {})}"
        )
        for issue in info.get("issues", []):
            lines.append(f"      - {issue}")
        for warning in info.get("warnings", []):
            lines.append(f"      ! {warning}")
    lines.append("")

    all_valid = all(info.get("valid", False) for info in optimizer.values())
    lines.append("5. Samlet vurdering")
    lines.append("-" * 60)
    if all_valid:
        lines.append("OK: Optimizer-holdene matcher appens player_pool, positioner, budget og maks. 4 fra land.")
    else:
        lines.append("OBS: Der er fejl/mismatch i mindst Ã©n formation. Se detaljer ovenfor.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    audit: dict[str, Any] = {}
    audit["required_files"] = audit_required_files()

    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(f"Mangler {PLAYER_POOL_PATH}")

    if not OPTIMAL_SQUADS_PATH.exists():
        raise FileNotFoundError(f"Mangler {OPTIMAL_SQUADS_PATH}")

    players = load_json(PLAYER_POOL_PATH)
    if not isinstance(players, list):
        raise ValueError("player_pool_v1.json er ikke en liste.")

    indexes = build_player_indexes(players)

    raw_optimizer = load_json(OPTIMAL_SQUADS_PATH)
    optimizer_by_formation = flatten_optimizer(raw_optimizer)

    audit["player_pool"] = audit_player_pool(players)
    audit["match_odds_probs"] = audit_match_odds_probs()
    audit["optimizer"] = audit_optimizer(optimizer_by_formation, indexes)

    OUT_JSON.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = format_report(audit)
    OUT_TXT.write_text(report, encoding="utf-8")

    print(report)
    print("")
    print(f"Skrev: {OUT_TXT}")
    print(f"Skrev: {OUT_JSON}")


if __name__ == "__main__":
    main()
