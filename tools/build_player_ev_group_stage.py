from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
FIXTURES_PATH = DATA_DIR / "fixtures_group.csv"
MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
IMPACT_REPORT_PATH = DATA_DIR / "player_ev_fixture_strength_impact_report.csv"

ROUND_WEIGHTS = {
    1: 1.00,
    2: 0.95,
    3: 0.90,
}

GOAL_POINTS = {
    "GK": 6.0,
    "DEF": 6.0,
    "MID": 5.0,
    "FWD": 4.0,
}
ASSIST_POINTS = 3.0
SHOT_ON_TARGET_POINTS = 1.0
CLEAN_SHEET_POINTS = {
    "GK": 2.8,
    "DEF": 2.2,
    "MID": 0.0,
    "FWD": 0.0,
}
YELLOW_CARD_POINTS = -1.0

IMPACT_FIELDS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "price",
    "old_ev",
    "new_ev",
    "ev_diff",
    "ev_diff_pct",
    "main_reason",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    text = txt(value).replace(",", ".")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_multiplier_lookup() -> dict[tuple[str, str], dict[str, float | str]]:
    _, rows = read_csv(MULTIPLIERS_PATH)
    lookup: dict[tuple[str, str], dict[str, float | str]] = {}

    for row in rows:
        match_id = txt(row.get("match_id"))
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        lookup[(match_id, home)] = {
            "opponent": away,
            "goal_multiplier": to_float(row.get("home_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("home_assist_multiplier"), 1.0),
            "clean_sheet_multiplier": to_float(row.get("home_clean_sheet_multiplier"), 1.0),
        }
        lookup[(match_id, away)] = {
            "opponent": home,
            "goal_multiplier": to_float(row.get("away_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("away_assist_multiplier"), 1.0),
            "clean_sheet_multiplier": to_float(row.get("away_clean_sheet_multiplier"), 1.0),
        }

    return lookup


def load_fixture_match_lookup() -> dict[tuple[str, str, str], str]:
    _, rows = read_csv(FIXTURES_PATH)
    lookup: dict[tuple[str, str, str], str] = {}
    for row in rows:
        match_id = txt(row.get("match_id"))
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        kickoff = txt(row.get("kickoff_dk"))
        lookup[(home, away, kickoff)] = match_id
        lookup[(away, home, kickoff)] = match_id
    return lookup


def match_id_for(row: dict[str, str], match_no: int, fixture_lookup: dict[tuple[str, str, str], str]) -> str:
    team = txt(row.get("team_id")).upper()
    opponent = txt(row.get(f"match_{match_no}_opponent_team")).upper()
    kickoff = txt(row.get(f"match_{match_no}_kickoff"))
    return fixture_lookup.get((team, opponent, kickoff), "")


def recompute_match_total(row: dict[str, str], match_no: int) -> float:
    position = txt(row.get("position")).upper()
    goal_ev = to_float(row.get(f"match_{match_no}_goal_ev"))
    assist_ev = to_float(row.get(f"match_{match_no}_assist_ev"))
    sot_ev = to_float(row.get(f"match_{match_no}_shots_on_target_ev"))
    clean_sheet_ev = to_float(row.get(f"match_{match_no}_clean_sheet_ev"))
    card_ev = to_float(row.get(f"match_{match_no}_card_ev"))
    result_ev = to_float(row.get(f"match_{match_no}_result_ev"))
    team_scores_ev = to_float(row.get(f"match_{match_no}_team_scores_ev"))
    opponent_scores_ev = to_float(row.get(f"match_{match_no}_opponent_scores_ev"))
    on_pitch_ev = to_float(row.get(f"match_{match_no}_on_pitch_ev"))

    return (
        goal_ev * GOAL_POINTS.get(position, 5.0)
        + assist_ev * ASSIST_POINTS
        + sot_ev * SHOT_ON_TARGET_POINTS
        + clean_sheet_ev * CLEAN_SHEET_POINTS.get(position, 0.0)
        + card_ev * YELLOW_CARD_POINTS
        + result_ev
        + team_scores_ev
        + opponent_scores_ev
        + on_pitch_ev
    )


def current_component_weighted_sum(row: dict[str, str]) -> float:
    return sum(to_float(row.get(f"match_{match_no}_weighted_match_ev")) for match_no in [1, 2, 3])


def current_component_total_sum(row: dict[str, str]) -> float:
    return sum(to_float(row.get(f"match_{match_no}_total_ev_next_match")) for match_no in [1, 2, 3])


def main() -> int:
    fields, rows = read_csv(PLAYER_EV_PATH)
    multiplier_lookup = load_multiplier_lookup()
    fixture_lookup = load_fixture_match_lookup()

    warnings: list[str] = []
    impact_rows: list[dict[str, Any]] = []
    updated_count = 0

    for row in rows:
        old_ev = to_float(row.get("weighted_group_stage_ev"))
        old_total_ev = to_float(row.get("total_ev_group_stage"))
        old_component_weighted = current_component_weighted_sum(row)
        old_component_total = current_component_total_sum(row)
        weighted_scale = old_ev / old_component_weighted if old_component_weighted else 1.0
        total_scale = old_total_ev / old_component_total if old_component_total else weighted_scale
        max_abs_reason_diff = 0.0
        main_reason = "neutral"
        any_multiplier = False

        for match_no in [1, 2, 3]:
            match_id = match_id_for(row, match_no, fixture_lookup)
            team = txt(row.get("team_id")).upper()
            multiplier = multiplier_lookup.get((match_id, team)) if match_id else None

            if not multiplier:
                opponent = txt(row.get(f"match_{match_no}_opponent_team"))
                if opponent:
                    warnings.append(f"missing_multiplier player={txt(row.get('player_id'))} team={team} match_no={match_no} opponent={opponent}")
                # Keep existing component totals for rows without a fixture match. Some legacy
                # EV rows have aggregate EV but no match breakdown.
                continue
            else:
                goal_multiplier = float(multiplier["goal_multiplier"])
                assist_multiplier = float(multiplier["assist_multiplier"])
                clean_sheet_multiplier = float(multiplier["clean_sheet_multiplier"])
                any_multiplier = True

            old_goal = to_float(row.get(f"match_{match_no}_goal_ev"))
            old_assist = to_float(row.get(f"match_{match_no}_assist_ev"))
            old_cs = to_float(row.get(f"match_{match_no}_clean_sheet_ev"))

            row[f"match_{match_no}_goal_ev"] = fmt(old_goal * goal_multiplier)
            row[f"match_{match_no}_assist_ev"] = fmt(old_assist * assist_multiplier)
            row[f"match_{match_no}_clean_sheet_ev"] = fmt(old_cs * clean_sheet_multiplier)

            reason_diff = abs(clean_sheet_multiplier - 1.0)
            reason = f"clean_sheet_multiplier={clean_sheet_multiplier:.3f}"
            if abs(goal_multiplier - 1.0) > reason_diff:
                reason_diff = abs(goal_multiplier - 1.0)
                reason = f"goal_multiplier={goal_multiplier:.3f}"
            if abs(assist_multiplier - 1.0) > reason_diff:
                reason_diff = abs(assist_multiplier - 1.0)
                reason = f"assist_multiplier={assist_multiplier:.3f}"
            if reason_diff > max_abs_reason_diff:
                max_abs_reason_diff = reason_diff
                main_reason = f"match_{match_no}_{reason}"

            total = recompute_match_total(row, match_no)
            weighted = total * ROUND_WEIGHTS[match_no]
            row[f"match_{match_no}_total_ev_next_match"] = fmt(total)
            row[f"match_{match_no}_weighted_match_ev"] = fmt(weighted)

        if any_multiplier:
            new_component_weighted = current_component_weighted_sum(row)
            new_component_total = current_component_total_sum(row)
            new_ev = max(0.0, new_component_weighted * weighted_scale)
            total_ev = max(0.0, new_component_total * total_scale)
            row["weighted_group_stage_ev"] = fmt(new_ev)
            row["optimizer_ev"] = fmt(new_ev)
            row["total_ev_group_stage"] = fmt(total_ev)
        else:
            new_ev = old_ev

        if any_multiplier:
            updated_count += 1

        diff = new_ev - old_ev
        pct = diff / old_ev if old_ev else 0.0
        impact_rows.append(
            {
                "player_id": txt(row.get("player_id")),
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "position": txt(row.get("position")),
                "price": txt(row.get("price")),
                "old_ev": fmt(old_ev),
                "new_ev": fmt(new_ev),
                "ev_diff": fmt(diff),
                "ev_diff_pct": fmt(pct),
                "main_reason": main_reason,
            }
        )

    rows.sort(key=lambda row: (to_float(row.get("weighted_group_stage_ev")), to_float(row.get("total_ev_group_stage"))), reverse=True)
    write_csv(PLAYER_EV_PATH, fields, rows)
    write_csv(IMPACT_REPORT_PATH, IMPACT_FIELDS, impact_rows)

    warnings_unique = list(dict.fromkeys(warnings))
    print(f"Skrevet: {PLAYER_EV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {IMPACT_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Antal spillere opdateret: {updated_count}")
    print(f"Warnings om manglende multipliers: {len(warnings_unique)}")

    print("Top 20 EV-stigninger:")
    for row in sorted(impact_rows, key=lambda item: to_float(item["ev_diff"]), reverse=True)[:20]:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['old_ev']} -> {row['new_ev']} | {row['ev_diff']} | {row['main_reason']}")

    print("Top 20 EV-fald:")
    for row in sorted(impact_rows, key=lambda item: to_float(item["ev_diff"]))[:20]:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['old_ev']} -> {row['new_ev']} | {row['ev_diff']} | {row['main_reason']}")

    if warnings_unique:
        print("Første warnings:")
        for warning in warnings_unique[:20]:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
