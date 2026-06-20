from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
FIXTURES_PATH = DATA_DIR / "fixtures_group.csv"
MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
MATCH_ODDS_PATH = DATA_DIR / "match_odds_probs.csv"
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

BASE_COMPONENTS = {
    "goal": "goal_multiplier",
    "assist": "assist_multiplier",
    "clean_sheet": "clean_sheet_multiplier",
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_fieldname(value: Any) -> str:
    return txt(value).lstrip("\ufeff")


def normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {normalize_fieldname(key): value for key, value in row.items()}


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
        fieldnames = [normalize_fieldname(field) for field in (reader.fieldnames or [])]
        rows = [normalize_row_keys(row) for row in reader]
        return fieldnames, rows


def git_executable() -> str | None:
    candidates = [
        shutil.which("git"),
        r"C:\Users\Administrator\AppData\Local\GitHubDesktop\app-3.5.12\resources\app\git\cmd\git.exe",
        r"C:\Users\Administrator\AppData\Local\GitHubDesktop\app-3.5.8\resources\app\git\cmd\git.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def load_head_ev_rows_from_git() -> tuple[list[str], list[dict[str, str]]]:
    git = git_executable()
    if not git:
        raise RuntimeError("Kan ikke finde git.exe til EV fallback.")

    result = subprocess.run(
        [git, "show", "HEAD:data/player_ev_group_stage_v1.csv"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    fieldnames = [normalize_fieldname(field) for field in (reader.fieldnames or [])]
    rows = [normalize_row_keys(row) for row in reader]
    return fieldnames, rows


def read_ev_rows_with_fallback(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    fields, rows = read_csv(path)
    if rows:
        return fields, rows, "working_tree"

    head_fields, head_rows = load_head_ev_rows_from_git()
    if not head_rows:
        raise RuntimeError(
            f"{path.relative_to(PROJECT_ROOT)} er header-only, og HEAD fallback indeholder heller ingen rækker."
        )
    return head_fields, head_rows, "git_head_fallback"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    normalized_fieldnames = [normalize_fieldname(field) for field in fieldnames]
    normalized_rows = [normalize_row_keys(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=normalized_fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def ensure_base_component_fields(fieldnames: list[str]) -> list[str]:
    next_fields = list(fieldnames)
    for match_no in [1, 2, 3]:
        for component in BASE_COMPONENTS:
            field = f"match_{match_no}_{component}_ev_base"
            if field not in next_fields:
                next_fields.append(field)
        for field in [
            f"match_{match_no}_goal_multiplier",
            f"match_{match_no}_assist_multiplier",
            f"match_{match_no}_clean_sheet_multiplier",
        ]:
            if field not in next_fields:
                next_fields.append(field)
    for field in ["p_6_points_after_2", "round3_rotation_factor"]:
        if field not in next_fields:
            next_fields.append(field)
    return next_fields


def round_for_match_id(match_id: Any) -> int:
    mid = int(txt(match_id) or "0")
    if 1 <= mid <= 24:
        return 1
    if 25 <= mid <= 48:
        return 2
    if 49 <= mid <= 72:
        return 3
    return 0


def load_team_round_win_probs() -> dict[str, dict[int, float]]:
    if not MATCH_ODDS_PATH.exists():
        return {}

    _, rows = read_csv(MATCH_ODDS_PATH)
    wins: dict[str, dict[int, float]] = {}
    for row in rows:
        match_id = txt(row.get("match_id"))
        rnd = round_for_match_id(match_id)
        if rnd not in {1, 2, 3}:
            continue
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        if home:
            wins.setdefault(home, {})[rnd] = to_float(row.get("home_win_prob_fair"))
        if away:
            wins.setdefault(away, {})[rnd] = to_float(row.get("away_win_prob_fair"))
    return wins


def round3_rotation_factor_for_team(team: str, team_round_win_probs: dict[str, dict[int, float]]) -> tuple[float, float]:
    p6 = team_round_win_probs.get(team, {}).get(1, 0.0) * team_round_win_probs.get(team, {}).get(2, 0.0)
    if p6 >= 0.55:
        return p6, 0.62
    if p6 >= 0.40:
        return p6, 0.74
    if p6 >= 0.25:
        return p6, 0.86
    return p6, 1.0


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


def load_fixture_match_lookup() -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str], str]]:
    _, rows = read_csv(FIXTURES_PATH)
    lookup: dict[tuple[str, str, str], str] = {}
    pair_candidates: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        match_id = txt(row.get("match_id"))
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        kickoff = txt(row.get("kickoff_dk"))
        lookup[(home, away, kickoff)] = match_id
        lookup[(away, home, kickoff)] = match_id
        pair_candidates.setdefault((home, away), set()).add(match_id)
        pair_candidates.setdefault((away, home), set()).add(match_id)

    pair_lookup = {
        pair: next(iter(match_ids))
        for pair, match_ids in pair_candidates.items()
        if len(match_ids) == 1
    }
    return lookup, pair_lookup


def match_id_for(
    row: dict[str, str],
    match_no: int,
    fixture_lookup: dict[tuple[str, str, str], str],
    fixture_pair_lookup: dict[tuple[str, str], str],
) -> str:
    team = txt(row.get("team_id")).upper()
    opponent = txt(row.get(f"match_{match_no}_opponent_team")).upper()
    kickoff = txt(row.get(f"match_{match_no}_kickoff"))
    if kickoff and kickoff not in {"0", "0.0"}:
        match_id = fixture_lookup.get((team, opponent, kickoff), "")
        if match_id:
            return match_id
    return fixture_pair_lookup.get((team, opponent), "")


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


def has_component_breakdown(row: dict[str, str]) -> bool:
    if abs(current_component_weighted_sum(row)) > 1e-9 or abs(current_component_total_sum(row)) > 1e-9:
        return True

    component_suffixes = [
        "goal_ev",
        "assist_ev",
        "shots_on_target_ev",
        "clean_sheet_ev",
        "card_ev",
        "result_ev",
        "team_scores_ev",
        "opponent_scores_ev",
        "on_pitch_ev",
        "start_minutes_ev",
    ]
    for match_no in [1, 2, 3]:
        for suffix in component_suffixes:
            if abs(to_float(row.get(f"match_{match_no}_{suffix}"))) > 1e-9:
                return True
    return False


def aggregate_ev_fallback(row: dict[str, str]) -> tuple[float, float]:
    weighted_candidates = [
        to_float(row.get("weighted_group_stage_ev")),
        to_float(row.get("optimizer_ev")),
        to_float(row.get("price_quality_ev")),
    ]
    total_candidates = [
        to_float(row.get("total_ev_group_stage")),
        to_float(row.get("weighted_group_stage_ev_before_price_quality")),
        to_float(row.get("optimizer_ev_before_price_quality")),
        to_float(row.get("model_ev_before_price_quality")),
        to_float(row.get("price_quality_raw_ev")),
    ]
    weighted = max(weighted_candidates)
    total = max(total_candidates)
    return weighted, total


def read_existing_multiplier(row: dict[str, str], match_no: int, component: str) -> float | None:
    field = f"match_{match_no}_{BASE_COMPONENTS[component]}"
    raw = txt(row.get(field))
    if not raw:
        return None
    value = to_float(raw, 0.0)
    if abs(value) <= 1e-9:
        return None
    return value


def seed_base_component_value(
    row: dict[str, str],
    match_no: int,
    component: str,
    current_multiplier: float,
) -> float:
    base_field = f"match_{match_no}_{component}_ev_base"
    base_raw = txt(row.get(base_field))
    if base_raw:
        return to_float(base_raw)

    adjusted_field = f"match_{match_no}_{component}_ev"
    adjusted_value = to_float(row.get(adjusted_field))
    previous_multiplier = read_existing_multiplier(row, match_no, component)

    if previous_multiplier is not None:
        return adjusted_value / previous_multiplier

    # Transition path for legacy rows without stable base fields or stored previous
    # multipliers. If a current fixture multiplier exists, assume the current adjusted
    # component was produced from that multiplier and back it out once; future runs
    # then use the persisted base field only.
    if abs(current_multiplier) > 1e-9 and abs(adjusted_value) > 1e-9:
        return adjusted_value / current_multiplier

    return adjusted_value


def main() -> int:
    fields, rows, row_source = read_ev_rows_with_fallback(PLAYER_EV_PATH)
    fields = ensure_base_component_fields(fields)
    multiplier_lookup = load_multiplier_lookup()
    fixture_lookup, fixture_pair_lookup = load_fixture_match_lookup()
    team_round_win_probs = load_team_round_win_probs()

    warnings: list[str] = []
    impact_rows: list[dict[str, Any]] = []
    updated_count = 0

    for row in rows:
        team = txt(row.get("team_id")).upper()
        p6, round3_rotation_factor = round3_rotation_factor_for_team(team, team_round_win_probs)
        row["p_6_points_after_2"] = fmt(p6)
        row["round3_rotation_factor"] = fmt(round3_rotation_factor)

        old_ev = to_float(row.get("weighted_group_stage_ev"))
        old_total_ev = to_float(row.get("total_ev_group_stage"))
        old_component_weighted = current_component_weighted_sum(row)
        old_component_total = current_component_total_sum(row)
        row_has_breakdown = has_component_breakdown(row)
        weighted_scale = old_ev / old_component_weighted if old_component_weighted else 1.0
        total_scale = old_total_ev / old_component_total if old_component_total else weighted_scale
        max_abs_reason_diff = 0.0
        main_reason = "neutral"
        any_multiplier = False

        for match_no in [1, 2, 3]:
            match_id = match_id_for(row, match_no, fixture_lookup, fixture_pair_lookup)
            multiplier = multiplier_lookup.get((match_id, team)) if match_id else None

            if not multiplier:
                opponent = txt(row.get(f"match_{match_no}_opponent_team"))
                if opponent:
                    warnings.append(f"missing_multiplier player={txt(row.get('player_id'))} team={team} match_no={match_no} opponent={opponent}")
                # Keep existing component totals for rows without a fixture match. Some legacy
                # EV rows have aggregate EV but no match breakdown.
                for component in BASE_COMPONENTS:
                    base_field = f"match_{match_no}_{component}_ev_base"
                    row[base_field] = fmt(seed_base_component_value(row, match_no, component, 1.0))
                continue
            else:
                goal_multiplier = float(multiplier["goal_multiplier"])
                assist_multiplier = float(multiplier["assist_multiplier"])
                clean_sheet_multiplier = float(multiplier["clean_sheet_multiplier"])
                any_multiplier = True

            base_goal = seed_base_component_value(row, match_no, "goal", goal_multiplier)
            base_assist = seed_base_component_value(row, match_no, "assist", assist_multiplier)
            base_cs = seed_base_component_value(row, match_no, "clean_sheet", clean_sheet_multiplier)

            row[f"match_{match_no}_goal_ev_base"] = fmt(base_goal)
            row[f"match_{match_no}_assist_ev_base"] = fmt(base_assist)
            row[f"match_{match_no}_clean_sheet_ev_base"] = fmt(base_cs)

            row[f"match_{match_no}_goal_multiplier"] = fmt(goal_multiplier)
            row[f"match_{match_no}_assist_multiplier"] = fmt(assist_multiplier)
            row[f"match_{match_no}_clean_sheet_multiplier"] = fmt(clean_sheet_multiplier)

            row[f"match_{match_no}_goal_ev"] = fmt(base_goal * goal_multiplier)
            row[f"match_{match_no}_assist_ev"] = fmt(base_assist * assist_multiplier)
            row[f"match_{match_no}_clean_sheet_ev"] = fmt(base_cs * clean_sheet_multiplier)

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

        if any_multiplier and row_has_breakdown:
            new_component_weighted = current_component_weighted_sum(row)
            new_component_total = current_component_total_sum(row)
            new_ev = max(0.0, new_component_weighted * weighted_scale)
            total_ev = max(0.0, new_component_total * total_scale)
            row["weighted_group_stage_ev"] = fmt(new_ev)
            row["optimizer_ev"] = fmt(new_ev)
            row["total_ev_group_stage"] = fmt(total_ev)
        else:
            new_ev, total_ev = aggregate_ev_fallback(row)
            row["weighted_group_stage_ev"] = fmt(new_ev)
            row["optimizer_ev"] = fmt(new_ev)
            row["total_ev_group_stage"] = fmt(total_ev)

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
    print(f"Kilderækker indlæst fra: {row_source}")
    print(f"Antal inputrækker: {len(rows)}")
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
