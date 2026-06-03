from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

FIXTURES_PATH = DATA_DIR / "fixtures_group.csv"
ODDS_PATH = DATA_DIR / "match_odds_probs.csv"
TEAM_POWER_PATH = DATA_DIR / "team_power.csv"
CLEAN_SHEET_PROBS_PATH = DATA_DIR / "clean_sheet_probs_bet365.csv"
OUT_PATH = DATA_DIR / "fixture_strength_multipliers.csv"

OUT_FIELDS = [
    "match_id",
    "home",
    "away",
    "kickoff_dk",
    "home_win_prob_fair",
    "draw_prob_fair",
    "away_win_prob_fair",
    "home_strength_advantage",
    "away_strength_advantage",
    "home_goal_multiplier",
    "away_goal_multiplier",
    "home_assist_multiplier",
    "away_assist_multiplier",
    "home_clean_sheet_prob_fair",
    "away_clean_sheet_prob_fair",
    "home_clean_sheet_multiplier",
    "away_clean_sheet_multiplier",
    "source",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any) -> float | None:
    text = txt(value).replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_odds() -> dict[str, dict[str, str]]:
    if not ODDS_PATH.exists():
        return {}
    return {txt(row.get("match_id")): row for row in read_csv(ODDS_PATH)}


def power_value(row: dict[str, str]) -> float | None:
    for field in [
        "team_power",
        "power",
        "rating",
        "strength",
        "strength_score",
        "team_rating",
        "elo",
    ]:
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


def load_team_power() -> dict[str, float]:
    if not TEAM_POWER_PATH.exists():
        return {}

    powers: dict[str, float] = {}
    for row in read_csv(TEAM_POWER_PATH):
        team = txt(row.get("team_id") or row.get("team") or row.get("country") or row.get("code")).upper()
        value = power_value(row)
        if team and value is not None:
            powers[team] = value
    return powers


def load_clean_sheet_probs() -> tuple[dict[tuple[str, str], float], float | None]:
    if not CLEAN_SHEET_PROBS_PATH.exists():
        return {}, None

    probs: dict[tuple[str, str], float] = {}
    values: list[float] = []
    for row in read_csv(CLEAN_SHEET_PROBS_PATH):
        match_id = txt(row.get("match_id"))
        team = txt(row.get("team_id")).upper()
        prob = to_float(row.get("clean_sheet_prob_fair"))
        if match_id and team and prob is not None:
            probs[(match_id, team)] = prob
            values.append(prob)

    if not values:
        return probs, None
    return probs, sum(values) / len(values)


def advantage_from_power(home: str, away: str, team_power: dict[str, float]) -> tuple[float, float, str]:
    home_power = team_power.get(home)
    away_power = team_power.get(away)
    if home_power is None or away_power is None:
        return 0.0, 0.0, "neutral_default"

    raw_diff = home_power - away_power
    scale = max(1.0, abs(home_power), abs(away_power))
    advantage = clamp(raw_diff / scale, -0.50, 0.50)
    return advantage, -advantage, "team_power_fallback"


def multipliers(advantage: float) -> tuple[float, float, float]:
    goal = clamp(1 + 0.70 * advantage, 0.75, 1.35)
    assist = clamp(1 + 0.50 * advantage, 0.80, 1.25)
    clean_sheet = clamp(1 + 1.10 * advantage, 0.55, 1.45)
    return goal, assist, clean_sheet


def clean_sheet_multiplier_from_prob(prob: float, baseline_prob: float) -> float:
    return clamp(1 + 1.35 * (prob - baseline_prob), 0.55, 1.45)


def build_rows(
    fixtures: list[dict[str, str]],
    odds_by_match: dict[str, dict[str, str]],
    team_power: dict[str, float],
    clean_sheet_probs: dict[tuple[str, str], float],
    clean_sheet_baseline_prob: float | None,
) -> tuple[list[dict[str, str]], int, int, int, int]:
    out_rows: list[dict[str, str]] = []
    odds_count = 0
    fallback_count = 0
    neutral_count = 0
    clean_sheet_count = 0

    for fixture in fixtures:
        match_id = txt(fixture.get("match_id"))
        home = txt(fixture.get("home")).upper()
        away = txt(fixture.get("away")).upper()
        odds = odds_by_match.get(match_id)

        home_prob = to_float(odds.get("home_win_prob_fair")) if odds else None
        draw_prob = to_float(odds.get("draw_prob_fair")) if odds else None
        away_prob = to_float(odds.get("away_win_prob_fair")) if odds else None

        if home_prob is not None and away_prob is not None:
            home_advantage = home_prob - away_prob
            away_advantage = away_prob - home_prob
            source = "match_odds_probs"
            odds_count += 1
        else:
            home_advantage, away_advantage, source = advantage_from_power(home, away, team_power)
            if source == "team_power_fallback":
                fallback_count += 1
            else:
                neutral_count += 1
            home_prob = None
            draw_prob = None
            away_prob = None

        home_goal, home_assist, home_clean_sheet = multipliers(home_advantage)
        away_goal, away_assist, away_clean_sheet = multipliers(away_advantage)
        source_parts = [source]

        home_clean_sheet_prob = clean_sheet_probs.get((match_id, home))
        away_clean_sheet_prob = clean_sheet_probs.get((match_id, away))
        if (
            clean_sheet_baseline_prob is not None
            and home_clean_sheet_prob is not None
            and away_clean_sheet_prob is not None
        ):
            home_clean_sheet = clean_sheet_multiplier_from_prob(home_clean_sheet_prob, clean_sheet_baseline_prob)
            away_clean_sheet = clean_sheet_multiplier_from_prob(away_clean_sheet_prob, clean_sheet_baseline_prob)
            source_parts.append("bet365_clean_sheet")
            clean_sheet_count += 1

        out_rows.append(
            {
                "match_id": match_id,
                "home": home,
                "away": away,
                "kickoff_dk": txt(fixture.get("kickoff_dk")),
                "home_win_prob_fair": fmt(home_prob),
                "draw_prob_fair": fmt(draw_prob),
                "away_win_prob_fair": fmt(away_prob),
                "home_strength_advantage": fmt(home_advantage),
                "away_strength_advantage": fmt(away_advantage),
                "home_goal_multiplier": fmt(home_goal),
                "away_goal_multiplier": fmt(away_goal),
                "home_assist_multiplier": fmt(home_assist),
                "away_assist_multiplier": fmt(away_assist),
                "home_clean_sheet_prob_fair": fmt(home_clean_sheet_prob),
                "away_clean_sheet_prob_fair": fmt(away_clean_sheet_prob),
                "home_clean_sheet_multiplier": fmt(home_clean_sheet),
                "away_clean_sheet_multiplier": fmt(away_clean_sheet),
                "source": "+".join(source_parts),
            }
        )

    return out_rows, odds_count, fallback_count, neutral_count, clean_sheet_count


def main() -> int:
    if not FIXTURES_PATH.exists():
        print(f"FEJL: Mangler {FIXTURES_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    fixtures = read_csv(FIXTURES_PATH)
    odds_by_match = load_odds()
    team_power = load_team_power()
    clean_sheet_probs, clean_sheet_baseline_prob = load_clean_sheet_probs()
    rows, odds_count, fallback_count, neutral_count, clean_sheet_count = build_rows(
        fixtures,
        odds_by_match,
        team_power,
        clean_sheet_probs,
        clean_sheet_baseline_prob,
    )

    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Antal kampe: {len(rows)}")
    print(f"Antal med odds: {odds_count}")
    print(f"Antal med fallback: {fallback_count}")
    print(f"Antal neutral default: {neutral_count}")
    print(f"Antal med bet365 clean sheet: {clean_sheet_count}")
    if clean_sheet_baseline_prob is not None:
        print(f"Bet365 clean sheet baseline: {clean_sheet_baseline_prob:.4f}")

    team_rows = []
    for row in rows:
        team_rows.append((row["home"], row["away"], row["match_id"], float(row["home_clean_sheet_multiplier"])))
        team_rows.append((row["away"], row["home"], row["match_id"], float(row["away_clean_sheet_multiplier"])))

    print("Top 10 højeste clean sheet multipliers:")
    for team, opponent, match_id, value in sorted(team_rows, key=lambda item: item[3], reverse=True)[:10]:
        print(f"- {team} vs {opponent} (kamp {match_id}): {value:.4f}")

    print("Top 10 laveste clean sheet multipliers:")
    for team, opponent, match_id, value in sorted(team_rows, key=lambda item: item[3])[:10]:
        print(f"- {team} vs {opponent} (kamp {match_id}): {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
