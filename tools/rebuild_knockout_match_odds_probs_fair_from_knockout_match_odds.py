from __future__ import annotations

import csv
from pathlib import Path


DATA = Path("data")
MATCH_ODDS = DATA / "knockout_match_odds.csv"
MATCH_ODDS_PROBS = DATA / "knockout_match_odds_probs.csv"


def txt(value: object) -> str:
    return "" if value is None else str(value).strip()


def fnum(value: object) -> float | None:
    try:
        return float(txt(value).replace(",", "."))
    except Exception:
        return None


def fair_probs(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    inv_h = 1 / home_odds
    inv_d = 1 / draw_odds
    inv_a = 1 / away_odds
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total


with MATCH_ODDS.open("r", encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

out: list[dict[str, object]] = []
for row in rows:
    h = fnum(row.get("home_win_odds"))
    d = fnum(row.get("draw_odds"))
    a = fnum(row.get("away_win_odds"))
    if not h or not d or not a or h <= 1 or d <= 1 or a <= 1:
        continue

    hp, dp, ap = fair_probs(h, d, a)
    home = txt(row.get("home")).upper()
    away = txt(row.get("away")).upper()

    out.append(
        {
            "match_id": txt(row.get("match_id")),
            "home": home,
            "away": away,
            "kickoff_dk": txt(row.get("kickoff_dk")),
            "home_team_id": txt(row.get("home_team_id")) or home,
            "away_team_id": txt(row.get("away_team_id")) or away,
            "home_win_odds": h,
            "draw_odds": d,
            "away_win_odds": a,
            "home_win_prob": hp,
            "draw_prob": dp,
            "away_win_prob": ap,
            "home_win_prob_fair": hp,
            "draw_prob_fair": dp,
            "away_win_prob_fair": ap,
            "source": txt(row.get("source")),
            "odds_fetched_at": txt(row.get("odds_fetched_at")),
            "odds_fetched_label": txt(row.get("odds_fetched_label")),
            "stage": txt(row.get("stage")),
            "group": txt(row.get("group")),
        }
    )

fields = [
    "match_id",
    "home",
    "away",
    "kickoff_dk",
    "home_team_id",
    "away_team_id",
    "home_win_odds",
    "draw_odds",
    "away_win_odds",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "home_win_prob_fair",
    "draw_prob_fair",
    "away_win_prob_fair",
    "source",
    "odds_fetched_at",
    "odds_fetched_label",
    "stage",
    "group",
]

with MATCH_ODDS_PROBS.open("w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out)

print(f"OK: skrev {len(out)} knockout-kampe til {MATCH_ODDS_PROBS}")
