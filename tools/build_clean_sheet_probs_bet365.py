from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INPUT_PATH = DATA / "clean_sheet_odds_bet365.csv"
FIXTURES_PATH = DATA / "fixtures_group.csv"
OUT_PATH = DATA / "clean_sheet_probs_bet365.csv"

SOURCE = "bet365 clean sheet yes/no 2026-06-02"

TEAM_ALIASES = {
    "algeria": "ALG",
    "argentina": "ARG",
    "australia": "AUS",
    "austria": "AUT",
    "belgium": "BEL",
    "bosnia-herzegovina": "BIH",
    "bosnia herzegovina": "BIH",
    "brazil": "BRA",
    "canada": "CAN",
    "cape verde": "CPV",
    "colombia": "COL",
    "croatia": "CRO",
    "curacao": "CUW",
    "curaçao": "CUW",
    "czechia": "CZE",
    "dr congo": "COD",
    "ecuador": "ECU",
    "egypt": "EGY",
    "england": "ENG",
    "france": "FRA",
    "germany": "GER",
    "ghana": "GHA",
    "haiti": "HAI",
    "iran": "IRN",
    "iraq": "IRQ",
    "ivory coast": "CIV",
    "japan": "JPN",
    "jordan": "JOR",
    "mexico": "MEX",
    "morocco": "MAR",
    "netherlands": "NED",
    "new zealand": "NZL",
    "norway": "NOR",
    "panama": "PAN",
    "paraguay": "PAR",
    "portugal": "POR",
    "qatar": "QAT",
    "saudi arabia": "KSA",
    "scotland": "SCO",
    "senegal": "SEN",
    "south africa": "RSA",
    "south korea": "KOR",
    "spain": "ESP",
    "sweden": "SWE",
    "switzerland": "SUI",
    "tunisia": "TUN",
    "turkiye": "TUR",
    "türkiye": "TUR",
    "turkey": "TUR",
    "uruguay": "URU",
    "usa": "USA",
    "uzbekistan": "UZB",
}

OUT_FIELDS = [
    "match_id",
    "home",
    "away",
    "kickoff_dk",
    "team_id",
    "opponent_team_id",
    "is_home",
    "match_raw",
    "team_raw",
    "clean_sheet_yes_odds",
    "clean_sheet_no_odds",
    "clean_sheet_yes_implied",
    "clean_sheet_no_implied",
    "clean_sheet_prob_fair",
    "source",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fix_mojibake(value: str) -> str:
    replacements = {
        "TÃ¼rkiye": "Türkiye",
        "CuraÃ§ao": "Curaçao",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    return value


def norm(value: Any) -> str:
    text = fix_mojibake(txt(value)).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_float(value: Any) -> float | None:
    raw = txt(value).replace(",", ".")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}".rstrip("0").rstrip(".")


def team_code(name: Any) -> str:
    key = norm(name)
    return TEAM_ALIASES.get(key, "")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fixture_lookup() -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(FIXTURES_PATH):
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        lookup[(home, away)] = row
        lookup[(away, home)] = row
    return lookup


def parse_match(raw_match: str) -> tuple[str, str]:
    parts = re.split(r"\s+v\s+", fix_mojibake(raw_match), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return "", ""
    return team_code(parts[0]), team_code(parts[1])


def build_rows() -> tuple[list[dict[str, str]], list[str]]:
    fixtures = fixture_lookup()
    rows: list[dict[str, str]] = []
    unmatched: list[str] = []

    for source_row in read_csv(INPUT_PATH):
        raw_match = txt(source_row.get("Kamp"))
        raw_team = txt(source_row.get("Hold"))
        match_home, match_away = parse_match(raw_match)
        team = team_code(raw_team)
        yes_odds = to_float(source_row.get("Rent bur - Ja"))
        no_odds = to_float(source_row.get("Rent bur - Nej"))

        fixture = fixtures.get((match_home, match_away))
        if not fixture or not team or yes_odds is None or no_odds is None:
            unmatched.append(f"{raw_match} | {raw_team}")
            continue

        implied_yes = 1.0 / yes_odds
        implied_no = 1.0 / no_odds
        fair_yes = implied_yes / (implied_yes + implied_no) if implied_yes + implied_no else None
        home = txt(fixture.get("home")).upper()
        away = txt(fixture.get("away")).upper()
        opponent = away if team == home else home

        rows.append(
            {
                "match_id": txt(fixture.get("match_id")),
                "home": home,
                "away": away,
                "kickoff_dk": txt(fixture.get("kickoff_dk")),
                "team_id": team,
                "opponent_team_id": opponent,
                "is_home": "1" if team == home else "0",
                "match_raw": fix_mojibake(raw_match),
                "team_raw": fix_mojibake(raw_team),
                "clean_sheet_yes_odds": fmt(yes_odds),
                "clean_sheet_no_odds": fmt(no_odds),
                "clean_sheet_yes_implied": fmt(implied_yes),
                "clean_sheet_no_implied": fmt(implied_no),
                "clean_sheet_prob_fair": fmt(fair_yes),
                "source": SOURCE,
            }
        )

    rows.sort(key=lambda row: (int(row["match_id"]), row["team_id"]))
    return rows, unmatched


def validate(rows: list[dict[str, str]], unmatched: list[str]) -> None:
    match_team_pairs = {(row["match_id"], row["team_id"]) for row in rows}
    fixture_rows = read_csv(FIXTURES_PATH)
    expected_pairs = {(txt(row["match_id"]), txt(row["home"]).upper()) for row in fixture_rows}
    expected_pairs |= {(txt(row["match_id"]), txt(row["away"]).upper()) for row in fixture_rows}
    missing_pairs = sorted(expected_pairs - match_team_pairs, key=lambda item: (int(item[0]), item[1]))
    counts = Counter(row["match_id"] for row in rows)
    duplicate_matches = [match_id for match_id, count in counts.items() if count != 2]

    print(f"Skrevet: {OUT_PATH.relative_to(ROOT)}")
    print(f"Indlæste odds-rækker: {len(rows)}")
    print(f"Matches mod fixtures/teams: {len(match_team_pairs)} / {len(expected_pairs)}")
    print(f"Umatchede input-rækker: {len(unmatched)}")
    print(f"Fixture/team-par uden bet365 CS: {len(missing_pairs)}")
    print(f"Kampe med andet end 2 rækker: {len(duplicate_matches)}")

    if unmatched:
        print("Umatchede holdnavne/input:")
        for item in unmatched[:20]:
            print(f"- {item}")

    if missing_pairs:
        print("Første manglende fixture/team-par:")
        for match_id, team in missing_pairs[:20]:
            print(f"- kamp {match_id}: {team}")

    print("Top 10 højeste clean sheet-sandsynligheder:")
    for row in sorted(rows, key=lambda item: float(item["clean_sheet_prob_fair"] or 0), reverse=True)[:10]:
        print(
            f"- kamp {row['match_id']} {row['team_id']} vs {row['opponent_team_id']}: "
            f"{float(row['clean_sheet_prob_fair']):.3f} "
            f"(odds {row['clean_sheet_yes_odds']}/{row['clean_sheet_no_odds']})"
        )

    extreme = []
    for row in rows:
        yes = float(row["clean_sheet_yes_odds"])
        no = float(row["clean_sheet_no_odds"])
        if yes <= 1.10 or no <= 1.10 or yes >= 15:
            extreme.append(row)

    print("Sanity for ekstreme odds:")
    if extreme:
        for row in extreme[:25]:
            print(
                f"- kamp {row['match_id']} {row['team_id']} vs {row['opponent_team_id']}: "
                f"ja={row['clean_sheet_yes_odds']} nej={row['clean_sheet_no_odds']} "
                f"fair={float(row['clean_sheet_prob_fair']):.3f}"
            )
    else:
        print("- Ingen ekstreme odds fundet.")


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"FEJL: Mangler {INPUT_PATH.relative_to(ROOT)}")
        return 1
    if not FIXTURES_PATH.exists():
        print(f"FEJL: Mangler {FIXTURES_PATH.relative_to(ROOT)}")
        return 1

    rows, unmatched = build_rows()
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    validate(rows, unmatched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
