from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FIXTURES_PATH = DATA / "fixtures_group.csv"
TEAM_NAMES_PATH = DATA / "worldcup_outright_odds.csv"
MANUAL_INPUT_PATH = DATA / "clean_sheet_odds_manual.csv"
OUT_ODDS_PATH = DATA / "clean_sheet_odds.csv"
OUT_PROBS_PATH = DATA / "clean_sheet_probs.csv"

ROUND_TARGET = 2

TEAM_ALIASES = {
    "algeriet": "ALG",
    "algeria": "ALG",
    "australia": "AUS",
    "australien": "AUS",
    "bosnien": "BIH",
    "bosnien hercegovina": "BIH",
    "bosnia herzegovina": "BIH",
    "brasilien": "BRA",
    "brazil": "BRA",
    "canada": "CAN",
    "colombia": "COL",
    "croatia": "CRO",
    "kroatien": "CRO",
    "curacao": "CUW",
    "curaçao": "CUW",
    "dr congo": "COD",
    "ecuador": "ECU",
    "egypten": "EGY",
    "egypt": "EGY",
    "elfenbenskysten": "CIV",
    "england": "ENG",
    "frankrig": "FRA",
    "france": "FRA",
    "ghana": "GHA",
    "haiti": "HAI",
    "holland": "NED",
    "iran": "IRN",
    "irak": "IRQ",
    "iraq": "IRQ",
    "japan": "JPN",
    "jordan": "JOR",
    "kap verde": "CPV",
    "mexico": "MEX",
    "marokko": "MAR",
    "new zealand": "NZL",
    "norge": "NOR",
    "panama": "PAN",
    "paraguay": "PAR",
    "portugal": "POR",
    "qatar": "QAT",
    "saudi arabien": "KSA",
    "saudi arabia": "KSA",
    "schweiz": "SUI",
    "scotland": "SCO",
    "senegal": "SEN",
    "spanien": "ESP",
    "spain": "ESP",
    "sverige": "SWE",
    "sydafrika": "RSA",
    "sydkorea": "KOR",
    "syd korea": "KOR",
    "tjekkiet": "CZE",
    "tunesien": "TUN",
    "tyrkiet": "TUR",
    "turkiye": "TUR",
    "uruguay": "URU",
    "usa": "USA",
    "usbekistan": "UZB",
    "uzbekistan": "UZB",
    "østrig": "AUT",
    "ostrig": "AUT",
    "?strig": "AUT",
}

TEAM_ID_ALIASES = {
    "DZA": "ALG",
}

TEAM_NAME_OVERRIDES = {
    "ALG": "Algeriet",
    "AUT": "Østrig",
}

MANUAL_FIELDS = [
    "round",
    "team_name",
    "opponent_name",
    "clean_sheet_yes_odds",
    "clean_sheet_no_odds",
    "source",
    "clean_sheet_fetched_label",
]

OUT_ODDS_FIELDS = [
    "round",
    "match_id",
    "team_id",
    "team_name",
    "opponent_team_id",
    "opponent_team_name",
    "clean_sheet_yes_odds",
    "clean_sheet_no_odds",
    "source",
    "clean_sheet_fetched_label",
    "clean_sheet_fetched_at",
]

OUT_PROBS_FIELDS = [
    "round",
    "match_id",
    "team_id",
    "team_name",
    "opponent_team_id",
    "opponent_team_name",
    "clean_sheet_yes_odds",
    "clean_sheet_no_odds",
    "clean_sheet_yes_prob_raw",
    "clean_sheet_no_prob_raw",
    "clean_sheet_yes_prob_novig",
    "clean_sheet_no_prob_novig",
    "source",
    "clean_sheet_fetched_label",
    "clean_sheet_fetched_at",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_float(value: Any) -> float | None:
    raw = txt(value).replace(",", ".")
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if number > 0 else None


def fmt_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: txt(row.get(field)) for field in fieldnames})


def round_from_match_id(match_id: int) -> int | None:
    if 1 <= match_id <= 24:
        return 1
    if 25 <= match_id <= 48:
        return 2
    if 49 <= match_id <= 72:
        return 3
    return None


def load_team_names() -> dict[str, str]:
    names: dict[str, str] = {}
    if not TEAM_NAMES_PATH.exists():
        return names
    for row in read_csv(TEAM_NAMES_PATH):
        raw_team_id = txt(row.get("team_id")).upper()
        team_id = TEAM_ID_ALIASES.get(raw_team_id, raw_team_id)
        team_name = TEAM_NAME_OVERRIDES.get(team_id, txt(row.get("team_name")))
        if team_id and team_name:
            names[team_id] = team_name
    return names


def team_id_from_name(name: str, reverse_names: dict[str, str]) -> str:
    key = norm(name)
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    if key in reverse_names:
        return reverse_names[key]
    return ""


def load_round2_fixture_rows() -> list[dict[str, str]]:
    fixtures: list[dict[str, str]] = []
    for row in read_csv(FIXTURES_PATH):
        match_id = int(txt(row.get("match_id")) or "0")
        if round_from_match_id(match_id) == ROUND_TARGET:
            fixtures.append(row)
    fixtures.sort(key=lambda row: int(txt(row.get("match_id"))))
    return fixtures


def scaffold_manual_rows() -> list[dict[str, str]]:
    team_names = load_team_names()
    rows: list[dict[str, str]] = []
    for fixture in load_round2_fixture_rows():
        home = txt(fixture.get("home")).upper()
        away = txt(fixture.get("away")).upper()
        rows.append(
            {
                "round": str(ROUND_TARGET),
                "team_name": team_names.get(home, home),
                "opponent_name": team_names.get(away, away),
                "clean_sheet_yes_odds": "",
                "clean_sheet_no_odds": "",
                "source": "",
                "clean_sheet_fetched_label": "",
            }
        )
        rows.append(
            {
                "round": str(ROUND_TARGET),
                "team_name": team_names.get(away, away),
                "opponent_name": team_names.get(home, home),
                "clean_sheet_yes_odds": "",
                "clean_sheet_no_odds": "",
                "source": "",
                "clean_sheet_fetched_label": "",
            }
        )
    return rows


def ensure_manual_input_file() -> None:
    if MANUAL_INPUT_PATH.exists():
        return
    write_csv(MANUAL_INPUT_PATH, MANUAL_FIELDS, scaffold_manual_rows())


def load_manual_lookup() -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    ensure_manual_input_file()
    team_names = load_team_names()
    reverse_names = {norm(name): code for code, name in team_names.items()}
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    unmatched: list[str] = []

    fixtures = load_round2_fixture_rows()
    valid_pairs = {
        (txt(row.get("home")).upper(), txt(row.get("away")).upper()) for row in fixtures
    } | {
        (txt(row.get("away")).upper(), txt(row.get("home")).upper()) for row in fixtures
    }

    for row in read_csv(MANUAL_INPUT_PATH):
        if txt(row.get("round")) != str(ROUND_TARGET):
            continue
        team_id = team_id_from_name(txt(row.get("team_name")), reverse_names)
        opponent_id = team_id_from_name(txt(row.get("opponent_name")), reverse_names)
        if not team_id or not opponent_id or (team_id, opponent_id) not in valid_pairs:
            unmatched.append(
                f"{txt(row.get('team_name'))} vs {txt(row.get('opponent_name'))}"
            )
            continue
        lookup[(team_id, opponent_id)] = row

    return lookup, unmatched


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], int, list[str], list[str]]:
    fixtures = load_round2_fixture_rows()
    team_names = load_team_names()
    manual_lookup, unmatched = load_manual_lookup()

    odds_rows: list[dict[str, str]] = []
    prob_rows: list[dict[str, str]] = []
    missing_blank_odds: list[str] = []
    matched_input_rows = 0

    for fixture in fixtures:
        match_id = txt(fixture.get("match_id"))
        home = txt(fixture.get("home")).upper()
        away = txt(fixture.get("away")).upper()
        for team_id, opponent_id in ((home, away), (away, home)):
            manual = manual_lookup.get((team_id, opponent_id), {})
            if manual:
                matched_input_rows += 1
            yes_odds = to_float(manual.get("clean_sheet_yes_odds"))
            no_odds = to_float(manual.get("clean_sheet_no_odds"))
            odds_row = {
                "round": str(ROUND_TARGET),
                "match_id": match_id,
                "team_id": team_id,
                "team_name": team_names.get(team_id, team_id),
                "opponent_team_id": opponent_id,
                "opponent_team_name": team_names.get(opponent_id, opponent_id),
                "clean_sheet_yes_odds": fmt_float(yes_odds),
                "clean_sheet_no_odds": fmt_float(no_odds),
                "source": txt(manual.get("source")),
                "clean_sheet_fetched_label": txt(manual.get("clean_sheet_fetched_label")),
                "clean_sheet_fetched_at": "",
            }
            odds_rows.append(odds_row)

            yes_prob_raw = 1.0 / yes_odds if yes_odds else None
            no_prob_raw = 1.0 / no_odds if no_odds else None
            yes_prob_novig = None
            no_prob_novig = None
            if yes_prob_raw is not None and no_prob_raw is not None:
                total = yes_prob_raw + no_prob_raw
                if total > 0:
                    yes_prob_novig = yes_prob_raw / total
                    no_prob_novig = no_prob_raw / total
            else:
                missing_blank_odds.append(
                    f"kamp {match_id}: {team_names.get(team_id, team_id)} vs {team_names.get(opponent_id, opponent_id)}"
                )

            prob_rows.append(
                {
                    **odds_row,
                    "clean_sheet_yes_prob_raw": fmt_float(yes_prob_raw),
                    "clean_sheet_no_prob_raw": fmt_float(no_prob_raw),
                    "clean_sheet_yes_prob_novig": fmt_float(yes_prob_novig),
                    "clean_sheet_no_prob_novig": fmt_float(no_prob_novig),
                }
            )

    return odds_rows, prob_rows, matched_input_rows, unmatched, missing_blank_odds


def main() -> int:
    odds_rows, prob_rows, matched_input_rows, unmatched, missing_blank_odds = build_rows()
    write_csv(OUT_ODDS_PATH, OUT_ODDS_FIELDS, odds_rows)
    write_csv(OUT_PROBS_PATH, OUT_PROBS_FIELDS, prob_rows)

    complete = [row for row in prob_rows if txt(row.get("clean_sheet_yes_prob_novig"))]
    print(f"Inputrækker: {sum(1 for row in read_csv(MANUAL_INPUT_PATH) if txt(row.get('round')) == str(ROUND_TARGET))}")
    print(f"Matchede fixture-rows: {matched_input_rows}")
    print(f"Manglende/blanke odds: {len(missing_blank_odds)}")
    print(f"Manglende fixture-matches: {len(unmatched)}")
    print("Top 10 clean sheet no-vig:")
    for row in sorted(complete, key=lambda item: float(item["clean_sheet_yes_prob_novig"]), reverse=True)[:10]:
        print(
            f"- kamp {row['match_id']} {row['team_name']} vs {row['opponent_team_name']}: "
            f"{row['clean_sheet_yes_prob_novig']} "
            f"(ja {row['clean_sheet_yes_odds']}, nej {row['clean_sheet_no_odds']})"
        )
    if unmatched:
        print("Umatchede inputrækker:")
        for item in unmatched:
            print(f"- {item}")
    if missing_blank_odds:
        print("Første blanke odds-rækker:")
        for item in missing_blank_odds[:20]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
