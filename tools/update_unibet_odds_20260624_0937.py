from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path


DATA_DIR = Path("data")
MATCH_ODDS_PATH = DATA_DIR / "match_odds.csv"
SNAPSHOT_LABEL = "Unibet 24.06 kl. 09:37"
SNAPSHOT_AT = "2026-06-24T09:37:00+02:00"
SOURCE = "Unibet"

ODDS_INPUT = [
    ("Bosnien", "Qatar", 1.41, 5.20, 8.00),
    ("Schweiz", "Canada", 2.35, 3.10, 3.45),
    ("Marokko", "Haiti", 1.19, 7.50, 17.00),
    ("Skotland", "Brasilien", 10.00, 5.60, 1.33),
    ("Sydafrika", "Sydkorea", 5.60, 3.80, 1.68),
    ("Tjekkiet", "Mexico", 3.90, 3.90, 1.93),
    ("Curacao", "Elfenbenskysten", 21.00, 10.00, 1.14),
    ("Ecuador", "Tyskland", 4.10, 4.00, 1.86),
    ("Japan", "Sverige", 1.92, 3.45, 4.50),
    ("Tunesien", "Holland", 23.00, 9.00, 1.14),
    ("Paraguay", "Australien", 3.10, 2.14, 4.10),
    ("Tyrkiet", "USA", 4.00, 4.20, 1.87),
    ("Norge", "Frankrig", 4.70, 4.70, 1.65),
    ("Senegal", "Irak", 1.23, 7.50, 13.00),
    ("Kap Verde", "Saudi Arabien", 2.43, 3.50, 3.10),
    ("Uruguay", "Spanien", 8.00, 4.40, 1.47),
    ("Egypten", "Iran", 2.40, 2.70, 4.10),
    ("New Zealand", "Belgien", 18.00, 8.00, 1.19),
    ("Kroatien", "Ghana", 1.79, 3.45, 5.80),
    ("Panama", "England", 19.00, 8.50, 1.17),
    ("Colombia", "Portugal", 3.95, 3.95, 1.92),
    ("DR Congo", "Usbekistan", 1.83, 4.20, 4.30),
    ("Algeriet", "Østrig", 3.90, 2.33, 2.85),
    ("Jordan", "Argentina", 17.00, 6.75, 1.22),
]

TEAM_ALIASES = {
    "bosnien": "BIH",
    "bosnien hercegovina": "BIH",
    "schweiz": "SUI",
    "canada": "CAN",
    "marokko": "MAR",
    "haiti": "HAI",
    "skotland": "SCO",
    "brasilien": "BRA",
    "sydafrika": "RSA",
    "sydkorea": "KOR",
    "tjekkiet": "CZE",
    "mexico": "MEX",
    "curacao": "CUW",
    "curaçao": "CUW",
    "elfenbenskysten": "CIV",
    "ecuador": "ECU",
    "tyskland": "GER",
    "japan": "JPN",
    "sverige": "SWE",
    "tunesien": "TUN",
    "holland": "NED",
    "paraguay": "PAR",
    "australien": "AUS",
    "tyrkiet": "TUR",
    "usa": "USA",
    "norge": "NOR",
    "frankrig": "FRA",
    "senegal": "SEN",
    "irak": "IRQ",
    "kap verde": "CPV",
    "saudi arabien": "KSA",
    "uruguay": "URU",
    "spanien": "ESP",
    "egypten": "EGY",
    "iran": "IRN",
    "new zealand": "NZL",
    "belgien": "BEL",
    "kroatien": "CRO",
    "ghana": "GHA",
    "panama": "PAN",
    "england": "ENG",
    "colombia": "COL",
    "portugal": "POR",
    "dr congo": "COD",
    "usbekistan": "UZB",
    "algeriet": "ALG",
    "ostrig": "AUT",
    "østrig": "AUT",
    "jordan": "JOR",
    "argentina": "ARG",
    "qatar": "QAT",
}

TEAM_ID_ALIASES = {
    "DZA": "ALG",
}


def normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "ø": "o",
        "ö": "o",
        "ó": "o",
        "á": "a",
        "é": "e",
        "í": "i",
        "ú": "u",
        "å": "a",
        "æ": "ae",
        "ã": "a",
        "ç": "c",
        "ô": "o",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = " ".join(text.replace("-", " ").replace("/", " ").split())
    return text


def canonical_team_id(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return TEAM_ID_ALIASES.get(text, text)


def code_from_name(value: str) -> str:
    normalized = normalize_name(value)
    code = TEAM_ALIASES.get(normalized, "")
    if not code:
        raise KeyError(f"Ukendt holdnavn i odds-input: {value}")
    return code


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_before_unibet_20260624_0937_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def format_odds(value: float) -> str:
    return f"{value:.2f}"


def latest_unibet_templates(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], list[dict[str, str]]]]:
    by_match_id: dict[str, dict[str, str]] = {}
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("source") != SOURCE:
            continue
        match_id = str(row.get("match_id") or "").strip()
        if not match_id:
            continue
        by_match_id[match_id] = row
        home_id = canonical_team_id(row.get("home_team_id") or row.get("home") or "")
        away_id = canonical_team_id(row.get("away_team_id") or row.get("away") or "")
        if home_id and away_id:
            by_pair.setdefault((home_id, away_id), []).append(row)
    return by_match_id, by_pair


def main() -> int:
    rows, fieldnames = read_csv(MATCH_ODDS_PATH)
    templates_by_match_id, templates_by_pair = latest_unibet_templates(rows)

    existing_snapshot_keys = {
        (str(row.get("match_id") or "").strip(), str(row.get("source") or "").strip(), str(row.get("odds_fetched_label") or "").strip())
        for row in rows
    }

    pending_rows: list[dict[str, str]] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    touched_match_ids: list[str] = []

    for home_name, away_name, home_odds, draw_odds, away_odds in ODDS_INPUT:
        home_code = code_from_name(home_name)
        away_code = code_from_name(away_name)
        pair = (home_code, away_code)
        candidates = templates_by_pair.get(pair, [])
        if not candidates:
            unmatched.append(f"{home_name} vs {away_name} ({home_code}-{away_code})")
            continue
        if len(candidates) > 1:
            # If multiple rows exist for same pair, use latest by file order only if match_ids are unique same fixture.
            unique_match_ids = {str(row.get('match_id') or '').strip() for row in candidates}
            if len(unique_match_ids) != 1:
                ambiguous.append(f"{home_name} vs {away_name} ({home_code}-{away_code}) -> {sorted(unique_match_ids)}")
                continue
        template = candidates[-1]
        match_id = str(template.get("match_id") or "").strip()
        key = (match_id, SOURCE, SNAPSHOT_LABEL)
        if key in existing_snapshot_keys:
            raise SystemExit(f"Snapshot findes allerede for match_id={match_id}, source={SOURCE}, label={SNAPSHOT_LABEL}")

        next_row = dict(template)
        next_row["source"] = SOURCE
        next_row["odds_fetched_label"] = SNAPSHOT_LABEL
        next_row["odds_fetched_at"] = SNAPSHOT_AT
        next_row["home_win_odds"] = format_odds(home_odds)
        next_row["draw_odds"] = format_odds(draw_odds)
        next_row["away_win_odds"] = format_odds(away_odds)
        pending_rows.append(next_row)
        touched_match_ids.append(match_id)

    if unmatched or ambiguous:
        raise SystemExit(
            "Afbrudt pga. match-problemer.\n"
            + (f"Unmatched: {unmatched}\n" if unmatched else "")
            + (f"Ambiguous: {ambiguous}\n" if ambiguous else "")
        )

    if len(pending_rows) != 24:
        raise SystemExit(f"Forventede 24 nye rækker, men fandt {len(pending_rows)}.")

    if len(set(touched_match_ids)) != 24:
        raise SystemExit("Der blev ikke matchet 24 unikke match_id'er.")

    backup = backup_file(MATCH_ODDS_PATH)
    updated_rows = rows + pending_rows
    write_csv(MATCH_ODDS_PATH, updated_rows, fieldnames)

    # Post-write sanity.
    final_rows, _ = read_csv(MATCH_ODDS_PATH)
    label_rows = [row for row in final_rows if row.get("source") == SOURCE and row.get("odds_fetched_label") == SNAPSHOT_LABEL]
    label_count = len(label_rows)
    duplicate_keys = {}
    for row in final_rows:
        if row.get("source") != SOURCE or row.get("odds_fetched_label") != SNAPSHOT_LABEL:
            continue
        dup_key = (str(row.get("match_id") or "").strip(), row.get("source"), row.get("odds_fetched_label"))
        duplicate_keys[dup_key] = duplicate_keys.get(dup_key, 0) + 1
    duplicate_hits = [key for key, count in duplicate_keys.items() if count > 1]

    print(f"backup: {backup}")
    print(f"inserted_rows: {len(pending_rows)}")
    print(f"label_count: {label_count}")
    print(f"unmatched: {unmatched}")
    print(f"ambiguous: {ambiguous}")
    print(f"duplicate_snapshot_keys: {duplicate_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
