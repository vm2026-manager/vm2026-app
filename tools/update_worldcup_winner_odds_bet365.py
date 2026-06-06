from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCE_NAME = "bet365"
SOURCE_DATE = "2026-06-06"

EXISTING_PATH = DATA / "worldcup_outright_odds.csv"
NEW_SOURCE_PATH = DATA / f"worldcup_outright_odds_{SOURCE_NAME}_{SOURCE_DATE.replace('-', '')}.csv"
AUDIT_PATH = DATA / f"worldcup_outright_odds_{SOURCE_NAME}_{SOURCE_DATE.replace('-', '')}_audit.md"

# Bet365 VM 2026 - samlet vinder, aflæst fra screenshot/PDF 2026-06-06.
ODDS = {
    "ESP": ("Spain", 5.50),
    "FRA": ("France", 6.00),
    "ENG": ("England", 7.50),
    "BRA": ("Brazil", 9.00),
    "POR": ("Portugal", 9.00),
    "ARG": ("Argentina", 10.00),
    "GER": ("Germany", 15.00),
    "NED": ("Netherlands", 21.00),
    "NOR": ("Norway", 26.00),
    "BEL": ("Belgium", 34.00),
    "COL": ("Colombia", 34.00),
    "JPN": ("Japan", 51.00),
    "MAR": ("Morocco", 51.00),
    "MEX": ("Mexico", 67.00),
    "USA": ("USA", 67.00),
    "URU": ("Uruguay", 67.00),
    "SUI": ("Switzerland", 81.00),
    "CRO": ("Croatia", 81.00),
    "TUR": ("Turkey", 81.00),
    "ECU": ("Ecuador", 101.00),
    "SEN": ("Senegal", 126.00),
    "SWE": ("Sweden", 126.00),
    "CAN": ("Canada", 126.00),
    "AUT": ("Austria", 151.00),
    "PAR": ("Paraguay", 151.00),
    "SCO": ("Scotland", 251.00),
    "EGY": ("Egypt", 301.00),
    "CIV": ("Ivory Coast", 301.00),
    "CZE": ("Czechia", 301.00),
    "BIH": ("Bosnia-Herzegovina", 351.00),
    "ALG": ("Algeria", 401.00),
    "KOR": ("South Korea", 401.00),
    "GHA": ("Ghana", 401.00),
    "TUN": ("Tunisia", 501.00),
    "AUS": ("Australia", 501.00),
    "IRN": ("Iran", 501.00),
    "COD": ("DR Congo", 751.00),
    "RSA": ("South Africa", 1001.00),
    "KSA": ("Saudi Arabia", 1001.00),
    "PAN": ("Panama", 1501.00),
    "IRQ": ("Iraq", 1501.00),
    "UZB": ("Uzbekistan", 1501.00),
    "QAT": ("Qatar", 2001.00),
    "CPV": ("Cape Verde", 2001.00),
    "NZL": ("New Zealand", 2501.00),
    "JOR": ("Jordan", 2501.00),
    "HAI": ("Haiti", 2501.00),
    "CUW": ("Curacao", 3501.00),
}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".backup_before_bet365_winner_odds_{stamp}.csv")
    shutil.copy2(path, backup)
    return backup


def write_source_file() -> None:
    rows = [
        {
            "team_id": team_id,
            "team_name": team_name,
            "bet365_win_odds": odds,
            "source": SOURCE_NAME,
            "source_date": SOURCE_DATE,
        }
        for team_id, (team_name, odds) in ODDS.items()
    ]
    write_csv_dicts(
        NEW_SOURCE_PATH,
        rows,
        ["team_id", "team_name", "bet365_win_odds", "source", "source_date"],
    )


def detect_team_id(row: dict[str, str]) -> str:
    for key in ["team_id", "team", "country_code", "code"]:
        value = str(row.get(key, "")).strip().upper()
        if value in ODDS:
            return value
    return ""


def update_existing_file() -> tuple[Path | None, list[dict[str, object]], list[str], list[str]]:
    if not EXISTING_PATH.exists():
        return None, [], [], []

    backup = make_backup(EXISTING_PATH)
    rows = read_csv_dicts(EXISTING_PATH)

    existing_fields = list(rows[0].keys()) if rows else []
    add_fields = [
        "bet365_win_odds",
        "winner_odds_source",
        "winner_odds_source_date",
    ]

    # Bevar eksisterende felter og tilføj nye til sidst.
    fieldnames = existing_fields[:]
    for field in add_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    updated_team_ids: list[str] = []
    unmatched_rows: list[str] = []

    for row in rows:
        team_id = detect_team_id(row)
        if not team_id:
            label = row.get("team_name") or row.get("country") or row.get("team_id") or json.dumps(row, ensure_ascii=False)
            unmatched_rows.append(str(label))
            continue

        _, odds = ODDS[team_id]
        row["bet365_win_odds"] = f"{odds:.2f}"
        row["winner_odds_source"] = SOURCE_NAME
        row["winner_odds_source_date"] = SOURCE_DATE
        updated_team_ids.append(team_id)

        # OBS: Vi opdaterer bevidst ikke eventuelle gamle kolonner som unibet_win_odds.
        # Modellen kan efterfølgende peges på bet365_win_odds eller en samlet/nyeste odds-kolonne.

    write_csv_dicts(EXISTING_PATH, rows, fieldnames)
    return backup, rows, updated_team_ids, unmatched_rows


def write_audit(
    backup: Path | None,
    updated_team_ids: list[str],
    unmatched_rows: list[str],
) -> None:
    missing_from_existing = sorted(set(ODDS) - set(updated_team_ids))

    lines = [
        "# Bet365 winner odds update audit",
        "",
        f"Source: {SOURCE_NAME}",
        f"Source date: {SOURCE_DATE}",
        f"Teams in Bet365 source: {len(ODDS)}",
        f"New source file: `{NEW_SOURCE_PATH.relative_to(ROOT)}`",
        f"Existing odds file: `{EXISTING_PATH.relative_to(ROOT)}`",
        f"Backup: `{backup.relative_to(ROOT)}`" if backup else "Backup: none, existing file not found",
        "",
        "## Update summary",
        "",
        f"Rows updated in existing file: {len(updated_team_ids)}",
        f"Bet365 teams not matched in existing file: {len(missing_from_existing)}",
        f"Existing rows without matched team_id: {len(unmatched_rows)}",
        "",
    ]

    if missing_from_existing:
        lines += [
            "## Bet365 teams not matched in existing file",
            "",
            *[f"- {team_id} - {ODDS[team_id][0]} ({ODDS[team_id][1]})" for team_id in missing_from_existing],
            "",
        ]

    if unmatched_rows:
        lines += [
            "## Existing rows without matched team_id",
            "",
            *[f"- {x}" for x in unmatched_rows[:100]],
            "",
        ]

    lines += [
        "## Top odds",
        "",
        "| Team | Odds |",
        "|---|---:|",
    ]
    for team_id, (team_name, odds) in sorted(ODDS.items(), key=lambda item: item[1][1]):
        lines.append(f"| {team_name} ({team_id}) | {odds:.2f} |")

    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    write_source_file()
    backup, _rows, updated_team_ids, unmatched_rows = update_existing_file()
    write_audit(backup, updated_team_ids, unmatched_rows)

    print("Bet365 winner odds update")
    print("-------------------------")
    print(f"Skrev ny kildefil: {NEW_SOURCE_PATH}")
    print(f"Skrev audit: {AUDIT_PATH}")
    if backup:
        print(f"Backup af eksisterende oddsfil: {backup}")
        print(f"Opdaterede rækker i eksisterende fil: {len(updated_team_ids)}")
    else:
        print("Ingen eksisterende worldcup_outright_odds.csv fundet.")
    print(f"Bet365-hold i alt: {len(ODDS)}")
    print("Færdig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())