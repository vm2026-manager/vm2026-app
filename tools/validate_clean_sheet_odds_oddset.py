from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "clean_sheet_odds_oddset.csv"


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any) -> float | None:
    raw = txt(value).replace(",", ".")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_rows() -> list[dict[str, str]]:
    if not INPUT.exists():
        raise FileNotFoundError(f"Mangler {INPUT}")
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def side_entries(rows: list[dict[str, str]], column: str) -> list[tuple[float, dict[str, str]]]:
    entries: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        value = to_float(row.get(column))
        if value is not None:
            entries.append((value, row))
    return sorted(entries, key=lambda item: item[0])


def print_side(label: str, rows: list[dict[str, str]], column: str, limit: int = 15) -> None:
    print(label)
    print("-" * len(label))
    for value, row in side_entries(rows, column)[:limit]:
        print(f"{row.get('match_id'):>2} {row.get('home')}-{row.get('away')} | {column}={value:g} | status={row.get('status')}")
    print()


def suspicious_rows(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    for row in rows:
        match = f"{row.get('match_id')} {row.get('home')}-{row.get('away')}"
        home_cs = to_float(row.get("home_clean_sheet_odds"))
        away_cs = to_float(row.get("away_clean_sheet_odds"))
        status = txt(row.get("status"))

        if status == "ok" and (home_cs is None or away_cs is None):
            issues.append(f"{match}: ok men tomme odds")
            continue
        if status == "partial" and home_cs is None and away_cs is None:
            issues.append(f"{match}: partial men begge odds tomme")
        if home_cs is not None and (home_cs < 1.01 or home_cs > 30):
            issues.append(f"{match}: mistænkelig home_clean_sheet_odds={home_cs:g}")
        if away_cs is not None and (away_cs < 1.01 or away_cs > 30):
            issues.append(f"{match}: mistænkelig away_clean_sheet_odds={away_cs:g}")
        if home_cs is not None and away_cs is not None and abs(home_cs - away_cs) < 0.001:
            issues.append(f"{match}: ens clean sheet-odds {home_cs:g}/{away_cs:g}")
    return issues


def main() -> int:
    rows = read_rows()
    counts = Counter(txt(row.get("status")) or "blank" for row in rows)

    print(f"Fil: {INPUT.relative_to(ROOT)}")
    print(f"Rækker: {len(rows)}")
    for key in ["ok", "partial", "missing_market", "missing_ou_tab", "match_not_found", "error", "blank"]:
        if counts.get(key, 0):
            print(f"{key}: {counts[key]}")
    for key, count in sorted(counts.items()):
        if key not in {"ok", "partial", "missing_market", "missing_ou_tab", "match_not_found", "error", "blank"}:
            print(f"{key}: {count}")
    print()

    print_side("Top 15 laveste home_clean_sheet_odds", rows, "home_clean_sheet_odds")
    print_side("Top 15 laveste away_clean_sheet_odds", rows, "away_clean_sheet_odds")

    combined: list[tuple[float, str, dict[str, str]]] = []
    for row in rows:
        for side, column in [("home", "home_clean_sheet_odds"), ("away", "away_clean_sheet_odds")]:
            value = to_float(row.get(column))
            if value is not None:
                combined.append((value, side, row))
    combined.sort(key=lambda item: item[0])

    print("Top 10 laveste clean sheet-odds samlet")
    print("--------------------------------------")
    for value, side, row in combined[:10]:
        team = row.get("home") if side == "home" else row.get("away")
        print(f"{row.get('match_id'):>2} {team} clean sheet vs {row.get('away') if side == 'home' else row.get('home')}: {value:g}")
    print()

    issues = suspicious_rows(rows)
    print("Mistænkelige rækker")
    print("-------------------")
    if issues:
        for issue in issues[:40]:
            print(f"- {issue}")
        if len(issues) > 40:
            print(f"... +{len(issues) - 40} flere")
    else:
        print("Ingen åbenlyse problemer fundet.")
    print()

    failing = [row for row in rows if txt(row.get("status")) != "ok"]
    print("Ikke-ok kampe")
    print("-------------")
    if failing:
        for row in failing:
            print(f"{row.get('match_id'):>2} {row.get('home')}-{row.get('away')} | {row.get('status')} | {txt(row.get('note'))[:120]}")
    else:
        print("Alle kampe er ok.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
