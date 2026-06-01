from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
IMPACT_PATH = DATA_DIR / "player_ev_fixture_strength_impact_report.csv"
MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
OUT_PATH = DATA_DIR / "fixture_strength_ev_sanity.md"


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any) -> float | None:
    text = txt(value).replace(",", ".")
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    return None if math.isnan(value_float) else value_float


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}".rstrip("0").rstrip(".")


def fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}%"


def md(value: Any) -> str:
    return txt(value).replace("|", "\\|")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md(value) for value in row) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stats_row(label: str, values: list[float]) -> list[Any]:
    if not values:
        return [label, 0, "", "", "", ""]
    return [
        label,
        len(values),
        fmt(statistics.mean(values)),
        fmt(statistics.median(values)),
        fmt(min(values)),
        fmt(max(values)),
    ]


def impact_table_rows(rows: list[dict[str, str]]) -> list[list[Any]]:
    return [
        [
            row.get("player_name", ""),
            row.get("team_id", ""),
            row.get("position", ""),
            row.get("price", ""),
            fmt(to_float(row.get("old_ev"))),
            fmt(to_float(row.get("new_ev"))),
            fmt(to_float(row.get("ev_diff"))),
            fmt_pct(to_float(row.get("ev_diff_pct"))),
            row.get("main_reason", ""),
        ]
        for row in rows
    ]


def main() -> int:
    ev_rows = read_csv(PLAYER_EV_PATH)
    impact_rows = read_csv(IMPACT_PATH)
    multiplier_rows = read_csv(MULTIPLIERS_PATH)
    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        player_pool = json.load(f)

    ev_diffs = [value for value in (to_float(row.get("ev_diff")) for row in impact_rows) if value is not None]
    ev_diff_pcts = [value for value in (to_float(row.get("ev_diff_pct")) for row in impact_rows) if value is not None]
    new_evs = [value for value in (to_float(row.get("new_ev")) for row in impact_rows) if value is not None]

    positive_count = sum(1 for value in ev_diffs if value > 0)
    zero_count = sum(1 for value in ev_diffs if value == 0)
    negative_count = sum(1 for value in ev_diffs if value < 0)
    negative_new_ev_count = sum(1 for value in new_evs if value < 0)

    top_rises = sorted(impact_rows, key=lambda row: to_float(row.get("ev_diff")) or 0.0, reverse=True)[:30]
    top_falls = sorted(impact_rows, key=lambda row: to_float(row.get("ev_diff")) or 0.0)[:30]
    cheap_rises = sorted(
        [row for row in impact_rows if (to_float(row.get("price")) or 0.0) <= 3_000_000],
        key=lambda row: to_float(row.get("ev_diff")) or 0.0,
        reverse=True,
    )[:20]
    expensive_falls = sorted(
        [row for row in impact_rows if (to_float(row.get("price")) or 0.0) >= 4_000_000],
        key=lambda row: to_float(row.get("ev_diff")) or 0.0,
    )[:20]

    team_sum: dict[str, float] = defaultdict(float)
    team_count: Counter[str] = Counter()
    position_values: dict[str, list[float]] = defaultdict(list)
    for row in impact_rows:
        diff = to_float(row.get("ev_diff"))
        if diff is None:
            continue
        team = txt(row.get("team_id")) or "UNKNOWN"
        position = txt(row.get("position")) or "UNKNOWN"
        team_sum[team] += diff
        team_count[team] += 1
        position_values[position].append(diff)

    team_rows = [
        [team, team_count[team], fmt(diff_sum)]
        for team, diff_sum in sorted(team_sum.items(), key=lambda item: item[1], reverse=True)
    ]
    position_rows = [
        [position, len(values), fmt(statistics.mean(values))]
        for position, values in sorted(position_values.items())
    ]

    outliers = []
    for row in impact_rows:
        diff = to_float(row.get("ev_diff")) or 0.0
        diff_pct = to_float(row.get("ev_diff_pct"))
        price = to_float(row.get("price")) or 0.0
        reasons = []
        if abs(diff) > 1.25:
            reasons.append("abs(ev_diff) > 1.25")
        if diff_pct is not None and diff_pct > 0.50:
            reasons.append("ev_diff_pct > 50%")
        if price >= 4_000_000 and diff < -0.50:
            reasons.append("høj pris og stort fald")
        if reasons:
            outlier_row = dict(row)
            outlier_row["_outlier_reason"] = "; ".join(reasons)
            outliers.append(outlier_row)
    outliers.sort(key=lambda row: abs(to_float(row.get("ev_diff")) or 0.0), reverse=True)

    positive_teams = [team for team, diff in team_sum.items() if diff > 0]
    negative_teams = [team for team, diff in team_sum.items() if diff < 0]
    mean_diff = statistics.mean(ev_diffs) if ev_diffs else 0.0
    max_abs_diff = max((abs(value) for value in ev_diffs), default=0.0)

    lines = [
        "# Fixture Strength EV Sanity",
        "",
        "Denne rapport er kun en sanity-check af fixture-strength impact. Den ændrer ikke modeldata.",
        "",
        "## 1. Datadækning",
        "",
        table(
            ["Kilde", "Rækker"],
            [
                ["player_ev_group_stage_v1.csv", len(ev_rows)],
                ["player_ev_fixture_strength_impact_report.csv", len(impact_rows)],
                ["fixture_strength_multipliers.csv", len(multiplier_rows)],
                ["player_pool_v1.json", len(player_pool)],
            ],
        ),
        "",
        "## 2. EV Diff Statistik",
        "",
        table(["Felt", "N", "Mean", "Median", "Min", "Max"], [stats_row("ev_diff", ev_diffs)]),
        "",
        "## 3. EV Diff Pct Statistik",
        "",
        table(["Felt", "N", "Mean", "Median", "Min", "Max"], [stats_row("ev_diff_pct", ev_diff_pcts)]),
        "",
        "## 4. Retning",
        "",
        table(
            ["Kategori", "Antal"],
            [
                ["ev_diff > 0", positive_count],
                ["ev_diff = 0", zero_count],
                ["ev_diff < 0", negative_count],
                ["new_ev < 0", negative_new_ev_count],
            ],
        ),
        "",
        "## 5. Top 30 EV-stigninger",
        "",
        table(["Spiller", "Hold", "Pos", "Pris", "Old EV", "New EV", "Diff", "Diff pct", "Main reason"], impact_table_rows(top_rises)),
        "",
        "## 6. Top 30 EV-fald",
        "",
        table(["Spiller", "Hold", "Pos", "Pris", "Old EV", "New EV", "Diff", "Diff pct", "Main reason"], impact_table_rows(top_falls)),
        "",
        "## 7. Top 20 EV-stigninger, pris <= 3.0 mio.",
        "",
        table(["Spiller", "Hold", "Pos", "Pris", "Old EV", "New EV", "Diff", "Diff pct", "Main reason"], impact_table_rows(cheap_rises)),
        "",
        "## 8. Top 20 EV-fald, pris >= 4.0 mio.",
        "",
        table(["Spiller", "Hold", "Pos", "Pris", "Old EV", "New EV", "Diff", "Diff pct", "Main reason"], impact_table_rows(expensive_falls)),
        "",
        "## 9. Holdniveau: sum ev_diff pr. team_id",
        "",
        table(["Team", "Spillere", "Sum ev_diff"], team_rows),
        "",
        "## 10. Positionniveau: mean ev_diff pr. position",
        "",
        table(["Position", "Spillere", "Mean ev_diff"], position_rows),
        "",
        "## 11. Potentielt mistænkelige outliers",
        "",
        table(
            ["Spiller", "Hold", "Pos", "Pris", "Old EV", "New EV", "Diff", "Diff pct", "Main reason", "Outlier reason"],
            [
                row + [outlier.get("_outlier_reason", "")]
                for row, outlier in zip(impact_table_rows(outliers[:50]), outliers[:50])
            ],
        ),
        "",
        "## 12. Kort vurdering",
        "",
        f"- Multipliers ser ud til at skubbe EV i den forventede retning: {len(positive_teams)} hold har samlet positiv ændring, og {len(negative_teams)} hold har samlet negativ ændring.",
        f"- Gennemsnitlig ændring er {fmt(mean_diff)}, medianen er {fmt(statistics.median(ev_diffs) if ev_diffs else 0.0)}, og der er {zero_count} uændrede spillere. Det peger på en moderat samlet effekt snarere end en total omskalering.",
        f"- Største absolutte spillerændring er {fmt(max_abs_diff)}. Outlier-listen bør gennemgås, men totalbilledet ser rimeligt ud som første fixture-strength lag.",
        "",
    ]

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"EV-output spillere: {len(ev_rows)}")
    print(f"Mean ev_diff: {fmt(mean_diff)}")
    print(f"Median ev_diff: {fmt(statistics.median(ev_diffs) if ev_diffs else 0.0)}")
    print(f"Stigning/fald/uændret: {positive_count}/{negative_count}/{zero_count}")
    print(f"new_ev < 0: {negative_new_ev_count}")
    print(f"Outliers: {len(outliers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
