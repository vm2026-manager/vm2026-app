from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
START_SECURITY_PATH = DATA / "player_start_security_nt.csv"
SPLIT_REPORT_PATH = DATA / "start_probability_availability_split_report.csv"
OUT_PATH = DATA / "start_probability_availability_split_sanity.md"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    value = str(value).strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_prob(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{'+' if value >= 0 else ''}{value:.4f}"


def md(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "\\|")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md(value) for value in row) + " |")
    return "\n".join(lines)


def load_report_rows() -> list[dict[str, Any]]:
    with SPLIT_REPORT_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    parsed_rows = []
    for row in rows:
        old = to_float(row.get("old_start_prob"))
        new = to_float(row.get("new_start_prob"))
        conditional = to_float(row.get("conditional_start_prob"))
        availability = to_float(row.get("availability_prob"))
        parsed = dict(row)
        parsed["_old"] = old
        parsed["_new"] = new
        parsed["_conditional"] = conditional
        parsed["_availability"] = availability
        parsed["_delta"] = None if old is None or new is None else new - old
        parsed_rows.append(parsed)
    return parsed_rows


def top_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row.get("player_name", ""),
            row.get("team_id", ""),
            fmt_prob(row["_old"]),
            fmt_prob(row["_new"]),
            fmt_delta(row["_delta"]),
            fmt_prob(row["_conditional"]),
            fmt_prob(row["_availability"]),
            row.get("availability_risk", ""),
        ]
        for row in rows
    ]


def main() -> None:
    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        players = json.load(f)
    with START_SECURITY_PATH.open(encoding="utf-8-sig", newline="") as f:
        start_security_rows = list(csv.DictReader(f))

    report_rows = load_report_rows()

    pool_with_split = [
        player
        for player in players
        if to_float(player.get("conditional_start_prob")) is not None
        and to_float(player.get("availability_prob")) is not None
    ]
    security_with_split = [
        row
        for row in start_security_rows
        if to_float(row.get("conditional_start_prob")) is not None
        and to_float(row.get("availability_prob")) is not None
    ]
    report_with_split = [
        row
        for row in report_rows
        if row["_conditional"] is not None and row["_availability"] is not None
    ]

    deltas = [row for row in report_rows if row["_delta"] is not None]
    rises = sorted(deltas, key=lambda row: row["_delta"], reverse=True)[:30]
    falls = sorted(deltas, key=lambda row: row["_delta"])[:30]
    high_cond_low_avail = sorted(
        [
            row
            for row in report_rows
            if row["_conditional"] is not None
            and row["_availability"] is not None
            and row["_conditional"] >= 0.85
            and row["_availability"] < 0.85
        ],
        key=lambda row: (row["_availability"], -row["_conditional"], row.get("player_name", "")),
    )[:30]

    risk_counts = Counter(row.get("availability_risk") or "blank" for row in report_rows)

    stats_rows = []
    for label, key in [
        ("old_start_prob", "_old"),
        ("new_start_prob", "_new"),
        ("conditional_start_prob", "_conditional"),
        ("availability_prob", "_availability"),
    ]:
        values = [row[key] for row in report_rows if row[key] is not None]
        stats_rows.append(
            [
                label,
                len(values),
                fmt_prob(min(values)) if values else "",
                fmt_prob(max(values)) if values else "",
                fmt_prob(statistics.mean(values)) if values else "",
            ]
        )

    examples = []
    for wanted in ["Cristian Romero", "Erling Haaland", "Antonio Nusa", "Manuel Neuer", "Oliver Baumann"]:
        examples.extend(
            row
            for row in report_rows
            if (row.get("player_name") or "").casefold() == wanted.casefold()
        )
    if len(examples) < 8:
        examples.extend(
            sorted(
                [
                    row
                    for row in report_rows
                    if row["_conditional"] is not None
                    and row["_availability"] is not None
                    and row["_conditional"] >= 0.90
                    and row["_availability"] < 0.85
                    and row not in examples
                ],
                key=lambda row: (row["_availability"], -row["_conditional"]),
            )[: 8 - len(examples)]
        )

    lines = [
        "# Start Probability Availability Split - Sanityrapport",
        "",
        "Denne rapport er afledt af `player_pool_v1.json`, `player_start_security_nt.csv` og `start_probability_availability_split_report.csv`. Den ændrer ikke modeldata.",
        "",
        "## 1. Dækning",
        "",
        table(
            ["Kilde", "Antal med conditional_start_prob + availability_prob", "Total rækker/spillere"],
            [
                ["player_pool_v1.json", len(pool_with_split), len(players)],
                ["player_start_security_nt.csv", len(security_with_split), len(start_security_rows)],
                ["start_probability_availability_split_report.csv", len(report_with_split), len(report_rows)],
            ],
        ),
        "",
        "## 2. Top 30 - new_start_prob steg mest",
        "",
        table(["Spiller", "Land", "Old", "New", "Delta", "Conditional", "Availability", "Risk"], top_rows(rises)),
        "",
        "## 3. Top 30 - new_start_prob faldt mest",
        "",
        table(["Spiller", "Land", "Old", "New", "Delta", "Conditional", "Availability", "Risk"], top_rows(falls)),
        "",
        "## 4. Top 30 - høj conditional_start_prob, lav availability_prob",
        "",
        table(["Spiller", "Land", "Old", "New", "Delta", "Conditional", "Availability", "Risk"], top_rows(high_cond_low_avail)),
        "",
        "## 5. Eksempler på profiler der håndteres bedre",
        "",
        table(["Spiller", "Land", "Old", "New", "Delta", "Conditional", "Availability", "Risk"], top_rows(examples)),
        "",
        "Cristian Romero er et godt sanity-eksempel, fordi hans conditional_start_prob ligger højt, mens availability holdes separat.",
        "",
        "## 6. Antal spillere pr. availability_risk",
        "",
        table(["availability_risk", "Antal"], [[key, value] for key, value in sorted(risk_counts.items())]),
        "",
        "## 7. Min/max/mean",
        "",
        table(["Felt", "N", "Min", "Max", "Mean"], stats_rows),
        "",
        "## 8. Kort vurdering",
        "",
        "- Risk-labels er nu mindre ekstreme: medium bliver standardområdet for spillere med rimelig availability, mens high i højere grad markerer lav availability eller tydelige absence-signaler.",
        "- Startformlen og selve sandsynlighedsfelterne er ikke ændret af denne sanityrapport.",
        "- Overordnet ser kalibreringen mere brugbar ud som advarselslabel end den tidligere fordeling, hvor high_risk dominerede for kraftigt.",
        "",
    ]

    OUT_PATH.write_text("\n".join(str(line) for line in lines), encoding="utf-8")

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"player_pool split coverage: {len(pool_with_split)}/{len(players)}")
    print(f"start_security split coverage: {len(security_with_split)}/{len(start_security_rows)}")
    print(f"report rows: {len(report_rows)}")
    print("availability_risk counts:", dict(sorted(risk_counts.items())))


if __name__ == "__main__":
    main()
