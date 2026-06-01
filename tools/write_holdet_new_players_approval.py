from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "holdet_new_players_priority_review.csv"
OUT_PATH = DATA_DIR / "holdet_new_players_approval.csv"

APPROVAL_FIELDS = [
    "approved_for_player_pool",
    "priority_bucket",
    "suggested_action",
    "holdet_player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "price",
    "is_out",
    "reason",
    "possible_duplicate_or_mismatch",
    "reviewer_note",
]

BUCKET_ORDER = {
    "must_review": 0,
    "likely_relevant": 1,
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_price(value: Any) -> int:
    cleaned = re.sub(r"[^\d]", "", txt(value))
    return int(cleaned) if cleaned else 0


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"FEJL: Mangler {INPUT_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    approval_rows = []
    for row in rows:
        bucket = txt(row.get("priority_bucket"))
        if bucket not in BUCKET_ORDER:
            continue
        approval_rows.append(
            {
                "approved_for_player_pool": "",
                "priority_bucket": bucket,
                "suggested_action": txt(row.get("suggested_action")),
                "holdet_player_id": txt(row.get("holdet_player_id")),
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "team_name": txt(row.get("team_name")),
                "position": txt(row.get("position")),
                "price": parse_price(row.get("price")),
                "is_out": txt(row.get("is_out")),
                "reason": txt(row.get("reason")),
                "possible_duplicate_or_mismatch": txt(row.get("possible_duplicate_or_mismatch")),
                "reviewer_note": "",
            }
        )

    approval_rows.sort(
        key=lambda row: (
            BUCKET_ORDER[row["priority_bucket"]],
            -parse_price(row["price"]),
            row["player_name"],
        )
    )

    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=APPROVAL_FIELDS)
        writer.writeheader()
        writer.writerows(approval_rows)

    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rækker: {len(approval_rows)}")
    print("Top 30:")
    for row in approval_rows[:30]:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['price']} | {row['priority_bucket']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
