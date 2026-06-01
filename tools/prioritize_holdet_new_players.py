from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NEW_PLAYERS_REVIEW_PATH = DATA_DIR / "holdet_new_players_review.csv"
MISMATCH_REPORT_PATH = DATA_DIR / "holdet_possible_name_or_team_mismatches.csv"
OUT_PATH = DATA_DIR / "holdet_new_players_priority_review.csv"
MISMATCHES_ONLY_PATH = DATA_DIR / "holdet_mismatches_only_review.csv"

OUT_FIELDS = [
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
]

STAR_NAME_MARKERS = {
    "bento",
    "bentancur",
    "bowen",
    "camavinga",
    "cole palmer",
    "de jong",
    "ekitike",
    "frenkie de jong",
    "gnabry",
    "joao palhinha",
    "joao pedro",
    "kolo muani",
    "le normand",
    "mitoma",
    "openda",
    "palmer",
    "palhinha",
    "richarlison",
    "serge gnabry",
    "trent alexander arnold",
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def parse_price(value: Any) -> int:
    cleaned = re.sub(r"[^\d]", "", txt(value))
    return int(cleaned) if cleaned else 0


def is_true(value: Any) -> bool:
    return txt(value).lower() in {"true", "1", "yes", "ja", "y"}


def bucket_for(price: int, out: bool) -> str:
    if out:
        return "low_priority"
    if price >= 4_000_000:
        return "must_review"
    if price >= 3_000_000:
        return "likely_relevant"
    if price >= 2_000_000:
        return "depth_or_watchlist"
    return "low_priority"


def is_star_profile(player_name: str) -> bool:
    name = norm(player_name)
    return any(marker in name for marker in STAR_NAME_MARKERS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not NEW_PLAYERS_REVIEW_PATH.exists():
        print(f"FEJL: Mangler {NEW_PLAYERS_REVIEW_PATH.relative_to(PROJECT_ROOT)}")
        return 1
    if not MISMATCH_REPORT_PATH.exists():
        print(f"FEJL: Mangler {MISMATCH_REPORT_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    all_review_rows = read_csv(NEW_PLAYERS_REVIEW_PATH)
    review_rows = [
        row
        for row in all_review_rows
        if txt(row.get("status")).lower() in {"", "potential_new"}
    ]
    mismatch_rows = read_csv(MISMATCH_REPORT_PATH)

    mismatch_by_id = {txt(row.get("holdet_player_id")): row for row in mismatch_rows}
    review_ids = {txt(row.get("holdet_player_id")) for row in review_rows}
    mismatches_only_rows = [
        row
        for row in mismatch_rows
        if txt(row.get("holdet_player_id")) not in review_ids
    ]
    out_rows: list[dict[str, Any]] = []

    for row in review_rows:
        holdet_player_id = txt(row.get("holdet_player_id"))
        price = parse_price(row.get("price"))
        out = is_true(row.get("is_out"))
        mismatch = mismatch_by_id.get(holdet_player_id)
        priority_bucket = bucket_for(price, out)

        possible_duplicate = ""
        reason = txt(row.get("reason"))
        if out:
            suggested_action = "ignore_for_now"
            reason = f"{reason}; is_out=True" if reason else "is_out=True"
        elif mismatch:
            suggested_action = "check_name_or_duplicate"
            possible_duplicate = " | ".join(
                part
                for part in [
                    txt(mismatch.get("closest_pool_player_name")),
                    txt(mismatch.get("closest_pool_team_id")),
                    txt(mismatch.get("closest_pool_position")),
                    txt(mismatch.get("reason")),
                ]
                if part
            )
        elif priority_bucket in {"must_review", "likely_relevant"} or is_star_profile(row.get("player_name", "")):
            suggested_action = "add_to_player_pool_review"
        elif priority_bucket == "depth_or_watchlist":
            suggested_action = "add_to_player_pool_review"
        else:
            suggested_action = "ignore_for_now"

        out_rows.append(
            {
                "priority_bucket": priority_bucket,
                "suggested_action": suggested_action,
                "holdet_player_id": holdet_player_id,
                "player_name": txt(row.get("player_name")),
                "team_id": txt(row.get("team_id")),
                "team_name": txt(row.get("team_name")),
                "position": txt(row.get("position")),
                "price": price,
                "is_out": txt(row.get("is_out")),
                "reason": reason,
                "possible_duplicate_or_mismatch": possible_duplicate,
            }
        )

    bucket_order = {
        "must_review": 0,
        "likely_relevant": 1,
        "depth_or_watchlist": 2,
        "low_priority": 3,
    }
    action_order = {
        "add_to_player_pool_review": 0,
        "check_name_or_duplicate": 1,
        "ignore_for_now": 2,
    }
    out_rows.sort(
        key=lambda row: (
            bucket_order.get(row["priority_bucket"], 99),
            action_order.get(row["suggested_action"], 99),
            -parse_price(row["price"]),
            row["player_name"],
        )
    )

    write_csv(OUT_PATH, OUT_FIELDS, out_rows)
    write_csv(MISMATCHES_ONLY_PATH, list(mismatch_rows[0].keys()) if mismatch_rows else [], mismatches_only_rows)

    print(f"Skrevet: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {MISMATCHES_ONLY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Priority-review rækker: {len(out_rows)}")
    print("Priority buckets:")
    for key, value in sorted(Counter(row["priority_bucket"] for row in out_rows).items(), key=lambda kv: bucket_order.get(kv[0], 99)):
        print(f"- {key}: {value}")
    print("Suggested actions:")
    for key, value in sorted(Counter(row["suggested_action"] for row in out_rows).items(), key=lambda kv: action_order.get(kv[0], 99)):
        print(f"- {key}: {value}")
    print("Top 50 must_review/likely_relevant:")
    top_rows = [row for row in out_rows if row["priority_bucket"] in {"must_review", "likely_relevant"}][:50]
    for row in top_rows:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['price']} | {row['priority_bucket']} | {row['suggested_action']}")
    print(f"Mismatches-only rækker: {len(mismatches_only_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
