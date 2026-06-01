from __future__ import annotations

import csv
import json
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PLAYER_POOL_PATH = ROOT / "data" / "player_pool_v1.json"
TM_SUMMARY_PATH = ROOT / "data" / "transfermarkt_national_team" / "player_national_team_usage_competitive_summary.csv"
TM_MATCHES_PATH = ROOT / "data" / "transfermarkt_national_team" / "player_national_team_matches_classified.csv"

BACKUP_PATH = ROOT / "data" / f"player_pool_v1.backup_before_tm_competitive_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_REPORT = ROOT / "data" / "transfermarkt_competitive_summary_merge_report.csv"
OUT_SPLIT_REPORT = ROOT / "data" / "start_probability_availability_split_report.csv"

SOURCE_TAG = f"transfermarkt_availability_split_{datetime.now().strftime('%Y_%m_%d')}"


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    value = txt(value).lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("’", "'").replace("`", "'").replace("´", "'")
    value = value.replace(".", "")
    value = value.replace("-", " ")
    value = " ".join(value.split())
    return value


def canonical_team(value: Any) -> str:
    raw = txt(value).upper()
    aliases = {
        "HOLDET_584": "CZE",
        "HOLDET_767": "CIV",
    }
    return aliases.get(raw, raw)


def key(name: Any, team: Any) -> str:
    return f"{norm(name)}||{canonical_team(team)}"


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def as_int_pct(prob: float | None) -> int | None:
    if prob is None:
        return None
    return int(round(prob * 100))


def classify_start_status(conditional_prob: float | None, availability_prob: float | None) -> str:
    if conditional_prob is None:
        return "ukendt - Transfermarkt landshold"
    if conditional_prob >= 0.88 and (availability_prob or 0.0) >= 0.80:
        return "sikker starter - Transfermarkt landshold"
    if conditional_prob >= 0.75:
        return "sandsynlig starter - Transfermarkt landshold"
    if conditional_prob >= 0.55:
        return "rotation/usikker - Transfermarkt landshold"
    return "sjældent startende - Transfermarkt landshold"


def classify_availability_risk(
    availability_prob: float | None,
    conditional_prob: float | None = None,
    sample_count: int | None = None,
    available_weight: float | None = None,
    absence_weight: float | None = None,
) -> str:
    if availability_prob is None or conditional_prob is None:
        return "unknown"

    sample_count = sample_count or 0
    available_weight = available_weight or 0.0
    absence_weight = absence_weight or 0.0
    total_weight = available_weight + absence_weight
    absence_share = absence_weight / total_weight if total_weight > 0 else 0.0
    strong_absence_signal = absence_weight >= 2.0 and absence_share >= 0.35

    if sample_count < 3 or available_weight <= 0:
        return "unknown"
    if availability_prob < 0.65 or strong_absence_signal:
        return "high_risk"
    if availability_prob >= 0.85 and conditional_prob >= 0.70:
        return "low_risk"
    return "medium_risk"


def is_true(value: Any) -> bool:
    return txt(value).lower() == "true"


def should_ignore_match(row: dict[str, str]) -> bool:
    if not txt(row.get("date_parsed")):
        return True

    row_text = txt(row.get("row_text")).lower()
    ignored_markers = [
        "postponed",
        "information not yet available",
        "not yet available",
        "cancelled",
        "abandoned",
    ]
    return any(marker in row_text for marker in ignored_markers)


def build_availability_splits(match_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    rows_by_player: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in match_rows:
        player_id = txt(row.get("player_id"))
        if not player_id or should_ignore_match(row):
            continue
        rows_by_player[player_id].append(row)

    splits: dict[str, dict[str, Any]] = {}
    for player_id, rows in rows_by_player.items():
        available_rows = []
        absence_rows = []
        available_weight = 0.0
        absence_weight = 0.0
        start_weight = 0.0

        for row in rows:
            started = is_true(row.get("started_estimate_clean")) or is_true(row.get("started_estimate"))
            on_bench = is_true(row.get("was_on_bench_clean")) or is_true(row.get("was_on_bench"))
            not_in_squad = is_true(row.get("was_not_in_squad_clean")) or is_true(row.get("was_not_in_squad"))
            has_position = is_true(row.get("has_position"))
            minutes = as_float(row.get("minutes_estimate"))
            absence_reason = txt(row.get("absence_reason")).lower()
            position_text = txt(row.get("position")).lower()
            weight = as_float(row.get("recency_weight")) or 0.25

            absent = (
                not_in_squad
                or absence_reason in {"injury", "suspension", "absence", "not_selected_or_unknown"}
                or "not in squad" in position_text
                or "injur" in position_text
                or "suspend" in position_text
            )

            available = (
                started
                or on_bench
                or (minutes is not None and minutes > 0)
                or (has_position and not absent)
            )

            if absent:
                absence_rows.append(row)
                absence_weight += weight
                continue

            if available:
                available_rows.append(row)
                available_weight += weight
                if started:
                    start_weight += weight

        if not available_rows and not absence_rows:
            continue

        conditional_raw = start_weight / available_weight if available_weight > 0 else None
        conditional_prob = None if conditional_raw is None else max(0.05, min(0.97, conditional_raw))

        availability_raw = (available_weight + 2.0) / (available_weight + absence_weight + 3.0)
        sample_count = len(available_rows) + len(absence_rows)
        high_sample = sample_count >= 12 and available_weight >= 3.0
        min_availability = 0.60 if high_sample else 0.35
        availability_prob = max(min_availability, min(1.0, availability_raw))

        if conditional_prob is None:
            final_start_prob = None
        else:
            final_start_prob = conditional_prob * (0.65 + 0.35 * availability_prob)
            final_start_prob = max(0.0, min(1.0, final_start_prob))

        splits[player_id] = {
            "conditional_start_prob": conditional_prob,
            "availability_prob": availability_prob,
            "start_prob": final_start_prob,
            "availability_risk": classify_availability_risk(
                availability_prob,
                conditional_prob,
                sample_count,
                available_weight,
                absence_weight,
            ),
            "availability_status": classify_availability_risk(
                availability_prob,
                conditional_prob,
                sample_count,
                available_weight,
                absence_weight,
            ),
            "available_rows": len(available_rows),
            "absence_rows": len(absence_rows),
            "available_weight": available_weight,
            "absence_weight": absence_weight,
            "start_weight": start_weight,
        }

    return splits


def main() -> None:
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(PLAYER_POOL_PATH)
    if not TM_SUMMARY_PATH.exists():
        raise FileNotFoundError(TM_SUMMARY_PATH)
    if not TM_MATCHES_PATH.exists():
        raise FileNotFoundError(TM_MATCHES_PATH)

    with PLAYER_POOL_PATH.open("r", encoding="utf-8-sig") as f:
        players = json.load(f)

    with TM_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        tm_rows = list(csv.DictReader(f))

    with TM_MATCHES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        match_rows = list(csv.DictReader(f))

    tm_by_key: dict[str, dict[str, str]] = {}
    for row in tm_rows:
        row_key = key(row.get("player_name"), row.get("team_id"))
        tm_by_key.setdefault(row_key, row)

    availability_splits = build_availability_splits(match_rows)
    shutil.copy2(PLAYER_POOL_PATH, BACKUP_PATH)

    applied = []
    skipped_no_match = 0

    for player in players:
        player_key = key(player.get("player_name"), player.get("team_id"))
        row = tm_by_key.get(player_key)
        split = availability_splits.get(txt(player.get("player_id")))

        if not row and not split:
            skipped_no_match += 1
            continue

        old_source = txt(player.get("start_prob_source"))
        old_prob = player.get("start_prob")

        summary_score = None
        if row:
            summary_score = (
                as_float(row.get("tm_recency_weighted_competitive_start_score"))
                or as_float(row.get("tm_weighted_competitive_start_score"))
            )

        conditional_prob = split.get("conditional_start_prob") if split else summary_score
        availability_prob = split.get("availability_prob") if split else None
        start_score = split.get("start_prob") if split else summary_score

        if start_score is None or conditional_prob is None:
            skipped_no_match += 1
            continue

        start_score = max(0.0, min(1.0, start_score))
        conditional_prob = max(0.05, min(0.97, float(conditional_prob)))
        if availability_prob is not None:
            availability_prob = max(0.0, min(1.0, float(availability_prob)))

        player["start_prob"] = round(start_score, 4)
        player["start_security"] = round(start_score, 4)
        player["start_probability_pct"] = as_int_pct(start_score)
        player["start_status"] = classify_start_status(conditional_prob, availability_prob)
        player["start_prob_source"] = SOURCE_TAG
        player["conditional_start_prob"] = round(conditional_prob, 4)
        player["availability_prob"] = round(availability_prob, 4) if availability_prob is not None else None
        player["availability_risk"] = split.get("availability_risk") if split else classify_availability_risk(availability_prob, conditional_prob)
        player["availability_status"] = split.get("availability_status") if split else classify_availability_risk(availability_prob, conditional_prob)

        if row:
            player["transfermarkt_competitive_rows"] = as_float(row.get("tm_competition_rows"))
            player["transfermarkt_competitive_starts"] = as_float(row.get("tm_competition_starts"))
            player["transfermarkt_competition_weight_sum"] = as_float(row.get("tm_competition_weight_sum"))
            player["transfermarkt_start_score"] = as_float(row.get("tm_weighted_competitive_start_score"))
            player["transfermarkt_recency_start_score"] = as_float(row.get("tm_recency_weighted_competitive_start_score"))

        player["transfermarkt_signal_confidence"] = 1.0
        if split:
            player["transfermarkt_available_rows"] = split["available_rows"]
            player["transfermarkt_absence_rows"] = split["absence_rows"]
            player["transfermarkt_available_weight"] = round(split["available_weight"], 4)
            player["transfermarkt_absence_weight"] = round(split["absence_weight"], 4)
            player["transfermarkt_start_weight"] = round(split["start_weight"], 4)

        applied.append(
            {
                "player_id": txt(player.get("player_id")),
                "player_name": txt(player.get("player_name")),
                "team_id": canonical_team(player.get("team_id")),
                "old_start_prob_source": old_source,
                "new_start_prob_source": SOURCE_TAG,
                "old_start_prob": old_prob,
                "new_start_prob": player.get("start_prob"),
                "conditional_start_prob": player.get("conditional_start_prob"),
                "availability_prob": player.get("availability_prob"),
                "availability_risk": player.get("availability_risk"),
                "tm_competition_rows": row.get("tm_competition_rows") if row else "",
                "tm_competition_starts": row.get("tm_competition_starts") if row else "",
            }
        )

    with PLAYER_POOL_PATH.open("w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)

    with OUT_SPLIT_REPORT.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "player_name",
            "team_id",
            "old_start_prob",
            "new_start_prob",
            "conditional_start_prob",
            "availability_prob",
            "availability_risk",
            "old_start_prob_source",
            "new_start_prob_source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in applied)

    print("Transfermarkt competitive summary merge")
    print("--------------------------------------")
    print(f"TM summary rows: {len(tm_rows)}")
    print(f"TM classified match rows: {len(match_rows)}")
    print(f"Availability splits: {len(availability_splits)}")
    print(f"Opdaterede spillere: {len(applied)}")
    print(f"Ingen sikkert match: {skipped_no_match}")
    print(f"Backup: {BACKUP_PATH.relative_to(ROOT)}")
    print(f"Availability split report: {OUT_SPLIT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
