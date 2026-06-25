from __future__ import annotations

import csv
import json
import shutil
import unicodedata
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from json_file_safety import write_json_strict


ROOT = Path(__file__).resolve().parents[1]

PLAYER_POOL_PATH = ROOT / "data" / "player_pool_v1.json"
TM_SUMMARY_PATH = ROOT / "data" / "transfermarkt_national_team" / "player_national_team_usage_competitive_summary.csv"
TM_MATCHES_PATH = ROOT / "data" / "transfermarkt_national_team" / "player_national_team_matches_classified.csv"
START_CONTEXT_OVERRIDES_PATH = ROOT / "data" / "start_signal_context_overrides.csv"

BACKUP_PATH = ROOT / "data" / f"player_pool_v1.backup_before_tm_competitive_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_REPORT = ROOT / "data" / "transfermarkt_competitive_summary_merge_report.csv"
OUT_SPLIT_REPORT = ROOT / "data" / "start_probability_availability_split_report.csv"
GK_AUDIT_CSV = ROOT / "data" / "goalkeeper_hierarchy_audit.csv"
GK_AUDIT_MD = ROOT / "data" / "goalkeeper_hierarchy_audit.md"
RECENT_NONSTARTER_AUDIT_CSV = ROOT / "data" / "recent_nonstarter_start_prob_audit.csv"
RECENT_NONSTARTER_AUDIT_MD = ROOT / "data" / "recent_nonstarter_start_prob_audit.md"

SOURCE_TAG = f"transfermarkt_availability_split_{datetime.now().strftime('%Y_%m_%d')}"
GK_SOURCE_TAG = f"{SOURCE_TAG}+gk_hierarchy_normalized"
GK_SANITY_TEAMS = {"ESP", "AUT", "GER", "FRA", "SUI", "ALG", "ARG"}
RECENT_AVAILABLE_WEIGHT = 0.70


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


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def load_start_context_overrides() -> dict[str, dict[str, str]]:
    if not START_CONTEXT_OVERRIDES_PATH.exists():
        return {}

    with START_CONTEXT_OVERRIDES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        player_id = txt(row.get("player_id"))
        if player_id:
            overrides[f"id::{player_id}"] = row
        overrides[key(row.get("player_name"), row.get("team_id"))] = row
    return overrides


def apply_start_context_override(player: dict[str, Any], override: dict[str, str]) -> None:
    numeric_fields = [
        "start_prob",
        "conditional_start_prob",
        "appearance_prob",
        "availability_prob",
    ]
    for field in numeric_fields:
        value = as_float(override.get(field))
        if value is not None:
            player[field] = round(max(0.0, min(1.0, value)), 4)

    if txt(override.get("availability_risk")):
        player["availability_risk"] = txt(override.get("availability_risk"))
        player["availability_status"] = txt(override.get("availability_risk"))
    if txt(override.get("round_specific_rotation_risk")):
        player["round_specific_rotation_risk"] = txt(override.get("round_specific_rotation_risk"))

    start_prob = as_float(player.get("start_prob"))
    conditional_prob = as_float(player.get("conditional_start_prob"))
    availability_prob = as_float(player.get("availability_prob"))
    player["start_security"] = round(start_prob, 4) if start_prob is not None else None
    player["start_probability_pct"] = as_int_pct(start_prob)
    player["start_status"] = classify_start_status(conditional_prob, availability_prob)
    player["start_prob_source"] = f"{SOURCE_TAG}+context_override"
    player["start_signal_context_note"] = txt(override.get("source_note"))


def has_valid_start_context_override(override: dict[str, str] | None) -> bool:
    if not override:
        return False
    for field in [
        "start_prob",
        "conditional_start_prob",
        "appearance_prob",
        "availability_prob",
    ]:
        if as_float(override.get(field)) is not None:
            return True
    return any(
        txt(override.get(field))
        for field in [
            "availability_risk",
            "round_specific_rotation_risk",
            "source_note",
        ]
    )


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


def classify_pool_start_status(start_prob: float) -> str:
    if start_prob >= 0.88:
        return "sikker starter - GK hierarchy"
    if start_prob >= 0.65:
        return "sandsynlig starter - GK hierarchy"
    if start_prob >= 0.25:
        return "rotation/usikker - GK hierarchy"
    return "reservekeeper - GK hierarchy"


def is_true(value: Any) -> bool:
    return txt(value).lower() == "true"


def is_out(player: dict[str, Any]) -> bool:
    return txt(player.get("holdet_is_out")).lower() in {"true", "1", "yes", "ja"}


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
        neutral_unavailable_rows = 0

        for row in rows:
            started = is_true(row.get("started_estimate_clean")) or is_true(row.get("started_estimate"))
            on_bench = is_true(row.get("was_on_bench_clean")) or is_true(row.get("was_on_bench"))
            not_in_squad = is_true(row.get("was_not_in_squad_clean")) or is_true(row.get("was_not_in_squad"))
            has_position = is_true(row.get("has_position"))
            minutes = as_float(row.get("minutes_estimate"))
            absence_reason = txt(row.get("absence_reason")).lower()
            participation_state = txt(row.get("participation_state")).lower()
            position_text = txt(row.get("position")).lower()
            weight = as_float(row.get("recency_weight")) or 0.25
            injury_or_suspension = (
                is_true(row.get("injury_or_suspension"))
                or participation_state in {"injured", "suspended"}
                or absence_reason in {"injury", "injury_or_fitness", "suspension"}
                or "injur" in position_text
                or "suspend" in position_text
            )

            if injury_or_suspension:
                neutral_unavailable_rows += 1
                continue

            absent = (
                not_in_squad
                or absence_reason in {"absence", "not_selected_or_unknown"}
                or "not in squad" in position_text
            )

            available = (
                started
                or on_bench
                or participation_state == "in squad"
                or is_true(row.get("selection_observation"))
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

        historical_start_rate = start_weight / available_weight if available_weight > 0 else None
        recent_available_rows = sorted(
            available_rows,
            key=lambda row: txt(row.get("date_parsed")),
            reverse=True,
        )[:3]
        recent_start_rate = (
            sum(
                1.0
                for row in recent_available_rows
                if is_true(row.get("started_estimate_clean")) or is_true(row.get("started_estimate"))
            )
            / len(recent_available_rows)
            if recent_available_rows
            else None
        )
        if historical_start_rate is None:
            conditional_raw = recent_start_rate
        elif recent_start_rate is None:
            conditional_raw = historical_start_rate
        else:
            conditional_raw = (
                RECENT_AVAILABLE_WEIGHT * recent_start_rate
                + (1.0 - RECENT_AVAILABLE_WEIGHT) * historical_start_rate
            )
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
            "historical_start_rate": historical_start_rate,
            "recent_available_rows": len(recent_available_rows),
            "recent_available_start_rate": recent_start_rate,
            "neutral_unavailable_rows": neutral_unavailable_rows,
        }

    return splits


def gk_raw_score(player: dict[str, Any]) -> float:
    start = as_float(player.get("start_prob")) or 0.0
    conditional = as_float(player.get("conditional_start_prob"))
    availability = as_float(player.get("availability_prob"))
    recency = as_float(player.get("transfermarkt_recency_start_score"))
    competitive = as_float(player.get("transfermarkt_start_score"))
    recent_rate = as_float(player.get("transfermarkt_recent_start_rate_since_2025"))

    signal_parts = []
    for weight, value in [
        (0.40, recency),
        (0.22, competitive),
        (0.18, conditional),
        (0.12, start),
        (0.08, recent_rate),
    ]:
        if value is not None:
            signal_parts.append((weight, max(0.0, min(1.0, value))))

    if signal_parts:
        weight_sum = sum(weight for weight, _ in signal_parts)
        base = sum(weight * value for weight, value in signal_parts) / weight_sum
    else:
        base = start

    availability_factor = 0.65 + 0.35 * max(0.0, min(1.0, availability if availability is not None else 0.75))
    if txt(player.get("start_prob_source")).endswith("+context_override") or "+context_override" in txt(player.get("start_prob_source")):
        if start < 0.35:
            base = min(base, start)
        else:
            base = max(base, start)
    return max(0.001, min(1.0, base * availability_factor))


def gk_team_metrics(players: list[dict[str, Any]]) -> dict[str, float]:
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        if txt(player.get("position")).upper() == "GK":
            by_team[canonical_team(player.get("team_id"))].append(player)

    sums = []
    multi_60 = 0
    multi_tier = 0
    for team_players in by_team.values():
        active = [p for p in team_players if not is_out(p)]
        total = sum(as_float(p.get("start_prob")) or 0.0 for p in active)
        sums.append(total)
        if sum(1 for p in active if (as_float(p.get("start_prob")) or 0.0) >= 0.60) >= 2:
            multi_60 += 1
        if sum(1 for p in active if (as_float(p.get("start_prob")) or 0.0) >= 0.65) >= 2:
            multi_tier += 1
    return {
        "teams_sum_gt_1_10": sum(1 for value in sums if value > 1.10),
        "teams_multi_gk_ge_0_60": multi_60,
        "teams_multi_probable_or_clear": multi_tier,
        "max_team_sum": max(sums) if sums else 0.0,
    }


def normalize_goalkeeper_hierarchy(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_by_id = {txt(player.get("player_id")): dict(player) for player in players if txt(player.get("position")).upper() == "GK"}
    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        if txt(player.get("position")).upper() == "GK":
            by_team[canonical_team(player.get("team_id"))].append(player)

    audit_rows: list[dict[str, Any]] = []
    for team_id, team_players in sorted(by_team.items()):
        active = [player for player in team_players if not is_out(player)]
        if not active:
            for player in team_players:
                player["start_prob"] = 0.0
                player["start_security"] = 0.0
                player["start_probability_pct"] = 0
            continue

        raw_scores = {txt(player.get("player_id")): gk_raw_score(player) for player in active}
        override_ids = {
            txt(player.get("player_id"))
            for player in active
            if "+context_override" in txt(player.get("start_prob_source"))
        }
        max_raw = max(raw_scores.values()) if raw_scores else 1.0
        weighted: dict[str, float] = {}
        for player in active:
            player_id = txt(player.get("player_id"))
            relative = raw_scores[player_id] / max(max_raw, 0.001)
            # Cubic sharpening enforces one active GK slot while preserving uncertainty
            # when the raw signals are genuinely close.
            weighted[player_id] = max(0.0001, relative**3.0)
            if player_id in override_ids and (as_float(player.get("start_prob")) or 0.0) >= 0.35:
                weighted[player_id] *= 1.75

        floor = 0.025 if len(active) <= 3 else 0.015
        raw_total = sum(weighted.values()) or 1.0
        floor_total = floor * max(len(active) - 1, 0)
        primary_id = max(weighted, key=weighted.get)
        normalized: dict[str, float] = {}
        for player in active:
            player_id = txt(player.get("player_id"))
            if player_id == primary_id:
                continue
            normalized[player_id] = floor + (1.0 - floor_total) * (weighted[player_id] / raw_total)
        normalized[primary_id] = max(0.0, 1.0 - sum(normalized.values()))

        # If the model is very uncertain, keep a real challenger but prevent two
        # "probable starter" goalkeepers.
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) > 1 and ordered[1][1] >= 0.60:
            excess = ordered[1][1] - 0.55
            normalized[ordered[1][0]] -= excess
            normalized[ordered[0][0]] += excess

        team_sum = sum(normalized.values())
        if team_sum > 0:
            normalized = {player_id: value / team_sum for player_id, value in normalized.items()}

        ranked = sorted(active, key=lambda player: normalized.get(txt(player.get("player_id")), 0.0), reverse=True)
        for rank, player in enumerate(ranked, start=1):
            player_id = txt(player.get("player_id"))
            raw = before_by_id.get(player_id, {})
            new_prob = round(max(0.0, min(1.0, normalized.get(player_id, 0.0))), 4)
            player["gk_start_prob_raw_before_normalization"] = raw.get("start_prob")
            player["gk_hierarchy_raw_score"] = round(raw_scores.get(player_id, 0.0), 6)
            player["gk_start_prob_normalized"] = new_prob
            player["gk_start_prob_normalized_at"] = datetime.now().strftime("%Y-%m-%d")
            player["gk_team_rank"] = rank
            player["start_prob"] = new_prob
            player["start_security"] = new_prob
            player["start_probability_pct"] = as_int_pct(new_prob)
            player["conditional_start_prob"] = new_prob
            player["start_status"] = classify_pool_start_status(new_prob)
            player["start_prob_source"] = GK_SOURCE_TAG

        for player in team_players:
            player_id = txt(player.get("player_id"))
            raw = before_by_id.get(player_id, {})
            if is_out(player):
                player["start_prob"] = 0.0
                player["start_security"] = 0.0
                player["start_probability_pct"] = 0
                player["conditional_start_prob"] = 0.0
                player["start_status"] = "ude - GK hierarchy"
                normalized_prob = 0.0
                rank = ""
                reason = "holdet_is_out_zeroed"
            else:
                normalized_prob = as_float(player.get("start_prob")) or 0.0
                rank = player.get("gk_team_rank", "")
                reason = "team_gk_exclusive_normalization"

            audit_rows.append(
                {
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": txt(player.get("player_name")),
                    "raw_start_prob": raw.get("start_prob", ""),
                    "raw_start_prob_source": raw.get("start_prob_source", ""),
                    "competitive_starts": player.get("transfermarkt_competitive_starts", ""),
                    "recent_starts": player.get("transfermarkt_recent_starts_since_2025", ""),
                    "availability_prob": player.get("availability_prob", ""),
                    "context_override": "yes" if "+context_override" in txt(raw.get("start_prob_source")) else "",
                    "team_gk_rank": rank,
                    "normalized_gk_start_prob": normalized_prob,
                    "normalized_prob_sum_team": round(sum(as_float(p.get("start_prob")) or 0.0 for p in team_players if not is_out(p)), 4),
                    "repair_reason": reason,
                }
            )

    return audit_rows


def write_gk_audit(audit_rows: list[dict[str, Any]], before_metrics: dict[str, float], after_metrics: dict[str, float]) -> None:
    fields = [
        "team_id",
        "player_id",
        "player_name",
        "raw_start_prob",
        "raw_start_prob_source",
        "competitive_starts",
        "recent_starts",
        "availability_prob",
        "context_override",
        "team_gk_rank",
        "normalized_gk_start_prob",
        "normalized_prob_sum_team",
        "repair_reason",
    ]
    with GK_AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)

    def table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
        lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(txt(row.get(field)) for field in fields) + " |")
        return lines

    sanity = [row for row in audit_rows if row["team_id"] in GK_SANITY_TEAMS]
    sanity.sort(key=lambda row: (row["team_id"], as_float(row.get("team_gk_rank")) or 99))
    lines = [
        "# Goalkeeper Hierarchy Audit",
        "",
        "GK-startchancer normaliseres pr. land efter alle individuelle Transfermarkt-signaler og context-overrides. Kun aktive keepere indgaar i sum-normaliseringen; `holdet_is_out=True` nulstilles.",
        "",
        "Metode: raw score = recency-weighted competitive start score, competitive start score, conditional start probability, existing start_prob, recent start rate og availability. Scores skarpes kubisk og normaliseres til cirka 1.00 pr. land med en lille reservefloor. Context-overrides loeftes som prioriteret input foer normalisering.",
        "",
        "## Foer/efter",
        "",
        f"- Lande hvor GK start_prob-sum > 1.10 foer: {int(before_metrics['teams_sum_gt_1_10'])}",
        f"- Lande hvor GK start_prob-sum > 1.10 efter: {int(after_metrics['teams_sum_gt_1_10'])}",
        f"- Lande med mindst to GK start_prob >= 0.60 foer: {int(before_metrics['teams_multi_gk_ge_0_60'])}",
        f"- Lande med mindst to GK start_prob >= 0.60 efter: {int(after_metrics['teams_multi_gk_ge_0_60'])}",
        f"- Lande med mindst to GK Sandsynlig/Klar starter foer: {int(before_metrics['teams_multi_probable_or_clear'])}",
        f"- Lande med mindst to GK Sandsynlig/Klar starter efter: {int(after_metrics['teams_multi_probable_or_clear'])}",
        f"- Maksimal GK start_prob-sum foer: {before_metrics['max_team_sum']:.4f}",
        f"- Maksimal GK start_prob-sum efter: {after_metrics['max_team_sum']:.4f}",
        "",
        "## Sanity-hold",
        "",
        *table(
            sanity,
            [
                "team_id",
                "team_gk_rank",
                "player_name",
                "raw_start_prob",
                "raw_start_prob_source",
                "competitive_starts",
                "recent_starts",
                "availability_prob",
                "context_override",
                "normalized_gk_start_prob",
                "normalized_prob_sum_team",
            ],
        ),
    ]
    GK_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recent_nonstarter_audit(
    before_players: list[dict[str, Any]],
    after_players: list[dict[str, Any]],
    splits: dict[str, dict[str, Any]],
    match_rows: list[dict[str, str]],
) -> None:
    before_by_id = {txt(player.get("player_id")): player for player in before_players}
    rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in match_rows:
        if txt(row.get("player_id")) and not should_ignore_match(row):
            rows_by_id[txt(row.get("player_id"))].append(row)

    audit_rows = []
    for player in after_players:
        player_id = txt(player.get("player_id"))
        split = splits.get(player_id)
        if not split or split.get("recent_available_start_rate") is None:
            continue
        old = before_by_id.get(player_id, {})
        recent_rows = []
        for row in sorted(rows_by_id.get(player_id, []), key=lambda item: txt(item.get("date_parsed")), reverse=True):
            state = txt(row.get("participation_state")).lower()
            neutral = is_true(row.get("injury_or_suspension")) or state in {"injured", "suspended"}
            available = (
                state == "in squad"
                or is_true(row.get("started_estimate_clean"))
                or is_true(row.get("started_estimate"))
                or (as_float(row.get("minutes_estimate")) or 0.0) > 0
                or is_true(row.get("was_on_bench_clean"))
                or is_true(row.get("was_on_bench"))
                or is_true(row.get("has_position"))
            )
            if neutral or not available:
                continue
            recent_rows.append(
                f"{txt(row.get('date_parsed'))}:{state}:start={1 if is_true(row.get('started_estimate_clean')) or is_true(row.get('started_estimate')) else 0}"
            )
            if len(recent_rows) == 3:
                break

        audit_rows.append(
            {
                "player_id": player_id,
                "player_name": txt(player.get("player_name")),
                "team_id": canonical_team(player.get("team_id")),
                "position": txt(player.get("position")),
                "recent_available_observations": "; ".join(recent_rows),
                "recent_available_start_rate": fmt(split.get("recent_available_start_rate")),
                "historical_weighted_start_rate": fmt(split.get("historical_start_rate")),
                "neutral_injury_or_suspension_rows": split.get("neutral_unavailable_rows", 0),
                "old_conditional_start_prob": fmt(old.get("conditional_start_prob")),
                "new_conditional_start_prob": fmt(player.get("conditional_start_prob")),
                "old_start_prob": fmt(old.get("start_prob")),
                "new_start_prob": fmt(player.get("start_prob")),
                "old_start_prob_source": txt(old.get("start_prob_source")),
                "new_start_prob_source": txt(player.get("start_prob_source")),
                "context_override": "yes" if "+context_override" in txt(player.get("start_prob_source")) else "",
            }
        )

    audit_rows.sort(
        key=lambda row: (
            as_float(row.get("recent_available_start_rate")) or 0.0,
            -(as_float(row.get("old_start_prob")) or 0.0),
        )
    )
    fields = list(audit_rows[0]) if audit_rows else []
    with RECENT_NONSTARTER_AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    rudiger = [row for row in audit_rows if row["player_id"] == "antonio_r_diger__ger"]
    recent_nonstarters = [
        row
        for row in audit_rows
        if (as_float(row.get("recent_available_start_rate")) or 0.0) <= 1 / 3
    ]
    sanity = rudiger + [row for row in recent_nonstarters if row["player_id"] != "antonio_r_diger__ger"][:5]
    lines = [
        "# Recent non-starter start probability audit",
        "",
        "## Modelændring",
        "",
        "- `in squad` uden minutter tæller som en tilgængelig ikke-start.",
        "- Skade og suspension er neutral utilgængelighed og indgår ikke i start-rate-nævneren.",
        f"- Conditional start-rate vægter de tre seneste tilgængelige observationer {RECENT_AVAILABLE_WEIGHT:.0%} og recency-vægtet historik {1 - RECENT_AVAILABLE_WEIGHT:.0%}.",
        "- Context-overrides anvendes bagefter og beholder højeste prioritet.",
        "",
        f"- Spillere med nyere start-rate højst 1/3: {len(recent_nonstarters)}",
        "",
        "## Rüdiger og fem øvrige sanity-spillere",
        "",
        "| player_name | team_id | recent_available_observations | recent_available_start_rate | historical_weighted_start_rate | old_start_prob | new_start_prob | old_conditional_start_prob | new_conditional_start_prob | context_override |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sanity:
        lines.append(
            "| "
            + " | ".join(
                txt(row.get(field))
                for field in [
                    "player_name",
                    "team_id",
                    "recent_available_observations",
                    "recent_available_start_rate",
                    "historical_weighted_start_rate",
                    "old_start_prob",
                    "new_start_prob",
                    "old_conditional_start_prob",
                    "new_conditional_start_prob",
                    "context_override",
                ]
            )
            + " |"
        )
    RECENT_NONSTARTER_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(PLAYER_POOL_PATH)
    if not TM_SUMMARY_PATH.exists():
        raise FileNotFoundError(TM_SUMMARY_PATH)
    if not TM_MATCHES_PATH.exists():
        raise FileNotFoundError(TM_MATCHES_PATH)

    with PLAYER_POOL_PATH.open("r", encoding="utf-8-sig") as f:
        players = json.load(f)
    before_players = deepcopy(players)

    with TM_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        tm_rows = list(csv.DictReader(f))

    with TM_MATCHES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        match_rows = list(csv.DictReader(f))

    tm_by_key: dict[str, dict[str, str]] = {}
    for row in tm_rows:
        row_key = key(row.get("player_name"), row.get("team_id"))
        tm_by_key.setdefault(row_key, row)

    availability_splits = build_availability_splits(match_rows)
    start_context_overrides = load_start_context_overrides()
    shutil.copy2(PLAYER_POOL_PATH, BACKUP_PATH)

    applied = []
    skipped_no_match = 0
    before_gk_metrics = gk_team_metrics(players)

    for player in players:
        player_key = key(player.get("player_name"), player.get("team_id"))
        override = start_context_overrides.get(f"id::{txt(player.get('player_id'))}") or start_context_overrides.get(player_key)
        has_override = has_valid_start_context_override(override)
        row = tm_by_key.get(player_key)
        split = availability_splits.get(txt(player.get("player_id")))

        if not row and not split and not has_override:
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
            if not has_override:
                skipped_no_match += 1
                continue
        else:
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

        if has_override and override:
            apply_start_context_override(player, override)

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
            player["transfermarkt_recent_3_available_start_score"] = round(split["recent_available_start_rate"], 4)
            player["transfermarkt_history_available_start_score"] = round(split["historical_start_rate"], 4)

        applied.append(
            {
                "player_id": txt(player.get("player_id")),
                "player_name": txt(player.get("player_name")),
                "team_id": canonical_team(player.get("team_id")),
                "old_start_prob_source": old_source,
                "new_start_prob_source": player.get("start_prob_source"),
                "old_start_prob": old_prob,
                "new_start_prob": player.get("start_prob"),
                "conditional_start_prob": player.get("conditional_start_prob"),
                "availability_prob": player.get("availability_prob"),
                "availability_risk": player.get("availability_risk"),
                "appearance_prob": player.get("appearance_prob"),
                "round_specific_rotation_risk": player.get("round_specific_rotation_risk"),
                "start_signal_context_note": player.get("start_signal_context_note"),
                "tm_competition_rows": row.get("tm_competition_rows") if row else "",
                "tm_competition_starts": row.get("tm_competition_starts") if row else "",
            }
        )

    gk_audit_rows = normalize_goalkeeper_hierarchy(players)
    after_gk_metrics = gk_team_metrics(players)
    write_gk_audit(gk_audit_rows, before_gk_metrics, after_gk_metrics)
    write_recent_nonstarter_audit(before_players, players, availability_splits, match_rows)

    write_json_strict(PLAYER_POOL_PATH, players)

    with OUT_SPLIT_REPORT.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "player_name",
            "team_id",
            "old_start_prob",
            "new_start_prob",
            "conditional_start_prob",
            "availability_prob",
            "availability_risk",
            "appearance_prob",
            "round_specific_rotation_risk",
            "start_signal_context_note",
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
    print(f"Context overrides: {len(start_context_overrides)}")
    print(f"Opdaterede spillere: {len(applied)}")
    print(f"Ingen sikkert match: {skipped_no_match}")
    print(f"GK teams sum > 1.10 before: {int(before_gk_metrics['teams_sum_gt_1_10'])}")
    print(f"GK teams sum > 1.10 after: {int(after_gk_metrics['teams_sum_gt_1_10'])}")
    print(f"GK teams with >=2 keepers >=0.60 before: {int(before_gk_metrics['teams_multi_gk_ge_0_60'])}")
    print(f"GK teams with >=2 keepers >=0.60 after: {int(after_gk_metrics['teams_multi_gk_ge_0_60'])}")
    print(f"Max GK team start_prob sum before: {before_gk_metrics['max_team_sum']:.4f}")
    print(f"Max GK team start_prob sum after: {after_gk_metrics['max_team_sum']:.4f}")
    print(f"Backup: {BACKUP_PATH.relative_to(ROOT)}")
    print(f"Availability split report: {OUT_SPLIT_REPORT.relative_to(ROOT)}")
    print(f"GK hierarchy audit: {GK_AUDIT_CSV.relative_to(ROOT)}")
    print(f"GK hierarchy report: {GK_AUDIT_MD.relative_to(ROOT)}")
    print(f"Recent non-starter audit: {RECENT_NONSTARTER_AUDIT_CSV.relative_to(ROOT)}")
    print(f"Recent non-starter report: {RECENT_NONSTARTER_AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
