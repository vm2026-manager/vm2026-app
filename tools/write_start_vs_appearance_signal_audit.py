from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
PLAYER_EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
START_SPLIT_REPORT_PATH = DATA_DIR / "start_probability_availability_split_report.csv"
START_SECURITY_PATH = DATA_DIR / "player_start_security_nt.csv"
OVERRIDES_PATH = DATA_DIR / "start_signal_context_overrides.csv"

OUT_CSV = DATA_DIR / "start_vs_appearance_signal_audit.csv"
OUT_MD = DATA_DIR / "start_vs_appearance_signal_audit.md"

SOURCE_TAG = "start_vs_appearance_context_override_2026_06_04"

WATCH_NAMES = {
    "mike maignan",
    "andreas schjelderup",
    "manu kone",
    "jurrien timber",
    "deniz undav",
    "ismael saibari",
    "ismaila sarr",
    "patrick wimmer",
}

CSV_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "position",
    "current_start_prob",
    "current_conditional_start_prob",
    "current_appearance_prob",
    "current_availability_risk",
    "before_start_prob",
    "before_conditional_start_prob",
    "before_appearance_prob",
    "before_availability_risk",
    "recent_starts",
    "recent_sub_appearances",
    "recent_unused_bench",
    "recent_absences",
    "injury_absences",
    "competitive_match_starts",
    "friendly_match_starts",
    "post_qualification_rotation_matches",
    "round_specific_rotation_risk",
    "suspected_issue",
    "recommended_action",
    "source_note",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def as_float(value: Any, default: float | None = None) -> float | None:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def as_int_pct(prob: float | None) -> int | None:
    if prob is None:
        return None
    return int(round(prob * 100))


def rounded_prob(value: Any) -> float | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return round(max(0.0, min(1.0, parsed)), 4)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key_for(player_name: Any, team_id: Any) -> str:
    return f"{norm(player_name)}||{txt(team_id).upper()}"


def load_overrides() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows = load_csv(OVERRIDES_PATH)
    by_id: dict[str, dict[str, str]] = {}
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        if txt(row.get("player_id")):
            by_id[txt(row.get("player_id"))] = row
        by_key[key_for(row.get("player_name"), row.get("team_id"))] = row
    return by_id, by_key


def status_from_prob(conditional_prob: float | None, availability_prob: float | None) -> str:
    conditional_prob = conditional_prob or 0.0
    availability_prob = availability_prob or 0.0
    if conditional_prob >= 0.88 and availability_prob >= 0.80:
        return "sikker starter - context override"
    if conditional_prob >= 0.75:
        return "sandsynlig starter - context override"
    if conditional_prob >= 0.55:
        return "rotation/usikker - context override"
    return "sjaeldent startende - context override"


def apply_override_to_player(player: dict[str, Any], override: dict[str, str]) -> None:
    for field in ["start_prob", "conditional_start_prob", "appearance_prob", "availability_prob"]:
        value = rounded_prob(override.get(field))
        if value is not None:
            player[field] = value

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
    player["start_status"] = status_from_prob(conditional_prob, availability_prob)
    player["start_prob_source"] = SOURCE_TAG
    player["start_signal_context_note"] = txt(override.get("source_note"))


def apply_override_to_ev_row(row: pd.Series, player: dict[str, Any]) -> pd.Series:
    for field in ["start_prob", "conditional_start_prob", "appearance_prob", "availability_prob"]:
        if field in row.index and player.get(field) not in (None, ""):
            row[field] = player.get(field)
    for field in ["availability_risk", "availability_status", "round_specific_rotation_risk", "start_prob_source"]:
        if field in row.index and player.get(field) not in (None, ""):
            row[field] = player.get(field)
    return row


def field_as_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field, "")
    if isinstance(value, float):
        return str(round(value, 4))
    return txt(value)


def audit_issue(player: dict[str, Any], override: dict[str, str] | None, raw_security: dict[str, str] | None) -> tuple[str, str]:
    name = norm(player.get("player_name"))
    if override:
        if name == "mike maignan":
            return (
                "normal rotation/friendly bench was treated like weak starter evidence",
                "restore established-GK start signal from existing raw start-security row and prior canonical layer",
            )
        if name == "andreas schjelderup":
            return (
                "sub appearances inflated start probability",
                "separate appearance_prob from start_prob using documented squad/start/sub split",
            )
        if name == "manu kone":
            return (
                "recent injury absences were not strong enough availability negatives",
                "mark high_risk and cap start/appearance signals until injury context clears",
            )
        return (
            "canonical availability split over-promoted an uncertain starter",
            "use existing raw country start-security row as safer start layer",
        )

    start_prob = as_float(player.get("start_prob"), 0.0) or 0.0
    conditional = as_float(player.get("conditional_start_prob"), 0.0) or 0.0
    risk = txt(player.get("availability_risk"))
    raw_start = as_float(raw_security.get("start_security") if raw_security else None)
    if raw_start is not None and start_prob - raw_start >= 0.18:
        return (
            "canonical split is materially higher than raw national start-security row",
            "manual review before using as secure starter",
        )
    if start_prob >= 0.65 and conditional >= 0.70 and risk == "high_risk":
        return (
            "high start signal conflicts with high availability risk",
            "review injury/absence context before optimizer rerun",
        )
    return ("no high-confidence data error found", "keep current signal; review only if new usage data arrives")


def build_raw_security_lookup() -> dict[str, dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for row in load_csv(START_SECURITY_PATH):
        if txt(row.get("player_id")):
            continue
        by_name.setdefault(norm(row.get("player_name")), row)
    return by_name


def update_split_report(players_by_id: dict[str, dict[str, Any]], overrides_by_id: dict[str, dict[str, str]]) -> None:
    rows = load_csv(START_SPLIT_REPORT_PATH)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    for extra in ["appearance_prob", "round_specific_rotation_risk", "start_signal_context_note"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    for row in rows:
        row_key = key_for(row.get("player_name"), row.get("team_id"))
        player = next(
            (
                p
                for p in players_by_id.values()
                if key_for(p.get("player_name"), p.get("team_id")) == row_key
            ),
            None,
        )
        if not player or txt(player.get("player_id")) not in overrides_by_id:
            continue
        row["new_start_prob"] = field_as_str(player, "start_prob")
        row["conditional_start_prob"] = field_as_str(player, "conditional_start_prob")
        row["availability_prob"] = field_as_str(player, "availability_prob")
        row["availability_risk"] = field_as_str(player, "availability_risk")
        row["new_start_prob_source"] = SOURCE_TAG
        row["appearance_prob"] = field_as_str(player, "appearance_prob")
        row["round_specific_rotation_risk"] = field_as_str(player, "round_specific_rotation_risk")
        row["start_signal_context_note"] = field_as_str(player, "start_signal_context_note")

    write_csv(START_SPLIT_REPORT_PATH, rows, fieldnames)


def write_markdown(rows: list[dict[str, Any]], changed_rows: list[dict[str, Any]], before_flags: int | None, after_flags: int | None) -> None:
    focus = {norm(row["player_name"]): row for row in rows if norm(row["player_name"]) in WATCH_NAMES}
    lines = [
        "# Start vs. appearance signal audit",
        "",
        "Denne audit skiller startchance, appearance/indhop og availability tydeligere ad. Der er ikke koert optimizer, strategi-output eller frontend.",
        "",
        "## Kort konklusion",
        "",
        "- Maignans lave startchance skyldtes, at normal rotation/friendly-bench blev vaegtet som svagt startsignal. Den er rettet med eksisterende raw start-security og tidligere canonical layer som kilde.",
        "- Schjelderup havde hoej appearance, men for hoej startchance. Start er nu skilt fra indhop.",
        "- Manu Kone havde skadefravaer, der ikke slog haardt nok igennem. Han er nu high_risk med lavere availability/start.",
        "- Patrick Wimmer og Ismaila Sarr var overpromoveret af canonical availability-splitten og er sat tilbage mod eksisterende raw national start-security.",
        "",
        "## Sanity cases",
        "",
        "| Spiller | Start foer | Start efter | Conditional foer | Conditional efter | Appearance efter | Risk efter | Handling |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    for name in [
        "mike maignan",
        "andreas schjelderup",
        "manu kone",
        "jurrien timber",
        "deniz undav",
        "ismael saibari",
        "ismaila sarr",
        "patrick wimmer",
    ]:
        row = focus.get(name)
        if not row:
            continue
        lines.append(
            "| {player_name} | {before_start_prob} | {current_start_prob} | {before_conditional_start_prob} | {current_conditional_start_prob} | {current_appearance_prob} | {current_availability_risk} | {recommended_action} |".format(
                **row
            )
        )

    lines += [
        "",
        "## Rettede hoej-sikkerhedsfejl",
        "",
    ]
    for row in changed_rows:
        lines.append(
            f"- {row['player_name']}: {row['suspected_issue']} -> {row['recommended_action']}."
        )

    unchanged_review = [
        row for row in rows
        if norm(row["player_name"]) in {"jurrien timber", "deniz undav", "ismael saibari"}
        and norm(row["player_name"]) not in {norm(changed["player_name"]) for changed in changed_rows}
    ]
    if unchanged_review:
        lines += ["", "## Ikke rettet, men plausibelt markeret", ""]
        for row in unchanged_review:
            lines.append(f"- {row['player_name']}: {row['suspected_issue']}; {row['recommended_action']}.")

    if before_flags is not None or after_flags is not None:
        lines += [
            "",
            "## Bubble-audit flags",
            "",
            f"- Foer denne start-audit: {before_flags if before_flags is not None else 'ukendt'}",
            f"- Efter genkoersel: {after_flags if after_flags is not None else 'koeres separat'}",
        ]

    lines += [
        "",
        "## Note om datagrundlag",
        "",
        "De ra Transfermarkt-matchfiler, som de gamle batch/classification-scriptnavne peger paa, findes ikke i repoet. Rettelserne er derfor lagt som et lille dokumenteret context-override-lag og tilsluttet det eksisterende merge-script, saa fremtidige merge-koersler kan bevare samme skelnen mellem start, indhop og skadefravaer.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_bubble_flags() -> int | None:
    path = DATA_DIR / "bubble_player_audit.csv"
    if not path.exists():
        return None
    rows = load_csv(path)
    return sum(1 for row in rows if txt(row.get("model_error_flag")).lower() in {"true", "1", "yes"})


def main() -> None:
    players = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))
    overrides_by_id, overrides_by_key = load_overrides()
    raw_security = build_raw_security_lookup()
    before_flags = count_bubble_flags()

    before_by_id: dict[str, dict[str, Any]] = {
        txt(player.get("player_id")): {
            "start_prob": player.get("start_prob"),
            "conditional_start_prob": player.get("conditional_start_prob"),
            "appearance_prob": player.get("appearance_prob"),
            "availability_risk": player.get("availability_risk"),
        }
        for player in players
    }

    changed_ids: set[str] = set()
    for player in players:
        player_id = txt(player.get("player_id"))
        override = overrides_by_id.get(player_id) or overrides_by_key.get(key_for(player.get("player_name"), player.get("team_id")))
        if not override:
            continue
        apply_override_to_player(player, override)
        changed_ids.add(player_id)

    PLAYER_POOL_PATH.write_text(json.dumps(players, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if PLAYER_EV_PATH.exists():
        ev_df = pd.read_csv(PLAYER_EV_PATH)
        players_by_id = {txt(player.get("player_id")): player for player in players}
        for idx, row in ev_df.iterrows():
            player_id = txt(row.get("player_id"))
            if player_id in changed_ids:
                ev_df.loc[idx] = apply_override_to_ev_row(row, players_by_id[player_id])
        ev_df.to_csv(PLAYER_EV_PATH, index=False)

    players_by_id = {txt(player.get("player_id")): player for player in players}
    update_split_report(players_by_id, overrides_by_id)

    audit_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    for player in players:
        name_norm = norm(player.get("player_name"))
        override = overrides_by_id.get(txt(player.get("player_id"))) or overrides_by_key.get(key_for(player.get("player_name"), player.get("team_id")))
        raw_row = raw_security.get(name_norm)
        issue, action = audit_issue(player, override, raw_row)

        include = bool(override) or name_norm in WATCH_NAMES or issue != "no high-confidence data error found"
        if not include:
            continue

        before = before_by_id.get(txt(player.get("player_id")), {})
        row = {
            "player_id": txt(player.get("player_id")),
            "player_name": txt(player.get("player_name")),
            "team": txt(player.get("team_id")),
            "position": txt(player.get("position")),
            "current_start_prob": field_as_str(player, "start_prob"),
            "current_conditional_start_prob": field_as_str(player, "conditional_start_prob"),
            "current_appearance_prob": field_as_str(player, "appearance_prob"),
            "current_availability_risk": field_as_str(player, "availability_risk"),
            "before_start_prob": field_as_str(before, "start_prob"),
            "before_conditional_start_prob": field_as_str(before, "conditional_start_prob"),
            "before_appearance_prob": field_as_str(before, "appearance_prob"),
            "before_availability_risk": field_as_str(before, "availability_risk"),
            "recent_starts": txt(override.get("recent_starts")) if override else "",
            "recent_sub_appearances": txt(override.get("recent_sub_appearances")) if override else "",
            "recent_unused_bench": txt(override.get("recent_unused_bench")) if override else "",
            "recent_absences": txt(override.get("recent_absences")) if override else txt(player.get("transfermarkt_absence_rows")),
            "injury_absences": txt(override.get("injury_absences")) if override else "",
            "competitive_match_starts": txt(override.get("competitive_match_starts")) if override else txt(raw_row.get("starts_def_used") if raw_row else ""),
            "friendly_match_starts": txt(override.get("friendly_match_starts")) if override else "",
            "post_qualification_rotation_matches": txt(override.get("post_qualification_rotation_matches")) if override else "",
            "round_specific_rotation_risk": field_as_str(player, "round_specific_rotation_risk"),
            "suspected_issue": issue,
            "recommended_action": action,
            "source_note": txt(override.get("source_note")) if override else "",
        }
        audit_rows.append(row)
        if txt(player.get("player_id")) in changed_ids:
            changed_rows.append(row)

    audit_rows.sort(key=lambda row: (0 if txt(row.get("player_id")) in changed_ids else 1, row["team"], row["player_name"]))
    write_csv(OUT_CSV, audit_rows, CSV_COLUMNS)
    write_markdown(audit_rows, changed_rows, before_flags, None)

    print("Start vs appearance signal audit")
    print("--------------------------------")
    print(f"Audit rows: {len(audit_rows)}")
    print(f"Context overrides applied: {len(changed_rows)}")
    print(f"Bubble model_error_flags before: {before_flags if before_flags is not None else 'unknown'}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
