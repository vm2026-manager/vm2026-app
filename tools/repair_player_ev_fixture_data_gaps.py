from __future__ import annotations

import csv
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
PRICE_DIAG_PATH = DATA / "price_quality_ev_diagnostics.csv"
FIXTURES_PATH = DATA / "fixtures_group.csv"
MULTIPLIERS_PATH = DATA / "fixture_strength_multipliers.csv"

QA_CSV = DATA / "player_ev_fixture_data_gaps.csv"
QA_MD = DATA / "player_ev_fixture_data_gap_report.md"

TEAM_ALIASES = {
    "HOLDET_584": "CZE",
    "HOLDET_767": "CIV",
}
ROUND_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.90}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def strip_accents(value: str) -> str:
    value = (
        value.replace("Æ", "Ae")
        .replace("æ", "ae")
        .replace("Ø", "O")
        .replace("ø", "o")
        .replace("Å", "A")
        .replace("å", "a")
    )
    text = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm_name(value: Any) -> str:
    text = strip_accents(txt(value)).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm_name(value)).strip("_")


def canonical_team(value: Any) -> str:
    team = txt(value).upper()
    return TEAM_ALIASES.get(team, team)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_player_pool() -> list[dict[str, Any]]:
    data = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("player_pool_v1.json forventes at vaere en liste.")
    return data


def write_player_pool(players: list[dict[str, Any]]) -> None:
    PLAYER_POOL_PATH.write_text(json.dumps(players, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fixtures() -> dict[str, list[dict[str, str]]]:
    _, rows = read_csv(FIXTURES_PATH)
    by_team: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        by_team.setdefault(home, []).append(
            {"match_id": txt(row.get("match_id")), "opponent": away, "kickoff_dk": txt(row.get("kickoff_dk")), "is_home": "1"}
        )
        by_team.setdefault(away, []).append(
            {"match_id": txt(row.get("match_id")), "opponent": home, "kickoff_dk": txt(row.get("kickoff_dk")), "is_home": "0"}
        )
    for fixtures in by_team.values():
        fixtures.sort(key=lambda r: int(r["match_id"]) if r["match_id"].isdigit() else 999)
    return by_team


def load_fixture_strength() -> dict[tuple[str, str], dict[str, float]]:
    _, rows = read_csv(MULTIPLIERS_PATH)
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        match_id = txt(row.get("match_id"))
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        lookup[(match_id, home)] = {
            "win_prob": to_float(row.get("home_win_prob_fair")),
            "goal_multiplier": to_float(row.get("home_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("home_assist_multiplier"), 1.0),
            "clean_sheet_multiplier": to_float(row.get("home_clean_sheet_multiplier"), 1.0),
            "clean_sheet_prob": to_float(row.get("home_clean_sheet_prob_fair")),
        }
        lookup[(match_id, away)] = {
            "win_prob": to_float(row.get("away_win_prob_fair")),
            "goal_multiplier": to_float(row.get("away_goal_multiplier"), 1.0),
            "assist_multiplier": to_float(row.get("away_assist_multiplier"), 1.0),
            "clean_sheet_multiplier": to_float(row.get("away_clean_sheet_multiplier"), 1.0),
            "clean_sheet_prob": to_float(row.get("away_clean_sheet_prob_fair")),
        }
    return lookup


def load_price_diag() -> dict[tuple[str, str, str], dict[str, str]]:
    if not PRICE_DIAG_PATH.exists():
        return {}
    _, rows = read_csv(PRICE_DIAG_PATH)
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (norm_name(row.get("player_name")), canonical_team(row.get("team_id")), txt(row.get("position")).upper())
        out[key] = row
    return out


def new_player_id(player_name: Any, team_id: str) -> str:
    return f"{slug(player_name)}__{team_id.lower()}"


def canonicalize_player_identity(row: dict[str, Any]) -> tuple[bool, str, str]:
    old_team = txt(row.get("team_id")).upper()
    new_team = canonical_team(old_team)
    old_id = txt(row.get("player_id"))
    changed = False
    if old_team != new_team:
        row["team_id"] = new_team
        changed = True
    if "__holdet_" in old_id.lower():
        row["player_id"] = new_player_id(row.get("player_name"), new_team)
        changed = True
    return changed, old_team, old_id


def has_round_context(row: dict[str, Any]) -> bool:
    return any(txt(row.get(f"match_{idx}_opponent_team")) for idx in [1, 2, 3])


def has_round_ev(row: dict[str, Any]) -> bool:
    return any(abs(to_float(row.get(f"match_{idx}_weighted_match_ev"))) > 1e-9 for idx in [1, 2, 3])


def ensure_columns(fields: list[str], rows: list[dict[str, Any]], wanted: list[str]) -> list[str]:
    out = list(fields)
    for col in wanted:
        if col not in out:
            out.append(col)
        for row in rows:
            row.setdefault(col, "")
    return out


def restore_ev_from_price_diag(row: dict[str, Any], price_diag: dict[tuple[str, str, str], dict[str, str]]) -> bool:
    current_ev = max(to_float(row.get("weighted_group_stage_ev")), to_float(row.get("optimizer_ev")))
    if current_ev > 0:
        return False
    key = (norm_name(row.get("player_name")), canonical_team(row.get("team_id")), txt(row.get("position")).upper())
    source = price_diag.get(key)
    if not source:
        return False
    source_ev = to_float(source.get("weighted_group_stage_ev"))
    if source_ev <= 0:
        return False

    before = to_float(source.get("model_ev_before_price_quality"))
    row["weighted_group_stage_ev"] = fmt(source_ev)
    row["optimizer_ev"] = fmt(source_ev)
    if to_float(row.get("total_ev_group_stage")) <= 0:
        row["total_ev_group_stage"] = fmt(before if before > 0 else source_ev)
    row["weighted_group_stage_ev_before_price_quality"] = txt(source.get("model_ev_before_price_quality")) or row.get("weighted_group_stage_ev_before_price_quality", "")
    row["price_quality_ev"] = txt(source.get("price_quality_ev")) or row.get("price_quality_ev", "")
    row["price_quality_repair_source"] = "price_quality_ev_diagnostics"
    return True


def fixture_weight(position: str, strength: dict[str, float]) -> float:
    if position in {"GK", "DEF"}:
        return max(0.05, 0.55 * strength.get("clean_sheet_multiplier", 1.0) + 0.45 * (0.75 + strength.get("win_prob", 0.33)))
    if position in {"MID", "FWD"}:
        return max(0.05, 0.55 * strength.get("goal_multiplier", 1.0) + 0.35 * strength.get("assist_multiplier", 1.0) + 0.10 * (0.75 + strength.get("win_prob", 0.33)))
    return 1.0


def fill_round_context(row: dict[str, Any], fixtures_by_team: dict[str, list[dict[str, str]]], strength_lookup: dict[tuple[str, str], dict[str, float]]) -> bool:
    team = canonical_team(row.get("team_id"))
    fixtures = fixtures_by_team.get(team, [])[:3]
    if len(fixtures) < 3 or has_round_context(row):
        return False

    total_weighted_ev = max(to_float(row.get("weighted_group_stage_ev")), to_float(row.get("optimizer_ev")))
    position = txt(row.get("position")).upper()
    raw_weights = []
    for fixture in fixtures:
        strength = strength_lookup.get((fixture["match_id"], team), {})
        raw_weights.append(fixture_weight(position, strength))
    weight_sum = sum(raw_weights) or 1.0

    for idx, fixture in enumerate(fixtures, start=1):
        strength = strength_lookup.get((fixture["match_id"], team), {})
        weighted_ev = total_weighted_ev * raw_weights[idx - 1] / weight_sum if total_weighted_ev > 0 else 0.0
        total_ev_next_match = weighted_ev / ROUND_WEIGHTS[idx]

        row[f"match_{idx}_opponent_team"] = fixture["opponent"]
        row[f"match_{idx}_kickoff"] = fixture["kickoff_dk"]
        row[f"match_{idx}_total_ev_next_match"] = fmt(total_ev_next_match)
        row[f"match_{idx}_weighted_match_ev"] = fmt(weighted_ev)
        row[f"match_{idx}_win_prob"] = fmt(strength.get("win_prob", 0.0))
        row[f"match_{idx}_clean_sheet_prob"] = fmt(strength.get("clean_sheet_prob", 0.0))
        row[f"match_{idx}_goal_multiplier"] = fmt(strength.get("goal_multiplier", 1.0))
        row[f"match_{idx}_assist_multiplier"] = fmt(strength.get("assist_multiplier", 1.0))
        row[f"match_{idx}_clean_sheet_multiplier"] = fmt(strength.get("clean_sheet_multiplier", 1.0))

    row["round_context_source"] = "distributed_from_existing_optimizer_ev"
    return True


def qa_rows(ev_rows: list[dict[str, Any]], fixture_teams: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in ev_rows:
        team = canonical_team(row.get("team_id"))
        optimizer_ev = max(to_float(row.get("optimizer_ev")), to_float(row.get("weighted_group_stage_ev")))
        round_context = has_round_context(row)
        round_ev = has_round_ev(row)
        valid_team = team in fixture_teams
        suspected = "ok"
        if txt(row.get("team_id")).upper().startswith("HOLDET_") or "__holdet_" in txt(row.get("player_id")).lower():
            suspected = "placeholder_team_or_player_id"
        elif not valid_team:
            suspected = "team_has_no_group_fixture"
        elif optimizer_ev <= 0 and not round_ev:
            suspected = "no_ev_source_for_valid_fixture_team"
        elif optimizer_ev > 0 and not round_context:
            suspected = "missing_round_fixture_context_for_existing_ev"
        elif optimizer_ev > 0 and not round_ev:
            suspected = "missing_round_ev_for_existing_ev"

        rows.append(
            {
                "player_id": txt(row.get("player_id")),
                "player_name": txt(row.get("player_name")),
                "team": team,
                "position": txt(row.get("position")),
                "price": txt(row.get("price")),
                "has_valid_team": "yes" if valid_team else "no",
                "has_fixture": "yes" if valid_team else "no",
                "has_round_context": "yes" if round_context else "no",
                "has_optimizer_ev": "yes" if optimizer_ev > 0 else "no",
                "has_fixture_ev": "yes" if round_ev else "no",
                "has_round_ev": "yes" if round_ev else "no",
                "suspected_root_cause": suspected,
            }
        )
    return rows


def count_gaps(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["suspected_root_cause"] != "ok")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fixtures_by_team = load_fixtures()
    fixture_teams = set(fixtures_by_team)
    strength_lookup = load_fixture_strength()
    price_diag = load_price_diag()

    ev_fields, ev_rows = read_csv(EV_PATH)
    before_qa = qa_rows(ev_rows, fixture_teams)
    before_gaps = count_gaps(before_qa)
    before_placeholders = sum(1 for row in before_qa if row["suspected_root_cause"] == "placeholder_team_or_player_id")
    before_missing_round = sum(
        1 for row in before_qa if row["suspected_root_cause"] in {"missing_round_fixture_context_for_existing_ev", "missing_round_ev_for_existing_ev"}
    )

    shutil.copy2(PLAYER_POOL_PATH, PLAYER_POOL_PATH.with_name(f"player_pool_v1.backup_before_fixture_gap_repair_{timestamp}.json"))
    shutil.copy2(EV_PATH, EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_fixture_gap_repair_{timestamp}.csv"))

    players = load_player_pool()
    pool_identity_changes = 0
    pool_ev_restored = 0
    for player in players:
        changed, _, _ = canonicalize_player_identity(player)
        if changed:
            pool_identity_changes += 1
        if restore_ev_from_price_diag(player, price_diag):
            pool_ev_restored += 1
    write_player_pool(players)

    repair_cols = [
        "round_context_source",
        "price_quality_repair_source",
    ]
    for idx in [1, 2, 3]:
        repair_cols.extend(
            [
                f"match_{idx}_win_prob",
                f"match_{idx}_clean_sheet_prob",
                f"match_{idx}_goal_multiplier",
                f"match_{idx}_assist_multiplier",
                f"match_{idx}_clean_sheet_multiplier",
            ]
        )
    ev_fields = ensure_columns(ev_fields, ev_rows, repair_cols)

    ev_identity_changes = 0
    ev_restored = 0
    round_context_filled = 0
    for row in ev_rows:
        changed, _, _ = canonicalize_player_identity(row)
        if changed:
            ev_identity_changes += 1
        if restore_ev_from_price_diag(row, price_diag):
            ev_restored += 1
        if fill_round_context(row, fixtures_by_team, strength_lookup):
            round_context_filled += 1

        row["team_norm"] = canonical_team(row.get("team_id")).lower()
        row["name_norm"] = norm_name(row.get("player_name"))
        row["position_norm"] = txt(row.get("position"))
        row["key_name_team_pos"] = f"{row['name_norm']}|{row['team_norm']}|{row['position_norm']}"
        row["key_name_team"] = f"{row['name_norm']}|{row['team_norm']}"
        row["key_token_team"] = f"{' '.join(sorted(row['name_norm'].split()))}|{row['team_norm']}"

    write_csv(EV_PATH, ev_fields, ev_rows)

    after_qa = qa_rows(ev_rows, fixture_teams)
    after_gaps = count_gaps(after_qa)
    write_csv(
        QA_CSV,
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "price",
            "has_valid_team",
            "has_fixture",
            "has_round_context",
            "has_optimizer_ev",
            "has_fixture_ev",
            "has_round_ev",
            "suspected_root_cause",
        ],
        after_qa,
    )

    after_missing_round = sum(
        1 for row in after_qa if row["suspected_root_cause"] in {"missing_round_fixture_context_for_existing_ev", "missing_round_ev_for_existing_ev"}
    )
    after_placeholders = sum(1 for row in after_qa if row["suspected_root_cause"] == "placeholder_team_or_player_id")
    no_ev_source = [row for row in after_qa if row["suspected_root_cause"] == "no_ev_source_for_valid_fixture_team"]
    missing_context = [row for row in after_qa if row["suspected_root_cause"].startswith("missing_round")]

    lines = [
        "# Player EV Fixture Data Gap Report",
        "",
        "Repairen canonicaliserer kun sikre Holdet-team aliases og fordeler eksisterende optimizer/weighted EV til runde-context. Den opfinder ikke ny spiller-EV.",
        "",
        "## Counts",
        "",
        f"- Gaps foer: {before_gaps}",
        f"- Gaps efter: {after_gaps}",
        f"- Placeholder IDs/teams foer: {before_placeholders}",
        f"- Placeholder IDs/teams efter: {after_placeholders}",
        f"- Missing round context for existing EV foer: {before_missing_round}",
        f"- Missing round context for existing EV efter: {after_missing_round}",
        f"- Player-pool identity changes: {pool_identity_changes}",
        f"- EV identity changes: {ev_identity_changes}",
        f"- EV restored from existing price-quality diagnostics: {ev_restored}",
        f"- Pool EV restored from existing price-quality diagnostics: {pool_ev_restored}",
        f"- Round contexts filled from existing optimizer EV: {round_context_filled}",
        "",
        "## Remaining no-EV-source examples",
        "",
    ]
    for row in no_ev_source[:25]:
        lines.append(f"- {row['player_name']} | {row['team']} | {row['position']} | {row['price']}")
    if len(no_ev_source) > 25:
        lines.append(f"- ... plus {len(no_ev_source) - 25} flere i CSV.")
    lines.extend(["", "## Remaining round-context examples", ""])
    for row in missing_context[:25]:
        lines.append(f"- {row['player_name']} | {row['team']} | {row['position']} | {row['suspected_root_cause']}")
    if len(missing_context) > 25:
        lines.append(f"- ... plus {len(missing_context) - 25} flere i CSV.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Kounde og Neuer har stadig ingen EV-kilde i eksisterende mellemdata; deres fixtures kan findes, men spiller-EV maa genbygges upstream.",
            "- Raphinha, Wesley Franca og Mahmoud Trezeguet havde optimizer_ev uden runde-context; runde-context er udfyldt fra eksisterende EV og fixture-multipliers.",
            "- HOLDET_584 er canonicaliseret til CZE i player_pool og EV-output.",
        ]
    )
    QA_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Skrevet: {PLAYER_POOL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {EV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {QA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Skrevet: {QA_MD.relative_to(PROJECT_ROOT)}")
    print(f"Gaps foer/efter: {before_gaps}/{after_gaps}")
    print(f"Placeholder foer/efter: {before_placeholders}/{after_placeholders}")
    print(f"Missing round context for existing EV foer/efter: {before_missing_round}/{after_missing_round}")
    print(f"Round contexts filled: {round_context_filled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
