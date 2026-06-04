from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"
OUT_CSV = DATA_DIR / "player_pool_vs_ev_start_prob_audit.csv"
OUT_MD = DATA_DIR / "player_pool_vs_ev_start_prob_audit.md"

TOP_PRIORITY_START_SOURCE_MARKERS = [
    "confirmed_lineup",
    "expected_lineup",
    "manual",
    "gk_hierarchy_normalized",
    "start_vs_appearance_context_override",
    "context_override",
]

HIGH_PRIORITY_START_SOURCE_MARKERS = [
    "transfermarkt_availability_split",
]

LOW_PRIORITY_START_SOURCE_MARKERS = [
    "team_minute_rank",
    "holdet_official_unmatched_default",
    "name+team",
    "legacy",
    "fallback",
]

SANITY_NAMES = {
    "Erling Haaland",
    "Martin Ødegaard",
    "Martin Odegaard",
    "Antonio Nusa",
    "Alexander Sørloth",
    "Alexander Sorloth",
    "Harry Kane",
    "Raphinha",
    "Thibaut Courtois",
    "Jules Koundé",
    "Jules Kounde",
    "Manuel Neuer",
    "Ladislav Krejčí",
    "Ladislav Krejci",
    "Vladimír Coufal",
    "Vladimir Coufal",
}


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


def fmt(value: Any, digits: int = 4) -> str:
    return str(round(to_float(value), digits))


def source_priority(source: Any) -> int:
    source_text = txt(source).lower()
    if "gk_hierarchy_normalized" in source_text:
        return 110
    if any(marker in source_text for marker in TOP_PRIORITY_START_SOURCE_MARKERS):
        return 100
    if any(marker in source_text for marker in HIGH_PRIORITY_START_SOURCE_MARKERS):
        return 90
    if any(marker in source_text for marker in LOW_PRIORITY_START_SOURCE_MARKERS):
        return 10
    if source_text:
        return 50
    return 0


def source_bucket(source: Any) -> str:
    source_text = txt(source).lower()
    for marker in LOW_PRIORITY_START_SOURCE_MARKERS:
        if marker in source_text:
            return marker
    for marker in TOP_PRIORITY_START_SOURCE_MARKERS:
        if marker in source_text:
            return marker
    for marker in HIGH_PRIORITY_START_SOURCE_MARKERS:
        if marker in source_text:
            return marker
    return source_text or "blank"


def load_pool_by_id() -> dict[str, dict[str, Any]]:
    data = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {PLAYER_POOL_PATH}")
    return {txt(player.get("player_id")): player for player in data}


def should_promote_pool_signal(pool: dict[str, Any], ev_row: pd.Series) -> bool:
    pool_start = to_float(pool.get("start_prob"), -1.0)
    ev_start = to_float(ev_row.get("start_prob"), -1.0)
    if pool_start < 0:
        return False
    pool_source = pool.get("start_prob_source")
    ev_source = ev_row.get("start_prob_source")
    if "gk_hierarchy_normalized" in txt(pool_source).lower() and abs(pool_start - ev_start) >= 0.0001:
        return True
    if source_priority(pool_source) <= source_priority(ev_source):
        return False
    if abs(pool_start - ev_start) >= 0.01:
        return True
    return txt(pool_source) != txt(ev_source)


def issue_for(pool: dict[str, Any] | None, ev_row: pd.Series) -> tuple[str, str]:
    if not pool:
        return "missing_player_pool_match", "Review player_id mapping."
    pool_start = to_float(pool.get("start_prob"), -1.0)
    ev_start = to_float(ev_row.get("start_prob"), -1.0)
    diff = pool_start - ev_start
    pool_source = pool.get("start_prob_source")
    ev_source = ev_row.get("start_prob_source")
    if should_promote_pool_signal(pool, ev_row):
        return (
            "better_player_pool_start_signal_should_override_legacy_ev_source",
            "Promote player_pool start_prob/source into EV start fields; keep EV components unchanged.",
        )
    if diff >= 0.20:
        return (
            "large_start_prob_mismatch_but_source_priority_not_higher",
            "Manual review; do not overwrite without better source priority.",
        )
    return "ok", "No action."


def build_audit_rows(ev_df: pd.DataFrame, pool_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, ev_row in ev_df.iterrows():
        player_id = txt(ev_row.get("player_id"))
        pool = pool_by_id.get(player_id)
        pool_start = to_float(pool.get("start_prob"), 0.0) if pool else 0.0
        ev_start = to_float(ev_row.get("start_prob"), 0.0)
        issue, action = issue_for(pool, ev_row)
        rows.append(
            {
                "player_id": player_id,
                "player_name": txt(ev_row.get("player_name")),
                "team_id": txt(ev_row.get("team_id")),
                "position": txt(ev_row.get("position")),
                "pool_start_prob": fmt(pool_start),
                "pool_start_prob_source": txt(pool.get("start_prob_source")) if pool else "",
                "ev_start_prob": fmt(ev_start),
                "ev_start_prob_source": txt(ev_row.get("start_prob_source")),
                "start_prob_diff": fmt(pool_start - ev_start),
                "minute_share": fmt(ev_row.get("minute_share"), 6),
                "suspected_issue": issue,
                "recommended_action": action,
            }
        )
    return rows


def count_serious(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row["suspected_issue"] == "better_player_pool_start_signal_should_override_legacy_ev_source"
        and to_float(row["start_prob_diff"]) >= 0.20
    )


def low_source_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if (
            row["suspected_issue"] == "better_player_pool_start_signal_should_override_legacy_ev_source"
            and to_float(row["start_prob_diff"]) >= 0.20
        ):
            counter[source_bucket(row["ev_start_prob_source"])] += 1
    return counter


def repair_ev_df(ev_df: pd.DataFrame, pool_by_id: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = ev_df.copy()
    changed: list[dict[str, Any]] = []
    if "minute_share" not in work.columns:
        work["minute_share"] = ""
    if "start_prob_source" not in work.columns:
        work["start_prob_source"] = ""

    for idx, ev_row in work.iterrows():
        player_id = txt(ev_row.get("player_id"))
        pool = pool_by_id.get(player_id)
        if not pool or not should_promote_pool_signal(pool, ev_row):
            continue
        old_start = to_float(ev_row.get("start_prob"))
        old_source = txt(ev_row.get("start_prob_source"))
        old_minute = to_float(ev_row.get("minute_share"))
        new_start = to_float(pool.get("start_prob"))
        new_source = txt(pool.get("start_prob_source"))
        new_minute = new_start / 11.0 if new_start > 0 else old_minute

        work.at[idx, "start_prob"] = round(new_start, 4)
        work.at[idx, "start_prob_source"] = new_source
        work.at[idx, "minute_share"] = round(new_minute, 6)

        changed.append(
            {
                "player_id": player_id,
                "player_name": txt(ev_row.get("player_name")),
                "team_id": txt(ev_row.get("team_id")),
                "old_start_prob": old_start,
                "new_start_prob": new_start,
                "old_start_prob_source": old_source,
                "new_start_prob_source": new_source,
                "old_minute_share": old_minute,
                "new_minute_share": new_minute,
                "match_1_goal_ev_before_after": txt(ev_row.get("match_1_goal_ev")),
                "match_1_weighted_ev_before_after": txt(ev_row.get("match_1_weighted_match_ev")),
            }
        )
    return work, changed


def write_audit_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "player_id",
        "player_name",
        "team_id",
        "position",
        "pool_start_prob",
        "pool_start_prob_source",
        "ev_start_prob",
        "ev_start_prob_source",
        "start_prob_diff",
        "minute_share",
        "suspected_issue",
        "recommended_action",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int = 40) -> list[str]:
    if not rows:
        return ["Ingen rækker."]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(txt(row.get(col)).replace("|", "/") for col in cols) + " |")
    return out


def write_audit_md(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], changed: list[dict[str, Any]], backup_path: Path) -> None:
    before_counts = low_source_counts(before_rows)
    after_counts = low_source_counts(after_rows)
    sanity = [row for row in after_rows if row["player_name"] in SANITY_NAMES]
    changed_by_id = {row["player_id"]: row for row in changed}
    sanity_changed = []
    for row in sanity:
        change = changed_by_id.get(row["player_id"], {})
        sanity_changed.append(
            {
                **row,
                "old_start_prob": fmt(change.get("old_start_prob", row["ev_start_prob"])),
                "new_start_prob": row["ev_start_prob"],
                "old_source": txt(change.get("old_start_prob_source", row["ev_start_prob_source"])),
                "new_source": row["ev_start_prob_source"],
                "goal_ev_changed": "no",
                "round_ev_changed": "no",
            }
        )

    lines = [
        "# Player Pool vs EV Start Probability Audit",
        "",
        "Audit og målrettet repair af EV-filens startfelter. Goal EV, weighted match EV og strategi-output er ikke genberegnet.",
        "",
        "## Rodårsag",
        "",
        "`tools/rebase_player_ev_to_holdet_master.py` brugte EV-rækkens eksisterende `start_prob` som førstevalg og player-pool signalet som fallback. Dermed kunne legacy/fallback-kilder som `team_minute_rank`, `name+team` og `holdet_official_unmatched_default` overskrive nyere dokumenterede Transfermarkt-/manual-/lineup-signaler.",
        "",
        "## Kildeprioritet efter rettelse",
        "",
        "1. confirmed_lineup / expected_lineup / manual / transfermarkt_availability_split / context_override",
        "2. andre dokumenterede ikke-fallback-kilder",
        "3. team_minute_rank / name+team / holdet_official_unmatched_default / legacy / fallback",
        "",
        "## Mismatch counts",
        "",
        f"- Alvorlige mismatches før: {count_serious(before_rows)}",
        f"- Alvorlige mismatches efter: {count_serious(after_rows)}",
        f"- team_minute_rank før/efter: {before_counts.get('team_minute_rank', 0)} / {after_counts.get('team_minute_rank', 0)}",
        f"- holdet_official_unmatched_default før/efter: {before_counts.get('holdet_official_unmatched_default', 0)} / {after_counts.get('holdet_official_unmatched_default', 0)}",
        f"- name+team før/efter: {before_counts.get('name+team', 0)} / {after_counts.get('name+team', 0)}",
        f"- Rækker påvirket: {len(changed)}",
        f"- Backup: `{backup_path.relative_to(ROOT)}`",
        "",
        "## Sanity-spillere",
        "",
        *md_table(
            sanity_changed,
            [
                "player_name",
                "team_id",
                "old_start_prob",
                "new_start_prob",
                "old_source",
                "new_source",
                "minute_share",
                "goal_ev_changed",
                "round_ev_changed",
            ],
            30,
        ),
        "",
        "## Noter",
        "",
        "- EV-komponenter som `match_1_goal_ev` og `match_1_weighted_match_ev` blev ikke ændret af denne repair.",
        "- `minute_share` blev kun synkroniseret til den nye dokumenterede startchance for de rækker, hvor startsignalet blev promoveret.",
        "- Optimizer og strategi-output blev ikke genkørt.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(PLAYER_POOL_PATH)
    if not EV_PATH.exists():
        raise FileNotFoundError(EV_PATH)

    pool_by_id = load_pool_by_id()
    ev_df = pd.read_csv(EV_PATH)
    before_rows = build_audit_rows(ev_df, pool_by_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_start_prob_source_repair_{timestamp}.csv")
    shutil.copy2(EV_PATH, backup_path)

    repaired_df, changed = repair_ev_df(ev_df, pool_by_id)
    repaired_df.to_csv(EV_PATH, index=False, encoding="utf-8-sig")

    after_rows = build_audit_rows(repaired_df, pool_by_id)
    write_audit_csv(after_rows)
    write_audit_md(before_rows, after_rows, changed, backup_path)

    print("EV start probability source repair")
    print("----------------------------------")
    print(f"Serious mismatches before: {count_serious(before_rows)}")
    print(f"Serious mismatches after: {count_serious(after_rows)}")
    print(f"Rows changed: {len(changed)}")
    print(f"Backup: {backup_path.relative_to(ROOT)}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
