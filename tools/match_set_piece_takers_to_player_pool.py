from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SET_PIECES_PATH = DATA / "set_piece_takers.csv"
ALIASES_PATH = DATA / "set_piece_taker_aliases.csv"
PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
OUT_MATCHED = DATA / "set_piece_takers_matched.csv"
OUT_UNMATCHED = DATA / "set_piece_takers_unmatched_or_ambiguous.csv"

INPUT_FIELDS = ["team_id", "land", "role", "player_name", "role_order", "source_note", "confidence", "is_confirmed"]
MATCHED_FIELDS = [
    *INPUT_FIELDS,
    "matched_player_id",
    "matched_player_name",
    "matched_team_id",
    "matched_position",
    "match_method",
]
UNMATCHED_FIELDS = [
    *INPUT_FIELDS,
    "match_status",
    "candidate_count",
    "candidate_names",
    "candidate_player_ids",
]

VALID_ROLES = {"penalty", "direct_fk", "corner_indirect_fk"}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_name(value: Any) -> str:
    raw = txt(value).lower()
    decomposed = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_player_pool() -> list[dict[str, Any]]:
    return json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8-sig"))


def load_aliases() -> dict[tuple[str, str], dict[str, str]]:
    if not ALIASES_PATH.exists():
        return {}
    aliases: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(ALIASES_PATH):
        team_id = txt(row.get("team_id")).upper()
        source_name = normalize_name(row.get("source_name"))
        if team_id and source_name:
            aliases[(team_id, source_name)] = row
    return aliases


def player_name(player: dict[str, Any]) -> str:
    return txt(player.get("player_name") or player.get("name"))


def player_position(player: dict[str, Any]) -> str:
    return txt(player.get("position") or player.get("holdet_position"))


def name_tokens(value: Any) -> list[str]:
    return normalize_name(value).split()


def safe_alias_candidates(team_players: list[dict[str, Any]], query_name: str) -> list[dict[str, Any]]:
    query_tokens = name_tokens(query_name)
    if not query_tokens:
        return []

    candidates: list[dict[str, Any]] = []
    if len(query_tokens) == 1:
        surname = query_tokens[0]
        for player in team_players:
            tokens = name_tokens(player_name(player))
            if tokens and tokens[-1] == surname:
                candidates.append(player)
        return candidates

    if len(query_tokens) == 2 and len(query_tokens[0]) == 1:
        initial, surname = query_tokens
        for player in team_players:
            tokens = name_tokens(player_name(player))
            if tokens and tokens[0].startswith(initial) and tokens[-1] == surname:
                candidates.append(player)
        return candidates

    for player in team_players:
        candidate = normalize_name(player_name(player))
        query = " ".join(query_tokens)
        if candidate.endswith(query):
            candidates.append(player)
    return candidates


def build_indexes(players: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        team_id = txt(player.get("team_id")).upper()
        name = player_name(player)
        if not team_id or not name:
            continue
        exact[(team_id, name.lower())].append(player)
        normalized[(team_id, normalize_name(name))].append(player)
    return exact, normalized


def base_row(row: dict[str, str]) -> dict[str, str]:
    return {field: txt(row.get(field)) for field in INPUT_FIELDS}


def validate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for row in rows:
        role = txt(row.get("role"))
        if role and role not in VALID_ROLES:
            invalid.append(
                {
                    **base_row(row),
                    "match_status": "invalid_role",
                    "candidate_count": 0,
                    "candidate_names": "",
                    "candidate_player_ids": "",
                }
            )
    return invalid


def match_rows(rows: list[dict[str, str]], players: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_index, normalized_index = build_indexes(players)
    aliases = load_aliases()
    players_by_id = {txt(player.get("player_id")): player for player in players if txt(player.get("player_id"))}
    players_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        players_by_team[txt(player.get("team_id")).upper()].append(player)
    matched: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = validate_rows(rows)

    for row in rows:
        role = txt(row.get("role"))
        if role and role not in VALID_ROLES:
            continue

        team_id = txt(row.get("team_id")).upper()
        name = txt(row.get("player_name"))
        if not team_id or not name:
            unresolved.append(
                {
                    **base_row(row),
                    "match_status": "missing_team_or_player",
                    "candidate_count": 0,
                    "candidate_names": "",
                    "candidate_player_ids": "",
                }
            )
            continue

        alias = aliases.get((team_id, normalize_name(name)))
        candidates: list[dict[str, Any]] = []
        method = "exact"
        if alias:
            resolved_id = txt(alias.get("resolved_player_id"))
            resolved_name = txt(alias.get("resolved_player_name"))
            if resolved_id and resolved_id in players_by_id:
                candidates = [players_by_id[resolved_id]]
                method = "alias_id"
            elif resolved_name:
                candidates = exact_index.get((team_id, resolved_name.lower()), [])
                method = "alias_name_exact"
                if not candidates:
                    candidates = normalized_index.get((team_id, normalize_name(resolved_name)), [])
                    method = "alias_name_normalized"

        if not candidates:
            candidates = exact_index.get((team_id, name.lower()), [])
            method = "exact"
        if not candidates:
            candidates = normalized_index.get((team_id, normalize_name(name)), [])
            method = "normalized"
        if not candidates:
            candidates = safe_alias_candidates(players_by_team.get(team_id, []), name)
            method = "safe_alias"

        if len(candidates) == 1:
            player = candidates[0]
            matched.append(
                {
                    **base_row(row),
                    "matched_player_id": txt(player.get("player_id")),
                    "matched_player_name": player_name(player),
                    "matched_team_id": txt(player.get("team_id")).upper(),
                    "matched_position": player_position(player),
                    "match_method": method,
                }
            )
            continue

        status = "unmatched" if not candidates else "ambiguous"
        unresolved.append(
            {
                **base_row(row),
                "match_status": status,
                "candidate_count": len(candidates),
                "candidate_names": "; ".join(player_name(player) for player in candidates),
                "candidate_player_ids": "; ".join(txt(player.get("player_id")) for player in candidates),
            }
        )

    return matched, unresolved


def main() -> int:
    if not SET_PIECES_PATH.exists():
        raise FileNotFoundError(f"Mangler input: {SET_PIECES_PATH.relative_to(ROOT)}")
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(f"Mangler input: {PLAYER_POOL_PATH.relative_to(ROOT)}")

    rows = read_csv(SET_PIECES_PATH)
    players = load_player_pool()
    matched, unresolved = match_rows(rows, players)

    write_csv(OUT_MATCHED, MATCHED_FIELDS, matched)
    write_csv(OUT_UNMATCHED, UNMATCHED_FIELDS, unresolved)

    examples = ", ".join(row["player_name"] for row in unresolved[:5] if row.get("player_name")) or "ingen"
    print(f"Dødboldsrækker: {len(rows)}")
    print(f"Matchede: {len(matched)}")
    print(f"Unmatched/ambiguous: {len(unresolved)}")
    print(f"Eksempler unmatched/ambiguous: {examples}")
    print(f"Skrevet: {OUT_MATCHED.relative_to(ROOT)}")
    print(f"Skrevet: {OUT_UNMATCHED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
