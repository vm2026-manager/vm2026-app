from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OLD_PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
HOLDET_PATH = DATA_DIR / "holdet_players_game_616_flat.csv"

PREVIEW_PATH = DATA_DIR / "player_pool_v1_from_holdet_616_PREVIEW.json"
DIAG_PATH = DATA_DIR / "build_player_pool_from_holdet_616_diagnostics.csv"
UNMATCHED_HOLDET_PATH = DATA_DIR / "build_player_pool_from_holdet_616_unmatched_holdet.csv"


TEAM_ALIASES = {
    "algeriet": "algeria",
    "algeria": "algeria",
    "argentina": "argentina",
    "australien": "australia",
    "australia": "australia",
    "østrig": "austria",
    "ostrig": "austria",
    "austria": "austria",
    "belgien": "belgium",
    "belgium": "belgium",
    "bosnien hercegovina": "bosnia and herzegovina",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "brasilien": "brazil",
    "brazil": "brazil",
    "canada": "canada",
    "cameroun": "cameroon",
    "kamerun": "cameroon",
    "cameroon": "cameroon",
    "kap verde": "cape verde",
    "cape verde": "cape verde",
    "colombia": "colombia",
    "costa rica": "costa rica",
    "curacao": "curacao",
    "curaçao": "curacao",
    "danmark": "denmark",
    "denmark": "denmark",
    "dr congo": "congo dr",
    "congo dr": "congo dr",
    "democratic republic of congo": "congo dr",
    "drc": "congo dr",
    "ecuador": "ecuador",
    "egypten": "egypt",
    "egypt": "egypt",
    "elfenbenskysten": "cote divoire",
    "cote divoire": "cote divoire",
    "côte divoire": "cote divoire",
    "ivory coast": "cote divoire",
    "england": "england",
    "frankrig": "france",
    "france": "france",
    "tyskland": "germany",
    "germany": "germany",
    "ghana": "ghana",
    "haiti": "haiti",
    "holland": "netherlands",
    "netherlands": "netherlands",
    "iran": "iran",
    "irak": "iraq",
    "iraq": "iraq",
    "italien": "italy",
    "italy": "italy",
    "japan": "japan",
    "jordan": "jordan",
    "sydkorea": "south korea",
    "south korea": "south korea",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "kroatien": "croatia",
    "croatia": "croatia",
    "marokko": "morocco",
    "morocco": "morocco",
    "mexico": "mexico",
    "new zealand": "new zealand",
    "norge": "norway",
    "norway": "norway",
    "panama": "panama",
    "paraguay": "paraguay",
    "polen": "poland",
    "poland": "poland",
    "portugal": "portugal",
    "qatar": "qatar",
    "saudi arabien": "saudi arabia",
    "saudi arabia": "saudi arabia",
    "skotland": "scotland",
    "scotland": "scotland",
    "serbien": "serbia",
    "serbia": "serbia",
    "spanien": "spain",
    "spain": "spain",
    "schweiz": "switzerland",
    "switzerland": "switzerland",
    "sverige": "sweden",
    "sweden": "sweden",
    "sydafrika": "south africa",
    "south africa": "south africa",
    "tunesien": "tunisia",
    "tunisia": "tunisia",
    "tyrkiet": "turkiye",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "türkiye": "turkiye",
    "ukraine": "ukraine",
    "uruguay": "uruguay",
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "usbekistan": "uzbekistan",
    "uzbekistan": "uzbekistan",
    "wales": "wales",
    "czechia": "czechia",
    "czech republic": "czechia",
}

TEAM_CODE_ALIASES = {
    "ALG": "algeria",
    "ARG": "argentina",
    "AUS": "australia",
    "AUT": "austria",
    "BEL": "belgium",
    "BIH": "bosnia and herzegovina",
    "BRA": "brazil",
    "CAN": "canada",
    "CIV": "cote divoire",
    "CMR": "cameroon",
    "COD": "congo dr",
    "COL": "colombia",
    "CPV": "cape verde",
    "CRC": "costa rica",
    "CRO": "croatia",
    "CUW": "curacao",
    "CZE": "czechia",
    "DEN": "denmark",
    "ECU": "ecuador",
    "EGY": "egypt",
    "ENG": "england",
    "ESP": "spain",
    "FRA": "france",
    "GER": "germany",
    "GHA": "ghana",
    "HAI": "haiti",
    "IRN": "iran",
    "IRQ": "iraq",
    "ITA": "italy",
    "JOR": "jordan",
    "JPN": "japan",
    "KOR": "south korea",
    "KSA": "saudi arabia",
    "MAR": "morocco",
    "MEX": "mexico",
    "NED": "netherlands",
    "NOR": "norway",
    "PAN": "panama",
    "PAR": "paraguay",
    "POL": "poland",
    "POR": "portugal",
    "QAT": "qatar",
    "RSA": "south africa",
    "SCO": "scotland",
    "SEN": "senegal",
    "SRB": "serbia",
    "SUI": "switzerland",
    "SWE": "sweden",
    "TUN": "tunisia",
    "TUR": "turkiye",
    "UKR": "ukraine",
    "URU": "uruguay",
    "USA": "usa",
    "UZB": "uzbekistan",
    "WAL": "wales",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = strip_accents(text)
    text = text.replace("ø", "o").replace("æ", "ae").replace("å", "a")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm_name(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"\b(jr|junior|sr|senior)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_sort_name(value: Any) -> str:
    tokens = [t for t in norm_name(value).split(" ") if t]
    return " ".join(sorted(tokens))


def compact_name(value: Any) -> str:
    return norm_name(value).replace(" ", "")


def canonical_team(value: Any) -> str:
    text = norm_text(value)
    return TEAM_ALIASES.get(text, text)


def name_score(a: Any, b: Any) -> float:
    a1 = norm_name(a)
    b1 = norm_name(b)
    if not a1 or not b1:
        return 0.0

    direct = SequenceMatcher(None, a1, b1).ratio()
    token = SequenceMatcher(None, token_sort_name(a1), token_sort_name(b1)).ratio()
    compact = SequenceMatcher(None, compact_name(a1), compact_name(b1)).ratio()

    return max(direct, token, compact)


def first_token(value: Any) -> str:
    name = norm_name(value)
    parts = name.split()
    return parts[0] if parts else ""


def last_token(value: Any) -> str:
    name = norm_name(value)
    parts = name.split()
    return parts[-1] if parts else ""


def is_safe_fuzzy_match(holdet_name: Any, old_name: Any, score: float) -> bool:
    """
    Fuzzy bruges kun konservativt.
    Det forhindrer fx Mohamed Alaa -> Mohamed Salah.
    """
    if score < 0.93:
        return False

    h_first = first_token(holdet_name)
    o_first = first_token(old_name)
    h_last = last_token(holdet_name)
    o_last = last_token(old_name)

    if not h_first or not o_first or not h_last or not o_last:
        return False

    # Mindst ét ydernavn skal være meget tæt.
    first_sim = SequenceMatcher(None, h_first, o_first).ratio()
    last_sim = SequenceMatcher(None, h_last, o_last).ratio()

    if max(first_sim, last_sim) < 0.88:
        return False

    # Ekstra værn mod generiske arabiske/navne med "mohamed/mohammad/al".
    generic_tokens = {"mohamed", "mohammad", "muhammad", "ahmed", "ali", "al"}
    h_tokens = set(norm_name(holdet_name).split())
    o_tokens = set(norm_name(old_name).split())

    shared_non_generic = (h_tokens & o_tokens) - generic_tokens
    if not shared_non_generic and (h_tokens & generic_tokens):
        return False

    return True


def load_old_pool() -> list[dict[str, Any]]:
    data = json.loads(OLD_PLAYER_POOL_PATH.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("players", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Kunne ikke finde spillerliste i player_pool_v1.json")


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def build_old_pool_df(old_players: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for idx, p in enumerate(old_players):
        player_name = get_first(p, ["player_name", "name", "full_name", "display_name"])
        team_id = get_first(p, ["team_id", "team", "country_code"])
        team_name = get_first(p, ["team_name", "country", "country_name"])
        position = get_first(p, ["position", "pos"])

        team_canon_from_name = canonical_team(team_name)
        team_canon_from_id = TEAM_CODE_ALIASES.get(str(team_id).upper(), canonical_team(team_id))
        team_canon = team_canon_from_name if team_canon_from_name else team_canon_from_id

        rows.append(
            {
                "old_index": idx,
                "old_player_id": p.get("player_id"),
                "old_player_name": player_name,
                "old_team_id": team_id,
                "old_team_name": team_name,
                "old_position": position,
                "old_team_canon": team_canon,
                "old_name_norm": norm_name(player_name),
                "old_name_token_sort": token_sort_name(player_name),
                "old_name_compact": compact_name(player_name),
            }
        )

    return pd.DataFrame(rows)


def build_team_code_map(old_df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for _, row in old_df.iterrows():
        team_canon = row.get("old_team_canon")
        team_id = row.get("old_team_id")

        if not isinstance(team_canon, str) or not team_canon:
            continue

        if isinstance(team_id, str) and team_id.strip():
            mapping.setdefault(team_canon, team_id.strip().upper())

    for code, canon in TEAM_CODE_ALIASES.items():
        mapping.setdefault(canon, code)

    # Hard guarantees for countries where Holdet team ids may collide with old numeric/proxy ids.
    mapping["croatia"] = "CRO"
    mapping["cote divoire"] = "CIV"
    mapping["congo dr"] = "COD"
    mapping["cape verde"] = "CPV"
    mapping["turkiye"] = "TUR"

    return mapping


def find_best_old_match(
    holdet_row: pd.Series,
    old_df: pd.DataFrame,
    used_old_indices: set[int],
) -> tuple[pd.Series | None, str, float]:
    h_name = holdet_row["player_name"]
    h_team = canonical_team(holdet_row["team_name"])

    candidates = old_df.loc[
        (old_df["old_team_canon"] == h_team)
        & (~old_df["old_index"].astype(int).isin(used_old_indices))
    ].copy()

    if candidates.empty:
        return None, "", 0.0

    exact = candidates.loc[candidates["old_name_norm"] == norm_name(h_name)]
    if not exact.empty:
        return exact.iloc[0], "exact_name_team", 1.0

    token = candidates.loc[candidates["old_name_token_sort"] == token_sort_name(h_name)]
    if not token.empty:
        return token.iloc[0], "token_name_team", 1.0

    candidates["score"] = candidates["old_player_name"].map(lambda x: name_score(h_name, x))
    candidates = candidates.sort_values("score", ascending=False)

    best = candidates.iloc[0]
    score = float(best["score"])
    old_name = best["old_player_name"]

    if is_safe_fuzzy_match(h_name, old_name, score):
        return best, "fuzzy_name_team", score

    return None, "", score


def make_player_id(player_name: str, team_id: str, holdet_player_id: Any) -> str:
    base = norm_name(player_name).replace(" ", "_")
    team = str(team_id).lower()
    if base and team:
        return f"{base}__{team}"
    return f"holdet_{holdet_player_id}"


def build_new_pool(
    old_players: list[dict[str, Any]],
    old_df: pd.DataFrame,
    holdet_df: pd.DataFrame,
    team_code_map: dict[str, str],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    new_players: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    used_old_indices: set[int] = set()

    for _, h in holdet_df.iterrows():
        h_team_canon = canonical_team(h["team_name"])
        team_id = team_code_map.get(h_team_canon, f"HOLDET_{int(h['holdet_team_id'])}")

        old_match, match_method, match_score = find_best_old_match(h, old_df, used_old_indices)
        copied_from_old = old_match is not None

        if copied_from_old:
            old_idx = int(old_match["old_index"])
            used_old_indices.add(old_idx)
            player = dict(old_players[old_idx])
            old_player_id = player.get("player_id")
        else:
            old_idx = None
            old_player_id = None
            player = {}

        player_name = str(h["player_name"])
        position = str(h["position"])
        price = int(h["price"])
        start_price = int(h["start_price"])

        if copied_from_old and old_player_id:
            player_id = str(old_player_id)
        else:
            player_id = make_player_id(player_name, team_id, h["holdet_player_id"])

        player.update(
            {
                "player_id": player_id,
                "player_name": player_name,
                "team_id": team_id,
                "team_name": str(h["team_name"]),
                "position": position,
                "price": price,
                "price_estimate": price,
                "price_source": "holdet_vm_2026_official",
                "position_source": "holdet_vm_2026_official",
                "holdet_game_id": 616,
                "holdet_player_id": int(h["holdet_player_id"]),
                "holdet_person_id": int(h["holdet_person_id"]),
                "holdet_team_id": int(h["holdet_team_id"]),
                "holdet_team_name": str(h["team_name"]),
                "holdet_position_id": int(h["holdet_position_id"]),
                "holdet_position": position,
                "holdet_start_price": start_price,
                "holdet_price": price,
                "holdet_is_out": bool(h["is_out"]),
                "has_holdet_vm_match": True,
                "official_holdet_master": True,
                "copied_from_old_player_pool": bool(copied_from_old),
                "old_player_id": old_player_id,
                "old_match_method": match_method if copied_from_old else "",
                "old_match_score": match_score if copied_from_old else None,
            }
        )

        # Fallbacks for nye officielle spillere, indtil EV-pipelinen genberegner alt.
        player.setdefault("start_prob", 0.25)
        player.setdefault("start_prob_source", "holdet_official_unmatched_default")
        player.setdefault("weighted_group_stage_ev", 0.0)
        player.setdefault("optimizer_ev", player.get("weighted_group_stage_ev", 0.0))

        new_players.append(player)

        diagnostics.append(
            {
                "holdet_player_id": int(h["holdet_player_id"]),
                "holdet_player_name": player_name,
                "holdet_team_name": str(h["team_name"]),
                "team_id": team_id,
                "position": position,
                "price": price,
                "copied_from_old_player_pool": bool(copied_from_old),
                "old_player_id": old_player_id,
                "old_player_name": old_match["old_player_name"] if copied_from_old else "",
                "old_team_id": old_match["old_team_id"] if copied_from_old else "",
                "old_team_name": old_match["old_team_name"] if copied_from_old else "",
                "old_match_method": match_method,
                "old_match_score": match_score,
            }
        )

    return new_players, pd.DataFrame(diagnostics)


def print_summary(new_players: list[dict[str, Any]], diag: pd.DataFrame) -> None:
    copied = int(diag["copied_from_old_player_pool"].sum())
    total = len(diag)
    unmatched = total - copied

    print("\nBUILD OFFICIAL PLAYER POOL FROM HOLDET 616")
    print(f"Officielle Holdet-spillere: {total}")
    print(f"Kopieret med gamle model-/EV-felter: {copied} / {total}")
    print(f"Nye officielle spillere uden gammelt match: {unmatched} / {total}")

    print("\nPositioner:")
    print(pd.Series([p.get("position") for p in new_players]).value_counts(dropna=False).to_string())

    print("\nHold:")
    print(pd.Series([p.get("team_id") for p in new_players]).nunique(), "unikke team_id")

    print("\nTeam-id check, Kroatien:")
    croatia_rows = diag.loc[diag["holdet_team_name"].astype(str).str.lower() == "croatia"]
    if croatia_rows.empty:
        print("(ingen Croatia-rækker fundet)")
    else:
        print(croatia_rows[["holdet_player_name", "holdet_team_name", "team_id"]].head(12).to_string(index=False))

    print("\nPris:")
    prices = pd.Series([p.get("price_estimate") for p in new_players], dtype="float")
    print(f"Min:    {int(prices.min()):,}".replace(",", "."))
    print(f"Median: {int(prices.median()):,}".replace(",", "."))
    print(f"Max:    {int(prices.max()):,}".replace(",", "."))

    print("\nMatchmetode:")
    print(diag["old_match_method"].replace("", "unmatched").value_counts(dropna=False).to_string())

    print("\nLaveste fuzzy-matchscore:")
    fuzzy = diag.loc[diag["old_match_method"] == "fuzzy_name_team"].copy()
    if fuzzy.empty:
        print("(ingen)")
    else:
        cols = [
            "holdet_player_name",
            "holdet_team_name",
            "position",
            "old_player_name",
            "old_team_name",
            "old_match_score",
        ]
        print(fuzzy.sort_values("old_match_score")[cols].head(25).to_string(index=False))

    print("\nFørste 30 officielle spillere uden gammelt match:")
    unmatched_df = diag.loc[~diag["copied_from_old_player_pool"]].head(30)
    if unmatched_df.empty:
        print("(ingen)")
    else:
        cols = ["holdet_player_name", "holdet_team_name", "team_id", "position", "price"]
        print(unmatched_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Overskriv player_pool_v1.json med Holdet-master")
    args = parser.parse_args()

    if not OLD_PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(f"Mangler {OLD_PLAYER_POOL_PATH}")

    if not HOLDET_PATH.exists():
        raise FileNotFoundError(f"Mangler {HOLDET_PATH}")

    old_players = load_old_pool()
    old_df = build_old_pool_df(old_players)
    holdet_df = pd.read_csv(HOLDET_PATH)

    team_code_map = build_team_code_map(old_df)

    new_players, diag = build_new_pool(
        old_players=old_players,
        old_df=old_df,
        holdet_df=holdet_df,
        team_code_map=team_code_map,
    )

    PREVIEW_PATH.write_text(
        json.dumps(new_players, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diag.to_csv(DIAG_PATH, index=False, encoding="utf-8-sig")

    unmatched = diag.loc[~diag["copied_from_old_player_pool"]].copy()
    unmatched.to_csv(UNMATCHED_HOLDET_PATH, index=False, encoding="utf-8-sig")

    print_summary(new_players, diag)

    print("\nFiler skrevet:")
    print(f"Preview:   {PREVIEW_PATH}")
    print(f"Diagnose:  {DIAG_PATH}")
    print(f"Unmatched: {UNMATCHED_HOLDET_PATH}")

    if not args.write:
        print("\nDRY RUN: player_pool_v1.json er IKKE ændret.")
        print("Hvis outputtet ser fornuftigt ud, kør igen med --write.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = OLD_PLAYER_POOL_PATH.with_name(
        f"player_pool_v1.backup_before_official_holdet_master_{timestamp}.json"
    )
    shutil.copy2(OLD_PLAYER_POOL_PATH, backup_path)

    OLD_PLAYER_POOL_PATH.write_text(
        json.dumps(new_players, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSKREV ÆNDRINGER")
    print(f"Backup:    {backup_path}")
    print(f"Opdateret: {OLD_PLAYER_POOL_PATH}")


if __name__ == "__main__":
    main()