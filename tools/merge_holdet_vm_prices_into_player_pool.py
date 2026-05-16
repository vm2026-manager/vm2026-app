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

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
HOLDET_FLAT_PATH = DATA_DIR / "holdet_players_game_616_flat.csv"

DIAG_CSV_PATH = DATA_DIR / "merge_holdet_vm_prices_diagnostics.csv"
UNMATCHED_POOL_CSV_PATH = DATA_DIR / "merge_holdet_vm_unmatched_player_pool.csv"
UNMATCHED_HOLDET_CSV_PATH = DATA_DIR / "merge_holdet_vm_unmatched_holdet.csv"


TEAM_ALIASES = {
    # ISO / short ids
    "alg": "algeria",
    "arg": "argentina",
    "aus": "australia",
    "aut": "austria",
    "bel": "belgium",
    "bih": "bosnia and herzegovina",
    "bra": "brazil",
    "can": "canada",
    "civ": "ivory coast",
    "cmr": "cameroon",
    "cod": "dr congo",
    "col": "colombia",
    "cpv": "cape verde",
    "crc": "costa rica",
    "cro": "croatia",
    "cuw": "curacao",
    "cze": "czechia",
    "den": "denmark",
    "ecu": "ecuador",
    "egy": "egypt",
    "eng": "england",
    "esp": "spain",
    "fra": "france",
    "ger": "germany",
    "gha": "ghana",
    "hai": "haiti",
    "irn": "iran",
    "irq": "iraq",
    "ita": "italy",
    "jor": "jordan",
    "jpn": "japan",
    "kor": "south korea",
    "ksa": "saudi arabia",
    "mar": "morocco",
    "mex": "mexico",
    "ned": "netherlands",
    "nld": "netherlands",
    "nor": "norway",
    "pan": "panama",
    "par": "paraguay",
    "por": "portugal",
    "qat": "qatar",
    "rsa": "south africa",
    "sen": "senegal",
    "sui": "switzerland",
    "swe": "sweden",
    "tun": "tunisia",
    "tur": "turkey",
    "ukr": "ukraine",
    "uru": "uruguay",
    "usa": "usa",
    "uzb": "uzbekistan",
    "wal": "wales",
    "wls": "wales",

    # Danish names from player_pool
    "algeriet": "algeria",
    "argentina": "argentina",
    "australien": "australia",
    "belgien": "belgium",
    "bosnien hercegovina": "bosnia and herzegovina",
    "brasilien": "brazil",
    "canada": "canada",
    "colombia": "colombia",
    "costa rica": "costa rica",
    "curacao": "curacao",
    "curaçao": "curacao",
    "danmark": "denmark",
    "dr congo": "dr congo",
    "ecuador": "ecuador",
    "egypten": "egypt",
    "elfenbenskysten": "ivory coast",
    "england": "england",
    "frankrig": "france",
    "ghana": "ghana",
    "haiti": "haiti",
    "holland": "netherlands",
    "iran": "iran",
    "irak": "iraq",
    "italien": "italy",
    "japan": "japan",
    "jordan": "jordan",
    "kamerun": "cameroon",
    "cameroun": "cameroon",
    "kap verde": "cape verde",
    "kroatien": "croatia",
    "marokko": "morocco",
    "mexico": "mexico",
    "new zealand": "new zealand",
    "norge": "norway",
    "panama": "panama",
    "paraguay": "paraguay",
    "polen": "poland",
    "portugal": "portugal",
    "qatar": "qatar",
    "saudi arabien": "saudi arabia",
    "schweiz": "switzerland",
    "serbien": "serbia",
    "skotland": "scotland",
    "spanien": "spain",
    "sverige": "sweden",
    "sydafrika": "south africa",
    "sydkorea": "south korea",
    "tunesien": "tunisia",
    "tyrkiet": "turkey",
    "tyskland": "germany",
    "ukraine": "ukraine",
    "uruguay": "uruguay",
    "usa": "usa",
    "usbekistan": "uzbekistan",
    "wales": "wales",
    "østrig": "austria",
    "ostrig": "austria",

    # English / API variants
    "algeria": "algeria",
    "austria": "austria",
    "australia": "australia",
    "belgium": "belgium",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "brazil": "brazil",
    "cameroon": "cameroon",
    "cape verde": "cape verde",
    "cote divoire": "ivory coast",
    "côte divoire": "ivory coast",
    "curacao": "curacao",
    "czech republic": "czechia",
    "czechia": "czechia",
    "democratic republic of congo": "dr congo",
    "denmark": "denmark",
    "dr congo": "dr congo",
    "drc": "dr congo",
    "egypt": "egypt",
    "france": "france",
    "germany": "germany",
    "haiti": "haiti",
    "iran": "iran",
    "iraq": "iraq",
    "ivory coast": "ivory coast",
    "korea republic": "south korea",
    "morocco": "morocco",
    "netherlands": "netherlands",
    "norway": "norway",
    "republic of korea": "south korea",
    "saudi arabia": "saudi arabia",
    "south africa": "south africa",
    "south korea": "south korea",
    "spain": "spain",
    "switzerland": "switzerland",
    "turkiye": "turkey",
    "turkey": "turkey",
    "united states": "usa",
    "united states of america": "usa",
    "uzbekistan": "uzbekistan",
}


POSITION_ALIASES = {
    "goalkeeper": "GK",
    "keeper": "GK",
    "gk": "GK",
    "malmand": "GK",
    "målmand": "GK",
    "defender": "DEF",
    "defense": "DEF",
    "defence": "DEF",
    "forsvar": "DEF",
    "def": "DEF",
    "midfielder": "MID",
    "midfield": "MID",
    "midtbanespiller": "MID",
    "midtbane": "MID",
    "mid": "MID",
    "forward": "FWD",
    "striker": "FWD",
    "attacker": "FWD",
    "attack": "FWD",
    "angriber": "FWD",
    "fwd": "FWD",
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
    text = norm_name(value)
    tokens = [t for t in text.split(" ") if t]
    return " ".join(sorted(tokens))


def compact_name(value: Any) -> str:
    return norm_name(value).replace(" ", "")


def norm_position(value: Any) -> str:
    text = norm_text(value).upper()
    if text in {"GK", "DEF", "MID", "FWD"}:
        return text

    lower = norm_text(value)
    return POSITION_ALIASES.get(lower, text)


def norm_team(value: Any) -> str:
    text = norm_text(value)
    return TEAM_ALIASES.get(text, text)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def name_score(pool_name: str, holdet_name: str) -> float:
    direct = similarity(pool_name, holdet_name)
    token = similarity(token_sort_name(pool_name), token_sort_name(holdet_name))
    compact = similarity(compact_name(pool_name), compact_name(holdet_name))
    return max(direct, token, compact)


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def load_player_pool(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("players", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Kan ikke finde spillerliste i {path}")


def save_player_pool(path: Path, players: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(players, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_pool_dataframe(players: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for i, p in enumerate(players):
        player_name = get_first(
            p,
            ["player_name", "name", "full_name", "display_name", "player"],
        )

        team_id = get_first(
            p,
            ["team_id", "team", "country_code", "country_id", "national_team_id"],
        )

        team_name = get_first(
            p,
            ["team_name", "country", "country_name", "national_team"],
        )

        position = get_first(
            p,
            ["position", "pos", "player_position"],
        )

        rows.append(
            {
                "pool_index": i,
                "pool_player_id": p.get("player_id"),
                "pool_player_name": player_name,
                "pool_team_id": team_id,
                "pool_team_name": team_name,
                "pool_position": position,
                "pool_price_before": get_first(
                    p,
                    ["price", "price_estimate", "price_value", "value"],
                ),
                "pool_name_norm": norm_name(player_name),
                "pool_name_token_sort": token_sort_name(player_name),
                "pool_name_compact": compact_name(player_name),
                "pool_team_norm": norm_team(team_name if team_name else team_id),
                "pool_position_norm": norm_position(position),
            }
        )

    return pd.DataFrame(rows)


def build_holdet_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["holdet_name_norm"] = df["player_name"].map(norm_name)
    df["holdet_name_token_sort"] = df["player_name"].map(token_sort_name)
    df["holdet_name_compact"] = df["player_name"].map(compact_name)
    df["holdet_team_norm"] = df["team_name"].map(norm_team)
    df["holdet_position_norm"] = df["position"].map(norm_position)

    df["holdet_key_name_team_pos"] = (
        df["holdet_name_norm"]
        + "|"
        + df["holdet_team_norm"]
        + "|"
        + df["holdet_position_norm"]
    )

    df["holdet_key_name_team"] = (
        df["holdet_name_norm"]
        + "|"
        + df["holdet_team_norm"]
    )

    df["holdet_key_tokens_team"] = (
        df["holdet_name_token_sort"]
        + "|"
        + df["holdet_team_norm"]
    )

    return df


def apply_row_match(
    merged: pd.DataFrame,
    pool_index: int,
    holdet_row: pd.Series,
    match_method: str,
    score: float | None = None,
) -> None:
    target_mask = merged["pool_index"] == pool_index

    update_cols = [
        "holdet_player_id",
        "holdet_person_id",
        "player_name",
        "holdet_team_id",
        "team_name",
        "holdet_position_id",
        "position_raw",
        "position",
        "start_price",
        "price",
        "points",
        "popularity",
        "is_out",
        "holdet_name_norm",
        "holdet_name_token_sort",
        "holdet_name_compact",
        "holdet_team_norm",
        "holdet_position_norm",
        "holdet_key_name_team_pos",
        "holdet_key_name_team",
        "holdet_key_tokens_team",
    ]

    for col in update_cols:
        if col in holdet_row.index and col in merged.columns:
            merged.loc[target_mask, col] = holdet_row[col]

    merged.loc[target_mask, "match_method"] = match_method
    merged.loc[target_mask, "match_score"] = score if score is not None else 1.0


def merge_data(pool_df: pd.DataFrame, holdet_df: pd.DataFrame) -> pd.DataFrame:
    pool_df = pool_df.copy()

    pool_df["pool_key_name_team_pos"] = (
        pool_df["pool_name_norm"]
        + "|"
        + pool_df["pool_team_norm"]
        + "|"
        + pool_df["pool_position_norm"]
    )

    pool_df["pool_key_name_team"] = (
        pool_df["pool_name_norm"]
        + "|"
        + pool_df["pool_team_norm"]
    )

    pool_df["pool_key_tokens_team"] = (
        pool_df["pool_name_token_sort"]
        + "|"
        + pool_df["pool_team_norm"]
    )

    holdet_exact = holdet_df.drop_duplicates("holdet_key_name_team_pos", keep="first")
    holdet_name_team = holdet_df.drop_duplicates("holdet_key_name_team", keep="first")
    holdet_tokens_team = holdet_df.drop_duplicates("holdet_key_tokens_team", keep="first")

    merged = pool_df.merge(
        holdet_exact,
        left_on="pool_key_name_team_pos",
        right_on="holdet_key_name_team_pos",
        how="left",
        suffixes=("", "_holdet"),
    )

    merged["match_method"] = merged["holdet_player_id"].apply(
        lambda x: "name_team_position" if pd.notna(x) else ""
    )
    merged["match_score"] = merged["holdet_player_id"].apply(
        lambda x: 1.0 if pd.notna(x) else None
    )

    # Fallback 1: exact name + team, ignoring position.
    missing = merged["holdet_player_id"].isna()
    fallback = pool_df.loc[missing].merge(
        holdet_name_team,
        left_on="pool_key_name_team",
        right_on="holdet_key_name_team",
        how="left",
        suffixes=("", "_holdet"),
    )

    for _, row in fallback.iterrows():
        if pd.notna(row.get("holdet_player_id")):
            apply_row_match(merged, int(row["pool_index"]), row, "name_team", 1.0)

    # Fallback 2: token-sorted name + team, good for names with different order.
    missing = merged["holdet_player_id"].isna()
    fallback_tokens = pool_df.loc[missing].merge(
        holdet_tokens_team,
        left_on="pool_key_tokens_team",
        right_on="holdet_key_tokens_team",
        how="left",
        suffixes=("", "_holdet"),
    )

    for _, row in fallback_tokens.iterrows():
        if pd.notna(row.get("holdet_player_id")):
            apply_row_match(merged, int(row["pool_index"]), row, "token_name_team", 1.0)

    # Fallback 3: fuzzy within same team. Prefer same position.
    used_holdet_ids = set(
        merged.loc[merged["holdet_player_id"].notna(), "holdet_player_id"]
        .astype(int)
        .tolist()
    )

    for _, pool_row in pool_df.loc[merged["holdet_player_id"].isna()].iterrows():
        pool_index = int(pool_row["pool_index"])
        pool_team = pool_row["pool_team_norm"]
        pool_pos = pool_row["pool_position_norm"]
        pool_name = pool_row["pool_name_norm"]

        candidates = holdet_df.loc[
            (holdet_df["holdet_team_norm"] == pool_team)
            & (~holdet_df["holdet_player_id"].astype(int).isin(used_holdet_ids))
        ].copy()

        if candidates.empty:
            continue

        candidates["candidate_score"] = candidates["holdet_name_norm"].map(
            lambda candidate_name: name_score(pool_name, candidate_name)
        )

        same_pos = candidates.loc[candidates["holdet_position_norm"] == pool_pos].copy()

        best_row = None
        best_score = 0.0
        method = ""

        if not same_pos.empty:
            same_pos = same_pos.sort_values("candidate_score", ascending=False)
            best_same_pos = same_pos.iloc[0]
            if float(best_same_pos["candidate_score"]) >= 0.88:
                best_row = best_same_pos
                best_score = float(best_same_pos["candidate_score"])
                method = "fuzzy_name_team_position"

        if best_row is None:
            candidates = candidates.sort_values("candidate_score", ascending=False)
            best_any_pos = candidates.iloc[0]
            if float(best_any_pos["candidate_score"]) >= 0.94:
                best_row = best_any_pos
                best_score = float(best_any_pos["candidate_score"])
                method = "fuzzy_name_team"

        if best_row is not None:
            apply_row_match(merged, pool_index, best_row, method, best_score)
            used_holdet_ids.add(int(best_row["holdet_player_id"]))

    merged["matched"] = merged["holdet_player_id"].notna()

    return merged


def update_players(players: list[dict[str, Any]], merged: pd.DataFrame) -> list[dict[str, Any]]:
    updated = [dict(p) for p in players]

    for _, row in merged.iterrows():
        if not bool(row["matched"]):
            continue

        idx = int(row["pool_index"])
        p = updated[idx]

        holdet_price = int(row["price"]) if pd.notna(row.get("price")) else None
        holdet_start_price = int(row["start_price"]) if pd.notna(row.get("start_price")) else None
        holdet_position = str(row["position"]) if pd.notna(row.get("position")) else None

        p["holdet_game_id"] = 616
        p["holdet_player_id"] = int(row["holdet_player_id"])
        p["holdet_person_id"] = int(row["holdet_person_id"])
        p["holdet_team_id"] = int(row["holdet_team_id"])
        p["holdet_team_name"] = row["team_name"]
        p["holdet_position_id"] = int(row["holdet_position_id"])
        p["holdet_position"] = holdet_position
        p["holdet_start_price"] = holdet_start_price
        p["holdet_price"] = holdet_price
        p["holdet_is_out"] = bool(row["is_out"])
        p["holdet_match_method"] = row["match_method"]
        p["holdet_match_score"] = float(row["match_score"]) if pd.notna(row.get("match_score")) else None
        p["has_holdet_vm_match"] = True

        if holdet_price is not None:
            p["price"] = holdet_price
            p["price_estimate"] = holdet_price
            p["price_source"] = "holdet_vm_2026_official"

        if holdet_position in {"GK", "DEF", "MID", "FWD"}:
            p["position"] = holdet_position
            p["position_source"] = "holdet_vm_2026_official"

    return updated


def print_summary(merged: pd.DataFrame, holdet_df: pd.DataFrame) -> None:
    total = len(merged)
    matched = int(merged["matched"].sum())
    unmatched = total - matched

    print("\nMERGE-SUMMARY")
    print(f"Player pool spillere: {total}")
    print(f"Holdet spillere: {len(holdet_df)}")
    print(f"Matchede player_pool-spillere: {matched} / {total}")
    print(f"Umatchede player_pool-spillere: {unmatched} / {total}")

    print("\nMatchmetode:")
    print(merged["match_method"].replace("", "unmatched").value_counts(dropna=False).to_string())

    print("\nLaveste fuzzy-matchscore:")
    fuzzy = merged.loc[merged["match_method"].astype(str).str.startswith("fuzzy", na=False)].copy()
    if fuzzy.empty:
        print("(ingen fuzzy matches)")
    else:
        cols = [
            "pool_player_name",
            "pool_team_name",
            "pool_position",
            "player_name",
            "team_name",
            "position",
            "match_method",
            "match_score",
        ]
        print(fuzzy.sort_values("match_score", ascending=True)[cols].head(30).to_string(index=False))

    print("\nPositioner efter Holdet-match:")
    if "position" in merged.columns:
        print(merged.loc[merged["matched"], "position"].value_counts(dropna=False).to_string())

    print("\nPris efter Holdet-match:")
    prices = pd.to_numeric(merged.loc[merged["matched"], "price"], errors="coerce")
    if len(prices.dropna()) > 0:
        print(f"Min:    {int(prices.min()):,}".replace(",", "."))
        print(f"Median: {int(prices.median()):,}".replace(",", "."))
        print(f"Max:    {int(prices.max()):,}".replace(",", "."))

    print("\nFørste 30 umatchet fra player_pool:")
    cols = [
        "pool_player_id",
        "pool_player_name",
        "pool_team_id",
        "pool_team_name",
        "pool_position",
        "pool_name_norm",
        "pool_team_norm",
        "pool_position_norm",
    ]
    unmatched_pool = merged.loc[~merged["matched"], cols].head(30)

    if unmatched_pool.empty:
        print("(ingen)")
    else:
        print(unmatched_pool.to_string(index=False))


def write_diagnostics(merged: pd.DataFrame, holdet_df: pd.DataFrame) -> None:
    merged.to_csv(DIAG_CSV_PATH, index=False, encoding="utf-8-sig")

    unmatched_pool = merged.loc[~merged["matched"]].copy()
    unmatched_pool.to_csv(UNMATCHED_POOL_CSV_PATH, index=False, encoding="utf-8-sig")

    matched_holdet_ids = set(
        merged.loc[merged["matched"], "holdet_player_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    unmatched_holdet = holdet_df.loc[
        ~holdet_df["holdet_player_id"].astype(int).isin(matched_holdet_ids)
    ].copy()

    unmatched_holdet.to_csv(UNMATCHED_HOLDET_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\nDiagnosefiler skrevet:")
    print(f"- {DIAG_CSV_PATH}")
    print(f"- {UNMATCHED_POOL_CSV_PATH}")
    print(f"- {UNMATCHED_HOLDET_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Skriv ændringer til player_pool_v1.json")
    args = parser.parse_args()

    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(f"Mangler {PLAYER_POOL_PATH}")

    if not HOLDET_FLAT_PATH.exists():
        raise FileNotFoundError(f"Mangler {HOLDET_FLAT_PATH}")

    players = load_player_pool(PLAYER_POOL_PATH)
    pool_df = build_pool_dataframe(players)
    holdet_df = build_holdet_dataframe(HOLDET_FLAT_PATH)
    merged = merge_data(pool_df, holdet_df)

    print_summary(merged, holdet_df)
    write_diagnostics(merged, holdet_df)

    if not args.write:
        print("\nDRY RUN: player_pool_v1.json er IKKE ændret.")
        print("Hvis matchraten og fuzzy-matchene ser gode ud, kør igen med --write.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = PLAYER_POOL_PATH.with_name(f"player_pool_v1.backup_before_holdet_616_{timestamp}.json")
    shutil.copy2(PLAYER_POOL_PATH, backup_path)

    updated_players = update_players(players, merged)
    save_player_pool(PLAYER_POOL_PATH, updated_players)

    print("\nSKREV ÆNDRINGER")
    print(f"Backup: {backup_path}")
    print(f"Opdateret: {PLAYER_POOL_PATH}")


if __name__ == "__main__":
    main()