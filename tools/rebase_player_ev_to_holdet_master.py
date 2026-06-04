from __future__ import annotations

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
EV_PATH = DATA_DIR / "player_ev_group_stage_v1.csv"

DIAG_PATH = DATA_DIR / "rebase_player_ev_to_holdet_master_diagnostics.csv"
UNMATCHED_PATH = DATA_DIR / "rebase_player_ev_to_holdet_master_unmatched.csv"


TEAM_ALIASES = {
    "cote divoire": "cote divoire",
    "côte divoire": "cote divoire",
    "ivory coast": "cote divoire",
    "congo dr": "congo dr",
    "dr congo": "congo dr",
    "democratic republic of congo": "congo dr",
    "turkiye": "turkiye",
    "türkiye": "turkiye",
    "turkey": "turkiye",
    "netherlands": "netherlands",
    "holland": "netherlands",
    "south korea": "south korea",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "croatia": "croatia",
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
    "HOLDET_584": "czechia",
    "HOLDET_767": "cote divoire",
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
    "NZL": "new zealand",
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
    return " ".join(sorted([t for t in norm_name(value).split(" ") if t]))


def compact_name(value: Any) -> str:
    return norm_name(value).replace(" ", "")


def canonical_team(value: Any) -> str:
    raw = str(value).strip()
    upper = raw.upper()
    if upper in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[upper]

    text = norm_text(raw)
    return TEAM_ALIASES.get(text, text)


def name_score(a: Any, b: Any) -> float:
    a1 = norm_name(a)
    b1 = norm_name(b)
    if not a1 or not b1:
        return 0.0

    return max(
        SequenceMatcher(None, a1, b1).ratio(),
        SequenceMatcher(None, token_sort_name(a1), token_sort_name(b1)).ratio(),
        SequenceMatcher(None, compact_name(a1), compact_name(b1)).ratio(),
    )


def load_player_pool() -> list[dict[str, Any]]:
    data = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("players", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Kan ikke finde spillerliste i {PLAYER_POOL_PATH}")


def get_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def pool_to_df(players: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for p in players:
        player_name = get_first(p, ["player_name", "name", "full_name", "display_name"], "")
        team_id = get_first(p, ["team_id", "team", "country_code"], "")
        team_name = get_first(p, ["team_name", "country", "country_name"], "")
        position = get_first(p, ["position", "pos"], "")

        rows.append(
            {
                "player_id": p.get("player_id"),
                "player_name": player_name,
                "team_id": team_id,
                "team_name": team_name,
                "position": position,
                "price": get_first(p, ["price", "price_estimate"], 0),
                "price_estimate": get_first(p, ["price_estimate", "price"], 0),
                "holdet_player_id": p.get("holdet_player_id"),
                "holdet_person_id": p.get("holdet_person_id"),
                "holdet_team_id": p.get("holdet_team_id"),
                "holdet_team_name": p.get("holdet_team_name"),
                "holdet_price": p.get("holdet_price"),
                "holdet_start_price": p.get("holdet_start_price"),
                "holdet_position": p.get("holdet_position"),
                "holdet_is_out": p.get("holdet_is_out"),
                "official_holdet_master": p.get("official_holdet_master", True),
                "copied_from_old_player_pool": p.get("copied_from_old_player_pool", False),
                "old_player_id": p.get("old_player_id"),
                "pool_weighted_group_stage_ev": p.get("weighted_group_stage_ev", 0.0),
                "pool_optimizer_ev": p.get("optimizer_ev", p.get("weighted_group_stage_ev", 0.0)),
                "pool_start_prob": p.get("start_prob", 0.25),
                "pool_start_prob_source": p.get("start_prob_source", "holdet_official_unmatched_default"),
            }
        )

    df = pd.DataFrame(rows)
    df["name_norm"] = df["player_name"].map(norm_name)
    df["team_norm"] = df["team_id"].map(canonical_team)
    df["team_name_norm"] = df["team_name"].map(canonical_team)
    df["position_norm"] = df["position"].astype(str)

    df["key_name_team_pos"] = df["name_norm"] + "|" + df["team_norm"] + "|" + df["position_norm"]
    df["key_name_team"] = df["name_norm"] + "|" + df["team_norm"]
    df["key_token_team"] = df["player_name"].map(token_sort_name) + "|" + df["team_norm"]

    return df


def prepare_ev_df(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()

    if "team_id" not in ev.columns and "team" in ev.columns:
        ev["team_id"] = ev["team"]

    if "player_name" not in ev.columns and "name" in ev.columns:
        ev["player_name"] = ev["name"]

    ev["name_norm"] = ev["player_name"].map(norm_name)
    ev["team_norm"] = ev["team_id"].map(canonical_team)
    ev["position_norm"] = ev["position"].astype(str)

    ev["key_name_team_pos"] = ev["name_norm"] + "|" + ev["team_norm"] + "|" + ev["position_norm"]
    ev["key_name_team"] = ev["name_norm"] + "|" + ev["team_norm"]
    ev["key_token_team"] = ev["player_name"].map(token_sort_name) + "|" + ev["team_norm"]

    return ev


def safe_numeric(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


TOP_PRIORITY_START_SOURCE_MARKERS = [
    "confirmed_lineup",
    "expected_lineup",
    "manual",
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


def start_source_priority(source: Any) -> int:
    text = str(source or "").strip().lower()
    if any(marker in text for marker in TOP_PRIORITY_START_SOURCE_MARKERS):
        return 100
    if any(marker in text for marker in HIGH_PRIORITY_START_SOURCE_MARKERS):
        return 90
    if any(marker in text for marker in LOW_PRIORITY_START_SOURCE_MARKERS):
        return 10
    if text:
        return 50
    return 0


def choose_start_signal(ev_row: dict[str, Any], pool_row: pd.Series) -> tuple[float, str]:
    ev_start = safe_numeric(ev_row.get("start_prob"), -1.0)
    ev_source = ev_row.get("start_prob_source", "")
    pool_start = safe_numeric(pool_row.get("pool_start_prob"), -1.0)
    pool_source = pool_row.get("pool_start_prob_source", "")

    if pool_start >= 0 and start_source_priority(pool_source) > start_source_priority(ev_source):
        return pool_start, pool_source
    if ev_start >= 0:
        return ev_start, ev_source or pool_source or "unknown"
    if pool_start >= 0:
        return pool_start, pool_source or "player_pool_fallback"
    return 0.25, "holdet_official_unmatched_default"


def match_ev_row(pool_row: pd.Series, ev_df: pd.DataFrame, used_ev_indices: set[int]) -> tuple[pd.Series | None, str, float]:
    available = ev_df.loc[~ev_df.index.isin(used_ev_indices)]

    exact = available.loc[available["key_name_team_pos"] == pool_row["key_name_team_pos"]]
    if not exact.empty:
        return exact.iloc[0], "name_team_position", 1.0

    name_team = available.loc[available["key_name_team"] == pool_row["key_name_team"]]
    if not name_team.empty:
        return name_team.iloc[0], "name_team", 1.0

    token = available.loc[available["key_token_team"] == pool_row["key_token_team"]]
    if not token.empty:
        return token.iloc[0], "token_name_team", 1.0

    candidates = available.loc[available["team_norm"] == pool_row["team_norm"]].copy()
    if candidates.empty:
        return None, "unmatched", 0.0

    candidates["score"] = candidates["player_name"].map(lambda x: name_score(pool_row["player_name"], x))
    candidates = candidates.sort_values("score", ascending=False)

    best = candidates.iloc[0]
    score = float(best["score"])

    if score >= 0.94:
        return best, "fuzzy_name_team", score

    return None, "unmatched", score


def main() -> None:
    if not PLAYER_POOL_PATH.exists():
        raise FileNotFoundError(f"Mangler {PLAYER_POOL_PATH}")

    if not EV_PATH.exists():
        raise FileNotFoundError(f"Mangler {EV_PATH}")

    players = load_player_pool()
    pool_df = pool_to_df(players)

    ev_df = pd.read_csv(EV_PATH)
    ev_df = prepare_ev_df(ev_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = EV_PATH.with_name(f"player_ev_group_stage_v1.backup_before_holdet_rebase_{timestamp}.csv")
    shutil.copy2(EV_PATH, backup_path)

    all_columns = list(ev_df.columns)
    helper_cols = {
        "name_norm",
        "team_norm",
        "position_norm",
        "key_name_team_pos",
        "key_name_team",
        "key_token_team",
    }
    output_cols = [c for c in all_columns if c not in helper_cols]

    required_front = [
        "player_id",
        "player_name",
        "team_id",
        "position",
        "start_prob",
        "start_prob_source",
        "minute_share",
        "weighted_group_stage_ev",
    ]

    for col in required_front:
        if col not in output_cols:
            output_cols.insert(0, col)

    output_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    used_ev_indices: set[int] = set()

    for _, pool_row in pool_df.iterrows():
        ev_match, match_method, match_score = match_ev_row(pool_row, ev_df, used_ev_indices)

        if ev_match is not None:
            used_ev_indices.add(int(ev_match.name))
            row = ev_match.to_dict()
        else:
            row = {}

        start_prob, start_prob_source = choose_start_signal(row, pool_row)

        weighted_ev = safe_numeric(
            row.get("weighted_group_stage_ev"),
            safe_numeric(pool_row.get("pool_weighted_group_stage_ev"), 0.0),
        )

        minute_share = safe_numeric(row.get("minute_share"), start_prob / 11.0 if start_prob > 0 else 0.0)

        row.update(
            {
                "player_id": pool_row["player_id"],
                "player_name": pool_row["player_name"],
                "team_id": pool_row["team_id"],
                "team_name": pool_row["team_name"],
                "position": pool_row["position"],
                "price": pool_row["price"],
                "price_estimate": pool_row["price_estimate"],
                "holdet_player_id": pool_row["holdet_player_id"],
                "holdet_person_id": pool_row["holdet_person_id"],
                "holdet_team_id": pool_row["holdet_team_id"],
                "holdet_team_name": pool_row["holdet_team_name"],
                "holdet_price": pool_row["holdet_price"],
                "holdet_start_price": pool_row["holdet_start_price"],
                "holdet_position": pool_row["holdet_position"],
                "holdet_is_out": pool_row["holdet_is_out"],
                "official_holdet_master": True,
                "ev_rebased_to_holdet_master": True,
                "ev_match_method": match_method,
                "ev_match_score": match_score,
                "start_prob": start_prob,
                "start_prob_source": start_prob_source,
                "minute_share": minute_share,
                "weighted_group_stage_ev": weighted_ev,
                "optimizer_ev": safe_numeric(row.get("optimizer_ev"), weighted_ev),
            }
        )

        for col in output_cols:
            if col not in row:
                row[col] = None

        output_rows.append(row)

        diag_rows.append(
            {
                "player_id": pool_row["player_id"],
                "player_name": pool_row["player_name"],
                "team_id": pool_row["team_id"],
                "position": pool_row["position"],
                "price": pool_row["price"],
                "ev_match_method": match_method,
                "ev_match_score": match_score,
                "weighted_group_stage_ev": weighted_ev,
                "start_prob": start_prob,
            }
        )

    out_df = pd.DataFrame(output_rows)

    front_cols = [
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "price",
        "price_estimate",
        "start_prob",
        "start_prob_source",
        "minute_share",
        "weighted_group_stage_ev",
        "optimizer_ev",
        "holdet_player_id",
        "holdet_team_name",
        "holdet_price",
        "official_holdet_master",
        "ev_rebased_to_holdet_master",
        "ev_match_method",
        "ev_match_score",
    ]

    final_cols = []
    for col in front_cols + output_cols + list(out_df.columns):
        if col in out_df.columns and col not in final_cols:
            final_cols.append(col)

    out_df = out_df[final_cols]
    out_df.to_csv(EV_PATH, index=False, encoding="utf-8-sig")

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(DIAG_PATH, index=False, encoding="utf-8-sig")

    unmatched = diag.loc[diag["ev_match_method"] == "unmatched"].copy()
    unmatched.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    print("\nREBASERET PLAYER EV TIL HOLDET-MASTER")
    print(f"Officielle spillere i player_pool: {len(pool_df)}")
    print(f"Rækker skrevet til EV-fil: {len(out_df)}")
    print("\nMatchmetode:")
    print(diag["ev_match_method"].value_counts(dropna=False).to_string())
    print("\nPositioner:")
    print(out_df["position"].value_counts(dropna=False).to_string())
    print("\nEV:")
    print(f"Sum:    {out_df['weighted_group_stage_ev'].sum():.3f}")
    print(f"Mean:   {out_df['weighted_group_stage_ev'].mean():.4f}")
    print(f"Median: {out_df['weighted_group_stage_ev'].median():.4f}")

    print("\nBackup:")
    print(backup_path)
    print("\nSkrev:")
    print(EV_PATH)
    print(DIAG_PATH)
    print(UNMATCHED_PATH)

    print("\nFørste 25 uden EV-match:")
    if unmatched.empty:
        print("(ingen)")
    else:
        print(unmatched.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
