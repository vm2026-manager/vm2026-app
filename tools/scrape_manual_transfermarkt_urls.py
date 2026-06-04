from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = PROJECT_ROOT / "tools" / "batch_transfermarkt_national_usage.py"
PLAYER_POOL_PATH = PROJECT_ROOT / "data" / "player_pool_v1.json"

OUT_DIR = PROJECT_ROOT / "data" / "transfermarkt_national_team"
SUMMARY_PATH = OUT_DIR / "player_national_team_usage_transfermarkt.csv"
CACHE_PATH = OUT_DIR / "transfermarkt_player_url_cache.csv"


MANUAL_PLAYERS = [
    {
        "player_name": "Bento Krepski",
        "team_id": "BRA",
        "position": "GK",
        "price_m": 5.0,
        "tm_player_id": "691906",
        "profile_url": "https://www.transfermarkt.com/bento/profil/spieler/691906",
        "national_url": "https://www.transfermarkt.com/bento/nationalmannschaft/spieler/691906",
    },
    {
        "player_name": "Ederson Moraes",
        "team_id": "BRA",
        "position": "GK",
        "price_m": 5.0,
        "tm_player_id": "238223",
        "profile_url": "https://www.transfermarkt.com/ederson/profil/spieler/238223",
        "national_url": "https://www.transfermarkt.com/ederson/nationalmannschaft/spieler/238223",
    },
    {
        "player_name": "Luis Suarez Charris",
        "team_id": "COL",
        "position": "FWD",
        "price_m": 4.5,
        "tm_player_id": "424784",
        "profile_url": "https://www.transfermarkt.com/luis-suarez/profil/spieler/424784",
        "national_url": "https://www.transfermarkt.com/luis-suarez/nationalmannschaft/spieler/424784",
    },
    {
        "player_name": "Neymar Jr.",
        "team_id": "BRA",
        "position": "FWD",
        "price_m": 5.5,
        "tm_player_id": "68290",
        "profile_url": "https://www.transfermarkt.com/neymar/profil/spieler/68290",
        "national_url": "https://www.transfermarkt.com/neymar/nationalmannschaft/spieler/68290",
    },
    {
        "player_name": "Kenan Yildiz",
        "team_id": "TUR",
        "position": "MID",
        "price_m": 4.0,
        "tm_player_id": "845654",
        "profile_url": "https://www.transfermarkt.com/kenan-y-ld-z/profil/spieler/845654",
        "national_url": "https://www.transfermarkt.com/kenan-y-ld-z/nationalmannschaft/spieler/845654",
    },
    {
        "player_name": "Alejandro 'Alex' Remiro",
        "team_id": "ESP",
        "position": "GK",
        "price_m": 4.0,
        "tm_player_id": "212862",
        "profile_url": "https://www.transfermarkt.com/alex-remiro/profil/spieler/212862",
        "national_url": "https://www.transfermarkt.com/alex-remiro/nationalmannschaft/spieler/212862",
    },
    {
        "player_name": "Alejandro 'Alex' Grimaldo",
        "team_id": "ESP",
        "position": "DEF",
        "price_m": 4.0,
        "tm_player_id": "193082",
        "profile_url": "https://www.transfermarkt.com/alejandro-grimaldo/profil/spieler/193082",
        "national_url": "https://www.transfermarkt.com/alejandro-grimaldo/nationalmannschaft/spieler/193082",
    },
    {
        "player_name": "Yassine 'Bono' Bounou",
        "team_id": "MAR",
        "position": "GK",
        "price_m": 3.5,
        "tm_player_id": "207834",
        "profile_url": "https://www.transfermarkt.com/yassine-bounou/profil/spieler/207834",
        "national_url": "https://www.transfermarkt.com/yassine-bounou/nationalmannschaft/spieler/207834",
    },
    {
        "player_name": "Wesley Franca",
        "team_id": "BRA",
        "position": "DEF",
        "price_m": 3.5,
        "tm_player_id": "964580",
        "profile_url": "https://www.transfermarkt.com/wesley/profil/spieler/964580",
        "national_url": "https://www.transfermarkt.com/wesley/nationalmannschaft/spieler/964580",
    },
    {
        "player_name": "Robin Risser",
        "team_id": "FRA",
        "position": "GK",
        "price_m": 4.0,
        "tm_player_id": "743515",
        "profile_url": "https://www.transfermarkt.com/robin-risser/profil/spieler/743515",
        "national_url": "https://www.transfermarkt.com/robin-risser/nationalmannschaft/spieler/743515",
    },
    {
        "player_name": "Sphephelo Yaya Sithole",
        "team_id": "RSA",
        "position": "MID",
        "price_m": 3.0,
        "tm_player_id": "401736",
        "profile_url": "https://www.transfermarkt.com/yaya-sithole/profil/spieler/401736",
        "national_url": "https://www.transfermarkt.com/yaya-sithole/nationalmannschaft/spieler/401736",
    },
    {
        "player_name": "Ryan Mendes Da Graça",
        "team_id": "CPV",
        "position": "FWD",
        "price_m": 3.0,
        "tm_player_id": "111627",
        "profile_url": "https://www.transfermarkt.com/ryan-mendes/profil/spieler/111627",
        "national_url": "https://www.transfermarkt.com/ryan-mendes/nationalmannschaft/spieler/111627",
    },
    {
        "player_name": "Roberto 'Gatito' Fernandez",
        "team_id": "PAR",
        "position": "GK",
        "price_m": 3.0,
        "tm_player_id": "107318",
        "profile_url": "https://www.transfermarkt.com/roberto-fernandez/profil/spieler/107318",
        "national_url": "https://www.transfermarkt.com/roberto-fernandez/nationalmannschaft/spieler/107318/verein_id/3581",
    },
    {
        "player_name": "Mohamed El-Shennawy",
        "team_id": "EGY",
        "position": "GK",
        "price_m": 3.0,
        "tm_player_id": "134573",
        "profile_url": "https://www.transfermarkt.com/mohamed-el-shenawy/profil/spieler/134573",
        "national_url": "https://www.transfermarkt.com/mohamed-el-shenawy/nationalmannschaft/spieler/134573",
    },
]


def slug(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_batch_module():
    spec = importlib.util.spec_from_file_location("batch_tm", BATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kan ikke importere {BATCH_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_player_pool() -> list[dict[str, Any]]:
    import json

    raw = json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["players", "items", "data"]:
            if isinstance(raw.get(key), list):
                return raw[key]

    return []


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def find_pool_player(name: str, team_id: str) -> dict[str, Any] | None:
    players = load_player_pool()
    wanted_name = norm(name)
    wanted_team = str(team_id or "").strip().upper()

    for player in players:
        p_name = norm(player.get("player_name") or player.get("name"))
        p_team = str(player.get("team_id") or "").strip().upper()

        if p_name == wanted_name and p_team == wanted_team:
            return player

    # Lidt blødere fallback
    for player in players:
        p_name = norm(player.get("player_name") or player.get("name"))
        p_team = str(player.get("team_id") or "").strip().upper()

        if wanted_name in p_name and p_team == wanted_team:
            return player

    return None


def get_player_id(player: dict[str, Any] | None, fallback_name: str, fallback_team: str) -> str:
    if player:
        for key in ["player_id", "id", "holdet_player_id"]:
            if player.get(key) is not None:
                return str(player.get(key))

    return f"manual_{slug(fallback_name)}_{fallback_team.lower()}"


def upsert_by_player_id(df: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    player_id = str(row.get("player_id", ""))

    if df.empty:
        return pd.DataFrame([row])

    if "player_id" in df.columns:
        df = df.loc[df["player_id"].astype(str) != player_id].copy()

    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tm = load_batch_module()

    summary_df = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()
    cache_df = pd.read_csv(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()

    print("MANUAL TRANSFERMARKT URL SCRAPER")
    print("=" * 80)

    for item in MANUAL_PLAYERS:
        pool_player = find_pool_player(item["player_name"], item["team_id"])
        player_id = get_player_id(pool_player, item["player_name"], item["team_id"])

        player = {
            "player_id": player_id,
            "player_name": item["player_name"],
            "team_id": item["team_id"],
            "position": item["position"],
            "price_m": item["price_m"],
        }

        print(f"{item['player_name']} | {item['team_id']} | player_id={player_id}")
        print(f"  {item['national_url']}")

        html = tm.fetch_html(item["national_url"])
        tables = tm.read_tables(html)
        match_table = tm.select_best_match_table(tables)

        if match_table is None:
            print("  FEJL: Ingen kampoversigt fundet")
            continue

        matches = tm.normalize_match_table(
            match_table,
            player,
            item["tm_player_id"],
            item["national_url"],
        )

        summary = tm.build_summary(
            player,
            item["tm_player_id"],
            item["national_url"],
            html,
            matches,
        )

        matches_path = OUT_DIR / (
            f"{slug(item['player_name'])}__{item['team_id']}_"
            f"{item['tm_player_id']}_national_matches.csv"
        )
        matches.to_csv(matches_path, index=False, encoding="utf-8-sig")

        summary_df = upsert_by_player_id(summary_df, summary)

        cache_row = {
            "player_id": player_id,
            "player_name": item["player_name"],
            "team_id": item["team_id"],
            "position": item["position"],
            "price_m": item["price_m"],
            "tm_player_id": item["tm_player_id"],
            "tm_profile_url": item["profile_url"],
            "tm_national_url": item["national_url"],
            "status": "ok_manual_url",
            "error": "",
            "last_attempt": "",
        }
        cache_df = upsert_by_player_id(cache_df, cache_row)

        print(
            f"  OK: caps={summary.get('tm_caps')} "
            f"recent20_start={summary.get('recent_20_start_share')} "
            f"usage={summary.get('national_team_usage_score')}"
        )
        print(f"  Skrev: {matches_path.name}")

    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    cache_df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")

    print("")
    print(f"Skrev summary: {SUMMARY_PATH}")
    print(f"Skrev cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
