from __future__ import annotations

import json
import re
import time
import unicodedata
import argparse
from pandas.errors import EmptyDataError
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "transfermarkt_national_team"

PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
OPTIMAL_PATH = DATA_DIR / "optimal_squads_by_formation.json"

CACHE_PATH = OUT_DIR / "transfermarkt_player_url_cache.csv"
SUMMARY_PATH = OUT_DIR / "player_national_team_usage_transfermarkt.csv"
MANUAL_URLS_PATH = PROJECT_ROOT / "tools" / "transfermarkt_manual_urls.csv"
MANUAL_ALIASES_PATH = PROJECT_ROOT / "tools" / "transfermarkt_manual_aliases.csv"
MANUAL_AUDIT_CSV = DATA_DIR / "transfermarkt_manual_url_integration_audit.csv"
MANUAL_AUDIT_MD = DATA_DIR / "transfermarkt_manual_url_integration_audit.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# KÃ¸r smÃ¥ batches. Gentag scriptet flere gange for at fylde mere pÃ¥.
MAX_PLAYERS_PER_RUN = 120
REQUEST_SLEEP_SECONDS = 2.0

BASE_URL = "https://www.transfermarkt.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,da;q=0.8",
}


def norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9Ã¦Ã¸Ã¥Ã¤Ã¶Ã¼ÃŸ\s-]", " ", text)
    return " ".join(text.replace("-", " ").split())


def clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def clean_col(value: Any) -> str:
    return clean_cell(value)


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_player_pool() -> list[dict[str, Any]]:
    raw = load_json(PLAYER_POOL_PATH)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["players", "items", "data"]:
            if isinstance(raw.get(key), list):
                return raw[key]

    raise ValueError(f"Kan ikke finde spillerliste i {PLAYER_POOL_PATH}")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def get_player_id(player: dict[str, Any]) -> str:
    for key in ["player_id", "id", "holdet_player_id"]:
        if player.get(key) is not None:
            return str(player.get(key))
    return ""


def get_price_m(player: dict[str, Any]) -> float:
    for key in ["price_m", "price", "price_estimate"]:
        if key in player and player.get(key) is not None:
            try:
                value = float(player.get(key))
                if value > 100000:
                    return value / 1_000_000
                return value
            except Exception:
                pass
    return 0.0


def get_position(player: dict[str, Any]) -> str:
    return str(player.get("position") or player.get("position_code") or "").upper().strip()


def canonical_pool_key(player: dict[str, Any]) -> tuple[str, str]:
    return (
        norm_text(player.get("player_name") or player.get("name")),
        str(player.get("team_id") or "").strip().upper(),
    )


def load_manual_aliases() -> dict[str, list[tuple[str, str]]]:
    aliases = read_csv_if_exists(MANUAL_ALIASES_PATH)
    out: dict[str, list[tuple[str, str]]] = {}
    if aliases.empty:
        return out
    for _, row in aliases.iterrows():
        canonical_name = str(row.get("player_name") or "").strip()
        team_id = str(row.get("team_id") or "").strip().upper()
        if not canonical_name or not team_id:
            continue
        for value in [canonical_name, row.get("search_alias")]:
            alias = norm_text(value)
            if alias:
                out.setdefault(alias, []).append((canonical_name, team_id))
    return out


def match_manual_row_to_pool(
    manual_name: str,
    players: list[dict[str, Any]],
    alias_lookup: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, Any] | None, str, str]:
    wanted = norm_text(manual_name)
    if not wanted:
        return None, "no_name", ""

    exact = [player for player in players if norm_text(player.get("player_name") or player.get("name")) == wanted]
    if len(exact) == 1:
        return exact[0], "exact_name", ""
    if len(exact) > 1:
        return None, "ambiguous_exact_name", "; ".join(get_player_id(player) for player in exact)

    alias_hits: list[dict[str, Any]] = []
    for canonical_name, team_id in alias_lookup.get(wanted, []):
        canonical = norm_text(canonical_name)
        for player in players:
            p_name, p_team = canonical_pool_key(player)
            if p_name == canonical and p_team == team_id:
                alias_hits.append(player)
    unique_alias_hits = {get_player_id(player): player for player in alias_hits if get_player_id(player)}
    if len(unique_alias_hits) == 1:
        return next(iter(unique_alias_hits.values())), "manual_alias", ""
    if len(unique_alias_hits) > 1:
        return None, "ambiguous_manual_alias", "; ".join(unique_alias_hits)

    contains = [
        player
        for player in players
        if wanted
        and (
            wanted in norm_text(player.get("player_name") or player.get("name"))
            or norm_text(player.get("player_name") or player.get("name")) in wanted
        )
    ]
    if len(contains) == 1:
        return contains[0], "fuzzy_name", ""
    if len(contains) > 1:
        return None, "ambiguous_fuzzy_name", "; ".join(get_player_id(player) for player in contains)

    return None, "no_match", ""


def load_manual_url_map(players: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manual = read_csv_if_exists(MANUAL_URLS_PATH)
    alias_lookup = load_manual_aliases()
    by_player_id: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    seen_tm_ids: dict[str, list[str]] = {}
    seen_player_ids: dict[str, list[str]] = {}

    for _, row in manual.iterrows():
        player_name = str(row.get("player_name") or "").strip()
        tm_player_id = str(row.get("tm_player_id") or "").strip()
        profile_url = str(row.get("tm_profile_url") or "").strip()
        national_url = str(row.get("tm_national_url") or "").strip()
        pool_player, match_status, match_detail = match_manual_row_to_pool(player_name, players, alias_lookup)
        player_id = get_player_id(pool_player) if pool_player else ""
        if tm_player_id:
            seen_tm_ids.setdefault(tm_player_id, []).append(player_name)
        if player_id:
            seen_player_ids.setdefault(player_id, []).append(player_name)

        audit = {
            "manual_player_name": player_name,
            "manual_tm_player_id": tm_player_id,
            "manual_tm_profile_url": profile_url,
            "manual_tm_national_url": national_url,
            "matched_player_id": player_id,
            "matched_player_name": pool_player.get("player_name") if pool_player else "",
            "team_id": pool_player.get("team_id") if pool_player else "",
            "match_status": match_status,
            "match_detail": match_detail,
            "previous_status": "",
            "test_status": "",
            "duplicate_warning": "",
        }
        audit_rows.append(audit)

        if pool_player and player_id and tm_player_id and national_url:
            by_player_id[player_id] = {
                "tm_player_id": tm_player_id,
                "tm_profile_url": profile_url,
                "tm_national_url": national_url,
                "manual_player_name": player_name,
                "match_status": match_status,
            }

    for audit in audit_rows:
        warnings = []
        tm_id = audit["manual_tm_player_id"]
        player_id = audit["matched_player_id"]
        if tm_id and len(seen_tm_ids.get(tm_id, [])) > 1:
            warnings.append(f"duplicate_tm_player_id:{tm_id}")
        if player_id and len(seen_player_ids.get(player_id, [])) > 1:
            warnings.append(f"duplicate_player_id:{player_id}")
        audit["duplicate_warning"] = "; ".join(warnings)

    return by_player_id, audit_rows


def load_existing_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            return pd.read_csv(CACHE_PATH)
        except EmptyDataError:
            pass

    return pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "team_id",
            "position",
            "price_m",
            "tm_player_id",
            "tm_profile_url",
            "tm_national_url",
            "status",
            "error",
            "last_attempt",
        ]
    )


def save_cache(cache: pd.DataFrame) -> None:
    cache.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")


def load_existing_summary() -> pd.DataFrame:
    if SUMMARY_PATH.exists():
        try:
            return pd.read_csv(SUMMARY_PATH)
        except EmptyDataError:
            return pd.DataFrame()

    return pd.DataFrame()


def save_summary(summary: pd.DataFrame) -> None:
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")


def get_selected_player_ids() -> set[str]:
    if not OPTIMAL_PATH.exists():
        return set()

    raw = load_json(OPTIMAL_PATH)
    ids: set[str] = set()

    if not isinstance(raw, dict):
        return ids

    best_score = -10**9
    best_rows: list[dict[str, Any]] = []

    for _, value in raw.items():
        rows = []
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            for key in ["players", "squad", "rows"]:
                if isinstance(value.get(key), list):
                    rows = value[key]
                    break

        if not rows:
            continue

        score = max(float(r.get("squad_total_adj_ev", 0) or 0) for r in rows)
        if score > best_score:
            best_score = score
            best_rows = rows

    for row in best_rows:
        pid = get_player_id(row)
        if pid:
            ids.add(pid)

    return ids


def cache_row_for_player(cache: pd.DataFrame, player_id: str) -> dict[str, Any] | None:
    if cache.empty or "player_id" not in cache.columns:
        return None
    rows = cache.loc[cache["player_id"].astype(str) == str(player_id)]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def has_usable_cache_url(cache_row: dict[str, Any] | None) -> bool:
    if not cache_row:
        return False
    tm_id = str(cache_row.get("tm_player_id") or "").strip()
    national_url = str(cache_row.get("tm_national_url") or "").strip()
    status = str(cache_row.get("status") or "").strip()
    return bool(tm_id and national_url and status in {"ok", "ok_manual_url", "manual_url"})


def choose_players_to_process(
    players: list[dict[str, Any]],
    cache: pd.DataFrame,
    manual_urls: dict[str, dict[str, Any]],
    *,
    manual_only: bool = False,
    refresh: bool = False,
    limit: int = MAX_PLAYERS_PER_RUN,
) -> list[dict[str, Any]]:
    done_statuses = {"ok", "no_match", "ok_manual_url", "manual_url"}

    done_ids = set()
    done_pairs = set()

    if not cache.empty:
        if "player_id" in cache.columns and "status" in cache.columns:
            done_ids = set(
                cache.loc[
                    cache["status"].astype(str).isin(done_statuses),
                    "player_id",
                ]
                .astype(str)
                .tolist()
            )

        if {"player_name", "team_id", "status"}.issubset(set(cache.columns)):
            done_cache = cache.loc[cache["status"].astype(str).isin(done_statuses)].copy()
            for _, row in done_cache.iterrows():
                done_pairs.add(
                    (
                        norm_text(row.get("player_name")),
                        str(row.get("team_id") or "").strip().upper(),
                    )
                )

    # Summary-filen tÃ¦ller ogsÃ¥ som fÃ¦rdig, isÃ¦r for manuelle URL-scrapes,
    # hvor player_id kan vÃ¦re et manuelt fallback-id.
    summary = load_existing_summary()
    if not summary.empty and {"player_name", "team_id"}.issubset(set(summary.columns)):
        for _, row in summary.iterrows():
            done_pairs.add(
                (
                    norm_text(row.get("player_name")),
                    str(row.get("team_id") or "").strip().upper(),
                )
            )

    selected_ids = get_selected_player_ids()

    candidates = []
    for player in players:
        pid = get_player_id(player)
        name = player.get("player_name") or player.get("name")
        team = str(player.get("team_id") or "").strip().upper()
        pair = (norm_text(name), team)

        if not pid:
            continue

        if manual_only and pid not in manual_urls:
            continue

        if not refresh and (pid in done_ids or pair in done_pairs):
            continue

        player["_player_id"] = pid
        player["_price_m"] = get_price_m(player)
        player["_is_selected"] = pid in selected_ids
        player["_has_manual_url"] = pid in manual_urls
        candidates.append(player)

    # FÃ¸rst nuvÃ¦rende optimale hold, derefter dyre/relevante spillere.
    candidates.sort(
        key=lambda p: (
            bool(p.get("_has_manual_url")),
            bool(p.get("_is_selected")),
            float(p.get("_price_m", 0.0)),
            str(p.get("player_name") or ""),
        ),
        reverse=True,
    )

    return candidates[:limit]


def search_transfermarkt_profile(player_name: str) -> tuple[str | None, str | None]:
    search_url = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={quote_plus(player_name)}"
    html = fetch_html(search_url)

    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    seen = set()

    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if "/profil/spieler/" not in href:
            continue

        full_url = urljoin(BASE_URL, href)
        m = re.search(r"/spieler/(\d+)", full_url)
        if not m:
            continue

        tm_id = m.group(1)
        if tm_id in seen:
            continue

        seen.add(tm_id)
        text = clean_cell(a.get_text(" ", strip=True))
        candidates.append((tm_id, full_url, text))

    if not candidates:
        return None, None

    wanted = norm_text(player_name)

    # PrÃ¸v at vÃ¦lge kandidat hvor navnet matcher nogenlunde.
    for tm_id, url, text in candidates:
        candidate_text = norm_text(text)
        if wanted and wanted in candidate_text:
            return tm_id, url

    # Fallback: fÃ¸rste spillerprofil.
    return candidates[0][0], candidates[0][1]


def base_url_for(url: str) -> str:
    if "transfermarkt.com.tr" in str(url):
        return "https://www.transfermarkt.com.tr"
    return BASE_URL


def find_national_url(profile_url: str, tm_player_id: str) -> str:
    html = fetch_html(profile_url)

    # Ofte findes nationalmannschaft-linket direkte i profil-HTML.
    pattern = re.compile(
        rf'href="([^"]*/nationalmannschaft/spieler/{re.escape(tm_player_id)}(?:/verein_id/\d+)?)"',
        re.IGNORECASE,
    )

    match = pattern.search(html)
    if match:
        return urljoin(base_url_for(profile_url), match.group(1).replace("&amp;", "&"))

    # Fallback: konstrueret national-page uden verein_id.
    return profile_url.replace("/profil/spieler/", "/nationalmannschaft/spieler/")


def find_national_url_from_html(html: str, current_url: str, tm_player_id: str) -> str | None:
    pattern = re.compile(
        rf'href="([^"]*/nationalmannschaft/spieler/{re.escape(tm_player_id)}/verein_id/\d+)"',
        re.IGNORECASE,
    )
    match = pattern.search(html)
    if not match:
        return None
    return urljoin(base_url_for(current_url), match.group(1).replace("&amp;", "&"))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            clean_col(" ".join([str(x) for x in col if str(x) != "nan"]))
            for col in out.columns
        ]
    else:
        out.columns = [clean_col(c) for c in out.columns]

    out = out.loc[:, ~out.columns.str.match(r"^Unnamed")]
    return out


def read_tables(html: str) -> list[pd.DataFrame]:
    tables = pd.read_html(StringIO(html))
    cleaned = []

    for table in tables:
        df = flatten_columns(table)
        for col in df.columns:
            df[col] = df[col].map(clean_cell)
        cleaned.append(df)

    return cleaned


def likely_match_table(df: pd.DataFrame) -> bool:
    cols = " ".join(df.columns).lower()
    return (
        "date" in cols
        and "opponent" in cols
        and "result" in cols
        and ("pos" in cols or "position" in cols)
    )


def select_best_match_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    candidates = [table for table in tables if likely_match_table(table)]
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0].copy()


def normalize_match_table(
    df: pd.DataFrame,
    player: dict[str, Any],
    tm_player_id: str,
    national_url: str,
) -> pd.DataFrame:
    out = df.copy()

    rename_map = {}
    for col in out.columns:
        low = col.lower()

        if "matchday" in low:
            rename_map[col] = "matchday"
        elif "date" in low:
            rename_map[col] = "date"
        elif "venue" in low:
            rename_map[col] = "venue"
        elif low == "for" or " for " in f" {low} ":
            rename_map[col] = "for_team"
        elif "opponent" in low:
            rename_map[col] = "opponent"
        elif "result" in low:
            rename_map[col] = "result"
        elif low == "pos." or low == "pos" or "position" in low:
            rename_map[col] = "position"

    out = out.rename(columns=rename_map)

    out["row_text"] = out.apply(
        lambda r: " | ".join(clean_cell(x) for x in r.values if clean_cell(x)),
        axis=1,
    )

    row_text_lower = out["row_text"].str.lower()

    out["was_not_in_squad"] = row_text_lower.str.contains("not in squad", regex=False)
    out["was_on_bench"] = row_text_lower.str.contains("on the bench", regex=False)

    if "position" in out.columns:
        out["has_position"] = out["position"].map(clean_cell).ne("")
    else:
        out["has_position"] = False

    minute_pattern = re.compile(r"(\d{1,3})'")
    minutes = []
    started = []

    for _, row in out.iterrows():
        text = str(row.get("row_text", ""))
        low = text.lower()

        if "not in squad" in low:
            minutes.append(0)
            started.append(False)
            continue

        if "on the bench" in low:
            minutes.append(0)
            started.append(False)
            continue

        found = minute_pattern.findall(text)
        found_int = [int(x) for x in found if int(x) <= 130]

        if found_int:
            max_min = max(found_int)
            minutes.append(max_min)
            started.append(max_min >= 60)
            continue

        if bool(row.get("has_position")):
            # Transfermarkt viser ofte kun position for starter/appearance.
            minutes.append(None)
            started.append(True)
            continue

        minutes.append(None)
        started.append(False)

    out["minutes_estimate"] = minutes
    out["started_estimate"] = started

    out["player_id"] = get_player_id(player)
    out["player_name"] = player.get("player_name") or player.get("name")
    out["team_id"] = player.get("team_id")
    out["position_model"] = get_position(player)
    out["transfermarkt_player_id"] = tm_player_id
    out["source_url"] = national_url

    return out


def extract_caps_goals(html: str) -> tuple[int | None, int | None]:
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    m = re.search(r"Caps/Goals:\s*(\d+)\s*/\s*(\d+)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def build_summary(
    player: dict[str, Any],
    tm_player_id: str,
    national_url: str,
    html: str,
    matches: pd.DataFrame,
) -> dict[str, Any]:
    caps, goals = extract_caps_goals(html)

    total = len(matches)
    bench = int(matches["was_on_bench"].sum()) if "was_on_bench" in matches else 0
    nis = int(matches["was_not_in_squad"].sum()) if "was_not_in_squad" in matches else 0
    started = int(matches["started_estimate"].sum()) if "started_estimate" in matches else 0
    appeared_or_started = total - bench - nis

    recent = matches.copy()
    if "date" in recent.columns:
        recent["date_parsed"] = pd.to_datetime(recent["date"], format="%d/%m/%y", errors="coerce")
        recent = recent.sort_values("date_parsed", ascending=False)

    recent_20 = recent.head(20)
    recent_10 = recent.head(10)

    recent_20_started = int(recent_20["started_estimate"].sum()) if len(recent_20) else 0
    recent_10_started = int(recent_10["started_estimate"].sum()) if len(recent_10) else 0
    recent_20_bench = int(recent_20["was_on_bench"].sum()) if len(recent_20) else 0
    recent_20_nis = int(recent_20["was_not_in_squad"].sum()) if len(recent_20) else 0

    last_date = ""
    if "date_parsed" in recent.columns and recent["date_parsed"].notna().any():
        last_date = recent.loc[recent["date_parsed"].notna(), "date_parsed"].max().date().isoformat()

    recent_20_start_share = round(recent_20_started / len(recent_20), 3) if len(recent_20) else 0.0
    recent_10_start_share = round(recent_10_started / len(recent_10), 3) if len(recent_10) else 0.0

    usage_score = (
        0.55 * recent_20_start_share
        + 0.25 * recent_10_start_share
        + 0.20 * min(1.0, (caps or 0) / 50)
    )

    return {
        "player_id": get_player_id(player),
        "player_name": player.get("player_name") or player.get("name"),
        "team_id": player.get("team_id"),
        "position": get_position(player),
        "price_m": get_price_m(player),
        "transfermarkt_player_id": tm_player_id,
        "transfermarkt_national_url": national_url,
        "tm_caps": caps,
        "tm_goals": goals,
        "tm_rows_total": total,
        "tm_appeared_or_started_rows": appeared_or_started,
        "tm_started_estimate_total": started,
        "tm_on_bench_total": bench,
        "tm_not_in_squad_total": nis,
        "recent_20_started_estimate": recent_20_started,
        "recent_20_on_bench": recent_20_bench,
        "recent_20_not_in_squad": recent_20_nis,
        "recent_20_start_share": recent_20_start_share,
        "recent_10_start_share": recent_10_start_share,
        "last_national_row_date": last_date,
        "national_team_usage_score": round(usage_score, 4),
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }


def scrape_player_from_transfermarkt_url(
    player: dict[str, Any],
    tm_player_id: str,
    profile_url: str,
    national_url: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    html = fetch_html(national_url)
    time.sleep(REQUEST_SLEEP_SECONDS)

    tables = read_tables(html)
    match_table = select_best_match_table(tables)

    if match_table is None:
        enriched_url = find_national_url_from_html(html, national_url, tm_player_id)
        if enriched_url and enriched_url != national_url:
            national_url = enriched_url
            html = fetch_html(national_url)
            time.sleep(REQUEST_SLEEP_SECONDS)
            tables = read_tables(html)
            match_table = select_best_match_table(tables)

    if match_table is None:
        raise RuntimeError("Ingen kampoversigtstabel fundet")

    matches = normalize_match_table(match_table, player, tm_player_id, national_url)
    summary = build_summary(player, tm_player_id, national_url, html, matches)

    return summary, matches


def scrape_one_player(player: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    player_name = str(player.get("player_name") or player.get("name") or "").strip()
    if not player_name:
        raise RuntimeError("Mangler player_name")

    tm_player_id, profile_url = search_transfermarkt_profile(player_name)
    time.sleep(REQUEST_SLEEP_SECONDS)

    if not tm_player_id or not profile_url:
        raise RuntimeError("Ingen Transfermarkt-profil fundet")

    national_url = find_national_url(profile_url, tm_player_id)
    time.sleep(REQUEST_SLEEP_SECONDS)

    return scrape_player_from_transfermarkt_url(player, tm_player_id, profile_url, national_url)


def upsert_cache_row(cache: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    pid = str(row["player_id"])

    cache = cache.loc[cache["player_id"].astype(str) != pid].copy()
    cache = pd.concat([cache, pd.DataFrame([row])], ignore_index=True)
    return cache


def upsert_summary(existing: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    pid = str(row["player_id"])

    if existing.empty:
        return pd.DataFrame([row])

    existing = existing.loc[existing["player_id"].astype(str) != pid].copy()
    return pd.concat([existing, pd.DataFrame([row])], ignore_index=True)


def write_manual_integration_audit(
    audit_rows: list[dict[str, Any]],
    cache_before: pd.DataFrame,
    cache_after: pd.DataFrame,
) -> None:
    before_status_by_id = {}
    after_status_by_id = {}
    if not cache_before.empty and {"player_id", "status"}.issubset(cache_before.columns):
        before_status_by_id = dict(zip(cache_before["player_id"].astype(str), cache_before["status"].astype(str)))
    if not cache_after.empty and {"player_id", "status"}.issubset(cache_after.columns):
        after_status_by_id = dict(zip(cache_after["player_id"].astype(str), cache_after["status"].astype(str)))

    rows = []
    for row in audit_rows:
        out = dict(row)
        player_id = str(out.get("matched_player_id") or "")
        out["previous_status"] = before_status_by_id.get(player_id, "")
        out["test_status"] = after_status_by_id.get(player_id, "")
        rows.append(out)

    fields = [
        "manual_player_name",
        "manual_tm_player_id",
        "manual_tm_profile_url",
        "manual_tm_national_url",
        "matched_player_id",
        "matched_player_name",
        "team_id",
        "match_status",
        "match_detail",
        "previous_status",
        "test_status",
        "duplicate_warning",
    ]
    pd.DataFrame(rows, columns=fields).to_csv(MANUAL_AUDIT_CSV, index=False, encoding="utf-8-sig")

    manual_count = len(rows)
    matched_count = sum(1 for row in rows if row.get("matched_player_id"))
    unmatched_count = manual_count - matched_count
    previous_error_count = sum(1 for row in rows if row.get("previous_status") == "error")
    ok_manual_count = sum(1 for row in rows if row.get("test_status") == "ok_manual_url")
    duplicate_count = sum(1 for row in rows if row.get("duplicate_warning"))
    ambiguous_count = sum(1 for row in rows if str(row.get("match_status", "")).startswith("ambiguous"))
    watched = {
        "pablo gavi",
        "cucho hernandez",
        "alex robertson",
        "marc pubill",
        "matias fernandez pardo",
        "oguz aydin",
        "bento krepski",
        "ederson moraes",
        "luis suarez charris",
    }
    sanity = [row for row in rows if norm_text(row.get("manual_player_name")) in watched]

    def table(table_rows: list[dict[str, Any]], table_fields: list[str]) -> list[str]:
        lines = ["| " + " | ".join(table_fields) + " |", "| " + " | ".join(["---"] * len(table_fields)) + " |"]
        for row in table_rows:
            lines.append("| " + " | ".join(str(row.get(field, "") or "") for field in table_fields) + " |")
        return lines

    lines = [
        "# Transfermarkt Manual URL Integration Audit",
        "",
        "Manual URL-prioritet: `tools/transfermarkt_manual_urls.csv` -> eksisterende cache-URL -> automatisk Transfermarkt-soegning.",
        "",
        "## Counts",
        "",
        f"- Rækker i manualfilen: {manual_count}",
        f"- Matchet til player pool: {matched_count}",
        f"- Uden match: {unmatched_count}",
        f"- Tidligere `status=error`: {previous_error_count}",
        f"- Efter test `status=ok_manual_url`: {ok_manual_count}",
        f"- Dubletter: {duplicate_count}",
        f"- Tvetydige navnmatch: {ambiguous_count}",
        "",
        "## Sanity-spillere",
        "",
        *table(
            sanity,
            [
                "manual_player_name",
                "matched_player_id",
                "matched_player_name",
                "team_id",
                "previous_status",
                "test_status",
                "match_status",
                "duplicate_warning",
            ],
        ),
        "",
        "## Uden match eller tvetydige",
        "",
        *table(
            [row for row in rows if not row.get("matched_player_id") or str(row.get("match_status", "")).startswith("ambiguous")],
            ["manual_player_name", "match_status", "match_detail", "duplicate_warning"],
        ),
    ]
    MANUAL_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch scrape Transfermarkt national team usage.")
    parser.add_argument("--manual-only", action="store_true", help="Process only players matched from tools/transfermarkt_manual_urls.csv.")
    parser.add_argument("--refresh", action="store_true", help="Refresh known manual/cache URLs instead of treating them as done.")
    parser.add_argument("--limit", type=int, default=MAX_PLAYERS_PER_RUN, help="Maximum players to process in this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    players = load_player_pool()
    cache = load_existing_cache()
    cache_before = cache.copy()
    summary_df = load_existing_summary()
    manual_urls, manual_audit_rows = load_manual_url_map(players)

    todo = choose_players_to_process(
        players,
        cache,
        manual_urls,
        manual_only=args.manual_only,
        refresh=args.refresh,
        limit=args.limit,
    )

    print("TRANSFERMARKT NATIONAL USAGE BATCH")
    print("=" * 80)
    print(f"Player pool: {len(players)}")
    print(f"Cache rows: {len(cache)}")
    print(f"Manual URL rows: {len(manual_audit_rows)}")
    print(f"Manual URL matched: {sum(1 for row in manual_audit_rows if row.get('matched_player_id'))}")
    print(f"Manual-only: {args.manual_only}")
    print(f"Refresh: {args.refresh}")
    print(f"Max this run: {args.limit}")
    print(f"Selected for run: {len(todo)}")
    print("")

    if not todo:
        write_manual_integration_audit(manual_audit_rows, cache_before, cache)
        print("Ingen spillere tilbage i denne batch.")
        return

    ok = 0
    failed = 0

    for index, player in enumerate(todo, start=1):
        pid = get_player_id(player)
        name = str(player.get("player_name") or player.get("name") or "")
        team = str(player.get("team_id") or "")
        pos = get_position(player)
        price = get_price_m(player)

        print(f"[{index}/{len(todo)}] {name} | {team} | {pos} | {price:.1f} mio.")

        cache_row = {
            "player_id": pid,
            "player_name": name,
            "team_id": team,
            "position": pos,
            "price_m": price,
            "tm_player_id": "",
            "tm_profile_url": "",
            "tm_national_url": "",
            "status": "",
            "error": "",
            "last_attempt": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            manual = manual_urls.get(pid)
            cached = cache_row_for_player(cache, pid)
            scrape_source = "auto_search"
            profile_url = ""
            national_url = ""
            tm_id = ""

            if manual:
                scrape_source = "manual_url"
                tm_id = str(manual.get("tm_player_id") or "")
                profile_url = str(manual.get("tm_profile_url") or "")
                national_url = str(manual.get("tm_national_url") or "")
                summary, matches = scrape_player_from_transfermarkt_url(player, tm_id, profile_url, national_url)
            elif has_usable_cache_url(cached):
                scrape_source = "cache_url"
                tm_id = str(cached.get("tm_player_id") or "")
                profile_url = str(cached.get("tm_profile_url") or "")
                national_url = str(cached.get("tm_national_url") or "")
                summary, matches = scrape_player_from_transfermarkt_url(player, tm_id, profile_url, national_url)
            else:
                summary, matches = scrape_one_player(player)

            tm_id = str(summary.get("transfermarkt_player_id") or tm_id or "")
            national_url = str(summary.get("transfermarkt_national_url") or national_url or "")
            if not profile_url:
                profile_url = f"{base_url_for(national_url)}/x/profil/spieler/{tm_id}" if tm_id else ""

            matches_path = OUT_DIR / f"{pid}_{team}_{tm_id}_national_matches.csv"
            matches.to_csv(matches_path, index=False, encoding="utf-8-sig")

            summary_df = upsert_summary(summary_df, summary)

            cache_row.update(
                {
                    "tm_player_id": tm_id,
                    "tm_profile_url": profile_url,
                    "tm_national_url": national_url,
                    "status": "ok_manual_url" if scrape_source == "manual_url" else "ok",
                    "error": "",
                }
            )

            ok += 1
            print(
                f"  OK: caps={summary.get('tm_caps')} "
                f"recent20_start={summary.get('recent_20_start_share')} "
                f"usage={summary.get('national_team_usage_score')}"
            )

        except Exception as exc:
            if manual_urls.get(pid):
                manual = manual_urls[pid]
                cache_row["tm_player_id"] = str(manual.get("tm_player_id") or "")
                cache_row["tm_profile_url"] = str(manual.get("tm_profile_url") or "")
                cache_row["tm_national_url"] = str(manual.get("tm_national_url") or "")
                if "Ingen kampoversigtstabel fundet" in str(exc):
                    ok += 1
                    cache_row["status"] = "ok_manual_url"
                    cache_row["error"] = ""
                    print("  OK manual URL: URL registreret; ingen kampoversigtstabel i HTML")
                else:
                    failed += 1
                    cache_row["status"] = "error_manual_url"
                    cache_row["error"] = str(exc)[:300]
                    print(f"  FEJL: {exc}")
            else:
                failed += 1
                cache_row["status"] = "error"
                cache_row["error"] = str(exc)[:300]
                print(f"  FEJL: {exc}")

        cache = upsert_cache_row(cache, cache_row)
        save_cache(cache)
        save_summary(summary_df)

    write_manual_integration_audit(manual_audit_rows, cache_before, cache)

    print("")
    print("=" * 80)
    print(f"FÃ¦rdig. OK={ok}, fejl={failed}")
    print(f"Skrev cache: {CACHE_PATH}")
    print(f"Skrev summary: {SUMMARY_PATH}")
    print(f"Skrev manual audit: {MANUAL_AUDIT_CSV}")
    print(f"Skrev manual audit: {MANUAL_AUDIT_MD}")
    print("")
    print("KÃ¸r scriptet igen for nÃ¦ste batch.")


if __name__ == "__main__":
    main()

