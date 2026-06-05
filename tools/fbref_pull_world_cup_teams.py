from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPETITION_URL = "https://fbref.com/en/comps/1/World-Cup-Stats"

OUTPUT_DIR = PROJECT_ROOT / "data" / "fbref_world_cup_2026"
HTML_DIR = OUTPUT_DIR / "html"
TEAM_TABLE_DIR = OUTPUT_DIR / "team_tables"
COMBINED_DIR = OUTPUT_DIR / "combined_tables"

INDEX_PATH = OUTPUT_DIR / "team_pages_index.csv"
TEAM_LINKS_PATH = OUTPUT_DIR / "world_cup_team_links.csv"

DEFAULT_SLEEP_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 60


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def ensure_directories() -> None:
    for path in [
        OUTPUT_DIR,
        HTML_DIR,
        TEAM_TABLE_DIR,
        COMBINED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def create_driver(headless: bool, timeout: int) -> webdriver.Chrome:
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-notifications")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(timeout)

    return driver


def wait_for_page(driver: webdriver.Chrome) -> None:
    WebDriverWait(driver, 25).until(
        lambda current_driver: current_driver.execute_script(
            "return document.readyState"
        )
        in {"interactive", "complete"}
    )


def detect_block_page(html: str, url: str) -> None:
    lowered = html.lower()

    block_terms = [
        "rate limit",
        "too many requests",
        "bot traffic",
        "access denied",
        "temporarily blocked",
        "cloudflare",
    ]

    found = [term for term in block_terms if term in lowered]

    if found:
        raise RuntimeError(
            f"FBref ser ud til at have returneret en blok-/rate-limit-side "
            f"for {url}. Fundet: {found}"
        )


def fetch_page(
    driver: webdriver.Chrome,
    url: str,
    retries: int = 3,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            wait_for_page(driver)

            # Giv eventuelle dynamiske elementer lidt tid.
            time.sleep(2)

            html = driver.page_source

            if len(html) < 1_000:
                raise RuntimeError(
                    f"Den hentede HTML-side virker for lille: {len(html)} tegn"
                )

            detect_block_page(html, url)

            return html

        except (TimeoutException, WebDriverException, RuntimeError) as exc:
            last_error = exc
            print(f"  Forsøg {attempt}/{retries} fejlede: {exc}")

            if attempt < retries:
                time.sleep(20 * attempt)

    raise RuntimeError(f"Kunne ikke hente {url}: {last_error}")


def get_visible_and_commented_soups(html: str) -> list[BeautifulSoup]:
    main_soup = BeautifulSoup(html, "html.parser")
    soups = [main_soup]

    for comment in main_soup.find_all(
        string=lambda text: isinstance(text, Comment)
    ):
        comment_text = str(comment)

        if "<table" in comment_text.lower() or "/en/squads/" in comment_text:
            soups.append(BeautifulSoup(comment_text, "html.parser"))

    return soups


def extract_team_links(html: str, base_url: str) -> pd.DataFrame:
    soups = get_visible_and_commented_soups(html)

    records_by_squad_id: dict[str, dict[str, str]] = {}

    for soup in soups:
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()

            if "/en/squads/" not in href:
                continue

            squad_match = re.search(r"/en/squads/([^/]+)/", href)

            if not squad_match:
                continue

            squad_id = squad_match.group(1)

            # Landsholdssiderne ender typisk med "-Stats".
            # Vi tillader både URL'er med og uden et årstal.
            if "-Stats" not in href:
                continue

            href_without_query = href.split("?")[0]
            full_url = urljoin(base_url, href_without_query)

            team_name = anchor.get_text(" ", strip=True)

            if not team_name:
                final_part = href_without_query.rstrip("/").split("/")[-1]
                team_name = final_part.removesuffix("-Stats").replace("-", " ")

            team_name = re.sub(r"\s+", " ", team_name).strip()

            if not team_name:
                continue

            # Samme land optræder ofte flere gange på VM-oversigten.
            # Behold kun én URL pr. squad-id.
            if squad_id not in records_by_squad_id:
                records_by_squad_id[squad_id] = {
                    "team_name": team_name,
                    "squad_id": squad_id,
                    "team_url": full_url,
                }

    result = pd.DataFrame(records_by_squad_id.values())

    if result.empty:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""

        squad_candidates = [
            str(anchor.get("href", ""))
            for anchor in soup.find_all("a", href=True)
            if "squad" in str(anchor.get("href", "")).lower()
        ][:30]

        diagnostic_path = HTML_DIR / "competition_page_no_team_links_debug.html"
        diagnostic_path.write_text(html, encoding="utf-8")

        raise RuntimeError(
            "Ingen landsholdslinks blev fundet på VM-siden. "
            f"Sidens titel: {title!r}. "
            f"Eksempel på squad-links: {squad_candidates}. "
            f"HTML er gemt til: {diagnostic_path}"
        )

    return (
        result.drop_duplicates(subset=["squad_id"])
        .sort_values(["team_name", "team_url"])
        .reset_index(drop=True)
    )


def iter_all_tables(html: str) -> list:
    tables = []
    seen_hashes: set[str] = set()

    for soup in get_visible_and_commented_soups(html):
        for table in soup.find_all("table"):
            table_html = str(table)

            table_hash = hashlib.sha1(
                table_html.encode("utf-8", errors="ignore")
            ).hexdigest()

            if table_hash in seen_hashes:
                continue

            seen_hashes.add(table_hash)
            tables.append(table)

    return tables


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        flattened: list[str] = []

        for column_tuple in result.columns:
            parts: list[str] = []

            for part in column_tuple:
                part_text = str(part).strip()

                if not part_text:
                    continue

                if part_text.lower() == "nan":
                    continue

                if part_text.startswith("Unnamed:"):
                    continue

                parts.append(part_text)

            flattened.append("__".join(parts) or "column")
    else:
        flattened = []

        for column in result.columns:
            column_text = str(column).strip()

            if column_text.startswith("Unnamed:"):
                column_text = "column"

            flattened.append(column_text or "column")

    counts: dict[str, int] = {}
    unique_columns: list[str] = []

    for column in flattened:
        counts[column] = counts.get(column, 0) + 1

        if counts[column] == 1:
            unique_columns.append(column)
        else:
            unique_columns.append(f"{column}_{counts[column]}")

    result.columns = unique_columns

    return result


def get_table_name(table, table_number: int) -> str:
    table_id = str(table.get("id", "")).strip()

    if table_id:
        return slugify(table_id)

    caption = table.find("caption")

    if caption:
        caption_text = caption.get_text(" ", strip=True)

        if caption_text:
            return slugify(caption_text)

    return f"table_{table_number:02d}"


def parse_and_save_team_tables(
    html: str,
    team_name: str,
    team_url: str,
    combined_frames: dict[str, list[pd.DataFrame]],
) -> int:
    team_slug = slugify(team_name)
    team_output_dir = TEAM_TABLE_DIR / team_slug
    team_output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for table_number, table in enumerate(iter_all_tables(html), start=1):
        table_name = get_table_name(table, table_number)

        try:
            frames = pd.read_html(StringIO(str(table)))
        except (ValueError, ImportError):
            continue

        if not frames:
            continue

        df = flatten_columns(frames[0])

        if df.empty:
            continue

        df.insert(0, "fbref_team_name", team_name)
        df.insert(1, "fbref_team_url", team_url)
        df.insert(2, "fbref_table_name", table_name)

        output_path = team_output_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        combined_frames[table_name].append(df)
        saved_count += 1

    return saved_count


def load_existing_team_tables_into_combined(
    combined_frames: dict[str, list[pd.DataFrame]],
) -> None:
    if not TEAM_TABLE_DIR.exists():
        return

    for csv_path in TEAM_TABLE_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        table_name = csv_path.stem
        combined_frames[table_name].append(df)


def save_combined_tables(
    combined_frames: dict[str, list[pd.DataFrame]],
) -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in COMBINED_DIR.glob("*_all_teams.csv"):
        try:
            old_file.unlink()
        except OSError:
            pass

    for table_name, frames in combined_frames.items():
        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True, sort=False)

        dedupe_columns = [
            column
            for column in [
                "fbref_team_name",
                "fbref_team_url",
                "fbref_table_name",
                "Player",
                "Squad",
                "Rk",
            ]
            if column in combined.columns
        ]

        if dedupe_columns:
            combined = combined.drop_duplicates(subset=dedupe_columns)
        else:
            combined = combined.drop_duplicates()

        output_path = COMBINED_DIR / f"{table_name}_all_teams.csv"
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")


def load_existing_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(INDEX_PATH)
    except Exception:
        return pd.DataFrame()


def save_index(records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(
        INDEX_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gem FBref World Cup-landesider og alle tilgængelige tabeller."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Hent også sider, som allerede er gemt lokalt.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Kør Chrome uden synligt browservindue.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Ventetid mellem FBref-sider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Begræns antal lande. 0 betyder alle.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Side-timeout i sekunder.",
    )

    args = parser.parse_args()

    ensure_directories()

    print("FBREF WORLD CUP TEAM SCRAPER")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Refresh: {args.refresh}")
    print(f"Headless: {args.headless}")
    print(f"Sleep: {args.sleep} sekunder")

    driver = create_driver(args.headless, args.timeout)

    try:
        competition_html_path = HTML_DIR / "world_cup_competition_page.html"

        if args.refresh or not competition_html_path.exists():
            print("\nHenter VM-oversigten...")

            competition_html = fetch_page(driver, COMPETITION_URL)

            competition_html_path.write_text(
                competition_html,
                encoding="utf-8",
            )
        else:
            print("\nBruger allerede gemt VM-oversigt.")

            competition_html = competition_html_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        team_links = extract_team_links(
            competition_html,
            COMPETITION_URL,
        )

        if args.limit > 0:
            team_links = team_links.head(args.limit)

        team_links.to_csv(
            TEAM_LINKS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Fundet landsholdslinks: {len(team_links)}")

        existing_index = load_existing_index()
        existing_records: dict[str, dict] = {}

        if not existing_index.empty and "team_url" in existing_index.columns:
            for _, row in existing_index.iterrows():
                existing_records[str(row["team_url"])] = row.to_dict()

        combined_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)

        for number, (_, team) in enumerate(team_links.iterrows(), start=1):
            team_name = str(team["team_name"])
            team_url = str(team["team_url"])
            squad_id = str(team.get("squad_id", ""))
            team_slug = slugify(team_name)

            html_path = HTML_DIR / f"{team_slug}.html"

            print()
            print(f"[{number}/{len(team_links)}] {team_name}")
            print(f"  {team_url}")

            fetch_status = "existing"
            error = ""
            tables_saved = 0

            try:
                if args.refresh or not html_path.exists():
                    html = fetch_page(driver, team_url)
                    html_path.write_text(html, encoding="utf-8")
                    fetch_status = "fetched"

                    time.sleep(max(args.sleep, 0))
                else:
                    html = html_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )

                tables_saved = parse_and_save_team_tables(
                    html=html,
                    team_name=team_name,
                    team_url=team_url,
                    combined_frames=combined_frames,
                )

                print(f"  Gemte tabeller: {tables_saved}")

            except Exception as exc:
                fetch_status = "error"
                error = str(exc)

                print(f"  FEJL: {exc}")

            record = {
                "team_name": team_name,
                "squad_id": squad_id,
                "team_url": team_url,
                "html_file": str(html_path.relative_to(PROJECT_ROOT)),
                "fetch_status": fetch_status,
                "tables_saved": tables_saved,
                "error": error,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

            existing_records[team_url] = record
            save_index(list(existing_records.values()))

        print("\nGenindlæser alle gemte landetabeller...")
        combined_frames = defaultdict(list)
        load_existing_team_tables_into_combined(combined_frames)

        print("Skriver kombinerede tabeller...")
        save_combined_tables(combined_frames)

        print()
        print("Færdig.")
        print(f"Landesider: {HTML_DIR}")
        print(f"Landetabeller: {TEAM_TABLE_DIR}")
        print(f"Kombinerede tabeller: {COMBINED_DIR}")
        print(f"Indeks: {INDEX_PATH}")

        return 0

    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())