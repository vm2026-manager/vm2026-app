from __future__ import annotations

import argparse
import csv
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FIXTURES_PATH = DATA / "fixtures_group.csv"
MANUAL_URLS_PATH = DATA / "oddset_match_urls.csv"
OUT_PATH = DATA / "oddset_match_urls_discovered.csv"
DEBUG_DIR = DATA / "debug_oddset_url_discovery"

START_URLS = [
    "https://danskespil.dk/oddset/sports/fodbold/verden/vm",
    "https://danskespil.dk/oddset/sports/competition/fodbold/verden/vm",
]

EVENT_RE = re.compile(
    r"https://danskespil\.dk/oddset/sports/event/\d+/fodbold/verden/vm/[^\"'#?\s<>]+",
    re.IGNORECASE,
)

TEAM_NAME_DA = {
    "ALG": ["algeriet", "algeria"],
    "ARG": ["argentina"],
    "AUS": ["australien", "australia"],
    "AUT": ["oestrig", "østrig", "austria"],
    "BEL": ["belgien", "belgium"],
    "BIH": ["bosnien-hercegovina", "bosnien", "bosnia-herzegovina", "bosnia"],
    "BRA": ["brasilien", "brazil"],
    "CAN": ["canada"],
    "CIV": ["elfenbenskysten", "ivory-coast", "cote-divoire", "cote-d-ivoire"],
    "COD": ["dr-congo", "congo-dr", "dem-rep-congo", "demokratiske-republik-congo"],
    "COL": ["colombia"],
    "CPV": ["kap-verde", "cape-verde"],
    "CRO": ["kroatien", "croatia"],
    "CUW": ["curacao", "curaçao"],
    "CZE": ["tjekkiet", "czech-republic", "czechia"],
    "ECU": ["ecuador"],
    "EGY": ["egypten", "egypt"],
    "ENG": ["england"],
    "ESP": ["spanien", "spain"],
    "FRA": ["frankrig", "france"],
    "GER": ["tyskland", "germany"],
    "GHA": ["ghana"],
    "HAI": ["haiti"],
    "IRN": ["iran"],
    "IRQ": ["irak", "iraq"],
    "JOR": ["jordan"],
    "JPN": ["japan"],
    "KOR": ["sydkorea", "south-korea", "korea-republic"],
    "KSA": ["saudi-arabien", "saudi-arabia"],
    "MAR": ["marokko", "morocco"],
    "MEX": ["mexico"],
    "NED": ["holland", "nederlandene", "netherlands"],
    "NOR": ["norge", "norway"],
    "NZL": ["new-zealand", "new-zealand"],
    "PAN": ["panama"],
    "PAR": ["paraguay"],
    "POR": ["portugal"],
    "QAT": ["qatar"],
    "RSA": ["sydafrika", "south-africa"],
    "SCO": ["skotland", "scotland"],
    "SEN": ["senegal"],
    "SUI": ["schweiz", "switzerland"],
    "SWE": ["sverige", "sweden"],
    "TUN": ["tunesien", "tunisia"],
    "TUR": ["tyrkiet", "turkey", "turkiye"],
    "URU": ["uruguay"],
    "USA": ["usa", "united-states"],
    "UZB": ["usbekistan", "uzbekistan"],
}


def norm_text(value: str) -> str:
    value = str(value or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("ø", "oe").replace("æ", "ae").replace("å", "aa")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def slug_variants(team_id: str) -> list[str]:
    raw = TEAM_NAME_DA.get(team_id, [team_id.lower()])
    out = []
    for item in raw:
        n = norm_text(item)
        if n and n not in out:
            out.append(n)
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def get_event_id(url: str) -> str:
    m = re.search(r"/event/(\d+)/", url)
    return m.group(1) if m else ""


def get_event_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    parts = path.split("/")
    if "vm" in parts:
        i = parts.index("vm")
        if i + 1 < len(parts):
            return norm_text(parts[i + 1])
    return norm_text(parts[-1] if parts else "")


def load_manual_urls() -> dict[str, dict]:
    manual = {}
    for row in read_csv(MANUAL_URLS_PATH):
        match_id = str(row.get("match_id", "")).strip()
        url = str(row.get("oddset_url", "")).strip()
        if match_id and url:
            manual[match_id] = row
    return manual


def accept_cookies(page) -> None:
    candidates = [
        "button:has-text('Accepter alle')",
        "button:has-text('Accepter')",
        "button:has-text('Tillad alle')",
        "button:has-text('OK')",
    ]
    for selector in candidates:
        try:
            btn = page.locator(selector).first
            if btn.count() and btn.is_visible(timeout=1000):
                btn.click(timeout=2000)
                page.wait_for_timeout(500)
                return
        except Exception:
            pass


def collect_event_urls_from_page(page, base_url: str, slow_ms: int, debug: bool, label: str) -> set[str]:
    urls: set[str] = set()

    try:
        page.goto(base_url, wait_until="domcontentloaded", timeout=45000)
        accept_cookies(page)
        page.wait_for_timeout(2500)
    except Exception as e:
        if debug:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            safe = norm_text(label)
            try:
                page.screenshot(path=str(DEBUG_DIR / f"{safe}_goto_error.png"), full_page=True)
            except Exception:
                pass
        print(f"ADVARSEL: Kunne ikke åbne {base_url}: {e}")
        return urls

    for i in range(12):
        try:
            html = page.content()
            for m in EVENT_RE.finditer(html):
                urls.add(m.group(0).split("?")[0])

            hrefs = page.eval_on_selector_all(
                "a[href]",
                """els => els.map(a => a.href).filter(Boolean)"""
            )
            for href in hrefs:
                href = urljoin(base_url, href).split("?")[0]
                if "/oddset/sports/event/" in href and "/fodbold/verden/vm/" in href:
                    urls.add(href)

            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(slow_ms)
        except Exception:
            pass

        # Try common "show more" buttons
        for text in ["Vis mere", "Se mere", "Flere"]:
            try:
                loc = page.locator(f"text={text}").first
                if loc.count() and loc.is_visible(timeout=500):
                    loc.click(timeout=1000)
                    page.wait_for_timeout(slow_ms)
            except Exception:
                pass

    if debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe = norm_text(label)
        try:
            page.screenshot(path=str(DEBUG_DIR / f"{safe}_done.png"), full_page=True)
            (DEBUG_DIR / f"{safe}_done.html").write_text(page.content(), encoding="utf-8")
        except Exception:
            pass

    return urls


def score_url_for_fixture(url: str, home: str, away: str) -> tuple[int, str]:
    slug = get_event_slug(url)
    home_vars = slug_variants(home)
    away_vars = slug_variants(away)

    exact_home_away = [f"{h}-{a}" for h in home_vars for a in away_vars]
    exact_away_home = [f"{a}-{h}" for h in home_vars for a in away_vars]

    if slug in exact_home_away:
        return 100, "exact_home_away_slug"
    if slug in exact_away_home:
        return 95, "exact_away_home_slug"

    has_home = any(h in slug for h in home_vars)
    has_away = any(a in slug for a in away_vars)

    if has_home and has_away:
        return 80, "contains_both_team_names"

    return 0, ""


def discover(args) -> list[dict]:
    fixtures = read_csv(FIXTURES_PATH)
    manual = load_manual_urls()

    found_urls: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, slow_mo=args.slow_ms)
        context = browser.new_context(
            viewport={"width": 1400, "height": 1000},
            locale="da-DK",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        for idx, start_url in enumerate(START_URLS, start=1):
            print(f"Crawler VM-side {idx}/{len(START_URLS)}: {start_url}")
            found_urls |= collect_event_urls_from_page(
                page,
                start_url,
                args.slow_ms,
                args.debug,
                f"start_{idx}",
            )

        # Also revisit manual URLs to ensure they are in candidate pool.
        for row in manual.values():
            url = str(row.get("oddset_url", "")).strip()
            if url:
                found_urls.add(url.split("?")[0])

        browser.close()

    print(f"Event-URL’er fundet/kendt: {len(found_urls)}")

    rows = []
    limit = args.limit if args.limit and args.limit > 0 else None
    fixtures_to_process = fixtures[:limit] if limit else fixtures

    for fx in fixtures_to_process:
        match_id = str(fx.get("match_id", "")).strip()
        home = str(fx.get("home", "")).strip()
        away = str(fx.get("away", "")).strip()
        kickoff = str(fx.get("kickoff_dk", "")).strip()

        if match_id in manual and manual[match_id].get("oddset_url"):
            url = str(manual[match_id].get("oddset_url")).strip()
            rows.append({
                "match_id": match_id,
                "home": home,
                "away": away,
                "kickoff_dk": kickoff,
                "home_name_da": "/".join(TEAM_NAME_DA.get(home, [home])),
                "away_name_da": "/".join(TEAM_NAME_DA.get(away, [away])),
                "oddset_url": url,
                "event_id": get_event_id(url),
                "discovery_status": "existing_manual",
                "note": "Fra data/oddset_match_urls.csv",
            })
            continue

        scored = []
        for url in found_urls:
            score, reason = score_url_for_fixture(url, home, away)
            if score:
                scored.append((score, reason, url))

        scored.sort(reverse=True, key=lambda x: x[0])

        if not scored:
            status = "not_found"
            url = ""
            event_id = ""
            note = "Ingen URL med begge holdnavne fundet"
        elif len(scored) == 1 or scored[0][0] > scored[1][0]:
            status = "found"
            _, reason, url = scored[0]
            event_id = get_event_id(url)
            note = reason
        else:
            status = "ambiguous"
            url = scored[0][2]
            event_id = get_event_id(url)
            note = "Flere lige gode kandidater: " + " | ".join(x[2] for x in scored[:5])

        rows.append({
            "match_id": match_id,
            "home": home,
            "away": away,
            "kickoff_dk": kickoff,
            "home_name_da": "/".join(TEAM_NAME_DA.get(home, [home])),
            "away_name_da": "/".join(TEAM_NAME_DA.get(away, [away])),
            "oddset_url": url,
            "event_id": event_id,
            "discovery_status": status,
            "note": note,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Begræns antal fixtures til test.")
    parser.add_argument("--headless", action="store_true", help="Kør browser headless.")
    parser.add_argument("--slow-ms", type=int, default=300, help="Playwright slow_mo / ventetid.")
    parser.add_argument("--debug", action="store_true", help="Gem debug screenshot/html.")
    parser.add_argument("--write", action="store_true", help="Skriv data/oddset_match_urls.csv med discovered URLs.")
    args = parser.parse_args()

    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Mangler {FIXTURES_PATH}")

    rows = discover(args)

    fieldnames = [
        "match_id",
        "home",
        "away",
        "kickoff_dk",
        "home_name_da",
        "away_name_da",
        "oddset_url",
        "event_id",
        "discovery_status",
        "note",
    ]

    write_csv(OUT_PATH, rows, fieldnames)

    from collections import Counter
    counts = Counter(row["discovery_status"] for row in rows)

    print("")
    print("Oddset URL discovery")
    print("--------------------")
    print(f"Fixtures behandlet: {len(rows)}")
    for key in ["existing_manual", "found", "ambiguous", "not_found"]:
        print(f"{key}: {counts.get(key, 0)}")
    print(f"Skrevet: {OUT_PATH.relative_to(ROOT)}")

    print("")
    print("Første 20:")
    for row in rows[:20]:
        print(
            f"{row['match_id']:>2} {row['home']}-{row['away']} | "
            f"{row['discovery_status']} | {row['oddset_url']}"
        )

    if args.write:
        write_rows = []
        for row in rows:
            write_rows.append({
                "match_id": row["match_id"],
                "home": row["home"],
                "away": row["away"],
                "kickoff_dk": row["kickoff_dk"],
                "oddset_url": row["oddset_url"] if row["discovery_status"] in {"existing_manual", "found"} else "",
            })
        write_csv(
            MANUAL_URLS_PATH,
            write_rows,
            ["match_id", "home", "away", "kickoff_dk", "oddset_url"],
        )
        print(f"Opdateret: {MANUAL_URLS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()