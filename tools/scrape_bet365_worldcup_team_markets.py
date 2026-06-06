from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "bet365_worldcup_team_markets"

START_URL = "https://www.bet365.dk/#/AS/B1/I%5E88/J%5E1/K%5E3/"

# Bruges kun til filnavne.
def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("ø", "oe").replace("å", "aa").replace("æ", "ae")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def visible_text(page: Page) -> str:
    return page.evaluate("() => document.body.innerText || ''")


def wait_for_user_ready(page: Page) -> None:
    print()
    print("Browseren åbner nu Bet365.")
    print("Gør manuelt dette i browseren, hvis nødvendigt:")
    print("  1. accepter cookies")
    print("  2. luk popups")
    print("  3. gå til VM 2026 > Teams")
    print("  4. sørg for at landelisten er synlig")
    print()
    input("Tryk ENTER her i PowerShell, når landelisten er klar...")


def get_team_candidates(page: Page) -> list[str]:
    """
    Brug kendt landeliste fra Bet365-vinderoddsfilen.
    Det er mere stabilt end at udlede landene fra Bet365' dynamiske innerText.
    """
    source_path = DATA_DIR / "worldcup_outright_odds_bet365_20260606.csv"

    if source_path.exists():
        import csv

        teams: list[str] = []
        with source_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = str(row.get("team_name", "")).strip()
                if team:
                    teams.append(team)

        if teams:
            return teams

    # Fallback hvis kildefilen ikke findes.
    return [
        "Algeria", "Argentina", "Australia", "Austria", "Belgium",
        "Bosnia-Herzegovina", "Brazil", "Canada", "Cape Verde",
        "Colombia", "Curacao", "Croatia", "Czechia", "DR Congo",
        "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
        "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
        "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
        "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
        "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
        "Sweden", "Switzerland", "Tunisia", "Turkey", "Uruguay",
        "USA", "Uzbekistan",
    ]
    """
    Bet365-DOM er dynamisk. Vi udleder landene fra synlig tekst.
    På Teams-siden står typisk land + FIFA-ranking.
    """
    text = visible_text(page)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    teams: list[str] = []
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and lines[i + 1].startswith("FIFA-rangering"):
            teams.append(line)

    # Rens dubletter, bevar rækkefølge.
    seen: set[str] = set()
    cleaned: list[str] = []
    for team in teams:
        key = team.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(team)

    return cleaned


def click_team(page: Page, team: str) -> bool:
    """
    Prøver flere klikmetoder. Vi undgår hardcodede CSS-klasser.
    """
    candidates = [
        page.get_by_text(team, exact=True),
        page.locator(f"text={team}").first,
    ]

    for loc in candidates:
        try:
            loc.click(timeout=5000)
            return True
        except Exception:
            pass

    # Fallback: klik via JS på elementer med præcis innerText.
    try:
        ok = page.evaluate(
            """
            (team) => {
                const elements = Array.from(document.querySelectorAll('*'));
                const el = elements.find(e => (e.innerText || '').trim() === team);
                if (!el) return false;
                el.scrollIntoView({block: 'center'});
                el.click();
                return true;
            }
            """,
            team,
        )
        return bool(ok)
    except Exception:
        return False


def save_team_snapshot(page: Page, team: str, index: int, refresh_wait: float) -> dict[str, str]:
    slug = safe_slug(team)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / f"{index:02d}_{slug}_{stamp}"

    time.sleep(refresh_wait)

    txt_path = base.with_suffix(".txt")
    html_path = base.with_suffix(".html")
    png_path = base.with_suffix(".png")

    text = visible_text(page)
    html = page.content()

    txt_path.write_text(text, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    page.screenshot(path=str(png_path), full_page=True)

    return {
        "team": team,
        "txt": str(txt_path.relative_to(PROJECT_ROOT)),
        "html": str(html_path.relative_to(PROJECT_ROOT)),
        "png": str(png_path.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=START_URL)
    parser.add_argument("--limit", type=int, default=0, help="0 = alle lande")
    parser.add_argument("--wait", type=float, default=2.0, help="Sekunder efter hvert klik")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            locale="da-DK",
        )
        page = context.new_page()

        print("Åbner:", args.url)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)

        wait_for_user_ready(page)

        teams = get_team_candidates(page)
        if not teams:
            print("Ingen lande fundet. Sørg for, at Teams-siden med landelisten er synlig.")
            return 1

        if args.limit and args.limit > 0:
            teams = teams[: args.limit]

        print()
        print(f"Fandt {len(teams)} lande:")
        for t in teams:
            print(" -", t)

        index_rows: list[dict[str, str]] = []

        for idx, team in enumerate(teams, start=1):
            print()
            print(f"[{idx}/{len(teams)}] Klikker: {team}")

            ok = click_team(page, team)
            if not ok:
                print(f"  FEJL: kunne ikke klikke {team}")
                index_rows.append(
                    {"team": team, "status": "click_failed", "txt": "", "html": "", "png": ""}
                )
                continue

            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightTimeoutError:
                pass

            try:
                snap = save_team_snapshot(page, team, idx, args.wait)
                snap["status"] = "ok"
                index_rows.append(snap)
                print("  OK:", snap["txt"], snap["png"])
            except Exception as exc:
                print("  FEJL ved gem:", exc)
                index_rows.append(
                    {"team": team, "status": f"save_failed: {exc}", "txt": "", "html": "", "png": ""}
                )

            # Forsøg at gå tilbage til teamlisten.
            try:
                page.go_back(wait_until="domcontentloaded", timeout=15000)
                time.sleep(1.0)
            except Exception:
                print("  Kunne ikke gå tilbage automatisk. Prøver at åbne start-URL igen.")
                page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(2.0)

        # Gem index.
        import csv

        index_path = OUT_DIR / "bet365_team_scrape_index.csv"
        fieldnames = ["team", "status", "txt", "html", "png"]
        with index_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in index_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        print()
        print("Færdig.")
        print("Index:", index_path)
        browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())