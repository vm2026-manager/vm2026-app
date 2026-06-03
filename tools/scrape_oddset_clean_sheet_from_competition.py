from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FIXTURES_PATH = DATA / "fixtures_group.csv"
OUT_PATH = DATA / "clean_sheet_odds_oddset.csv"
DEBUG_DIR = DATA / "debug_oddset_clean_sheet_competition"

COMPETITION_URL = "https://danskespil.dk/oddset/sports/competition/25260/fodbold/verden/vm/matches"

TEAM_NAME_DA = {
    "ALG": "Algeriet",
    "ARG": "Argentina",
    "AUS": "Australien",
    "AUT": "Østrig",
    "BEL": "Belgien",
    "BIH": "Bosnien-Hercegovina",
    "BRA": "Brasilien",
    "CAN": "Canada",
    "CIV": "Elfenbenskysten",
    "COD": "DR Congo",
    "COL": "Colombia",
    "CPV": "Kap Verde",
    "CRO": "Kroatien",
    "CUW": "Curaçao",
    "CZE": "Tjekkiet",
    "ECU": "Ecuador",
    "EGY": "Egypten",
    "ENG": "England",
    "ESP": "Spanien",
    "FRA": "Frankrig",
    "GER": "Tyskland",
    "GHA": "Ghana",
    "HAI": "Haiti",
    "IRN": "Iran",
    "IRQ": "Irak",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "Sydkorea",
    "KSA": "Saudi-Arabien",
    "MAR": "Marokko",
    "MEX": "Mexico",
    "NED": "Holland",
    "NOR": "Norge",
    "NZL": "New Zealand",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "RSA": "Sydafrika",
    "SCO": "Skotland",
    "SEN": "Senegal",
    "SUI": "Schweiz",
    "SWE": "Sverige",
    "TUN": "Tunesien",
    "TUR": "Tyrkiet",
    "URU": "Uruguay",
    "USA": "USA",
    "UZB": "Usbekistan",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip().replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", value)
    if not m:
        return None
    return float(m.group(0))


def accept_cookies(page) -> None:
    for selector in [
        "button:has-text('Accepter alle')",
        "button:has-text('Accepter')",
        "button:has-text('Tillad alle')",
        "button:has-text('OK')",
    ]:
        try:
            loc = page.locator(selector).first
            if loc.count() and loc.is_visible(timeout=1000):
                loc.click(timeout=2000)
                page.wait_for_timeout(1000)
                return
        except Exception:
            pass


def save_debug(page, match_id: str, label: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DEBUG_DIR / f"match_{match_id}_{label}_{stamp}"
    try:
        page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass
    try:
        base.with_suffix(".html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def click_text_if_visible(page, text: str, timeout: int = 3000) -> bool:
    try:
        loc = page.get_by_text(text, exact=False)
        for i in range(min(loc.count(), 20)):
            candidate = loc.nth(i)
            try:
                if candidate.is_visible(timeout=500):
                    candidate.click(timeout=timeout)
                    page.wait_for_timeout(1500)
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def open_competition(page) -> None:
    page.goto(COMPETITION_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(5000)
    accept_cookies(page)
    page.wait_for_timeout(3000)


def find_and_open_match(page, home_name: str, away_name: str, match_id: str, slow_ms: int) -> bool:
    """
    Finder kampen på VM-kampoversigten og klikker ind på den.

    Kræver at samme kampblok indeholder både hjemmehold og udehold. Det
    forhindrer, at fx en senere Jordan-kamp åbner den første synlige
    kamp med Jordan.
    """
    # Start øverst hver gang.
    try:
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)
    except Exception:
        pass

    def click_exact_match_card() -> bool:
        try:
            clicked = page.evaluate(
                """({homeName, awayName}) => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                    const home = norm(homeName);
                    const away = norm(awayName);
                    const elements = Array.from(document.querySelectorAll('div, a, button, [role="button"]'));
                    const candidates = elements
                        .map((el, idx) => {
                            const text = norm(el.innerText || el.textContent || '');
                            if (!text.includes(home) || !text.includes(away) || !text.includes('kampvinder')) {
                                return null;
                            }
                            if (text.length > 500) {
                                return null;
                            }
                            const rect = el.getBoundingClientRect();
                            return {
                                idx,
                                area: Math.max(1, rect.width * rect.height),
                                height: rect.height,
                                width: rect.width,
                            };
                        })
                        .filter(Boolean)
                        .sort((a, b) => {
                            const aGood = a.width > 300 && a.height > 40 ? 0 : 1;
                            const bGood = b.width > 300 && b.height > 40 ? 0 : 1;
                            if (aGood !== bGood) return aGood - bGood;
                            return a.area - b.area;
                        });
                    if (!candidates.length) return false;
                    const el = elements[candidates[0].idx];
                    el.scrollIntoView({block: 'center', inline: 'center'});
                    setTimeout(() => el.click(), 100);
                    return true;
                }""",
                {"homeName": home_name, "awayName": away_name},
            )
            if clicked:
                page.wait_for_timeout(4500)
                return "/sports/event/" in page.url
        except Exception:
            pass
        return False

    if click_exact_match_card():
        return True

    for scroll_round in range(40):
        text = page.get_by_text(re.compile(rf"{re.escape(home_name)}\s+{re.escape(away_name)}", re.IGNORECASE)).first

        try:
            if text.count() and text.is_visible(timeout=800):
                # Klik først på selve kampteksten. På Oddset åbner rækken typisk kampen.
                try:
                    text.click(timeout=3000)
                    page.wait_for_timeout(4000)
                except Exception:
                    pass

                # Hvis URL nu er event-side, er vi inde.
                if "/sports/event/" in page.url:
                    return True

                # Hvis ikke, klik på nærmeste område omkring teksten via JS.
                try:
                    box = text.bounding_box()
                    if box:
                        page.mouse.click(box["x"] + 40, box["y"] + 25)
                        page.wait_for_timeout(4000)
                except Exception:
                    pass

                if "/sports/event/" in page.url:
                    return True

                # Tredje forsøg: klik på odds-/markedstæller til højre på samme horisontale niveau.
                try:
                    box = text.bounding_box()
                    if box:
                        page.mouse.click(1225, box["y"] + 25)
                        page.wait_for_timeout(4000)
                except Exception:
                    pass

                if "/sports/event/" in page.url:
                    return True

        except Exception:
            pass

        # Scroll ned og prøv igen.
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(slow_ms)

    save_debug(page, match_id, "match_not_found")
    return False


def click_ou_maal(page, match_id: str) -> bool:
    """
    Klikker O/U Mål-fanen på event-siden.
    """
    for text in ["O/U Mål", "O/U", "Antal mål"]:
        if click_text_if_visible(page, text, timeout=4000):
            page.wait_for_timeout(3000)
            return True

    # Hvis fanen ikke er synlig, kan markedet være længere nede eller under søgning.
    try:
        page.mouse.wheel(0, 800)
        page.wait_for_timeout(1500)
    except Exception:
        pass

    for text in ["O/U Mål", "O/U", "Antal mål"]:
        if click_text_if_visible(page, text, timeout=4000):
            page.wait_for_timeout(3000)
            return True

    save_debug(page, match_id, "ou_tab_not_found")
    return False


def extract_team_under_0_5_from_text(page_text: str, team_name: str) -> Optional[float]:
    """
    Prøver at finde:
    '[Team] Antal mål - over/under'
    og derefter oddset for 'U 0.5'.

    Vi bruger en ret bred tekst-regex, fordi DOM'en på Oddset er dynamisk.
    """
    text = page_text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Tag et vindue fra team-markedets overskrift og frem.
    header_patterns = [
        f"{re.escape(team_name)} Antal mål - over/under",
        f"{re.escape(team_name)} Antal mål",
        f"{re.escape(team_name)}.*?over/under",
    ]

    for hp in header_patterns:
        m = re.search(hp, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        window = text[m.start(): m.start() + 2500]

        # Typisk tekst:
        # O 0.5 1,12
        # U 0.5 5,50
        patterns = [
            r"U\s*0[,.]5\s*(\d+[,.]\d+)",
            r"Under\s*0[,.]5\s*(\d+[,.]\d+)",
        ]
        for pat in patterns:
            mm = re.search(pat, window, flags=re.IGNORECASE)
            if mm:
                return parse_float(mm.group(1))

    return None


def scrape_event_page(page, match_id: str, home_name: str, away_name: str) -> tuple[Optional[float], Optional[float], str, str]:
    """
    Returnerer:
    home_team_under_0_5, away_team_under_0_5, status, note
    """
    if not click_ou_maal(page, match_id):
        return None, None, "missing_ou_tab", "Kunne ikke klikke O/U Mål"

    # Fold evt. 'Vis mere' ud et par gange.
    for _ in range(3):
        clicked = click_text_if_visible(page, "Vis mere", timeout=1500)
        if not clicked:
            break

    page.wait_for_timeout(1500)
    text = page.locator("body").inner_text(timeout=10000)

    home_u05 = extract_team_under_0_5_from_text(text, home_name)
    away_u05 = extract_team_under_0_5_from_text(text, away_name)

    if home_u05 is not None and away_u05 is not None:
        return home_u05, away_u05, "ok", ""
    if home_u05 is not None or away_u05 is not None:
        save_debug(page, match_id, "partial")
        return home_u05, away_u05, "partial", "Kun den ene hold-total U 0.5 blev fundet"

    save_debug(page, match_id, "missing_market")
    return None, None, "missing_market", "Kunne ikke finde holdenes U 0.5 i O/U Mål"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--slow-ms", type=int, default=700)
    parser.add_argument("--start-match-id", type=str, default="")
    args = parser.parse_args()

    fixtures = read_csv(FIXTURES_PATH)

    if args.start_match_id:
        seen_start = False
        filtered = []
        for fx in fixtures:
            if str(fx.get("match_id")) == args.start_match_id:
                seen_start = True
            if seen_start:
                filtered.append(fx)
        fixtures = filtered

    if args.limit and args.limit > 0:
        fixtures = fixtures[: args.limit]

    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, slow_mo=args.slow_ms)
        context = browser.new_context(
            viewport={"width": 1500, "height": 1100},
            locale="da-DK",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        for i, fx in enumerate(fixtures, 1):
            match_id = str(fx.get("match_id", "")).strip()
            home = str(fx.get("home", "")).strip()
            away = str(fx.get("away", "")).strip()
            kickoff = str(fx.get("kickoff_dk", "")).strip()

            home_name = TEAM_NAME_DA.get(home, home)
            away_name = TEAM_NAME_DA.get(away, away)

            print(f"[{i}/{len(fixtures)}] {match_id} {home}-{away} | {home_name} - {away_name}")

            try:
                open_competition(page)

                opened = find_and_open_match(page, home_name, away_name, match_id, args.slow_ms)
                if not opened:
                    rows.append({
                        "match_id": match_id,
                        "home": home,
                        "away": away,
                        "kickoff_dk": kickoff,
                        "home_clean_sheet_odds": "",
                        "away_clean_sheet_odds": "",
                        "home_team_under_0_5_odds": "",
                        "away_team_under_0_5_odds": "",
                        "source": "Oddset team total under 0.5",
                        "scraped_at": datetime.now().isoformat(timespec="seconds"),
                        "oddset_url": "",
                        "status": "match_not_found",
                        "note": f"Kunne ikke åbne kampen fra competition-siden: {home_name}-{away_name}",
                    })
                    print("  FEJL: match_not_found")
                    continue

                event_url = page.url
                home_u05, away_u05, status, note = scrape_event_page(page, match_id, home_name, away_name)

                # Definition:
                # Home clean sheet = away team under 0.5
                # Away clean sheet = home team under 0.5
                home_cs = away_u05
                away_cs = home_u05

                rows.append({
                    "match_id": match_id,
                    "home": home,
                    "away": away,
                    "kickoff_dk": kickoff,
                    "home_clean_sheet_odds": home_cs if home_cs is not None else "",
                    "away_clean_sheet_odds": away_cs if away_cs is not None else "",
                    "home_team_under_0_5_odds": home_u05 if home_u05 is not None else "",
                    "away_team_under_0_5_odds": away_u05 if away_u05 is not None else "",
                    "source": "Oddset team total under 0.5",
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                    "oddset_url": event_url,
                    "status": status,
                    "note": note,
                })

                print(f"  {status} | home_cs={home_cs} | away_cs={away_cs}")

            except Exception as e:
                save_debug(page, match_id, "exception")
                rows.append({
                    "match_id": match_id,
                    "home": home,
                    "away": away,
                    "kickoff_dk": kickoff,
                    "home_clean_sheet_odds": "",
                    "away_clean_sheet_odds": "",
                    "home_team_under_0_5_odds": "",
                    "away_team_under_0_5_odds": "",
                    "source": "Oddset team total under 0.5",
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                    "oddset_url": page.url,
                    "status": "error",
                    "note": str(e)[:500],
                })
                print(f"  FEJL: {e}")

        browser.close()

    fieldnames = [
        "match_id",
        "home",
        "away",
        "kickoff_dk",
        "home_clean_sheet_odds",
        "away_clean_sheet_odds",
        "home_team_under_0_5_odds",
        "away_team_under_0_5_odds",
        "source",
        "scraped_at",
        "oddset_url",
        "status",
        "note",
    ]
    write_csv(OUT_PATH, rows, fieldnames)

    ok = sum(1 for r in rows if r["status"] == "ok")
    partial = sum(1 for r in rows if r["status"] == "partial")
    errors = len(rows) - ok - partial

    print("")
    print("Færdig")
    print("------")
    print(f"Rækker: {len(rows)}")
    print(f"OK: {ok}")
    print(f"Partial: {partial}")
    print(f"Fejl/mangler: {errors}")
    print(f"Skrevet: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
