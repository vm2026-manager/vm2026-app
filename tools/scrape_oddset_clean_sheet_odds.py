from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "oddset_match_urls.csv"
OUTPUT_PATH = DATA_DIR / "clean_sheet_odds_oddset.csv"
DEBUG_DIR = DATA_DIR / "debug_oddset_clean_sheet"

SOURCE = "Oddset team total under 0.5"

TEAM_NAME_ALIASES = {
    "MEX": ["Mexico"],
    "RSA": ["Sydafrika", "South Africa"],
    "KOR": ["Sydkorea", "Korea Republic", "S. Korea"],
    "CZE": ["Tjekkiet", "Czech Republic", "Czechia"],
    "SUI": ["Schweiz"],
    "QAT": ["Qatar"],
    "BRA": ["Brasilien"],
    "MAR": ["Marokko"],
    "SCO": ["Skotland"],
    "HAI": ["Haiti"],
    "GER": ["Tyskland"],
    "CUW": ["Curaçao", "Curacao"],
    "ESP": ["Spanien"],
    "CPV": ["Kap Verde"],
    "URU": ["Uruguay"],
    "KSA": ["Saudi-Arabien", "Saudi Arabien"],
    "FRA": ["Frankrig"],
    "SEN": ["Senegal"],
    "NOR": ["Norge"],
    "IRQ": ["Irak"],
    "ARG": ["Argentina"],
    "ALG": ["Algeriet"],
    "POR": ["Portugal"],
    "BEL": ["Belgien"],
    "USA": ["USA"],
    "JPN": ["Japan"],
}

OUT_FIELDS = [
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


@dataclass
class ScrapeResult:
    home_under: str = ""
    away_under: str = ""
    status: str = "missing_market"
    note: str = ""
    debug_files: list[str] | None = None


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def decimal_odds(value: str) -> str:
    return value.replace(",", ".")


def read_input(limit: int | None) -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        return rows[:limit]
    return rows


def team_names(code: str) -> list[str]:
    code = txt(code).upper()
    return TEAM_NAME_ALIASES.get(code, [code])


def odds_pattern() -> str:
    return r"(\d{1,2}(?:[,.]\d{1,2})?)"


def extract_under_0_5_from_market(page_text: str, team_code: str) -> tuple[str, str]:
    compact = normalize_ws(page_text)
    notes: list[str] = []

    for team_name in team_names(team_code):
        market_regex = re.compile(
            rf"{re.escape(team_name)}\s+Antal mål\s*-\s*over/under(?P<body>.{{0,2500}}?)(?=(?:[A-ZÆØÅ][\wÆØÅæøå.' -]{{1,40}}\s+Antal mål\s*-\s*over/under)|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = market_regex.search(compact)
        if not match:
            notes.append(f"market_not_found:{team_name}")
            continue

        body = match.group("body")
        under_patterns = [
            rf"(?:U|Under)\s*0[,.]5\s+{odds_pattern()}",
            rf"0[,.]5\s+(?:U|Under)\s+{odds_pattern()}",
        ]
        for pattern in under_patterns:
            under_match = re.search(pattern, body, flags=re.IGNORECASE)
            if under_match:
                return decimal_odds(under_match.group(1)), f"matched:{team_name}"

        notes.append(f"under_0_5_not_found:{team_name}")

    return "", "; ".join(notes)


async def click_if_visible(page: Any, labels: list[str], timeout_ms: int = 1500) -> str:
    for label in labels:
        locators = [
            page.get_by_role("button", name=re.compile(label, re.I)),
            page.get_by_text(re.compile(label, re.I)),
        ]
        for locator in locators:
            try:
                count = await locator.count()
                for i in range(min(count, 20)):
                    candidate = locator.nth(i)
                    try:
                        if await candidate.is_visible(timeout=500):
                            await candidate.click(timeout=timeout_ms)
                            return label
                    except Exception:
                        continue
            except Exception:
                continue
    return ""


async def accept_cookies(page: Any) -> str:
    return await click_if_visible(
        page,
        [
            "Accepter alle",
            "Acceptér alle",
            "Tillad alle",
            "Godkend alle",
            "OK",
            "Accepter",
            "Acceptér",
        ],
        timeout_ms=1200,
    )


async def open_ou_tab(page: Any) -> str:
    clicked = await click_if_visible(page, ["O/U Mål", "Over/Under", "Over/under", "Mål"], timeout_ms=1500)
    if clicked:
        try:
            await page.wait_for_timeout(1200)
        except Exception:
            pass
    return clicked


async def save_debug(page: Any, row: dict[str, str], reason: str) -> list[str]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    safe_match_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", txt(row.get("match_id")) or "unknown")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DEBUG_DIR / f"match_{safe_match_id}_{reason}_{stamp}"
    screenshot_path = base.with_suffix(".png")
    html_path = base.with_suffix(".html")
    files: list[str] = []
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
        files.append(str(screenshot_path.relative_to(PROJECT_ROOT)))
    except Exception:
        pass
    try:
        html_path.write_text(await page.content(), encoding="utf-8")
        files.append(str(html_path.relative_to(PROJECT_ROOT)))
    except Exception:
        pass
    return files


async def scrape_row(browser: Any, row: dict[str, str], args: argparse.Namespace) -> ScrapeResult:
    context = await browser.new_context(
        locale="da-DK",
        timezone_id="Europe/Copenhagen",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
    )
    page = await context.new_page()
    debug_files: list[str] = []
    note_parts: list[str] = []

    try:
        await page.goto(row["oddset_url"], wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(2500)

        cookie = await accept_cookies(page)
        if cookie:
            note_parts.append(f"cookie_clicked:{cookie}")
            await page.wait_for_timeout(800)

        tab = await open_ou_tab(page)
        if tab:
            note_parts.append(f"tab_clicked:{tab}")

        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            note_parts.append("networkidle_timeout")

        page_text = await page.locator("body").inner_text(timeout=20_000)
        home_under, home_note = extract_under_0_5_from_market(page_text, row["home"])
        away_under, away_note = extract_under_0_5_from_market(page_text, row["away"])

        if home_note:
            note_parts.append(f"home:{home_note}")
        if away_note:
            note_parts.append(f"away:{away_note}")

        if home_under and away_under:
            status = "ok"
        elif home_under or away_under:
            status = "partial"
        else:
            status = "missing_market"

        if status != "ok" or args.debug:
            debug_files = await save_debug(page, row, status)
            if debug_files:
                note_parts.append("debug=" + "|".join(debug_files))

        return ScrapeResult(
            home_under=home_under,
            away_under=away_under,
            status=status,
            note="; ".join(part for part in note_parts if part),
            debug_files=debug_files,
        )
    except Exception as exc:
        debug_files = await save_debug(page, row, "error")
        return ScrapeResult(
            status="missing_market",
            note=f"error:{type(exc).__name__}:{exc}; debug={'|'.join(debug_files)}",
            debug_files=debug_files,
        )
    finally:
        await context.close()


async def run(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[str]]:
    from playwright.async_api import async_playwright

    rows = read_input(args.limit)
    output_rows: list[dict[str, str]] = []
    all_debug_files: list[str] = []
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless, slow_mo=args.slow_ms)
        try:
            for row in rows:
                result = await scrape_row(browser, row, args)
                if result.debug_files:
                    all_debug_files.extend(result.debug_files)

                home_under = result.home_under
                away_under = result.away_under
                output_rows.append(
                    {
                        "match_id": txt(row.get("match_id")),
                        "home": txt(row.get("home")),
                        "away": txt(row.get("away")),
                        "kickoff_dk": txt(row.get("kickoff_dk")),
                        "home_clean_sheet_odds": away_under,
                        "away_clean_sheet_odds": home_under,
                        "home_team_under_0_5_odds": home_under,
                        "away_team_under_0_5_odds": away_under,
                        "source": SOURCE,
                        "scraped_at": scraped_at,
                        "oddset_url": txt(row.get("oddset_url")),
                        "status": result.status,
                        "note": result.note,
                    }
                )
        finally:
            await browser.close()

    return output_rows, all_debug_files


def write_output(rows: list[dict[str, str]]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--slow-ms", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import asyncio

        rows, debug_files = asyncio.run(run(args))
    except ModuleNotFoundError as exc:
        print(f"FEJL: Mangler Python Playwright ({exc}). Installer/kør fx: python -m pip install playwright && python -m playwright install chromium")
        return 1

    write_output(rows)

    print(f"Skrevet: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Kampe forsøgt: {len(rows)}")
    for row in rows:
        print(
            f"- match {row['match_id']} {row['home']}-{row['away']}: "
            f"{row['status']} | home_cs={row['home_clean_sheet_odds'] or '-'} | "
            f"away_cs={row['away_clean_sheet_odds'] or '-'}"
        )
    if debug_files:
        print("Debug-filer:")
        for path in debug_files:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
