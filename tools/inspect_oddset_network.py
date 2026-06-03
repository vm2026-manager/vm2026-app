from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OUT_CSV = DATA / "oddset_network_requests_full.csv"
OUT_TXT = DATA / "oddset_network_text_responses.txt"
HAR_PATH = DATA / "oddset_network.har"

TEST_URLS = [
    "https://danskespil.dk/oddset/sports/event/9290632/fodbold/verden/vm/mexico-sydafrika",
    "https://danskespil.dk/oddset/sports/event/9654758/fodbold/verden/vm/sydkorea-tjekkiet",
]


def safe_text_response(resp) -> str:
    try:
        ctype = (resp.headers.get("content-type") or "").lower()
        if any(x in ctype for x in ["json", "text", "javascript", "html"]):
            return resp.text()[:20000]
    except Exception:
        return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--slow-ms", type=int, default=300)
    parser.add_argument("--wait-ms", type=int, default=25000)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    rows = []
    text_blocks = []
    ws_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, slow_mo=args.slow_ms)

        context = browser.new_context(
            viewport={"width": 1500, "height": 1100},
            locale="da-DK",
            record_har_path=str(HAR_PATH),
            record_har_content="attach",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        page = context.new_page()

        def on_request(req):
            rows.append({
                "kind": "request",
                "method": req.method,
                "resource_type": req.resource_type,
                "status": "",
                "url": req.url,
                "post_data": (req.post_data or "")[:2000],
                "content_type": "",
            })

        def on_response(resp):
            ctype = resp.headers.get("content-type", "")
            rows.append({
                "kind": "response",
                "method": "",
                "resource_type": "",
                "status": str(resp.status),
                "url": resp.url,
                "post_data": "",
                "content_type": ctype,
            })

            txt = safe_text_response(resp)
            if txt:
                lower = txt.lower()
                # Gem bredt, men især alt der kan relatere til kamp/odds/markets
                if any(x in lower for x in [
                    "mexico", "sydafrika", "sydkorea", "tjekkiet",
                    "9290632", "9654758", "market", "selection",
                    "odds", "under", "0.5", "event", "openbet"
                ]):
                    text_blocks.append(
                        f"\n\n===== {resp.status} {resp.url} =====\n"
                        f"content-type: {ctype}\n\n"
                        f"{txt}"
                    )

        def on_websocket(ws):
            ws_rows.append({"event": "websocket", "url": ws.url})
            rows.append({
                "kind": "websocket",
                "method": "",
                "resource_type": "websocket",
                "status": "",
                "url": ws.url,
                "post_data": "",
                "content_type": "",
            })

            def on_frame_sent(payload):
                rows.append({
                    "kind": "ws_sent",
                    "method": "",
                    "resource_type": "websocket",
                    "status": "",
                    "url": ws.url,
                    "post_data": str(payload)[:2000],
                    "content_type": "",
                })

            def on_frame_received(payload):
                rows.append({
                    "kind": "ws_received",
                    "method": "",
                    "resource_type": "websocket",
                    "status": "",
                    "url": ws.url,
                    "post_data": str(payload)[:2000],
                    "content_type": "",
                })

            ws.on("framesent", on_frame_sent)
            ws.on("framereceived", on_frame_received)

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("websocket", on_websocket)

        for url in TEST_URLS:
            print(f"Åbner: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Vent på app/scripts
            page.wait_for_timeout(args.wait_ms)

            # Prøv at trigge lazy loading
            for _ in range(4):
                page.mouse.wheel(0, 1600)
                page.wait_for_timeout(1500)

            # Prøv klik på O/U Mål hvis synlig
            for txt in ["O/U Mål", "Mål", "Alle"]:
                try:
                    loc = page.get_by_text(txt, exact=False).first
                    if loc.count() and loc.is_visible(timeout=1500):
                        loc.click(timeout=3000)
                        page.wait_for_timeout(5000)
                except Exception:
                    pass

        context.close()
        browser.close()

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["kind", "method", "resource_type", "status", "url", "post_data", "content_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUT_TXT.write_text("\n".join(text_blocks), encoding="utf-8")

    print("")
    print("Skrevet:")
    print(f"- {OUT_CSV}")
    print(f"- {OUT_TXT}")
    print(f"- {HAR_PATH}")
    print("")
    print(f"Rows: {len(rows)}")
    print(f"Text blocks: {len(text_blocks)}")
    print(f"WebSockets: {len(ws_rows)}")


if __name__ == "__main__":
    main()