from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CSV_PATH = DATA / "oddset_network_requests_full.csv"
TXT_PATH = DATA / "oddset_network_text_responses.txt"
HAR_PATH = DATA / "oddset_network.har"

OUT = DATA / "oddset_network_clues.txt"

PATTERNS = [
    "9290632", "9654758",
    "mexico", "sydafrika", "sydkorea", "tjekkiet",
    "under", "0.5", "0,5",
    "antal mål", "maal", "mål",
    "market", "selection", "odds", "price",
    "openbet", "eventmarket", "event-market",
    "sportsbook", "coupon", "offer",
]

NOISE = [
    "ensighten", "privacy", "google", "gtm", "varify",
    "png", "svg", "jpg", "woff", "css",
    "operationsmessengerservices",
    "onboarding/get",
]


def hit(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in PATTERNS)


def noisy(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in NOISE)


def snippet(text: str, pattern: str, radius: int = 600) -> str:
    low = text.lower()
    i = low.find(pattern.lower())
    if i == -1:
        return ""
    a = max(0, i - radius)
    b = min(len(text), i + len(pattern) + radius)
    return text[a:b]


parts = []

parts.append("=== CSV relevante requests/responses ===\n")

if CSV_PATH.exists():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "")
            post = row.get("post_data", "")
            alltxt = f"{url}\n{post}"
            if hit(alltxt) and not noisy(alltxt):
                parts.append(
                    f"\nKIND: {row.get('kind')}\n"
                    f"STATUS: {row.get('status')}\n"
                    f"TYPE: {row.get('resource_type')}\n"
                    f"CONTENT: {row.get('content_type')}\n"
                    f"URL: {url}\n"
                    f"POST: {post[:1500]}\n"
                )

parts.append("\n\n=== TEXT RESPONSE SNIPPETS ===\n")

if TXT_PATH.exists():
    txt = TXT_PATH.read_text(encoding="utf-8", errors="ignore")
    for pat in PATTERNS:
        snips = []
        start = 0
        low = txt.lower()
        while True:
            i = low.find(pat.lower(), start)
            if i == -1 or len(snips) >= 8:
                break
            a = max(0, i - 500)
            b = min(len(txt), i + 800)
            s = txt[a:b]
            if not noisy(s):
                snips.append(s)
            start = i + len(pat)
        if snips:
            parts.append(f"\n\n--- Pattern: {pat} ---\n")
            for n, s in enumerate(snips, 1):
                parts.append(f"\n[SNIP {n}]\n{s}\n")

parts.append("\n\n=== HAR SNIPPETS ===\n")

if HAR_PATH.exists():
    har = HAR_PATH.read_text(encoding="utf-8", errors="ignore")
    for pat in ["9290632", "9654758", "mexico", "sydafrika", "under", "0.5", "market", "selection", "odds", "openbet"]:
        snips = []
        start = 0
        low = har.lower()
        while True:
            i = low.find(pat.lower(), start)
            if i == -1 or len(snips) >= 5:
                break
            a = max(0, i - 500)
            b = min(len(har), i + 1000)
            s = har[a:b]
            if not noisy(s):
                snips.append(s)
            start = i + len(pat)
        if snips:
            parts.append(f"\n\n--- HAR Pattern: {pat} ---\n")
            for n, s in enumerate(snips, 1):
                parts.append(f"\n[HAR SNIP {n}]\n{s}\n")

OUT.write_text("\n".join(parts), encoding="utf-8")

print(f"Skrevet: {OUT}")
print("Vis første relevante linjer med:")
print(r'Get-Content ".\data\oddset_network_clues.txt" -TotalCount 220')