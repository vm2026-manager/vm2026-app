from pathlib import Path
from datetime import datetime
import csv
import json
import shutil

DATA = Path("data")

MATCH_ODDS = DATA / "match_odds.csv"
MATCH_ODDS_PROBS = DATA / "match_odds_probs.csv"
FRESHNESS = DATA / "data_freshness.json"

FETCHED_AT = "2026-06-14T12:31:07+02:00"
FETCHED_LABEL = "14.06 kl. 12:31"
SOURCE = "Unibet"

ODDS = [
    ("GER", "CUW", 1.05, 23.00, 46.00),
    ("NED", "JPN", 2.08, 3.60, 3.85),
    ("CIV", "ECU", 3.40, 2.85, 2.63),
    ("SWE", "TUN", 1.96, 3.50, 4.35),
    ("ESP", "CPV", 1.10, 13.00, 26.00),
    ("BEL", "EGY", 1.62, 4.00, 6.25),
    ("KSA", "URU", 8.50, 4.60, 1.46),
    ("IRN", "NZL", 1.86, 3.50, 4.90),
    ("FRA", "SEN", 1.48, 4.60, 8.00),
    ("IRQ", "NOR", 17.00, 7.50, 1.20),

    ("ARG", "ALG", 1.38, 5.00, 10.00),
    ("AUT", "JOR", 1.30, 6.10, 10.00),
    ("POR", "COD", 1.29, 5.80, 12.00),
    ("ENG", "CRO", 1.70, 3.15, 5.50),
    ("GHA", "PAN", 2.14, 3.40, 3.75),
    ("UZB", "COL", 9.00, 4.70, 1.42),
    ("CZE", "RSA", 1.74, 4.00, 5.00),
    ("SUI", "BIH", 1.60, 4.10, 6.40),
    ("CAN", "QAT", 1.27, 6.00, 14.00),
    ("MEX", "KOR", 2.00, 3.50, 4.10),
    ("USA", "AUS", 1.60, 4.35, 5.80),
    ("SCO", "MAR", 5.30, 3.60, 1.78),
    ("BRA", "HAI", 1.12, 11.00, 26.00),
    ("TUR", "PAR", 2.08, 3.40, 4.00),
    ("NED", "SWE", 1.68, 4.10, 5.60),
    ("GER", "CIV", 1.61, 4.20, 6.00),
    ("ECU", "CUW", 1.22, 6.75, 17.00),
    ("TUN", "JPN", 5.60, 3.80, 1.73),
    ("ESP", "KSA", 1.13, 9.50, 29.00),
    ("BEL", "IRN", 1.41, 4.90, 9.00),
    ("URU", "CPV", 1.43, 4.70, 8.50),
    ("NZL", "EGY", 5.40, 3.70, 1.75),
    ("ARG", "AUT", 1.64, 4.00, 5.80),

    ("FRA", "IRQ", 1.13, 10.50, 26.00),
    ("NOR", "SEN", 2.18, 3.55, 3.50),
    ("JOR", "ALG", 7.00, 4.50, 1.52),
    ("POR", "UZB", 1.28, 6.25, 13.00),
    ("ENG", "GHA", 1.36, 5.10, 10.00),
    ("PAN", "CRO", 6.75, 4.00, 1.58),
    ("COL", "COD", 1.49, 4.30, 8.00),
    ("BIH", "QAT", 1.54, 4.20, 7.00),
    ("SUI", "CAN", 2.20, 3.45, 3.65),
    ("MAR", "HAI", 1.35, 5.30, 10.50),
    ("SCO", "BRA", 7.00, 4.90, 1.44),
    ("RSA", "KOR", 6.25, 4.10, 1.60),
    ("CZE", "MEX", 5.00, 3.65, 1.80),
    ("CUW", "CIV", 15.00, 6.40, 1.25),
    ("ECU", "GER", 5.00, 3.85, 1.76),
    ("JPN", "SWE", 2.15, 3.50, 3.75),
    ("TUN", "NED", 7.00, 4.60, 1.52),
    ("PAR", "AUS", 2.23, 3.35, 3.55),
    ("TUR", "USA", 2.80, 3.75, 2.55),
    ("NOR", "FRA", 4.40, 3.75, 1.87),
    ("SEN", "IRQ", 1.44, 4.60, 8.50),
    ("CPV", "KSA", 2.55, 3.35, 2.85),
    ("URU", "ESP", 5.40, 3.95, 1.65),
    ("EGY", "IRN", 2.28, 3.15, 3.70),

    ("NZL", "BEL", 10.50, 5.80, 1.32),
    ("CRO", "GHA", 1.67, 3.80, 5.60),
    ("PAN", "ENG", 10.50, 5.40, 1.32),
    ("COL", "POR", 3.60, 3.35, 2.16),
    ("COD", "UZB", 2.35, 3.25, 3.20),
    ("ALG", "AUT", 3.50, 3.35, 2.23),
    ("JOR", "ARG", 15.00, 7.00, 1.21),
]

def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            path,
            path.with_name(f"{path.stem}.backup_before_repair_unibet_20260614_1231_{stamp}{path.suffix}")
        )

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])

def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def add_fields(fields, extra):
    out = list(fields)
    for f in extra:
        if f not in out:
            out.append(f)
    return out

def clean(v):
    return "" if v is None else str(v).strip()

def fnum(v):
    return float(str(v).replace(",", "."))

backup(MATCH_ODDS)
backup(MATCH_ODDS_PROBS)
backup(FRESHNESS)

rows, fields = read_csv(MATCH_ODDS)
fields = add_fields(fields, [
    "match_id", "home", "away", "kickoff_dk",
    "home_win_odds", "draw_odds", "away_win_odds",
    "source", "odds_fetched_at", "odds_fetched_label",
    "home_team_id", "away_team_id",
])

# Først fjern de dårlige Unibet-rækker fra den tidligere kørsel.
before = len(rows)
rows = [
    r for r in rows
    if not (
        clean(r.get("source")) == SOURCE
        and clean(r.get("odds_fetched_label")) == FETCHED_LABEL
    )
]
removed = before - len(rows)

# Byg fixture-lookup fra eksisterende rækker med korrekt home/away.
fixture_by_pair = {}
fixture_by_match_id = {}

for r in rows:
    home = clean(r.get("home")).upper()
    away = clean(r.get("away")).upper()
    if not home or not away:
        continue

    key = (home, away)
    if key not in fixture_by_pair:
        fixture_by_pair[key] = {
            "match_id": clean(r.get("match_id")),
            "kickoff_dk": clean(r.get("kickoff_dk")),
        }

    mid = clean(r.get("match_id"))
    if mid and mid not in fixture_by_match_id:
        fixture_by_match_id[mid] = {
            "home": home,
            "away": away,
            "kickoff_dk": clean(r.get("kickoff_dk")),
        }

inserted = 0
missing = []

for home, away, h, d, a in ODDS:
    key = (home, away)
    fixture = fixture_by_pair.get(key)

    if not fixture:
        missing.append(f"{home}-{away}")
        continue

    r = {f: "" for f in fields}
    r["match_id"] = fixture["match_id"]
    r["home"] = home
    r["away"] = away
    r["kickoff_dk"] = fixture["kickoff_dk"]
    r["home_team_id"] = home
    r["away_team_id"] = away
    r["home_win_odds"] = h
    r["draw_odds"] = d
    r["away_win_odds"] = a
    r["source"] = SOURCE
    r["odds_fetched_at"] = FETCHED_AT
    r["odds_fetched_label"] = FETCHED_LABEL
    rows.append(r)
    inserted += 1

write_csv(MATCH_ODDS, rows, fields)

# Rebuild match_odds_probs med både home/away og team_id-felter.
prob_fields = [
    "match_id", "home", "away", "kickoff_dk",
    "home_team_id", "away_team_id",
    "home_win_odds", "draw_odds", "away_win_odds",
    "home_win_prob", "draw_prob", "away_win_prob",
    "source", "odds_fetched_at", "odds_fetched_label",
]

prob_rows = []

for r in rows:
    try:
        h = fnum(r.get("home_win_odds"))
        d = fnum(r.get("draw_odds"))
        a = fnum(r.get("away_win_odds"))
    except Exception:
        continue

    if h <= 1 or d <= 1 or a <= 1:
        continue

    inv_h, inv_d, inv_a = 1 / h, 1 / d, 1 / a
    total = inv_h + inv_d + inv_a

    home = clean(r.get("home")).upper()
    away = clean(r.get("away")).upper()

    prob_rows.append({
        "match_id": clean(r.get("match_id")),
        "home": home,
        "away": away,
        "kickoff_dk": clean(r.get("kickoff_dk")),
        "home_team_id": clean(r.get("home_team_id")) or home,
        "away_team_id": clean(r.get("away_team_id")) or away,
        "home_win_odds": h,
        "draw_odds": d,
        "away_win_odds": a,
        "home_win_prob": inv_h / total,
        "draw_prob": inv_d / total,
        "away_win_prob": inv_a / total,
        "source": clean(r.get("source")),
        "odds_fetched_at": clean(r.get("odds_fetched_at")),
        "odds_fetched_label": clean(r.get("odds_fetched_label")),
    })

write_csv(MATCH_ODDS_PROBS, prob_rows, prob_fields)

fresh = {}
if FRESHNESS.exists():
    try:
        fresh = json.loads(FRESHNESS.read_text(encoding="utf-8"))
    except Exception:
        fresh = {}

fresh["unibet_odds_fetched_at"] = FETCHED_AT
fresh["unibet_odds_fetched_label"] = FETCHED_LABEL
fresh["match_odds_source"] = SOURCE
fresh["match_odds_updated_at"] = datetime.now().isoformat(timespec="seconds")
FRESHNESS.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")

print("OK: Repareret Unibet-odds")
print(f"Fjernede dårlige Unibet-rækker: {removed}")
print(f"Indsatte korrekte Unibet-rækker: {inserted}")
print(f"Manglende fixture-match: {len(missing)}")
if missing:
    for x in missing:
        print(f"- {x}")
print(f"match_odds.csv rækker: {len(rows)}")
print(f"match_odds_probs.csv rækker: {len(prob_rows)}")
print(f"Kilde: {SOURCE} · hentet {FETCHED_LABEL}")