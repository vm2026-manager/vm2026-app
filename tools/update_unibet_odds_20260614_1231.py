from pathlib import Path
from datetime import datetime
import csv
import json
import shutil

ROOT = Path(".")
DATA = ROOT / "data"

MATCH_ODDS = DATA / "match_odds.csv"
MATCH_ODDS_PROBS = DATA / "match_odds_probs.csv"
FIXTURES = DATA / "fixtures_group.csv"
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

def backup(path: Path):
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.stem}.backup_before_unibet_20260614_1231_{stamp}{path.suffix}"))

def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])

def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def add_fields(fieldnames, fields):
    out = list(fieldnames)
    for field in fields:
        if field not in out:
            out.append(field)
    return out

def fixture_lookup():
    rows, cols = read_csv(FIXTURES)
    home_cols = ["home_team_id", "home_team", "home", "home_code", "home_team_code"]
    away_cols = ["away_team_id", "away_team", "away", "away_code", "away_team_code"]
    id_cols = ["match_id", "fixture_id", "id"]
    ko_cols = ["kickoff", "kickoff_utc", "kickoff_time", "start_time"]

    def first(cols2):
        return next((c for c in cols2 if c in cols), None)

    hc, ac, ic, kc = first(home_cols), first(away_cols), first(id_cols), first(ko_cols)
    lookup = {}
    if not hc or not ac:
        return lookup

    for r in rows:
        key = (str(r.get(hc, "")).strip().upper(), str(r.get(ac, "")).strip().upper())
        lookup[key] = {
            "match_id": r.get(ic, "") if ic else "",
            "kickoff": r.get(kc, "") if kc else "",
        }
    return lookup

backup(MATCH_ODDS)
backup(MATCH_ODDS_PROBS)
backup(FRESHNESS)

fixtures = fixture_lookup()
rows, fields = read_csv(MATCH_ODDS)

needed = [
    "match_id", "home_team_id", "away_team_id",
    "home_win_odds", "draw_odds", "away_win_odds",
    "source", "odds_fetched_at", "odds_fetched_label",
]
fields = add_fields(fields, needed)

index = {}
for i, r in enumerate(rows):
    key = (str(r.get("home_team_id", "")).strip().upper(), str(r.get("away_team_id", "")).strip().upper())
    if key != ("", ""):
        index[key] = i

updated = 0
inserted = 0

for home, away, h, d, a in ODDS:
    key = (home, away)
    fx = fixtures.get(key, {})
    if key in index:
        r = rows[index[key]]
        updated += 1
    else:
        r = {f: "" for f in fields}
        rows.append(r)
        index[key] = len(rows) - 1
        inserted += 1

    r["match_id"] = r.get("match_id") or fx.get("match_id", "")
    r["home_team_id"] = home
    r["away_team_id"] = away
    r["home_win_odds"] = h
    r["draw_odds"] = d
    r["away_win_odds"] = a
    r["source"] = SOURCE
    r["odds_fetched_at"] = FETCHED_AT
    r["odds_fetched_label"] = FETCHED_LABEL

write_csv(MATCH_ODDS, rows, fields)

# Rebuild simple normalized probability file from match_odds.csv
prob_rows = []
for r in rows:
    try:
        h = float(str(r.get("home_win_odds", "")).replace(",", "."))
        d = float(str(r.get("draw_odds", "")).replace(",", "."))
        a = float(str(r.get("away_win_odds", "")).replace(",", "."))
    except ValueError:
        continue

    inv_h, inv_d, inv_a = 1 / h, 1 / d, 1 / a
    total = inv_h + inv_d + inv_a

    prob_rows.append({
        "match_id": r.get("match_id", ""),
        "home_team_id": r.get("home_team_id", ""),
        "away_team_id": r.get("away_team_id", ""),
        "home_win_odds": h,
        "draw_odds": d,
        "away_win_odds": a,
        "home_win_prob": inv_h / total,
        "draw_prob": inv_d / total,
        "away_win_prob": inv_a / total,
        "source": r.get("source", SOURCE),
        "odds_fetched_at": r.get("odds_fetched_at", FETCHED_AT),
        "odds_fetched_label": r.get("odds_fetched_label", FETCHED_LABEL),
    })

write_csv(
    MATCH_ODDS_PROBS,
    prob_rows,
    [
        "match_id", "home_team_id", "away_team_id",
        "home_win_odds", "draw_odds", "away_win_odds",
        "home_win_prob", "draw_prob", "away_win_prob",
        "source", "odds_fetched_at", "odds_fetched_label",
    ],
)

fresh = {}
if FRESHNESS.exists():
    fresh = json.loads(FRESHNESS.read_text(encoding="utf-8"))

fresh["unibet_odds_fetched_at"] = FETCHED_AT
fresh["unibet_odds_fetched_label"] = FETCHED_LABEL
fresh["match_odds_source"] = SOURCE
fresh["match_odds_updated_at"] = datetime.now().isoformat(timespec="seconds")

FRESHNESS.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")

print("OK: Unibet-odds opdateret")
print(f"Odds i script: {len(ODDS)}")
print(f"Opdaterede rækker: {updated}")
print(f"Nye rækker: {inserted}")
print(f"match_odds_probs rækker: {len(prob_rows)}")
print(f"Kilde: {SOURCE} · hentet {FETCHED_LABEL}")