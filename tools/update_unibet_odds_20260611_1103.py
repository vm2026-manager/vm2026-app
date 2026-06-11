import csv
import json
from pathlib import Path
from datetime import datetime

DATA = Path("data")

MATCH_ODDS = DATA / "match_odds.csv"
MATCH_ODDS_PROBS = DATA / "match_odds_probs.csv"
FIXTURES = DATA / "fixtures_group.csv"
FRESHNESS = DATA / "data_freshness.json"

FETCHED_AT_ISO = "2026-06-11T11:03:03+02:00"
FETCHED_LABEL = "11.06 kl. 11:03"
SOURCE = "Unibet"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path in [MATCH_ODDS, MATCH_ODDS_PROBS, FRESHNESS]:
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup_before_unibet_20260611_1103_{timestamp}{path.suffix}")
        backup.write_text(path.read_text(encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8"), encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8")
        print("Backup:", backup)

TEAM = {
    "Mexico": "MEX",
    "Sydafrika": "RSA",
    "Sydkorea": "KOR",
    "Tjekkiet": "CZE",
    "Canada": "CAN",
    "Bosnien": "BIH",
    "USA": "USA",
    "Paraguay": "PAR",
    "Qatar": "QAT",
    "Schweiz": "SUI",
    "Brasilien": "BRA",
    "Marokko": "MAR",
    "Haiti": "HAI",
    "Skotland": "SCO",
    "Australia": "AUS",
    "Australien": "AUS",
    "Tyrkiet": "TUR",
    "Tyskland": "GER",
    "Curacao": "CUW",
    "Curaçao": "CUW",
    "Holland": "NED",
    "Japan": "JPN",
    "Elfenbenskysten": "CIV",
    "Ecuador": "ECU",
    "Sverige": "SWE",
    "Tunesien": "TUN",
    "Spanien": "ESP",
    "Kap Verde": "CPV",
    "Belgien": "BEL",
    "Egypten": "EGY",
    "Saudi Arabien": "KSA",
    "Uruguay": "URU",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Frankrig": "FRA",
    "Senegal": "SEN",
    "Irak": "IRQ",
    "Norge": "NOR",
    "Argentina": "ARG",
    "Algeriet": "ALG",
    "Østrig": "AUT",
    "Jordan": "JOR",
    "Portugal": "POR",
    "DR Congo": "COD",
    "Congo DR": "COD",
    "England": "ENG",
    "Kroatien": "CRO",
    "Ghana": "GHA",
    "Panama": "PAN",
    "Uzbekistan": "UZB",
    "Colombia": "COL",
}

# Odds aflæst fra Unibet-PDF/screenshot hentet 11.06 kl. 11:03.
# Format: dato/tid er kun til audit; match_id/kickoff hentes fra fixtures_group.csv.
UNIBET_ODDS = [
    ("Mexico", "Sydafrika", 1.42, 4.70, 9.00),
    ("Sydkorea", "Tjekkiet", 2.75, 3.15, 2.95),
    ("Canada", "Bosnien", 1.85, 3.50, 5.00),
    ("USA", "Paraguay", 1.95, 3.40, 4.50),
    ("Qatar", "Schweiz", 16.00, 7.00, 1.23),
    ("Brasilien", "Marokko", 1.65, 3.90, 6.10),
    ("Haiti", "Skotland", 6.40, 4.35, 1.58),
    ("Australia", "Tyrkiet", 5.60, 3.85, 1.70),
    ("Tyskland", "Curacao", 1.04, 25.00, 51.00),
    ("Holland", "Japan", 2.10, 3.70, 3.55),
    ("Elfenbenskysten", "Ecuador", 3.75, 2.80, 2.45),
    ("Sverige", "Tunesien", 1.93, 3.50, 4.50),
    ("Spanien", "Kap Verde", 1.11, 12.50, 26.00),
    ("Belgien", "Egypten", 1.62, 4.00, 6.10),
    ("Saudi Arabien", "Uruguay", 8.00, 4.70, 1.47),
    ("Iran", "New Zealand", 1.87, 3.55, 4.70),
    ("Frankrig", "Senegal", 1.45, 4.60, 8.00),
    ("Irak", "Norge", 17.00, 7.50, 1.21),

    ("Argentina", "Algeriet", 1.40, 4.90, 10.00),
    ("Østrig", "Jordan", 1.32, 5.75, 11.00),
    ("Portugal", "DR Congo", 1.29, 6.00, 12.00),
    ("England", "Kroatien", 1.74, 3.80, 5.20),
    ("Ghana", "Panama", 2.12, 3.50, 3.80),
    ("Uzbekistan", "Colombia", 9.50, 4.80, 1.42),
    ("Tjekkiet", "Sydafrika", 2.02, 3.40, 4.35),
    ("Schweiz", "Bosnien", 1.61, 4.00, 6.40),
    ("Canada", "Qatar", 1.32, 5.30, 12.00),
    ("Mexico", "Sydkorea", 1.83, 3.60, 4.90),
    ("USA", "Australia", 1.81, 3.80, 4.70),
    ("Skotland", "Marokko", 4.30, 3.40, 2.02),
    ("Brasilien", "Haiti", 1.09, 13.00, 34.00),
    ("Tyrkiet", "Paraguay", 2.28, 3.30, 3.55),
    ("Holland", "Sverige", 1.68, 4.10, 5.60),
    ("Tyskland", "Elfenbenskysten", 1.61, 4.30, 6.10),
    ("Ecuador", "Curacao", 1.22, 6.75, 17.00),
    ("Tunesien", "Japan", 5.30, 3.70, 1.76),
    ("Spanien", "Saudi Arabien", 1.12, 10.00, 30.00),
    ("Belgien", "Iran", 1.41, 4.90, 9.00),
    ("Uruguay", "Kap Verde", 1.47, 4.50, 8.00),
    ("New Zealand", "Egypten", 5.40, 3.70, 1.75),
    ("Argentina", "Østrig", 1.71, 3.85, 5.50),
    ("Frankrig", "Irak", 1.14, 10.00, 23.00),
    ("Norge", "Senegal", 2.18, 3.55, 3.50),
    ("Jordan", "Algeriet", 6.50, 4.25, 1.56),
    ("Portugal", "Uzbekistan", 1.27, 6.25, 12.00),
    ("England", "Ghana", 1.36, 5.10, 10.00),
    ("Panama", "Kroatien", 7.00, 4.20, 1.54),
    ("Colombia", "DR Congo", 1.53, 4.20, 7.50),
    ("Bosnien", "Qatar", 1.65, 3.85, 6.25),
    ("Schweiz", "Canada", 2.20, 3.45, 3.65),
    ("Marokko", "Haiti", 1.35, 5.30, 10.50),
    ("Skotland", "Brasilien", 8.00, 4.50, 1.49),
    ("Sydafrika", "Sydkorea", 4.25, 3.50, 1.98),

    ("Tjekkiet", "Mexico", 5.00, 3.65, 1.84),
    ("Curacao", "Elfenbenskysten", 15.00, 6.40, 1.25),
    ("Ecuador", "Tyskland", 5.00, 3.85, 1.76),
    ("Japan", "Sverige", 2.10, 3.50, 3.75),
    ("Tunesien", "Holland", 7.00, 4.60, 1.52),
    ("Paraguay", "Australia", 2.23, 3.35, 3.55),
    ("Tyrkiet", "USA", 2.75, 3.50, 2.63),
    ("Norge", "Frankrig", 4.40, 3.75, 1.87),
    ("Senegal", "Irak", 1.46, 4.50, 8.00),
    ("Kap Verde", "Saudi Arabien", 2.70, 3.40, 2.80),
    ("Uruguay", "Spanien", 5.80, 4.00, 1.68),
    ("Egypten", "Iran", 2.28, 3.15, 3.70),
    ("New Zealand", "Belgien", 10.50, 5.80, 1.32),
    ("Kroatien", "Ghana", 1.68, 3.90, 6.00),
    ("Panama", "England", 11.00, 5.75, 1.30),
    ("Colombia", "Portugal", 3.70, 3.40, 2.15),
    ("DR Congo", "Uzbekistan", 2.38, 3.30, 3.30),
    ("Algeriet", "Østrig", 3.60, 3.35, 2.20),
    ("Jordan", "Argentina", 18.00, 7.00, 1.22),
]

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def fmt(x):
    return f"{float(x):.2f}".rstrip("0").rstrip(".")

def fair_probs(h, x, a):
    inv_h = 1 / float(h)
    inv_x = 1 / float(x)
    inv_a = 1 / float(a)
    total = inv_h + inv_x + inv_a
    return inv_h / total, inv_x / total, inv_a / total

fixtures = read_csv(FIXTURES)
fixture_by_pair = {(r["home"], r["away"]): r for r in fixtures}

new_unibet_rows = []
missing = []

for home_name, away_name, h, x, a in UNIBET_ODDS:
    home = TEAM[home_name]
    away = TEAM[away_name]
    fixture = fixture_by_pair.get((home, away))

    if not fixture:
        missing.append((home_name, away_name, home, away))
        continue

    new_unibet_rows.append({
        "match_id": fixture["match_id"],
        "home": home,
        "away": away,
        "kickoff_dk": fixture["kickoff_dk"],
        "home_win_odds": fmt(h),
        "draw_odds": fmt(x),
        "away_win_odds": fmt(a),
        "home_clean_sheet_odds": "",
        "away_clean_sheet_odds": "",
        "over_2_5_odds": "",
        "under_2_5_odds": "",
        "source": SOURCE,
        "odds_fetched_at": FETCHED_AT_ISO,
        "odds_fetched_label": FETCHED_LABEL,
    })

if missing:
    print("Mangler fixture-match for:")
    for m in missing:
        print(m)
    raise SystemExit("Stopper uden at skrive, fordi ikke alle Unibet-rækker kunne matches.")

if len(new_unibet_rows) != 72:
    raise SystemExit(f"Forventede 72 Unibet-rækker, fik {len(new_unibet_rows)}")

# 1) match_odds.csv: fjern gamle Unibet-rækker, tilføj nye
match_rows = read_csv(MATCH_ODDS)
match_fields = list(match_rows[0].keys()) if match_rows else [
    "match_id", "home", "away", "kickoff_dk",
    "home_win_odds", "draw_odds", "away_win_odds",
    "home_clean_sheet_odds", "away_clean_sheet_odds",
    "over_2_5_odds", "under_2_5_odds", "source"
]

for f in ["odds_fetched_at", "odds_fetched_label"]:
    if f not in match_fields:
        match_fields.append(f)

kept = [r for r in match_rows if str(r.get("source", "")).strip().lower() != "unibet"]

# Sørg for samme felter i alle rækker
for r in kept:
    for f in match_fields:
        r.setdefault(f, "")

for r in new_unibet_rows:
    for f in match_fields:
        r.setdefault(f, "")

write_csv(MATCH_ODDS, kept + new_unibet_rows, match_fields)

# 2) match_odds_probs.csv: brug Unibet som app/tooltip-odds og beregn fair probs
probs_rows = []
for r in new_unibet_rows:
    hp, xp, ap = fair_probs(r["home_win_odds"], r["draw_odds"], r["away_win_odds"])
    probs_rows.append({
        "match_id": r["match_id"],
        "home": r["home"],
        "away": r["away"],
        "kickoff_dk": r["kickoff_dk"],
        "home_win_odds": r["home_win_odds"],
        "draw_odds": r["draw_odds"],
        "away_win_odds": r["away_win_odds"],
        "home_clean_sheet_odds": "",
        "away_clean_sheet_odds": "",
        "over_2_5_odds": "",
        "under_2_5_odds": "",
        "home_win_prob_fair": f"{hp:.4f}",
        "draw_prob_fair": f"{xp:.4f}",
        "away_win_prob_fair": f"{ap:.4f}",
        "home_cs_prob": "",
        "away_cs_prob": "",
        "over_2_5_prob": "",
        "under_2_5_prob": "",
        "source": "Unibet",
        "odds_fetched_at": FETCHED_AT_ISO,
        "odds_fetched_label": FETCHED_LABEL,
    })

probs_fields = [
    "match_id", "home", "away", "kickoff_dk",
    "home_win_odds", "draw_odds", "away_win_odds",
    "home_clean_sheet_odds", "away_clean_sheet_odds",
    "over_2_5_odds", "under_2_5_odds",
    "home_win_prob_fair", "draw_prob_fair", "away_win_prob_fair",
    "home_cs_prob", "away_cs_prob",
    "over_2_5_prob", "under_2_5_prob",
    "source", "odds_fetched_at", "odds_fetched_label",
]
write_csv(MATCH_ODDS_PROBS, probs_rows, probs_fields)

# 3) freshness
fresh = {}
if FRESHNESS.exists():
    try:
        fresh = json.loads(FRESHNESS.read_text(encoding="utf-8"))
    except Exception:
        fresh = {}

fresh["unibet_odds_fetched_at"] = FETCHED_AT_ISO
fresh["unibet_odds_fetched_label"] = FETCHED_LABEL
fresh["match_odds_source"] = "Unibet"
fresh["match_odds_updated_at"] = datetime.now().isoformat(timespec="seconds")

FRESHNESS.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")

print("Unibet-odds opdateret.")
print("Rækker i match_odds.csv:", len(kept + new_unibet_rows))
print("Unibet-rækker:", len(new_unibet_rows))
print("Rækker i match_odds_probs.csv:", len(probs_rows))
print("Kilde:", SOURCE)
print("Hentet:", FETCHED_LABEL)
print()
print("Stikprøve:")
for r in new_unibet_rows[:5]:
    print(r["match_id"], r["home"], r["away"], r["home_win_odds"], r["draw_odds"], r["away_win_odds"])
