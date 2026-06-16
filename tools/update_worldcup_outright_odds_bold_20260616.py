from pathlib import Path
from datetime import datetime
import csv
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTRIGHT = ROOT / "data" / "worldcup_outright_odds.csv"

FETCHED_LABEL = "16.06 kl. 12:20"
FETCHED_AT = "2026-06-16T12:20:52+02:00"
SOURCE_NOTE = "Bold.dk odds-sammenligning, VM 2026 vinderodds"

# Format:
# team_id, team_name_da, bookmaker_1, bookmaker_2, bwin, bookmaker_4
# Kolonnenavne er holdt neutrale, fordi PDF'en viser en sammenligning fra flere bookmakere.
OUTRIGHT_UPDATES = [
    ("FRA", "Frankrig", 5.40, 5.40, 5.00, 5.50),
    ("ESP", "Spanien", 5.55, 5.55, 6.00, 6.00),
    ("ENG", "England", 7.95, 7.95, 8.00, 8.50),
    ("POR", "Portugal", 8.40, 8.40, 8.00, 8.50),
    ("BRA", "Brasilien", 10.90, 10.90, 10.00, 12.00),
    ("ARG", "Argentina", 10.80, 10.80, 10.00, 11.00),
    ("GER", "Tyskland", 15.40, 15.40, 13.00, 15.00),
    ("NED", "Holland", 18.10, 18.10, 19.00, 19.00),
    ("NOR", "Norge", 29.80, 29.80, 34.00, 34.00),
    ("USA", "USA", 40.00, 40.00, 34.00, 41.00),
    ("BEL", "Belgien", 40.00, 40.00, 34.00, 41.00),
    ("MAR", "Marokko", 40.00, 40.00, 41.00, 41.00),
    ("MEX", "Mexico", 50.00, 50.00, 51.00, 41.00),
    ("COL", "Colombia", 41.00, 41.00, 41.00, 51.00),
    ("JPN", "Japan", 50.00, 50.00, 51.00, 46.00),
    ("SUI", "Schweiz", 79.00, 79.00, 67.00, 91.00),
    ("URU", "Uruguay", 67.00, 67.00, 67.00, 81.00),
    ("SWE", "Sverige", 88.00, 88.00, 81.00, 101.00),
    ("SEN", "Senegal", 120.00, 120.00, 81.00, 121.00),
    ("CRO", "Kroatien", 81.00, 81.00, 81.00, 101.00),
    ("AUS", "Australien", 86.00, 86.00, 101.00, 151.00),
    ("CIV", "Elfenbenskysten", 190.00, 190.00, 101.00, 141.00),
    ("ECU", "Ecuador", 150.00, 150.00, 101.00, 131.00),

    ("TUR", "Tyrkiet", 150.00, 150.00, 126.00, 151.00),
    ("CAN", "Canada", 135.00, 135.00, 151.00, 301.00),
    ("AUT", "Østrig", 150.00, 150.00, 151.00, 151.00),
    ("SCO", "Skotland", 195.00, 195.00, 151.00, 201.00),
    ("KOR", "Sydkorea", None, None, None, 201.00),
    ("EGY", "Egypten", 250.00, 250.00, 201.00, 401.00),
    ("BIH", "Bosnien-Hercegovina", 300.00, 300.00, 251.00, None),
    ("DZA", "Algeriet", 375.00, 375.00, 251.00, 401.00),
    ("CZE", "Tjekkiet", 475.00, 475.00, 301.00, 501.00),
    ("PAR", "Paraguay", 400.00, 400.00, 301.00, 401.00),
    ("GHA", "Ghana", 400.00, 400.00, 501.00, 501.00),
    ("NZL", "New Zealand", 975.00, 975.00, 501.00, 1001.00),
    ("IRN", "Iran", 700.00, 700.00, 501.00, 751.00),
    ("TUN", "Tunesien", 750.00, 750.00, 751.00, 751.00),
    ("KSA", "Saudi-Arabien", 925.00, 925.00, 751.00, 1001.00),
    ("COD", "DR Congo", None, None, 751.00, 751.00),
    ("RSA", "Sydafrika", 1800.00, 1800.00, 1001.00, 1001.00),
    ("QAT", "Qatar", 1700.00, 1700.00, 1001.00, 1001.00),
    ("HAI", "Haiti", 3000.00, 3000.00, 2501.00, 1001.00),
    ("CUW", "Curacao", 3100.00, 3100.00, 2501.00, 1001.00),
    ("CPV", "Kap Verde", 1400.00, 1400.00, 1001.00, 1001.00),
    ("IRQ", "Irak", 1400.00, 1400.00, 1001.00, None),
    ("JOR", "Jordan", 2200.00, 2200.00, 1001.00, 1001.00),
    ("UZB", "Usbekistan", 1400.00, 1400.00, 1001.00, 1001.00),
    ("PAN", "Panama", 1400.00, 1400.00, 1001.00, 1001.00),
]


ALIASES = {
    "FRANKRIG": "FRA", "FRANCE": "FRA",
    "SPANIEN": "ESP", "SPAIN": "ESP",
    "ENGLAND": "ENG",
    "PORTUGAL": "POR",
    "BRASILIEN": "BRA", "BRAZIL": "BRA",
    "ARGENTINA": "ARG",
    "TYSKLAND": "GER", "GERMANY": "GER",
    "HOLLAND": "NED", "NETHERLANDS": "NED",
    "NORGE": "NOR", "NORWAY": "NOR",
    "USA": "USA",
    "BELGIEN": "BEL", "BELGIUM": "BEL",
    "MAROKKO": "MAR", "MOROCCO": "MAR",
    "MEXICO": "MEX",
    "COLOMBIA": "COL",
    "JAPAN": "JPN",
    "SCHWEIZ": "SUI", "SWITZERLAND": "SUI",
    "URUGUAY": "URU",
    "SVERIGE": "SWE", "SWEDEN": "SWE",
    "SENEGAL": "SEN",
    "KROATIEN": "CRO", "CROATIA": "CRO",
    "AUSTRALIEN": "AUS", "AUSTRALIA": "AUS",
    "ELFENBENSKYSTEN": "CIV", "IVORY COAST": "CIV",
    "ECUADOR": "ECU",
    "TYRKIET": "TUR", "TURKEY": "TUR",
    "CANADA": "CAN",
    "ØSTRIG": "AUT", "AUSTRIA": "AUT",
    "SKOTLAND": "SCO", "SCOTLAND": "SCO",
    "SYDKOREA": "KOR", "SOUTH KOREA": "KOR",
    "EGYPTEN": "EGY", "EGYPT": "EGY",
    "BOSNIEN-HERCEGOVINA": "BIH", "BOSNIEN": "BIH", "BOSNIA": "BIH",
    "ALGERIET": "DZA", "ALGERIA": "DZA", "ALG": "DZA",
    "TJEKKIET": "CZE", "CZECHIA": "CZE",
    "PARAGUAY": "PAR",
    "GHANA": "GHA",
    "NEW ZEALAND": "NZL", "NEWZEALAND": "NZL",
    "IRAN": "IRN", "IRI": "IRN",
    "TUNESIEN": "TUN", "TUNISIA": "TUN",
    "SAUDI-ARABIEN": "KSA", "SAUDI ARABIEN": "KSA", "SAUDI ARABIA": "KSA",
    "DR CONGO": "COD", "DRCONGO": "COD",
    "SYDAFRIKA": "RSA", "SOUTH AFRICA": "RSA",
    "QATAR": "QAT",
    "HAITI": "HAI",
    "CURACAO": "CUW", "CURAÇAO": "CUW",
    "KAP VERDE": "CPV", "KAPVERDE": "CPV", "CAPE VERDE": "CPV",
    "IRAK": "IRQ", "IRAQ": "IRQ",
    "JORDAN": "JOR",
    "USBEKISTAN": "UZB", "UZBEKISTAN": "UZB",
    "PANAMA": "PAN",
}


def canon(value):
    raw = str(value or "").strip().upper()
    key = raw.replace("_", " ").replace("-", " ")
    compact = key.replace(" ", "")
    return ALIASES.get(raw) or ALIASES.get(key) or ALIASES.get(compact) or raw


def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.2f}"


def add_field(fieldnames, name):
    if name not in fieldnames:
        fieldnames.append(name)


def odds_values(row):
    vals = [
        row["bold_bookmaker_1_win_odds"],
        row["bold_bookmaker_2_win_odds"],
        row["bold_bwin_win_odds"],
        row["bold_bookmaker_4_win_odds"],
    ]
    out = []
    for v in vals:
        if v in ("", None):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def implied_prob_from_odds(odds):
    if not odds:
        return ""
    return f"{100.0 / float(odds):.2f}"


def main():
    if not OUTRIGHT.exists():
        raise FileNotFoundError(OUTRIGHT)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = OUTRIGHT.with_name(f"worldcup_outright_odds.backup_before_bold_update_{stamp}.csv")
    shutil.copy2(OUTRIGHT, backup)

    with OUTRIGHT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for col in [
        "team_id",
        "team_name",
        "bold_bookmaker_1_win_odds",
        "bold_bookmaker_2_win_odds",
        "bold_bwin_win_odds",
        "bold_bookmaker_4_win_odds",
        "market_min_win_odds",
        "market_max_win_odds",
        "market_avg_win_odds",
        "market_median_win_odds",
        "model_win_odds",
        "model_win_implied_pct",
        "outright_odds_source_note",
        "outright_odds_fetched_label",
        "outright_odds_fetched_at",
    ]:
        add_field(fieldnames, col)

    # Eksisterende rækker matches på team_id eller team_name.
    by_team = {}
    for row in rows:
        keys = [
            canon(row.get("team_id")),
            canon(row.get("team")),
            canon(row.get("team_name")),
            canon(row.get("country")),
            canon(row.get("land")),
        ]
        for key in keys:
            if key:
                by_team[key] = row

    updated = 0
    added = 0

    for team_id, team_name, b1, b2, bwin, b4 in OUTRIGHT_UPDATES:
        key = canon(team_id)
        row = by_team.get(key) or by_team.get(canon(team_name))

        if row is None:
            row = {field: "" for field in fieldnames}
            rows.append(row)
            added += 1

        row["team_id"] = key
        if not row.get("team_name"):
            row["team_name"] = team_name

        row["bold_bookmaker_1_win_odds"] = fmt(b1)
        row["bold_bookmaker_2_win_odds"] = fmt(b2)
        row["bold_bwin_win_odds"] = fmt(bwin)
        row["bold_bookmaker_4_win_odds"] = fmt(b4)

        vals = [
            v for v in [b1, b2, bwin, b4]
            if v is not None
        ]

        if vals:
            market_min = min(vals)
            market_max = max(vals)
            market_avg = sum(vals) / len(vals)
            market_median = statistics.median(vals)

            row["market_min_win_odds"] = fmt(market_min)
            row["market_max_win_odds"] = fmt(market_max)
            row["market_avg_win_odds"] = fmt(market_avg)
            row["market_median_win_odds"] = fmt(market_median)

            # Brug median som modelværdi, så én skæv bookmaker ikke styrer hele styrkesignalet.
            row["model_win_odds"] = fmt(market_median)
            row["model_win_implied_pct"] = implied_prob_from_odds(market_median)

            # Legacy-kompatibilitet:
            # Flere scripts har tidligere brugt unibet_win_odds som vinderodds-kolonne.
            # Vi opdaterer den til samme median, så modellen får de nye odds uden scriptændring.
            if "unibet_win_odds" in fieldnames:
                row["unibet_win_odds"] = fmt(market_median)

        row["outright_odds_source_note"] = SOURCE_NOTE
        row["outright_odds_fetched_label"] = FETCHED_LABEL
        row["outright_odds_fetched_at"] = FETCHED_AT

        updated += 1

    with OUTRIGHT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Backup:", backup)
    print("Opdaterede lande:", updated)
    print("Tilføjede rækker:", added)
    print("Samlet antal rækker:", len(rows))

    print("\nStikprøver:")
    for wanted in ["FRA", "ESP", "ENG", "GER", "NOR", "IRN", "JOR"]:
        row = next((r for r in rows if canon(r.get("team_id")) == wanted), None)
        if row:
            print(
                f"{wanted}: model_win_odds={row.get('model_win_odds')} "
                f"median={row.get('market_median_win_odds')} "
                f"kilde={row.get('outright_odds_fetched_label')}"
            )

    print("\nFærdig. Kør IKKE EV/optimizer endnu, før vi har tjekket EV-idempotens.")

if __name__ == "__main__":
    main()
