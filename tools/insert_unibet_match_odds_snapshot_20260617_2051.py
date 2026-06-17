import csv, shutil
from pathlib import Path

MATCH_ODDS = Path("data/match_odds.csv")
AUDIT_OUT = Path("data/unibet_match_odds_insert_snapshot_audit_20260617_2051.csv")

SOURCE = "Unibet"
BASE_LABEL = "15.06 kl. 20:06"
NEW_LABEL = "17.06 kl. 20:51"
NEW_AT = "17.06 kl. 20:51"

ODDS = {
    ("ENG","CRO"): (1.74, 3.75, 5.60),
    ("GHA","PAN"): (2.43, 3.25, 3.40),
    ("UZB","COL"): (10.00, 5.10, 1.38),
    ("CZE","RSA"): (1.74, 3.90, 5.10),
    ("SUI","BIH"): (1.52, 4.30, 7.50),
    ("CAN","QAT"): (1.27, 6.10, 13.00),
    ("MEX","KOR"): (2.08, 3.45, 4.00),
    ("USA","AUS"): (1.63, 4.10, 5.80),
    ("SCO","MAR"): (6.00, 3.75, 1.70),
    ("BRA","HAI"): (1.11, 11.00, 29.00),
    ("TUR","PAR"): (2.02, 3.40, 4.20),
    ("NED","SWE"): (1.80, 3.75, 5.10),
    ("GER","CIV"): (1.55, 4.40, 6.40),
    ("ECU","CUW"): (1.11, 12.00, 26.00),
    ("TUN","JPN"): (7.50, 4.25, 1.50),
    ("ESP","KSA"): (1.12, 11.00, 26.00),
    ("BEL","IRN"): (1.40, 4.90, 9.00),
    ("URU","CPV"): (1.49, 4.35, 8.00),
    ("NZL","EGY"): (6.00, 4.00, 1.66),
    ("ARG","AUT"): (1.57, 4.10, 6.75),
    ("FRA","IRQ"): (1.10, 12.50, 29.00),
    ("NOR","SEN"): (2.35, 3.55, 3.10),
    ("JOR","DZA"): (6.50, 4.10, 1.58),
    ("POR","UZB"): (1.24, 6.50, 14.00),
    ("ENG","GHA"): (1.33, 5.50, 10.50),
    ("PAN","CRO"): (6.75, 4.00, 1.58),
    ("COL","COD"): (1.49, 4.30, 8.00),
    ("BIH","QAT"): (1.62, 3.85, 6.10),
    ("SUI","CAN"): (2.15, 3.40, 3.55),
    ("MAR","HAI"): (1.32, 5.50, 10.00),
    ("SCO","BRA"): (7.00, 4.80, 1.46),
    ("RSA","KOR"): (6.10, 4.10, 1.58),
    ("CZE","MEX"): (4.60, 3.60, 1.84),
    ("CUW","CIV"): (20.00, 8.00, 1.16),
    ("ECU","GER"): (5.00, 3.85, 1.76),
    ("JPN","SWE"): (2.12, 3.45, 3.55),
    ("TUN","NED"): (10.00, 5.40, 1.32),
    ("PAR","AUS"): (2.23, 3.35, 3.55),
    ("TUR","USA"): (2.85, 3.60, 2.40),
    ("NOR","FRA"): (4.35, 3.80, 1.84),
    ("SEN","IRQ"): (1.36, 5.00, 9.50),
    ("CPV","KSA"): (2.75, 3.70, 2.48),
    ("URU","ESP"): (6.25, 4.25, 1.55),
    ("EGY","IRN"): (2.15, 3.15, 3.85),
    ("NZL","BEL"): (12.50, 6.40, 1.25),
    ("CRO","GHA"): (1.67, 3.80, 5.60),
    ("PAN","ENG"): (11.00, 5.75, 1.29),
    ("COL","POR"): (3.65, 3.35, 2.14),
    ("COD","UZB"): (2.25, 3.35, 3.30),
    ("DZA","AUT"): (3.05, 3.20, 2.48),
    ("JOR","ARG"): (15.00, 7.00, 1.21),
}

def fmt(v):
    return f"{float(v):.2f}"

backup = MATCH_ODDS.with_suffix(".backup_before_insert_snapshot_20260617_2051.csv")
shutil.copy2(MATCH_ODDS, backup)

with MATCH_ODDS.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    cols = list(reader.fieldnames or [])

# Idempotens: fjern evt. tidligere forsøg med samme nye snapshot
rows = [
    r for r in rows
    if not (
        r.get("source") == SOURCE
        and r.get("odds_fetched_label") == NEW_LABEL
        and r.get("odds_fetched_at") == NEW_AT
    )
]

template = {}
for r in rows:
    if r.get("source") != SOURCE:
        continue
    if r.get("odds_fetched_label") != BASE_LABEL:
        continue
    pair = ((r.get("home") or "").strip().upper(), (r.get("away") or "").strip().upper())
    if pair in ODDS and pair not in template:
        template[pair] = r

audit = []
new_rows = []
for pair, (h, x, a) in ODDS.items():
    if pair not in template:
        audit.append({"match": f"{pair[0]}-{pair[1]}", "inserted": "FALSE", "reason": "missing_base_snapshot"})
        continue

    nr = dict(template[pair])
    nr["source"] = SOURCE
    nr["odds_fetched_label"] = NEW_LABEL
    nr["odds_fetched_at"] = NEW_AT
    nr["home_win_odds"] = fmt(h)
    nr["draw_odds"] = fmt(x)
    nr["away_win_odds"] = fmt(a)

    # Ikke bland clean sheet eller over/under ind i 1X2-opdateringen
    nr["home_clean_sheet_odds"] = ""
    nr["away_clean_sheet_odds"] = ""
    nr["over_2_5_odds"] = ""
    nr["under_2_5_odds"] = ""

    new_rows.append(nr)
    audit.append({"match": f"{pair[0]}-{pair[1]}", "inserted": "TRUE", "reason": ""})

rows.extend(new_rows)

with MATCH_ODDS.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

with AUDIT_OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["match", "inserted", "reason"])
    writer.writeheader()
    writer.writerows(audit)

print("backup:", backup)
print("base_label:", BASE_LABEL)
print("new_label:", NEW_LABEL)
print("odds_input:", len(ODDS))
print("inserted_rows:", len(new_rows))
print("missing_base:", len([a for a in audit if a["inserted"] == "FALSE"]))
print("audit:", AUDIT_OUT)
