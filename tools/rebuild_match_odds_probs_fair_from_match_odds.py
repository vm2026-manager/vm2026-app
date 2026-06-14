from pathlib import Path
from datetime import datetime
import csv
import shutil

DATA = Path("data")
MATCH_ODDS = DATA / "match_odds.csv"
MATCH_ODDS_PROBS = DATA / "match_odds_probs.csv"

PREFERRED_SOURCE = "Unibet"
PREFERRED_LABEL = "14.06 kl. 12:31"

def txt(v):
    return "" if v is None else str(v).strip()

def fnum(v):
    try:
        return float(txt(v).replace(",", "."))
    except Exception:
        return None

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])

def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def fair_probs(home_odds, draw_odds, away_odds):
    inv_h = 1 / home_odds
    inv_d = 1 / draw_odds
    inv_a = 1 / away_odds
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total

if not MATCH_ODDS.exists():
    raise SystemExit(f"Mangler {MATCH_ODDS}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if MATCH_ODDS_PROBS.exists():
    shutil.copy2(
        MATCH_ODDS_PROBS,
        MATCH_ODDS_PROBS.with_name(f"{MATCH_ODDS_PROBS.stem}.backup_before_fair_rebuild_{stamp}{MATCH_ODDS_PROBS.suffix}")
    )

rows, fields = read_csv(MATCH_ODDS)

valid = []
for r in rows:
    match_id = txt(r.get("match_id"))
    home = txt(r.get("home")).upper()
    away = txt(r.get("away")).upper()
    h = fnum(r.get("home_win_odds"))
    d = fnum(r.get("draw_odds"))
    a = fnum(r.get("away_win_odds"))

    if not match_id or not home or not away:
        continue
    if not h or not d or not a or h <= 1 or d <= 1 or a <= 1:
        continue

    source = txt(r.get("source"))
    label = txt(r.get("odds_fetched_label"))

    # Rank: brug Unibet 14.06 først; ellers behold eksisterende odds som fallback
    if source == PREFERRED_SOURCE and label == PREFERRED_LABEL:
        rank = 3
    elif source.lower() in {"unibet"}:
        rank = 2
    else:
        rank = 1

    valid.append((match_id, rank, r, h, d, a))

best = {}
for match_id, rank, r, h, d, a in valid:
    old = best.get(match_id)
    if old is None or rank >= old[0]:
        best[match_id] = (rank, r, h, d, a)

out = []
for match_id in sorted(best, key=lambda x: int(x) if str(x).isdigit() else str(x)):
    rank, r, h, d, a = best[match_id]
    hp, dp, ap = fair_probs(h, d, a)

    home = txt(r.get("home")).upper()
    away = txt(r.get("away")).upper()

    out.append({
        "match_id": match_id,
        "home": home,
        "away": away,
        "kickoff_dk": txt(r.get("kickoff_dk")),
        "home_team_id": txt(r.get("home_team_id")) or home,
        "away_team_id": txt(r.get("away_team_id")) or away,

        "home_win_odds": h,
        "draw_odds": d,
        "away_win_odds": a,

        # Gamle kolonner
        "home_win_prob": hp,
        "draw_prob": dp,
        "away_win_prob": ap,

        # Vigtige kolonner som build_fixture_strength_multipliers.py faktisk læser
        "home_win_prob_fair": hp,
        "draw_prob_fair": dp,
        "away_win_prob_fair": ap,

        "source": txt(r.get("source")),
        "odds_fetched_at": txt(r.get("odds_fetched_at")),
        "odds_fetched_label": txt(r.get("odds_fetched_label")),
    })

fields_out = [
    "match_id", "home", "away", "kickoff_dk",
    "home_team_id", "away_team_id",
    "home_win_odds", "draw_odds", "away_win_odds",
    "home_win_prob", "draw_prob", "away_win_prob",
    "home_win_prob_fair", "draw_prob_fair", "away_win_prob_fair",
    "source", "odds_fetched_at", "odds_fetched_label",
]

write_csv(MATCH_ODDS_PROBS, out, fields_out)

unibet_count = sum(
    1 for r in out
    if r["source"] == PREFERRED_SOURCE and r["odds_fetched_label"] == PREFERRED_LABEL
)

print("OK: match_odds_probs.csv genbygget med *_fair-kolonner")
print(f"Kampe i probs: {len(out)}")
print(f"Unibet {PREFERRED_LABEL}: {unibet_count}")
print(f"Fallback/ældre odds: {len(out) - unibet_count}")
print("Nu bør build_fixture_strength_multipliers.py kunne læse 1X2-odds.")