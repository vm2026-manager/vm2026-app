import json
import csv
from pathlib import Path

files = [
    "data/player_pool_v1.json",
    "data/player_ev_group_stage_v1.csv",
    "data/player_start_security_nt.csv",
    "data/start_signal_context_overrides.csv",
    "data/bruttoliste_med_score_og_start_filled.csv",
    "data/bruttoliste_med_score_og_start.csv",
]

print("\n=== ROBERTO ALVARADO IN DATA FILES ===\n")

for file in files:
    path = Path(file)
    if not path.exists():
        print(f"--- {file}: MISSING")
        continue

    print(f"\n--- {file} ---")

    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                items = data.get("players") or data.get("data") or []
            else:
                items = data

            found = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                txt = json.dumps(item, ensure_ascii=False).lower()
                if "alvarado" in txt:
                    found = True
                    print(json.dumps(item, ensure_ascii=False, indent=2))
            if not found:
                print("No Alvarado row found")

        else:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                found = False
                for row in reader:
                    txt = str(row).lower()
                    if "alvarado" in txt:
                        found = True
                        print(row)
                if not found:
                    print("No Alvarado row found")

    except Exception as e:
        print("ERROR:", e)

print("\n=== INDEX WARNING LOGIC ===\n")
index = Path("index.html").read_text(encoding="utf-8", errors="replace")
terms = ["Tjek start", "check start", "start warning", "startWarning", "warning", "start_prob", "conditional_start_prob", "high risk"]
for term in terms:
    print(f"\n--- term: {term} ---")
    lower = index.lower()
    needle = term.lower()
    start = 0
    hits = 0
    while True:
        pos = lower.find(needle, start)
        if pos == -1:
            break
        hits += 1
        snippet = index[max(0, pos-350):pos+650]
        print(f"\nHIT {hits} at char {pos}:\n{snippet}\n")
        start = pos + len(needle)
        if hits >= 8:
            print("... more hits omitted")
            break
