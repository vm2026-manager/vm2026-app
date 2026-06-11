from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_tooltip_odds_timestamp_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old1 = '''          source: selected.source || "odds"
        };'''
new1 = '''          source: selected.source || "odds",
          oddsFetchedLabel: selected.odds_fetched_label || selected.oddsFetchedLabel || "",
          oddsFetchedAt: selected.odds_fetched_at || selected.oddsFetchedAt || ""
        };'''

if old1 in text:
    text = text.replace(old1, new1, 1)
    changes.append("oddsByTeamPair entry får oddsFetchedLabel/oddsFetchedAt")
elif "oddsFetchedLabel: selected.odds_fetched_label" in text:
    print("Allerede patchet: oddsByTeamPair entry")
else:
    raise SystemExit("Kunne ikke finde oddsByTeamPair source-blok.")

old2 = '''          source: row ? row.source : ""
        });'''
new2 = '''          source: row ? row.source : "",
          oddsFetchedLabel: row ? (row.oddsFetchedLabel || row.odds_fetched_label || "") : "",
          oddsFetchedAt: row ? (row.oddsFetchedAt || row.odds_fetched_at || "") : ""
        });'''

if old2 in text:
    text = text.replace(old2, new2, 1)
    changes.append("fixture rows får oddsFetchedLabel/oddsFetchedAt")
elif "oddsFetchedLabel: row ? (row.oddsFetchedLabel" in text:
    print("Allerede patchet: fixture rows")
else:
    raise SystemExit("Kunne ikke finde fixture source-blok.")

old3 = '''      var source = fixtures.find(function (f) { return f.source; });
      var sourceText = source ? source.source : "odds";'''
new3 = '''      var source = fixtures.find(function (f) { return f.source; });
      var sourceText = source ? source.source : "odds";
      if (source && source.oddsFetchedLabel) {
        sourceText = sourceText + " · hentet " + source.oddsFetchedLabel;
      }'''

if old3 in text:
    text = text.replace(old3, new3, 1)
    changes.append("tooltip viser source + hentet-tidspunkt")
elif 'sourceText = sourceText + " · hentet " + source.oddsFetchedLabel' in text:
    print("Allerede patchet: tooltip sourceText")
else:
    raise SystemExit("Kunne ikke finde tooltip sourceText-blok.")

p.write_text(text, encoding="utf-8")

print("OK: Tooltip-tidsstempel patchet.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in ["oddsFetchedLabel", "· hentet", "odds_fetched_label"]:
    print(needle, "=>", text.count(needle))
