from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_danish_kickoff_parser_{stamp}.html")
shutil.copy2(p, backup)

old = '''    function kickoffToDate(value) {
      if (!value) return null;
      const normalized = String(value).trim().replace(" ", "T");
      const date = new Date(normalized);
      return Number.isNaN(date.getTime()) ? null : date;
    }'''

new = '''    function kickoffToDate(value) {
      if (!value) return null;

      const raw = String(value).trim();
      if (!raw) return null;

      // ISO-ish formats, e.g. 2026-06-13T21:00:00 or 2026-06-13 21:00
      const normalized = raw.replace(" ", "T");
      let date = new Date(normalized);
      if (!Number.isNaN(date.getTime())) return date;

      // Danish display formats used in player/tooltip data:
      // 13.06 21.00
      // 13.06.2026 21.00
      // 13/06 21:00
      const m = raw.match(/^(\\d{1,2})[\\.\\/\\-](\\d{1,2})(?:[\\.\\/\\-](\\d{2,4}))?\\s+(\\d{1,2})[\\.:](\\d{2})/);
      if (m) {
        const day = Number(m[1]);
        const month = Number(m[2]);
        let year = m[3] ? Number(m[3]) : 2026;
        if (year < 100) year += 2000;
        const hour = Number(m[4]);
        const minute = Number(m[5]);

        date = new Date(year, month - 1, day, hour, minute, 0, 0);
        if (!Number.isNaN(date.getTime())) return date;
      }

      return null;
    }'''

if old not in text:
    raise SystemExit("Kunne ikke finde den eksisterende kickoffToDate()-blok.")

text = text.replace(old, new, 1)

old_sort = '''          .sort(function (a, b) {
            var ad = new Date(String(a.kickoff_dk || a.kickoff || "").replace(" ", "T"));
            var bd = new Date(String(b.kickoff_dk || b.kickoff || "").replace(" ", "T"));
            return (Number.isFinite(ad.getTime()) ? ad.getTime() : 0) - (Number.isFinite(bd.getTime()) ? bd.getTime() : 0);
          })'''

new_sort = '''          .sort(function (a, b) {
            var ad = typeof kickoffToDate === "function" ? kickoffToDate(a.kickoff_dk || a.kickoff) : null;
            var bd = typeof kickoffToDate === "function" ? kickoffToDate(b.kickoff_dk || b.kickoff) : null;
            return (ad ? ad.getTime() : 0) - (bd ? bd.getTime() : 0);
          })'''

if old_sort in text:
    text = text.replace(old_sort, new_sort, 1)

p.write_text(text, encoding="utf-8")

print("OK: kickoffToDate kan nu læse dansk datoformat som 13.06 21.00.")
print(f"Backup: {backup}")
print("")
print("Sanity:")
for needle in [
    "13.06 21.00",
    "year = m[3] ? Number(m[3]) : 2026",
    "kickoffToDate(a.kickoff_dk || a.kickoff)",
]:
    print(needle, "=>", text.count(needle))
