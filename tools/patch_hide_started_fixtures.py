from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_hide_started_fixtures_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old1 = '''    function getNow() {
      return new Date();
    }

    function getNextFixtureForTeam(teamId) {'''

new1 = '''    const HIDE_FIXTURE_AFTER_KICKOFF_MINUTES = 0;

    function getNow() {
      return new Date();
    }

    function isUpcomingKickoffDate(kickoffDate) {
      if (!kickoffDate || Number.isNaN(kickoffDate.getTime())) return true;
      return kickoffDate.getTime() > getNow().getTime() + HIDE_FIXTURE_AFTER_KICKOFF_MINUTES * 60 * 1000;
    }

    function isUpcomingFixture(fixture) {
      if (!fixture) return false;
      return isUpcomingKickoffDate(fixture.kickoffDate || kickoffToDate(fixture.kickoff_dk || fixture.kickoff));
    }

    function getNextFixtureForTeam(teamId) {'''

if old1 in text:
    text = text.replace(old1, new1, 1)
    changes.append("Tilføjet fælles kommende-kamp helper")
elif "function isUpcomingFixture(fixture)" in text:
    print("Kommende-kamp helper findes allerede.")
else:
    raise SystemExit("Kunne ikke finde getNow/getNextFixtureForTeam-blokken.")

old2 = '''        .filter(f => f.stage === "GROUP")
        .filter(f => canonicalTeamId(f.home) === team || canonicalTeamId(f.away) === team)
        .filter(f => f.kickoffDate >= now)
        .sort((a, b) => a.kickoffDate - b.kickoffDate);'''

new2 = '''        .filter(f => f.stage === "GROUP")
        .filter(f => canonicalTeamId(f.home) === team || canonicalTeamId(f.away) === team)
        .filter(f => isUpcomingFixture(f))
        .sort((a, b) => (a.kickoffDate || kickoffToDate(a.kickoff_dk || a.kickoff) || 0) - (b.kickoffDate || kickoffToDate(b.kickoff_dk || b.kickoff) || 0));'''

if old2 in text:
    text = text.replace(old2, new2, 1)
    changes.append("getNextFixtureForTeam bruger nu isUpcomingFixture")
elif ".filter(f => isUpcomingFixture(f))" in text:
    print("getNextFixtureForTeam ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde relevantFixtures-filteret.")

old3 = '''    function isFuture(value) {
      var raw = String(value || "");
      if (!raw) return true;
      var d = new Date(raw.replace(" ", "T"));
      if (!Number.isFinite(d.getTime())) return true;
      return d.getTime() > Date.now();
    }'''

new3 = '''    function isFuture(value) {
      var raw = String(value || "");
      if (!raw) return true;

      if (typeof kickoffToDate === "function" && typeof isUpcomingKickoffDate === "function") {
        var parsed = kickoffToDate(raw);
        if (parsed && Number.isFinite(parsed.getTime())) {
          return isUpcomingKickoffDate(parsed);
        }
      }

      var d = new Date(raw.replace(" ", "T"));
      if (!Number.isFinite(d.getTime())) return true;
      return d.getTime() > Date.now();
    }'''

if old3 in text:
    text = text.replace(old3, new3, 1)
    changes.append("Tooltip-isFuture bruger nu samme kommende-kamp logik")
elif 'typeof isUpcomingKickoffDate === "function"' in text:
    print("isFuture ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde isFuture-blokken.")

# Sørg for at addFixture faktisk afviser startede kampe.
# Vi forsøger først en meget målrettet indsættelse efter function addFixture(...)-linjen.
if "function addFixture(" in text and "if (!isFuture(kickoff)) return;" not in text:
    pattern = re.compile(r'(function addFixture\([^\)]*\) \{\s*)')
    text, n = pattern.subn(r'\1\n        if (!isFuture(kickoff)) return;\n', text, count=1)
    if n != 1:
        raise SystemExit("Kunne ikke indsætte isFuture-check i addFixture.")
    changes.append("addFixture afviser nu startede/spillede kampe")
elif "if (!isFuture(kickoff)) return;" in text:
    print("addFixture har allerede isFuture-check.")
else:
    raise SystemExit("Kunne ikke finde addFixture.")

# Sikr at fallback fra global fixtures også kun bruger kommende kampe og sorterer.
old4 = '''      if (!out.length && Array.isArray(fixtures)) {
        fixtures'''
new4 = '''      if (!out.length && Array.isArray(fixtures)) {
        fixtures
          .filter(function (f) { return isFuture(f.kickoff_dk || f.kickoff); })
          .sort(function (a, b) {
            var ad = new Date(String(a.kickoff_dk || a.kickoff || "").replace(" ", "T"));
            var bd = new Date(String(b.kickoff_dk || b.kickoff || "").replace(" ", "T"));
            return (Number.isFinite(ad.getTime()) ? ad.getTime() : 0) - (Number.isFinite(bd.getTime()) ? bd.getTime() : 0);
          })'''
if old4 in text:
    text = text.replace(old4, new4, 1)
    changes.append("Tooltip fallback-fixtures filtreres/sorteres på kommende kickoff")
elif ".filter(function (f) { return isFuture(f.kickoff_dk || f.kickoff); })" in text:
    print("Tooltip fallback ser allerede patchet ud.")
else:
    print("ADVARSEL: Kunne ikke finde fallback-fixtures start. Hopper over denne del.")

old5 = '''        setInterval(() => {
          sanitizeSearchField();
          renderTradeList();
        }, 60000);'''

new5 = '''        setInterval(() => {
          sanitizeSearchField();
          render();
        }, 60000);'''

if old5 in text:
    text = text.replace(old5, new5, 1)
    changes.append("Minut-refresh renderer hele siden, så startede kampe forsvinder automatisk")
elif "render();" in text and "}, 60000);" in text:
    print("Minut-refresh ser muligvis allerede patchet ud.")
else:
    print("ADVARSEL: Kunne ikke finde minut-refresh-blokken. Hopper over denne del.")

p.write_text(text, encoding="utf-8")

print("OK: Startede/spillede kampe skjules automatisk.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "HIDE_FIXTURE_AFTER_KICKOFF_MINUTES",
    "function isUpcomingFixture",
    ".filter(f => isUpcomingFixture(f))",
    'typeof isUpcomingKickoffDate === "function"',
    "if (!isFuture(kickoff)) return;",
]:
    print(needle, "=>", text.count(needle))
