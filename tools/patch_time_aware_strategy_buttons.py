from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_time_aware_strategy_buttons_{stamp}.html")
shutil.copy2(p, backup)

helper_marker = "/* TIME_AWARE_STRATEGY_BUTTONS_JS_START */"

helper = '''
/* TIME_AWARE_STRATEGY_BUTTONS_JS_START */
const GROUP_ROUND_FIXTURE_COUNT = 24;

function getSortedGroupFixturesForStrategyState() {
  if (!Array.isArray(fixtures) || !fixtures.length) return [];

  return fixtures
    .filter(f => String(f.stage || "").toUpperCase() === "GROUP")
    .slice()
    .sort((a, b) => {
      const ad = a.kickoffDate || kickoffToDate(a.kickoff_dk || a.kickoff);
      const bd = b.kickoffDate || kickoffToDate(b.kickoff_dk || b.kickoff);
      return (ad ? ad.getTime() : 0) - (bd ? bd.getTime() : 0);
    });
}

function getGroupRoundFixtures(roundNo) {
  const sorted = getSortedGroupFixturesForStrategyState();
  const start = (roundNo - 1) * GROUP_ROUND_FIXTURE_COUNT;
  return sorted.slice(start, start + GROUP_ROUND_FIXTURE_COUNT);
}

function hasUpcomingGroupRoundFixtures(roundNo) {
  const roundFixtures = getGroupRoundFixtures(roundNo);
  if (!roundFixtures.length) return true;
  return roundFixtures.some(f => isUpcomingFixture(f));
}

function hasUpcomingGroupStageFixtures() {
  const sorted = getSortedGroupFixturesForStrategyState();
  if (!sorted.length) return true;
  return sorted.some(f => isUpcomingFixture(f));
}

function isStrategyTimeRelevant(strategyKey) {
  const key = String(strategyKey || "");

  const round1Remaining = hasUpcomingGroupRoundFixtures(1);
  const round2Remaining = hasUpcomingGroupRoundFixtures(2);
  const groupRemaining = hasUpcomingGroupStageFixtures();

  if (key === "next_round" || key === "practical_start") {
    return round1Remaining;
  }

  if (key === "round1_2") {
    return round1Remaining || round2Remaining;
  }

  if (key === "group_stage") {
    return groupRemaining;
  }

  return true;
}

function getVisibleStrategyKeys(strategyKeys) {
  const keys = Array.isArray(strategyKeys) ? strategyKeys : [];
  const visible = keys.filter(isStrategyTimeRelevant);

  if (visible.length) return visible;

  const longRun = keys.find(key => String(key) === "long_run");
  return longRun ? [longRun] : keys;
}

function ensureActiveStrategyVisible(strategyKeys) {
  const visible = getVisibleStrategyKeys(strategyKeys);

  if (visible.length && !visible.includes(activeStrategyKey)) {
    activeStrategyKey = visible[0];
    try {
      localStorage.setItem("vm2026_active_strategy", activeStrategyKey);
    } catch (error) {
      // Ignore localStorage errors.
    }
  }

  return visible;
}
/* TIME_AWARE_STRATEGY_BUTTONS_JS_END */
'''

changes = []

if helper_marker not in text:
    anchor = "    function renderStrategyButtons() {"
    if anchor not in text:
        raise SystemExit("Kunne ikke finde renderStrategyButtons().")
    text = text.replace(anchor, helper + "\n\n" + anchor, 1)
    changes.append("Tilføjet tidsstyrede strategi-helpers")
else:
    changes.append("Tidsstyrede strategi-helpers fandtes allerede")

pattern = re.compile(
    r'(strategyButtons\.innerHTML\s*=\s*)Object\.keys\(([^)]+)\)(\.map\(strategyKey\s*=>)',
    flags=re.MULTILINE
)

if "ensureActiveStrategyVisible(Object.keys(" not in text:
    text, n = pattern.subn(
        r'\1ensureActiveStrategyVisible(Object.keys(\2))\3',
        text,
        count=1
    )
    if n != 1:
        raise SystemExit("Kunne ikke patch'e strategyButtons.innerHTML.")
    changes.append("renderStrategyButtons filtrerer nu irrelevante strategier")
else:
    changes.append("renderStrategyButtons var allerede patchet")

p.write_text(text, encoding="utf-8")

print("OK: Strategiknapper er nu tidsstyrede.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "TIME_AWARE_STRATEGY_BUTTONS_JS_START",
    "GROUP_ROUND_FIXTURE_COUNT",
    "function isStrategyTimeRelevant",
    "ensureActiveStrategyVisible(Object.keys(",
    "key === \\"round1_2\\"",
    "key === \\"group_stage\\"",
]:
    print(needle, "=>", text.count(needle))
