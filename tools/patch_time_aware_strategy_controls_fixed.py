from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_time_aware_strategy_controls_fixed_{stamp}.html")
shutil.copy2(p, backup)

helper_marker = "/* TIME_AWARE_STRATEGY_CONTROLS_JS_START */"

helper = '''
/* TIME_AWARE_STRATEGY_CONTROLS_JS_START */
const STRATEGY_ROUND_FIXTURE_COUNT = 24;

function getSortedGroupFixturesForStrategyControls() {
  if (!Array.isArray(fixtures) || !fixtures.length) return [];

  const groupFixtures = fixtures
    .filter(f => String(f.stage || "").toUpperCase() === "GROUP")
    .slice()
    .sort((a, b) => {
      const ad = a.kickoffDate || kickoffToDate(a.kickoff_dk || a.kickoff);
      const bd = b.kickoffDate || kickoffToDate(b.kickoff_dk || b.kickoff);
      return (ad ? ad.getTime() : 0) - (bd ? bd.getTime() : 0);
    });

  return groupFixtures.length ? groupFixtures : fixtures.slice().sort((a, b) => {
    const ad = a.kickoffDate || kickoffToDate(a.kickoff_dk || a.kickoff);
    const bd = b.kickoffDate || kickoffToDate(b.kickoff_dk || b.kickoff);
    return (ad ? ad.getTime() : 0) - (bd ? bd.getTime() : 0);
  });
}

function getStrategyRoundFixtures(roundNo) {
  const sorted = getSortedGroupFixturesForStrategyControls();
  const start = (roundNo - 1) * STRATEGY_ROUND_FIXTURE_COUNT;
  return sorted.slice(start, start + STRATEGY_ROUND_FIXTURE_COUNT);
}

function hasUpcomingStrategyRound(roundNo) {
  const roundFixtures = getStrategyRoundFixtures(roundNo);
  if (!roundFixtures.length) return true;
  return roundFixtures.some(f => isUpcomingFixture(f));
}

function hasUpcomingStrategyGroupStage() {
  const sorted = getSortedGroupFixturesForStrategyControls();
  if (!sorted.length) return true;
  return sorted.some(f => isUpcomingFixture(f));
}

function isStrategyStillRelevantByTime(strategyKey) {
  const key = String(strategyKey || "");

  if (key === "next_round") {
    return hasUpcomingStrategyRound(1);
  }

  // Vises som "1. + 2. runde" i UI.
  if (key === "practical_start") {
    return hasUpcomingStrategyRound(1) || hasUpcomingStrategyRound(2);
  }

  if (key === "group_stage") {
    return hasUpcomingStrategyGroupStage();
  }

  // long_run og ukendte strategier bevares.
  return true;
}

function getVisibleStrategyKeysByTime(strategyKeys) {
  const keys = Array.isArray(strategyKeys) ? strategyKeys : [];
  const visible = keys.filter(isStrategyStillRelevantByTime);

  if (visible.length) return visible;

  const longRun = keys.find(key => String(key) === "long_run");
  return longRun ? [longRun] : keys;
}

function ensureActiveStrategyStillVisible(strategyKeys) {
  const visible = getVisibleStrategyKeysByTime(strategyKeys);

  if (visible.length && !visible.includes(activeStrategyKey)) {
    activeStrategyKey = visible[0];
    frontendCaptainOverride = null;

    try {
      localStorage.setItem("vm2026_active_strategy", activeStrategyKey);
    } catch (error) {
      // Ignore localStorage errors.
    }
  }

  return visible;
}
/* TIME_AWARE_STRATEGY_CONTROLS_JS_END */
'''

changes = []

if helper_marker not in text:
    m = re.search(r'(^[ \t]*function\s+renderStrategyControls\s*\(\)\s*\{)', text, flags=re.MULTILINE)
    if not m:
        raise SystemExit("Kunne ikke finde function renderStrategyControls().")
    text = text[:m.start()] + helper + "\n\n" + text[m.start():]
    changes.append("Tilføjet tidsstyrede strategi-helpers")
else:
    changes.append("Tidsstyrede strategi-helpers fandtes allerede")

# Find renderStrategyControls body.
m = re.search(r'(^[ \t]*function\s+renderStrategyControls\s*\(\)\s*\{)', text, flags=re.MULTILINE)
if not m:
    raise SystemExit("Kunne ikke finde renderStrategyControls() efter helper-indsættelse.")

start = m.start()
brace_start = text.find("{", m.end() - 1)
depth = 0
end = None

for i in range(brace_start, len(text)):
    ch = text[i]
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break

if end is None:
    raise SystemExit("Kunne ikke afgrænse renderStrategyControls().")

func = text[start:end]

if "ensureActiveStrategyStillVisible(USER_STRATEGIES)" not in func:
    func = func.replace(
        "{",
        "{\n      const visibleStrategyKeys = ensureActiveStrategyStillVisible(USER_STRATEGIES);",
        1
    )
    changes.append("renderStrategyControls beregner nu synlige strategier")
else:
    changes.append("visibleStrategyKeys fandtes allerede i renderStrategyControls")

if "USER_STRATEGIES.map" in func:
    func = func.replace("USER_STRATEGIES.map", "visibleStrategyKeys.map", 1)
    changes.append("Strategiknapper bygges nu kun fra synlige strategier")
elif "visibleStrategyKeys.map" in func:
    changes.append("Strategiknapper brugte allerede visibleStrategyKeys")
else:
    raise SystemExit("Kunne ikke finde USER_STRATEGIES.map i renderStrategyControls().")

text = text[:start] + func + text[end:]
p.write_text(text, encoding="utf-8")

print("OK: Strategiknapper filtreres nu efter runde/tid.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
needles = [
    "TIME_AWARE_STRATEGY_CONTROLS_JS_START",
    "STRATEGY_ROUND_FIXTURE_COUNT",
    "function isStrategyStillRelevantByTime",
    "ensureActiveStrategyStillVisible(USER_STRATEGIES)",
    "visibleStrategyKeys.map",
    'key === "next_round"',
    'key === "practical_start"',
    'key === "group_stage"',
]
for needle in needles:
    print(needle + " => " + str(text.count(needle)))
