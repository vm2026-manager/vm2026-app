from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_strategy_button_postprocess_{stamp}.html")
shutil.copy2(p, backup)

marker = "/* STRATEGY_BUTTON_POSTPROCESS_JS_START */"

helper = '''
/* STRATEGY_BUTTON_POSTPROCESS_JS_START */
function getStrategyControlSortedGroupFixtures() {
  if (!Array.isArray(fixtures) || !fixtures.length) return [];

  const groupFixtures = fixtures
    .filter(f => String(f.stage || "").toUpperCase() === "GROUP")
    .slice();

  const source = groupFixtures.length ? groupFixtures : fixtures.slice();

  return source.sort((a, b) => {
    const ad = a.kickoffDate || kickoffToDate(a.kickoff_dk || a.kickoff);
    const bd = b.kickoffDate || kickoffToDate(b.kickoff_dk || b.kickoff);
    return (ad ? ad.getTime() : 0) - (bd ? bd.getTime() : 0);
  });
}

function getStrategyControlRoundFixtures(roundNo) {
  const sorted = getStrategyControlSortedGroupFixtures();
  const start = (roundNo - 1) * 24;
  return sorted.slice(start, start + 24);
}

function hasStrategyControlUpcomingRound(roundNo) {
  const roundFixtures = getStrategyControlRoundFixtures(roundNo);
  if (!roundFixtures.length) return true;
  return roundFixtures.some(f => isUpcomingFixture(f));
}

function hasStrategyControlUpcomingGroupStage() {
  const sorted = getStrategyControlSortedGroupFixtures();
  if (!sorted.length) return true;
  return sorted.some(f => isUpcomingFixture(f));
}

function isStrategyButtonRelevantNow(strategyKey) {
  const key = String(strategyKey || "");

  if (key === "next_round") return true;
  if (key === "long_run") return true;

  // UI-label: "1. + 2. runde"
  if (key === "practical_start") {
    return hasStrategyControlUpcomingRound(1) || hasStrategyControlUpcomingRound(2);
  }

  if (key === "group_stage") {
    return hasStrategyControlUpcomingGroupStage();
  }

  return true;
}

function getStrategyButtonLabelNow(strategyKey, currentText) {
  const key = String(strategyKey || "");

  if (key === "next_round" && !hasStrategyControlUpcomingRound(1)) {
    return "Næste kamp";
  }

  return currentText;
}

function postProcessTimeAwareStrategyButtons() {
  if (!strategyButtons) return;

  const buttons = Array.from(strategyButtons.querySelectorAll(".strategy-btn"));
  if (!buttons.length) return;

  let firstVisibleKey = null;
  let activeIsVisible = false;

  buttons.forEach(button => {
    const key = String(button.dataset.strategyKey || "");
    const visible = isStrategyButtonRelevantNow(key);

    button.hidden = !visible;
    button.style.display = visible ? "" : "none";

    if (visible) {
      if (!firstVisibleKey) firstVisibleKey = key;

      const label = getStrategyButtonLabelNow(key, button.textContent || "");
      if (label && button.textContent !== label) {
        button.textContent = label;
      }

      if (key === activeStrategyKey) {
        activeIsVisible = true;
      }
    }
  });

  if (!activeIsVisible && firstVisibleKey) {
    activeStrategyKey = firstVisibleKey;
    frontendCaptainOverride = null;

    try {
      localStorage.setItem("vm2026_active_strategy", activeStrategyKey);
    } catch (error) {
      // Ignore localStorage errors.
    }

    buttons.forEach(button => {
      const key = String(button.dataset.strategyKey || "");
      button.classList.toggle("active", key === activeStrategyKey);
    });
  }
}
/* STRATEGY_BUTTON_POSTPROCESS_JS_END */
'''

changes = []

if marker not in text:
    m = re.search(r'(^[ \t]*function\s+renderStrategyControls\s*\(\)\s*\{)', text, flags=re.MULTILINE)
    if not m:
        raise SystemExit("Kunne ikke finde renderStrategyControls().")
    text = text[:m.start()] + helper + "\n\n" + text[m.start():]
    changes.append("Tilføjet postprocess-helper til strategiknapper")
else:
    changes.append("Postprocess-helper fandtes allerede")

old = '''      renderStrategyControls();
      renderActiveSlotBadge();'''

new = '''      renderStrategyControls();
      postProcessTimeAwareStrategyButtons();
      renderActiveSlotBadge();'''

if old in text:
    text = text.replace(old, new, 1)
    changes.append("render() kalder nu postProcessTimeAwareStrategyButtons()")
elif "postProcessTimeAwareStrategyButtons();" in text:
    changes.append("render() kaldte allerede postprocess-helper")
else:
    raise SystemExit("Kunne ikke indsætte postprocess-kald i render().")

p.write_text(text, encoding="utf-8")

print("OK: Strategiknapper postprocesses nu efter tid/runde.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "STRATEGY_BUTTON_POSTPROCESS_JS_START",
    "function postProcessTimeAwareStrategyButtons",
    "postProcessTimeAwareStrategyButtons();",
    'key === "next_round"',
    'key === "practical_start"',
    'key === "group_stage"',
    "Næste kamp",
]:
    print(needle + " => " + str(text.count(needle)))
