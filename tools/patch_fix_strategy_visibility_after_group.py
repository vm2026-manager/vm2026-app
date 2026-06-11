from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_fix_strategy_visibility_after_group_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old_next = '''  if (key === "next_round") {
    return hasUpcomingStrategyRound(1);
  }'''

new_next = '''  if (key === "next_round") {
    return true;
  }'''

if old_next in text:
    text = text.replace(old_next, new_next, 1)
    changes.append("Næste kamp/næste runde er nu altid synlig")
elif '''if (key === "next_round") {
    return true;
  }''' in text:
    changes.append("Næste kamp/næste runde var allerede altid synlig")
else:
    raise SystemExit("Kunne ikke finde next_round-logikken.")

# Gør labelen dynamisk: efter runde 1 kaldes next_round for Næste kamp.
# Først prøver vi at patche strategyDisplayName-funktionen, hvis den findes.
m = re.search(r'function\s+strategyDisplayName\s*\(\s*strategyKey\s*\)\s*\{', text)
if m and "getTimeAwareStrategyDisplayName" not in text:
    insert = '''
function getTimeAwareStrategyDisplayName(strategyKey) {
  const key = String(strategyKey || "");

  if (key === "next_round" && !hasUpcomingStrategyRound(1)) {
    return "Næste kamp";
  }

  return strategyDisplayNames[key] || key;
}

'''
    text = text[:m.start()] + insert + text[m.start():]
    changes.append("Tilføjet dynamisk strateginavn-helper")

# Erstat kald til strategyDisplayName(strategyKey), men kun hvis helperen findes.
if "getTimeAwareStrategyDisplayName" in text and "getTimeAwareStrategyDisplayName(strategyKey)" not in text:
    if "strategyDisplayName(strategyKey)" in text:
        text = text.replace("strategyDisplayName(strategyKey)", "getTimeAwareStrategyDisplayName(strategyKey)", 1)
        changes.append("Strategiknapper bruger nu dynamisk navn")
    else:
        changes.append("Kunne ikke finde strategyDisplayName(strategyKey), label kan være styret andetsteds")
elif "getTimeAwareStrategyDisplayName(strategyKey)" in text:
    changes.append("Strategiknapper brugte allerede dynamisk navn")

# Hvis display name-data direkte overskriver til 'Næste runde (runde 1)', så sikrer vi via fallback-replace i knap-rendering.
# Denne ekstra patch er forsigtig og påvirker kun next_round-knappen.
if "data-strategy-key=\"${strategyKey}\"" in text and "const displayName = getTimeAwareStrategyDisplayName(strategyKey);" not in text:
    # Prøv at finde map-blokken og indsætte displayName, hvis den bruger inline navn.
    text = text.replace(
        "visibleStrategyKeys.map(strategyKey => {",
        "visibleStrategyKeys.map(strategyKey => {\n        const displayName = getTimeAwareStrategyDisplayName(strategyKey);",
        1
    )
    if "const displayName = getTimeAwareStrategyDisplayName(strategyKey);" in text:
        changes.append("Indsat displayName i strategi-map")
        text = text.replace("${escapeHtml(strategyDisplayName(strategyKey))}", "${escapeHtml(displayName)}", 1)

p.write_text(text, encoding="utf-8")

print("OK: Strategivisning rettet.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    'if (key === "next_round")',
    "return true;",
    "getTimeAwareStrategyDisplayName",
    "Næste kamp",
    'key === "group_stage"',
]:
    print(needle + " => " + str(text.count(needle)))
