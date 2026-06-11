from pathlib import Path
from datetime import datetime
import re
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_manual_bank_budget_total_{stamp}.html")
shutil.copy2(p, backup)

old_block_pattern = re.compile(
    r"""function getManualBankAmount\(\) \{\s*
      return manualBankMillions === null \? null : Math\.round\(manualBankMillions \* 1000000\);\s*
    \}\s*

    function getFullOptimizationBudget\(\) \{\s*
      const manualBank = getManualBankAmount\(\);\s*
      return manualBank === null \? BUDGET_TOTAL : getSpentBudget\(\) \+ manualBank;\s*
    \}\s*

    function getAutofillBankLeft\(baseSpent\) \{\s*
      const manualBank = getManualBankAmount\(\);\s*
      return manualBank === null \? BUDGET_TOTAL - baseSpent : manualBank;\s*
    \}""",
    re.MULTILINE,
)

new_block = """function getManualBankAmount() {
      return manualBankMillions === null ? null : Math.round(manualBankMillions * 1000000);
    }

    function getEffectiveBudgetTotal() {
      const manualBudgetTotal = getManualBankAmount();
      return manualBudgetTotal === null ? BUDGET_TOTAL : manualBudgetTotal;
    }

    function getFullOptimizationBudget() {
      return getEffectiveBudgetTotal();
    }

    function getAutofillBankLeft(baseSpent) {
      return getEffectiveBudgetTotal() - baseSpent;
    }"""

text, n1 = old_block_pattern.subn(new_block, text, count=1)
if n1 != 1:
    raise SystemExit(f"Kunne ikke patche budget-funktionsblokken. Antal hits: {n1}")

old_display_pattern = re.compile(
    r"""function getDisplayedBankLeft\(\) \{\s*
      const manualBank = getManualBankAmount\(\);\s*
      return manualBank === null \? getBankLeft\(\) : manualBank;\s*
    \}""",
    re.MULTILINE,
)

new_display = """function getDisplayedBankLeft() {
      return getEffectiveBudgetTotal() - getSpentBudget();
    }"""

text, n2 = old_display_pattern.subn(new_display, text, count=1)
if n2 != 1:
    raise SystemExit(f"Kunne ikke patche getDisplayedBankLeft. Antal hits: {n2}")

text = text.replace(
    'placeholder="fx 1,2" aria-label="Aktuel bank i millioner kroner"',
    'placeholder="fx 50,5" aria-label="Samlet budgetramme i millioner kroner"'
)

p.write_text(text, encoding="utf-8")

print("OK: Manuel bank er nu samlet budgetramme.")
print(f"Backup: {backup}")
print("Ændret:")
print("- getEffectiveBudgetTotal() tilføjet")
print("- getFullOptimizationBudget() bruger samlet budgetramme")
print("- getAutofillBankLeft(baseSpent) = samlet budgetramme - baseSpent")
print("- getDisplayedBankLeft() = samlet budgetramme - brugt")
print("- placeholder ændret til fx 50,5")
