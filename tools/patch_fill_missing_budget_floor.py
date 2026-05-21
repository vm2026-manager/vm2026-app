from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "index.html"


OLD_BLOCK = '''      const candidateSquad = { ...squad, ...solution };
      const issues = getSquadConstraintIssues(candidateSquad);
      if (issues.length) {
        showStatus(`Udfyldningen blev stoppet, fordi resultatet ville være ugyldigt: ${issues.join(" · ")}`, "error");
        return;
      }

      pushHistory();
      squad = candidateSquad;
      activeSlotKey = null;
      activeSlotText.textContent = "Ingen valgt";
      clearStatus();
      saveState();
      render();
      scheduleAutosave();
'''


NEW_BLOCK = '''      const candidateSquad = { ...squad, ...solution };
      const issues = getSquadConstraintIssues(candidateSquad);
      if (issues.length) {
        showStatus(`Udfyldningen blev stoppet, fordi resultatet ville være ugyldigt: ${issues.join(" · ")}`, "error");
        return;
      }

      const spentAfterFill = getSpentBudget(candidateSquad);
      const bankAfterFill = BUDGET_TOTAL - spentAfterFill;
      const maxAcceptableBankAfterFill = 1500000;

      if (bankAfterFill > maxAcceptableBankAfterFill) {
        showStatus(
          `Udfyldningen blev stoppet, fordi den efterlod ${formatMoney(bankAfterFill)} i banken. Vælg evt. dyrere låste spillere eller justér manuelt. Målet er maks. ca. ${formatMoney(maxAcceptableBankAfterFill)} i banken.`,
          "error"
        );
        return;
      }

      pushHistory();
      squad = candidateSquad;
      activeSlotKey = null;
      activeSlotText.textContent = "Ingen valgt";
      clearStatus();
      saveState();
      render();
      scheduleAutosave();
'''


def main() -> None:
    text = INDEX_PATH.read_text(encoding="utf-8")

    if OLD_BLOCK not in text:
        raise RuntimeError("Fandt ikke den forventede blok i index.html. Stopper uden ændring.")

    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    INDEX_PATH.write_text(text, encoding="utf-8")

    print("OK: Fyld mangler optimalt stopper nu, hvis fuldt hold efterlader mere end 1,5 mio. i banken.")
    print(f"Skrev: {INDEX_PATH}")


if __name__ == "__main__":
    main()