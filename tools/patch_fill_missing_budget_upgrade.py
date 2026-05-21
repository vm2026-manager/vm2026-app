from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "index.html"


HELPER_INSERT_AFTER = '''    function fillPartiallyGreedy(emptySlots, usedIdsBase, teamCountsBase, bankLeft) {
      const partialSolution = {};
      const usedIds = new Set(usedIdsBase);
      const teamCounts = { ...teamCountsBase };
      let remainingBank = bankLeft;

      const remainingSlots = [...emptySlots];

      while (remainingSlots.length) {
        let bestPick = null;
        let bestSlotIndex = -1;

        for (let i = 0; i < remainingSlots.length; i++) {
          const slot = remainingSlots[i];
          const candidates = getEligiblePlayersForSlot(slot, usedIds, teamCounts, remainingBank);
          if (!candidates.length) continue;

          const candidate = candidates[0];
          if (!bestPick || getPlayerScore(candidate) > getPlayerScore(bestPick.player)) {
            bestPick = { slot, player: candidate };
            bestSlotIndex = i;
          }
        }

        if (!bestPick) break;

        partialSolution[bestPick.slot.key] = bestPick.player.player_id;
        usedIds.add(String(bestPick.player.player_id));
        const teamKey = bestPick.player.team_id || bestPick.player.team_name;
        teamCounts[teamKey] = (teamCounts[teamKey] || 0) + 1;
        remainingBank -= getPlayerPrice(bestPick.player);

        remainingSlots.splice(bestSlotIndex, 1);
      }

      return partialSolution;
    }
'''


HELPER_FUNCTION = '''
    function improveFilledSolutionBudgetUse(solution, emptySlots, usedIdsBase, teamCountsBase, bankLeft) {
      const maxAcceptableBankAfterFill = 1500000;
      const improved = { ...solution };
      const slotByKey = Object.fromEntries(emptySlots.map(slot => [slot.key, slot]));

      function buildState(currentSolution) {
        const usedIds = new Set(usedIdsBase);
        const teamCounts = { ...teamCountsBase };
        let extraSpent = 0;

        for (const [slotKey, playerId] of Object.entries(currentSolution)) {
          const player = getPlayerById(playerId);
          if (!player) continue;

          usedIds.add(String(player.player_id));
          const teamKey = player.team_id || player.team_name || "";
          teamCounts[teamKey] = (teamCounts[teamKey] || 0) + 1;
          extraSpent += getPlayerPrice(player);
        }

        return { usedIds, teamCounts, extraSpent };
      }

      for (let round = 0; round < 30; round++) {
        const state = buildState(improved);
        const remainingBank = bankLeft - state.extraSpent;

        if (remainingBank <= maxAcceptableBankAfterFill) {
          return improved;
        }

        let bestSwap = null;

        for (const [slotKey, currentPlayerId] of Object.entries(improved)) {
          const slot = slotByKey[slotKey];
          const currentPlayer = getPlayerById(currentPlayerId);
          if (!slot || !currentPlayer) continue;

          const currentPrice = getPlayerPrice(currentPlayer);
          const currentScore = getPlayerScore(currentPlayer);

          const usedIdsWithoutCurrent = new Set(state.usedIds);
          usedIdsWithoutCurrent.delete(String(currentPlayer.player_id));

          const teamCountsWithoutCurrent = { ...state.teamCounts };
          const currentTeamKey = currentPlayer.team_id || currentPlayer.team_name || "";
          if (currentTeamKey) {
            teamCountsWithoutCurrent[currentTeamKey] = (teamCountsWithoutCurrent[currentTeamKey] || 1) - 1;
            if (teamCountsWithoutCurrent[currentTeamKey] <= 0) delete teamCountsWithoutCurrent[currentTeamKey];
          }

          const slotBank = remainingBank + currentPrice;
          const candidates = getEligiblePlayersForSlot(slot, usedIdsWithoutCurrent, teamCountsWithoutCurrent, slotBank)
            .slice(0, 450);

          for (const candidate of candidates) {
            if (String(candidate.player_id) === String(currentPlayer.player_id)) continue;

            const candidatePrice = getPlayerPrice(candidate);
            const extraCost = candidatePrice - currentPrice;
            if (extraCost <= 0 || extraCost > remainingBank) continue;

            const bankAfterSwap = remainingBank - extraCost;
            const candidateScore = getPlayerScore(candidate);
            const scoreDelta = candidateScore - currentScore;

            // Prioritet:
            // 1) Ram maks 1,5 mio. i bank, hvis muligt
            // 2) Brug mere budget
            // 3) Undgå for stort kvalitetstab
            const hitsBudgetTarget = bankAfterSwap <= maxAcceptableBankAfterFill ? 1 : 0;
            const qualityPenalty = Math.max(0, -scoreDelta);
            const metric =
              hitsBudgetTarget * 1000000000
              + extraCost / 1000
              + scoreDelta * 1000
              - qualityPenalty * 3000;

            if (!bestSwap || metric > bestSwap.metric) {
              bestSwap = {
                slotKey,
                player: candidate,
                metric,
                bankAfterSwap,
                extraCost,
                scoreDelta
              };
            }
          }
        }

        if (!bestSwap) {
          return improved;
        }

        improved[bestSwap.slotKey] = bestSwap.player.player_id;
      }

      return improved;
    }
'''


OLD_FINAL_BLOCK = '''      const candidateSquad = { ...squad, ...solution };
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


NEW_FINAL_BLOCK = '''      solution = improveFilledSolutionBudgetUse(solution, emptySlots, usedIdsBase, teamCountsBase, bankLeft);

      const candidateSquad = { ...squad, ...solution };
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
          `Der findes en gyldig fuld udfyldning, men appen kunne ikke finde en fornuftig løsning, der bruger budgettet nok. Den bedste fundne løsning efterlod ${formatMoney(bankAfterFill)} i banken. Justér evt. én af dine låste spillere manuelt.`,
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

    if "function improveFilledSolutionBudgetUse" not in text:
        if HELPER_INSERT_AFTER not in text:
            raise RuntimeError("Fandt ikke fillPartiallyGreedy-blokken.")
        text = text.replace(HELPER_INSERT_AFTER, HELPER_INSERT_AFTER + "\n" + HELPER_FUNCTION, 1)

    if OLD_FINAL_BLOCK not in text:
        raise RuntimeError("Fandt ikke final budget-blokken. Har du allerede ændret den?")

    text = text.replace(OLD_FINAL_BLOCK, NEW_FINAL_BLOCK, 1)
    INDEX_PATH.write_text(text, encoding="utf-8")

    print("OK: Fyld mangler forsøger nu at opgradere spillere, før den stopper pga. for stort restbudget.")
    print(f"Skrev: {INDEX_PATH}")


if __name__ == "__main__":
    main()