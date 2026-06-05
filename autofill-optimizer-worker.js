const MAX_ACCEPTABLE_BANK = 500000;
const SCORE_PRICE_LAMBDAS = [
  0,
  0.000002,
  0.000004,
  0.000006,
  0.000008,
  0.000010,
  0.000012,
  0.000015,
  0.000020,
  0.000030,
  0.000040
];

function playerSort(a, b) {
  const scoreDiff = b.score - a.score;
  if (scoreDiff !== 0) return scoreDiff;
  const startDiff = b.start - a.start;
  if (startDiff !== 0) return startDiff;
  return a.price - b.price;
}

function getEligiblePlayers(players, position, usedIds, teamCounts, maxPerTeam, maxPrice = Infinity) {
  return players
    .filter(player => player.position === position)
    .filter(player => !usedIds.has(player.id))
    .filter(player => (teamCounts[player.team] || 0) < maxPerTeam)
    .filter(player => player.price <= maxPrice)
    .sort(playerSort);
}

function greedyIncumbent(slots, candidatesByPosition, bankLeft, usedIdsBase, teamCountsBase, maxPerTeam) {
  const solution = {};
  const usedIds = new Set(usedIdsBase);
  const teamCounts = { ...teamCountsBase };
  let spent = 0;
  let score = 0;

  for (const slot of slots) {
    const candidate = candidatesByPosition[slot.position].find(player =>
      !usedIds.has(player.id)
      && (teamCounts[player.team] || 0) < maxPerTeam
      && spent + player.price <= bankLeft
    );
    if (!candidate) return null;
    solution[slot.key] = candidate.id;
    usedIds.add(candidate.id);
    teamCounts[candidate.team] = (teamCounts[candidate.team] || 0) + 1;
    spent += candidate.price;
    score += candidate.score;
  }
  return { solution, score, spent };
}

function cheapestFeasibleIncumbent(
  slots,
  cheapestByPosition,
  bankLeft,
  usedIdsBase,
  teamCountsBase,
  maxPerTeam,
  deadline
) {
  const solution = {};
  const usedIds = new Set(usedIdsBase);
  const teamCounts = { ...teamCountsBase };
  const lastIndexByPosition = {};
  let visitedNodes = 0;

  function dfs(index, spent, score) {
    visitedNodes++;
    if (visitedNodes % 5000 === 0 && performance.now() > deadline) return null;
    if (index === slots.length) {
      return { solution: { ...solution }, score, spent, visitedNodes };
    }
    const slot = slots[index];
    const candidates = cheapestByPosition[slot.position];
    const startIndex = lastIndexByPosition[slot.position] ?? 0;
    for (let candidateIndex = startIndex; candidateIndex < candidates.length; candidateIndex++) {
      const player = candidates[candidateIndex];
      const nextSpent = spent + player.price;
      if (nextSpent > bankLeft) break;
      if (usedIds.has(player.id)) continue;
      const currentTeamCount = teamCounts[player.team] || 0;
      if (currentTeamCount >= maxPerTeam) continue;

      solution[slot.key] = player.id;
      usedIds.add(player.id);
      teamCounts[player.team] = currentTeamCount + 1;
      const previousPositionIndex = lastIndexByPosition[slot.position];
      lastIndexByPosition[slot.position] = candidateIndex + 1;
      const result = dfs(index + 1, nextSpent, score + player.score);
      if (result) return result;

      if (previousPositionIndex === undefined) delete lastIndexByPosition[slot.position];
      else lastIndexByPosition[slot.position] = previousPositionIndex;
      if (currentTeamCount === 0) delete teamCounts[player.team];
      else teamCounts[player.team] = currentTeamCount;
      usedIds.delete(player.id);
      delete solution[slot.key];
    }
    return null;
  }

  const result = dfs(0, 0, 0);
  return result || { solution: null, score: -Infinity, spent: Infinity, visitedNodes };
}

function improveBudgetUse(solution, slots, playersById, candidatesByPosition, bankLeft, usedIdsBase, teamCountsBase, maxPerTeam) {
  const improved = { ...solution };
  const slotByKey = Object.fromEntries(slots.map(slot => [slot.key, slot]));

  function buildState() {
    const usedIds = new Set(usedIdsBase);
    const teamCounts = { ...teamCountsBase };
    let spent = 0;
    for (const playerId of Object.values(improved)) {
      const player = playersById[playerId];
      if (!player) continue;
      usedIds.add(player.id);
      teamCounts[player.team] = (teamCounts[player.team] || 0) + 1;
      spent += player.price;
    }
    return { usedIds, teamCounts, spent };
  }

  for (let round = 0; round < 30; round++) {
    const state = buildState();
    const remainingBank = bankLeft - state.spent;
    if (remainingBank <= MAX_ACCEPTABLE_BANK) return improved;

    let bestSwap = null;
    for (const [slotKey, currentPlayerId] of Object.entries(improved)) {
      const slot = slotByKey[slotKey];
      const current = playersById[currentPlayerId];
      if (!slot || !current) continue;

      const usedWithoutCurrent = new Set(state.usedIds);
      usedWithoutCurrent.delete(current.id);
      const countsWithoutCurrent = { ...state.teamCounts };
      countsWithoutCurrent[current.team] = (countsWithoutCurrent[current.team] || 1) - 1;
      if (countsWithoutCurrent[current.team] <= 0) delete countsWithoutCurrent[current.team];

      const candidates = getEligiblePlayers(
        candidatesByPosition[slot.position],
        slot.position,
        usedWithoutCurrent,
        countsWithoutCurrent,
        maxPerTeam,
        remainingBank + current.price
      );

      for (const candidate of candidates) {
        if (candidate.id === current.id) continue;
        const extraCost = candidate.price - current.price;
        if (extraCost <= 0 || extraCost > remainingBank) continue;
        const bankAfterSwap = remainingBank - extraCost;
        const scoreDelta = candidate.score - current.score;
        const qualityPenalty = Math.max(0, -scoreDelta);
        const metric =
          (bankAfterSwap <= MAX_ACCEPTABLE_BANK ? 1 : 0) * 1000000000
          + extraCost / 1000
          + scoreDelta * 1000
          - qualityPenalty * 3000;
        if (!bestSwap || metric > bestSwap.metric) {
          bestSwap = { slotKey, playerId: candidate.id, metric };
        }
      }
    }
    if (!bestSwap) return improved;
    improved[bestSwap.slotKey] = bestSwap.playerId;
  }
  return improved;
}

export function solveAutofill(payload, hooks = {}) {
  const startedAt = performance.now();
  const deadline = startedAt + Math.max(1000, Number(payload.maxDurationMs || 18000));
  const {
    slots,
    players,
    bankLeft,
    usedIds: usedIdValues,
    teamCounts: teamCountsBase,
    maxPerTeam
  } = payload;
  const usedIdsBase = new Set(usedIdValues);
  const playersById = Object.fromEntries(players.map(player => [player.id, player]));
  const positions = [...new Set(slots.map(slot => slot.position))];
  const candidatesByPosition = Object.fromEntries(
    positions.map(position => [
      position,
      getEligiblePlayers(players, position, usedIdsBase, teamCountsBase, maxPerTeam)
    ])
  );

  if (slots.some(slot => !candidatesByPosition[slot.position].length)) {
    return { solution: null, stats: { evaluatedCombinations: 0, visitedNodes: 0, durationMs: performance.now() - startedAt } };
  }

  const orderedSlots = [...slots].sort((a, b) => {
    const countDiff = candidatesByPosition[a.position].length - candidatesByPosition[b.position].length;
    return countDiff || a.position.localeCompare(b.position) || a.key.localeCompare(b.key);
  });
  const greedy = greedyIncumbent(
    orderedSlots,
    candidatesByPosition,
    bankLeft,
    usedIdsBase,
    teamCountsBase,
    maxPerTeam
  );
  let visitedNodes = 0;
  let evaluatedCombinations = greedy ? 1 : 0;
  let budgetPrunes = 0;
  let scorePrunes = 0;
  let teamPrunes = 0;
  let symmetryPrunes = 0;
  let firstIncumbentMs = greedy ? performance.now() - startedAt : null;

  const cheapestByPosition = Object.fromEntries(
    positions.map(position => [
      position,
      [...candidatesByPosition[position]].sort((a, b) => a.price - b.price)
    ])
  );
  const feasibility = greedy || cheapestFeasibleIncumbent(
    orderedSlots,
    cheapestByPosition,
    bankLeft,
    usedIdsBase,
    teamCountsBase,
    maxPerTeam,
    deadline
  );
  let bestSolution = feasibility?.solution || null;
  let bestScore = feasibility?.score ?? -Infinity;
  if (bestSolution && firstIncumbentMs === null) {
    firstIncumbentMs = performance.now() - startedAt;
    evaluatedCombinations++;
  }
  hooks.onProgress?.({
    visitedNodes,
    bestScore: Number.isFinite(bestScore) ? bestScore : null,
    hasSolution: Boolean(bestSolution),
    elapsedMs: performance.now() - startedAt
  });

  const maxNeededByPosition = {};
  for (const slot of orderedSlots) {
    maxNeededByPosition[slot.position] = (maxNeededByPosition[slot.position] || 0) + 1;
  }

  function buildSuffixBestTables(values, maxCount, preferHigher) {
    const tables = Array.from({ length: values.length + 1 }, () =>
      Array(maxCount + 1).fill(preferHigher ? -Infinity : Infinity)
    );
    tables[values.length][0] = 0;
    for (let start = values.length - 1; start >= 0; start--) {
      tables[start][0] = 0;
      for (let count = 1; count <= maxCount; count++) {
        const skip = tables[start + 1][count];
        const tail = tables[start + 1][count - 1];
        const take = Number.isFinite(tail) ? values[start] + tail : tail;
        tables[start][count] = preferHigher
          ? Math.max(skip, take)
          : Math.min(skip, take);
      }
    }
    return tables;
  }

  const scoreSuffixBounds = {};
  const costSuffixBounds = {};
  const lagrangeSuffixBounds = {};
  for (const position of positions) {
    const candidates = candidatesByPosition[position];
    const maxCount = maxNeededByPosition[position];
    scoreSuffixBounds[position] = buildSuffixBestTables(
      candidates.map(player => player.score),
      maxCount,
      true
    );
    costSuffixBounds[position] = buildSuffixBestTables(
      candidates.map(player => player.price),
      maxCount,
      false
    );
    lagrangeSuffixBounds[position] = SCORE_PRICE_LAMBDAS.map(lambda =>
      buildSuffixBestTables(
        candidates.map(player => player.score - lambda * player.price),
        maxCount,
        true
      )
    );
  }
  function remainingPositionCounts(fromIndex) {
    const counts = {};
    for (let index = fromIndex; index < orderedSlots.length; index++) {
      const position = orderedSlots[index].position;
      counts[position] = (counts[position] || 0) + 1;
    }
    return counts;
  }

  function optimisticScore(fromIndex, lastIndexByPosition) {
    let total = 0;
    for (const [position, count] of Object.entries(remainingPositionCounts(fromIndex))) {
      const startIndex = lastIndexByPosition[position] ?? 0;
      const bound = scoreSuffixBounds[position][startIndex]?.[count] ?? -Infinity;
      if (!Number.isFinite(bound)) return -Infinity;
      total += bound;
    }
    return total;
  }

  function budgetAwareOptimisticScore(fromIndex, lastIndexByPosition, remainingBudget) {
    const remainingCounts = remainingPositionCounts(fromIndex);
    let bestUpperBound = Infinity;
    for (let lambdaIndex = 0; lambdaIndex < SCORE_PRICE_LAMBDAS.length; lambdaIndex++) {
      const lambda = SCORE_PRICE_LAMBDAS[lambdaIndex];
      let adjustedTotal = lambda * remainingBudget;
      let feasibleRelaxation = true;
      for (const [position, count] of Object.entries(remainingCounts)) {
        const startIndex = lastIndexByPosition[position] ?? 0;
        const bound = lagrangeSuffixBounds[position][lambdaIndex][startIndex]?.[count] ?? -Infinity;
        if (!Number.isFinite(bound)) {
          feasibleRelaxation = false;
          break;
        }
        adjustedTotal += bound;
      }
      if (feasibleRelaxation) bestUpperBound = Math.min(bestUpperBound, adjustedTotal);
    }
    return bestUpperBound;
  }

  function minimumRemainingCost(fromIndex, lastIndexByPosition) {
    let total = 0;
    for (const [position, count] of Object.entries(remainingPositionCounts(fromIndex))) {
      const startIndex = lastIndexByPosition[position] ?? 0;
      const bound = costSuffixBounds[position][startIndex]?.[count] ?? Infinity;
      if (!Number.isFinite(bound)) return Infinity;
      total += bound;
    }
    return total;
  }

  function dfs(index, solution, usedIds, teamCounts, spent, score, lastIndexByPosition) {
    visitedNodes++;
    if (visitedNodes % 2000 === 0) {
      hooks.onProgress?.({
        visitedNodes,
        bestScore: Number.isFinite(bestScore) ? bestScore : null,
        hasSolution: Boolean(bestSolution),
        elapsedMs: performance.now() - startedAt
      });
    }
    if (visitedNodes % 5000 === 0 && performance.now() > deadline) {
      throw new Error("Worker-beregningen overskred tidsgrænsen.");
    }
    if (index === orderedSlots.length) {
      evaluatedCombinations++;
      if (score > bestScore) {
        bestScore = score;
        bestSolution = { ...solution };
        if (firstIncumbentMs === null) firstIncumbentMs = performance.now() - startedAt;
      }
      return;
    }
    const relaxedScoreBound = Math.min(
      optimisticScore(index, lastIndexByPosition),
      budgetAwareOptimisticScore(
        index,
        lastIndexByPosition,
        bankLeft - spent
      )
    );
    if (score + relaxedScoreBound <= bestScore) {
      scorePrunes++;
      return;
    }
    if (spent + minimumRemainingCost(index, lastIndexByPosition) > bankLeft) {
      budgetPrunes++;
      return;
    }

    const slot = orderedSlots[index];
    const candidates = candidatesByPosition[slot.position];
    const startIndex = lastIndexByPosition[slot.position] ?? 0;
    symmetryPrunes += startIndex;
    for (let candidateIndex = startIndex; candidateIndex < candidates.length; candidateIndex++) {
      const player = candidates[candidateIndex];
      if (spent + player.price > bankLeft) {
        budgetPrunes++;
        continue;
      }
      if (usedIds.has(player.id)) continue;
      const currentTeamCount = teamCounts[player.team] || 0;
      if (currentTeamCount >= maxPerTeam) {
        teamPrunes++;
        continue;
      }

      solution[slot.key] = player.id;
      usedIds.add(player.id);
      teamCounts[player.team] = currentTeamCount + 1;
      const previousPositionIndex = lastIndexByPosition[slot.position];
      lastIndexByPosition[slot.position] = candidateIndex + 1;

      dfs(
        index + 1,
        solution,
        usedIds,
        teamCounts,
        spent + player.price,
        score + player.score,
        lastIndexByPosition
      );

      if (previousPositionIndex === undefined) delete lastIndexByPosition[slot.position];
      else lastIndexByPosition[slot.position] = previousPositionIndex;
      if (currentTeamCount === 0) delete teamCounts[player.team];
      else teamCounts[player.team] = currentTeamCount;
      usedIds.delete(player.id);
      delete solution[slot.key];
    }
  }

  dfs(0, {}, new Set(usedIdsBase), { ...teamCountsBase }, 0, 0, {});
  const improved = bestSolution
    ? improveBudgetUse(
        bestSolution,
        orderedSlots,
        playersById,
        candidatesByPosition,
        bankLeft,
        usedIdsBase,
        teamCountsBase,
        maxPerTeam
      )
    : null;
  return {
    solution: improved,
    stats: {
      evaluatedCombinations,
      visitedNodes,
      durationMs: performance.now() - startedAt,
      firstIncumbentMs,
      bestScore: Number.isFinite(bestScore) ? bestScore : null,
      budgetPrunes,
      scorePrunes,
      teamPrunes,
      symmetryPrunes,
      memoizationHits: 0,
      greedyIncumbentFound: Boolean(greedy),
      feasibilityNodes: greedy ? 0 : feasibility.visitedNodes
    }
  };
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = event => {
    const { requestId, payload, forceError } = event.data || {};
    try {
      if (forceError) throw new Error("Kontrolleret worker-fejl");
      let lastProgressAt = -Infinity;
      const result = solveAutofill(payload, {
        onProgress(progress) {
          const now = performance.now();
          if (now - lastProgressAt < 350) return;
          lastProgressAt = now;
          self.postMessage({ requestId, ok: true, progress });
        }
      });
      self.postMessage({ requestId, ok: true, ...result });
    } catch (error) {
      self.postMessage({
        requestId,
        ok: false,
        error: error && error.message ? error.message : String(error)
      });
    }
  };
}
