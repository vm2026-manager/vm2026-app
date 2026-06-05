const MAX_ACCEPTABLE_BANK = 500000;

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

export function solveAutofill(payload) {
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
  let bestSolution = greedy?.solution || null;
  let bestScore = greedy?.score ?? -Infinity;
  let visitedNodes = 0;
  let evaluatedCombinations = greedy ? 1 : 0;

  const cheapestByPosition = Object.fromEntries(
    positions.map(position => [
      position,
      [...candidatesByPosition[position]].sort((a, b) => a.price - b.price)
    ])
  );

  function remainingPositionCounts(fromIndex) {
    const counts = {};
    for (let index = fromIndex; index < orderedSlots.length; index++) {
      const position = orderedSlots[index].position;
      counts[position] = (counts[position] || 0) + 1;
    }
    return counts;
  }

  function optimisticScore(fromIndex, usedIds, lastIndexByPosition) {
    let total = 0;
    for (const [position, count] of Object.entries(remainingPositionCounts(fromIndex))) {
      const startIndex = lastIndexByPosition[position] ?? 0;
      const available = candidatesByPosition[position]
        .slice(startIndex)
        .filter(player => !usedIds.has(player.id));
      if (available.length < count) return -Infinity;
      for (let index = 0; index < count; index++) total += available[index].score;
    }
    return total;
  }

  function minimumRemainingCost(fromIndex, usedIds, lastIndexByPosition) {
    let total = 0;
    for (const [position, count] of Object.entries(remainingPositionCounts(fromIndex))) {
      const startIndex = lastIndexByPosition[position] ?? 0;
      const allowedIds = new Set(
        candidatesByPosition[position].slice(startIndex).map(player => player.id)
      );
      const available = cheapestByPosition[position]
        .filter(player => allowedIds.has(player.id) && !usedIds.has(player.id));
      if (available.length < count) return Infinity;
      for (let index = 0; index < count; index++) total += available[index].price;
    }
    return total;
  }

  function dfs(index, solution, usedIds, teamCounts, spent, score, lastIndexByPosition) {
    visitedNodes++;
    if (visitedNodes % 5000 === 0 && performance.now() > deadline) {
      throw new Error("Worker-beregningen overskred tidsgrænsen.");
    }
    if (index === orderedSlots.length) {
      evaluatedCombinations++;
      if (score > bestScore) {
        bestScore = score;
        bestSolution = { ...solution };
      }
      return;
    }
    if (score + optimisticScore(index, usedIds, lastIndexByPosition) <= bestScore) return;
    if (spent + minimumRemainingCost(index, usedIds, lastIndexByPosition) > bankLeft) return;

    const slot = orderedSlots[index];
    const candidates = candidatesByPosition[slot.position];
    const startIndex = lastIndexByPosition[slot.position] ?? 0;
    for (let candidateIndex = startIndex; candidateIndex < candidates.length; candidateIndex++) {
      const player = candidates[candidateIndex];
      if (spent + player.price > bankLeft) continue;
      if (usedIds.has(player.id)) continue;
      const currentTeamCount = teamCounts[player.team] || 0;
      if (currentTeamCount >= maxPerTeam) continue;

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
      durationMs: performance.now() - startedAt
    }
  };
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = event => {
    const { requestId, payload, forceError } = event.data || {};
    try {
      if (forceError) throw new Error("Kontrolleret worker-fejl");
      self.postMessage({ requestId, ok: true, ...solveAutofill(payload) });
    } catch (error) {
      self.postMessage({
        requestId,
        ok: false,
        error: error && error.message ? error.message : String(error)
      });
    }
  };
}
