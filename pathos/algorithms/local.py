from __future__ import annotations
import time
import random
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register
from pathos.core.parallel import batch_map


@register
class HillClimbing(Algorithm):
    """Hill Climbing local search — greedily improves state via neighbors.

    Requires: successors, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 15.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 15

    @classmethod
    def score_for(cls, space: Any) -> float:
        """Bump HC above TabuSearch (18) on vector-state pure-optimization
        problems. Empirically (benchmarks/FINDINGS.md §2b) HC matches or
        beats TS on every TourSpace size from 5 to 25 cities and is
        4-5× faster — the rigorous-sounding "TS escapes local optima"
        argument doesn't pay off on smooth 2-opt landscapes. Goal-bearing
        problems are unaffected because the goal-preference filter in
        Solver._select already eliminates HC there.
        """
        if (
            Capability.GOAL not in space.capabilities
            and isinstance(space._initial, (list, tuple))
        ):
            return float(cls.power_rank) + 5  # 15+5=20, above TabuSearch=18
        return float(cls.power_rank)

    def __init__(self, space: Any, max_restarts: int = 1, max_sideways: int = 0) -> None:
        super().__init__(space)
        self.max_restarts = max_restarts
        self.max_sideways = max_sideways

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        best_state = self.space._initial
        best_cost = self.space._evaluate(best_state)
        expanded = 0

        for _ in range(self.max_restarts):
            state = self.space._initial
            cost = self.space._evaluate(state)
            sideways = 0
            while True:
                neighbors = list(self.space._successors(state))
                expanded += 1
                if not neighbors:
                    break
                best_n = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))
                n_cost = self.space._evaluate(best_n[1])
                if n_cost < cost:
                    state, cost = best_n[1], n_cost
                    sideways = 0
                elif n_cost == cost and sideways < self.max_sideways:
                    state, cost = best_n[1], n_cost
                    sideways += 1
                else:
                    break
            if cost < best_cost:
                best_cost, best_state = cost, state

        return SearchResult(
            best_state, None, best_cost, "HillClimbing",
            expanded, time.perf_counter() - t0, self._goal_reached(best_state),
        )


@register
class TabuSearch(Algorithm):
    """Tabu Search — escapes local optima via short-term memory of visited states.

    Requires: successors, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 18.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 18

    def __init__(self, space: Any, max_iter: int = 100, tabu_size: int = 10) -> None:
        super().__init__(space)
        self.max_iter = max_iter
        self.tabu_size = tabu_size

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        current = self.space._initial
        best = current
        best_cost = self.space._evaluate(current)
        tabu: list[Any] = [current]
        expanded = 0

        for _ in range(self.max_iter):
            neighbors = [
                (a, child) for a, child in self.space._successors(current)
                if child not in tabu
            ]
            expanded += 1
            if not neighbors:
                break
            action, current = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))
            cost = self.space._evaluate(current)
            if cost < best_cost:
                best_cost, best = cost, current
            tabu.append(current)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)

        return SearchResult(
            best, None, best_cost, "TabuSearch",
            expanded, time.perf_counter() - t0, self._goal_reached(best),
        )


@register
class LocalBeamSearch(Algorithm):
    """Local Beam Search — maintains k parallel states and expands best neighbors.

    Requires: successors, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 16.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 16

    def __init__(self, space: Any, k: int = 5, max_iter: int = 100) -> None:
        super().__init__(space)
        self.k = k
        self.max_iter = max_iter

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        beam = [initial] * self.k
        best = initial
        best_cost = self.space._evaluate(initial)
        expanded = 0

        for _ in range(self.max_iter):
            candidates = []
            for state in beam:
                for _, child in self.space._successors(state):
                    candidates.append(child)
                expanded += 1
            if not candidates:
                break
            candidate_costs = batch_map(self.space._evaluate, candidates, self._n_workers)
            ranked = sorted(zip(candidate_costs, candidates))[: self.k]
            beam = [s for _, s in ranked]
            if ranked[0][0] < best_cost:
                best_cost, best = ranked[0][0], beam[0]

        return SearchResult(
            best, None, best_cost, "LocalBeamSearch",
            expanded, time.perf_counter() - t0, self._goal_reached(best),
        )
