from __future__ import annotations
import dataclasses
import math
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
            if self.space._cancel_requested():
                break
            state = self.space._initial
            cost = self.space._evaluate(state)
            sideways = 0
            while True:
                if self.space._cancel_requested():
                    break
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
            if self.space._cancel_requested():
                break
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
            if self.space._cancel_requested():
                break
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


@register
class AnytimeLocal(Algorithm):
    """Anytime Local Search — meta-algorithm that delivers best-effort
    under a wall-clock budget for pure-optimization spaces.

    Runs a cascade `[HillClimbing, SimulatedAnnealing, TabuSearch]`:
    HillClimbing is a cheap fast-probe (greedy descent, gets stuck in
    local optima); SimulatedAnnealing and TabuSearch are escape phases
    that explore around the incumbent. The best (lowest-cost) result
    across all phases is returned — UNLIKE AnytimeCSP, lower-cost
    semantics is meaningful for local search, so the AnytimeAStar
    incumbent rule applies.

    Wins auto-selection only when `space._mode == "auto"` — score_for
    returns -inf otherwise so users explicitly opting into "exact" or
    "approximate" keep the base-algorithm pick.

    Requires only the intersection of HC/SA/Tabu: SUCCESSORS+EVALUATE.
    Does NOT declare GOAL, so on goal-bearing spaces the goal-honoring
    filter in `Solver._select` keeps AnytimeLocal out — letting
    AnytimeAStar (with HEURISTIC) or uninformed goal algorithms win
    instead. This is intentional: pure-optimization is AnytimeLocal's
    lane.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.EVALUATE})
    power_rank = 0  # irrelevant — score_for short-circuits

    # Populated immediately below this class definition.
    SCHEDULE: list[tuple[type[Algorithm], dict[str, Any]]] = []

    @classmethod
    def score_for(cls, space: Any) -> float:
        if space._mode == "auto":
            return 1000.0
        return -math.inf

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        best: SearchResult | None = None
        for alg_cls, kwargs in self.SCHEDULE:
            if self.space._cancel_requested():
                break
            phase_result = alg_cls(self.space, **kwargs).solve()
            if self._is_better(phase_result, best):
                best = phase_result
        if best is None:
            return SearchResult.not_found(
                "AnytimeLocal", 0, time.perf_counter() - t0,
            )
        return dataclasses.replace(
            best,
            algorithm="AnytimeLocal",
            elapsed=time.perf_counter() - t0,
        )

    @staticmethod
    def _is_better(candidate: SearchResult, best: SearchResult | None) -> bool:
        """Incumbent comparison: lower cost wins. Mirrors AnytimeAStar.

        - A `found=False` candidate never displaces anything.
        - If `best` is None and candidate is found, candidate wins.
        - Otherwise lower cost wins; `None` and `inf` treated as worst.
        """
        if not candidate.found:
            return False
        if best is None:
            return True
        c_cost = candidate.cost if candidate.cost is not None else math.inf
        b_cost = best.cost if best.cost is not None else math.inf
        return c_cost < b_cost


# Cascade body populated at import time after SimulatedAnnealing is in
# scope (cross-module — see pathos/algorithms/__init__.py).
from pathos.algorithms.evolutionary import SimulatedAnnealing  # noqa: E402

AnytimeLocal.SCHEDULE = [
    (HillClimbing, {"max_restarts": 3}),
    (SimulatedAnnealing, {"max_iter": 500, "T0": 100.0, "cooling": 0.99}),
    (TabuSearch, {"max_iter": 200, "tabu_size": 20}),
]
