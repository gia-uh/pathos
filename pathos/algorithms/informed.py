from __future__ import annotations
import heapq
import time
import math
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class AStar(Algorithm):
    """A* search — optimal pathfinding with admissible heuristic.

    Requires: successors, goal, heuristic, evaluate.
    Selects the path minimizing g(n) + h(n) where g is actual cost
    and h is an admissible heuristic estimate to goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 30 (preferred over BFS/DFS/Greedy when available).
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 30

    @classmethod
    def score_for(cls, space: Any) -> float:
        # Admissible: the user explicitly asking for "approximate" wants
        # speed, so cede to bounded-suboptimal siblings.
        if space._mode == "approximate":
            return float(cls.power_rank) - 10.0
        return float(cls.power_rank)

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        h0 = self.space._heuristic(initial)
        frontier: list[Any] = [(h0, 0, 0.0, initial, [])]
        g_score: dict[Any, float] = {initial: 0.0}
        counter = 1
        expanded = 0

        while frontier:
            f, _, g, state, path = heapq.heappop(frontier)
            if g > g_score.get(state, math.inf):
                continue
            if self.space._goal(state):
                return SearchResult(
                    state, path, g, "AStar", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                new_g = g + self.space._evaluate(child)
                if new_g < g_score.get(child, math.inf):
                    g_score[child] = new_g
                    f_val = new_g + self.space._heuristic(child)
                    heapq.heappush(frontier, (f_val, counter, new_g, child, path + [(action, child)]))
                    counter += 1

        return SearchResult.not_found("AStar", expanded, time.perf_counter() - t0)


@register
class GreedyBestFirst(Algorithm):
    """Greedy Best-First search — fast but not optimal; follows heuristic greedily.

    Requires: successors, goal, heuristic.

    Attributes:
        requires: Capability set needed.
        power_rank: 20.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.HEURISTIC})
    power_rank = 20

    @classmethod
    def score_for(cls, space: Any) -> float:
        # Unbounded suboptimal: gets a small bump in approximate mode so
        # it can outrank demoted-A*, but stays below WeightedA*'s bump
        # because no quality bound is provided.
        if space._mode == "approximate":
            return float(cls.power_rank) + 5.0
        return float(cls.power_rank)

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        frontier: list[Any] = [(self.space._heuristic(initial), 0, initial, [])]
        visited: set[Any] = set()
        counter = 1
        expanded = 0

        while frontier:
            _, _, state, path = heapq.heappop(frontier)
            if state in visited:
                continue
            visited.add(state)
            if self.space._goal(state):
                return SearchResult(
                    state, path, float(len(path)), "GreedyBestFirst",
                    expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                if child not in visited:
                    heapq.heappush(frontier, (
                        self.space._heuristic(child), counter, child, path + [(action, child)]
                    ))
                    counter += 1

        return SearchResult.not_found("GreedyBestFirst", expanded, time.perf_counter() - t0)


@register
class WeightedAStar(Algorithm):
    """Weighted A* — trades optimality for speed via inflated heuristic.

    Requires: successors, goal, heuristic, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 28.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 28

    @classmethod
    def score_for(cls, space: Any) -> float:
        # ε-bounded suboptimal: the explicit target of `optimality="approximate"`.
        # Bump above admissible A* / IDA* / Bidirectional in approximate mode.
        if space._mode == "approximate":
            return float(cls.power_rank) + 10.0
        return float(cls.power_rank)

    def __init__(self, space: Any, weight: float = 2.0) -> None:
        super().__init__(space)
        self.weight = weight

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        h0 = self.space._heuristic(initial)
        frontier: list[Any] = [(h0 * self.weight, 0, 0.0, initial, [])]
        g_score: dict[Any, float] = {initial: 0.0}
        counter = 1
        expanded = 0

        while frontier:
            _, _, g, state, path = heapq.heappop(frontier)
            if g > g_score.get(state, math.inf):
                continue
            if self.space._goal(state):
                return SearchResult(
                    state, path, g, "WeightedAStar", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                new_g = g + self.space._evaluate(child)
                if new_g < g_score.get(child, math.inf):
                    g_score[child] = new_g
                    f = new_g + self.weight * self.space._heuristic(child)
                    heapq.heappush(frontier, (f, counter, new_g, child, path + [(action, child)]))
                    counter += 1

        return SearchResult.not_found("WeightedAStar", expanded, time.perf_counter() - t0)


@register
class IDAstar(Algorithm):
    """IDA* — iterative deepening A*, memory-efficient optimal search.

    Requires: successors, goal, heuristic, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 25.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 25

    @classmethod
    def score_for(cls, space: Any) -> float:
        # Admissible: same demote as A* in approximate mode.
        if space._mode == "approximate":
            return float(cls.power_rank) - 10.0
        return float(cls.power_rank)

    def _search(self, path: list[Any], g: float, bound: float) -> tuple[float | str, list[Any] | None]:
        state = path[-1]
        f = g + self.space._heuristic(state)
        if f > bound:
            return f, None
        if self.space._goal(state):
            return "FOUND", path[:]
        minimum = math.inf
        for action, child in self.space._successors(state):
            if child not in path:
                path.append(child)
                result, found_path = self._search(path, g + self.space._evaluate(child), bound)
                if result == "FOUND":
                    return "FOUND", found_path
                if isinstance(result, float):
                    minimum = min(minimum, result)
                path.pop()
        return minimum, None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        bound = self.space._heuristic(initial)
        path = [initial]
        expanded = 0
        while True:
            result, found = self._search(path, 0.0, bound)
            if result == "FOUND" and found:
                steps = [(found[i], found[i]) for i in range(1, len(found))]
                return SearchResult(
                    found[-1], steps, float(len(steps)),
                    "IDAstar", expanded, time.perf_counter() - t0, True,
                )
            if result == math.inf:
                return SearchResult.not_found("IDAstar", expanded, time.perf_counter() - t0)
            bound = float(result)  # result is float here (str "FOUND" branch returned above)
            expanded += 1


@register
class BidirectionalAStar(Algorithm):
    """Bidirectional A* — searches from both ends to meet in the middle.

    Requires: successors, goal, heuristic, evaluate, reverse_successors.

    Attributes:
        requires: Capability set needed.
        power_rank: 35.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE,
                          Capability.REVERSE_SUCCESSORS})
    power_rank = 35

    @classmethod
    def score_for(cls, space: Any) -> float:
        # Admissible: demote in approximate mode just like A* / IDA*.
        if space._mode == "approximate":
            return float(cls.power_rank) - 10.0
        return float(cls.power_rank)

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        # Find goal state via BFS first to get a concrete goal node
        from pathos.algorithms.uninformed import BFS
        bfs_result = BFS(self.space).solve()
        if not bfs_result.found:
            return SearchResult.not_found("BidirectionalAStar", 0, time.perf_counter() - t0)
        goal = bfs_result.solution

        fwd_dist: dict[Any, float] = {initial: 0.0}
        bwd_dist: dict[Any, float] = {goal: 0.0}
        fwd_prev: dict[Any, Any] = {}
        bwd_prev: dict[Any, Any] = {}
        fwd_open: list[Any] = [(self.space._heuristic(initial), 0, initial)]
        bwd_open: list[Any] = [(self.space._heuristic(goal), 1, goal)]
        counter = 2
        mu = math.inf
        expanded = 0

        while fwd_open or bwd_open:
            if fwd_open:
                _, _, s = heapq.heappop(fwd_open)
                expanded += 1
                for a, child in self.space._successors(s):
                    ng = fwd_dist[s] + self.space._evaluate(child)
                    if ng < fwd_dist.get(child, math.inf):
                        fwd_dist[child] = ng
                        fwd_prev[child] = s
                        heapq.heappush(fwd_open, (ng + self.space._heuristic(child), counter, child))
                        counter += 1
                        if child in bwd_dist:
                            mu = min(mu, ng + bwd_dist[child])
            if bwd_open:
                _, _, s = heapq.heappop(bwd_open)
                expanded += 1
                for a, prev in self.space._reverse_successors(s):
                    ng = bwd_dist[s] + self.space._evaluate(s)
                    if ng < bwd_dist.get(prev, math.inf):
                        bwd_dist[prev] = ng
                        bwd_prev[prev] = s
                        heapq.heappush(bwd_open, (ng + self.space._heuristic(prev), counter, prev))
                        counter += 1
                        if prev in fwd_dist:
                            mu = min(mu, fwd_dist[prev] + ng)
            if fwd_open and bwd_open:
                best_f = fwd_open[0][0] if fwd_open else math.inf
                best_b = bwd_open[0][0] if bwd_open else math.inf
                if best_f + best_b >= mu:
                    break

        if mu == math.inf:
            return SearchResult.not_found("BidirectionalAStar", expanded, time.perf_counter() - t0)
        return SearchResult(goal, [], mu, "BidirectionalAStar", expanded, time.perf_counter() - t0, True)


@register
class AnytimeAStar(Algorithm):
    """Anytime A* — meta-algorithm that delivers best-effort under a
    wall-clock budget.

    Runs a cascade of progressively-tighter A* variants
    [GreedyBestFirst, WeightedAStar(5), WeightedAStar(3),
     WeightedAStar(2), WeightedAStar(1.5), AStar], keeping the best
    incumbent across phases. Exits cleanly when
    space._cancel_requested() is set (either by the Solver's SIGALRM
    or by an outer meta-algorithm).

    Wins auto-selection only when space._mode == "auto" — its
    score_for returns -inf otherwise so users explicitly opting into
    "exact" or "approximate" keep the base-algorithm pick.

    Requires the same capabilities as AStar (the heaviest of the cascade).
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL,
                          Capability.HEURISTIC, Capability.EVALUATE})
    power_rank = 0  # irrelevant — score_for short-circuits

    @classmethod
    def score_for(cls, space: Any) -> float:
        if space._mode == "auto":
            return 1000.0
        return -math.inf

    def solve(self) -> SearchResult:
        # Cascade body lives in Task 8. This scaffold returns not_found
        # so tests verifying selection (not behaviour) can pass.
        t0 = time.perf_counter()
        return SearchResult.not_found(
            "AnytimeAStar", 0, time.perf_counter() - t0,
        )
