from __future__ import annotations
import time
from collections import deque
from typing import Any
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


@register
class BFS(Algorithm):
    """Breadth-First Search — complete, optimal for unit-cost graphs.

    Requires: successors, goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 10.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 10

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        if self.space._goal(initial):
            return SearchResult(initial, [], 0.0, "BFS", 0, 0.0, True)

        frontier: deque[tuple[Any, list[Any]]] = deque([(initial, [])])
        visited: set[Any] = {initial}
        expanded = 0

        while frontier:
            state, path = frontier.popleft()
            expanded += 1
            for action, child in self.space._successors(state):
                if child in visited:
                    continue
                new_path = path + [(action, child)]
                if self.space._goal(child):
                    return SearchResult(
                        child, new_path, float(len(new_path)),
                        "BFS", expanded, time.perf_counter() - t0, True,
                    )
                visited.add(child)
                frontier.append((child, new_path))

        return SearchResult.not_found("BFS", expanded, time.perf_counter() - t0)


@register
class DFS(Algorithm):
    """Depth-First Search — memory-efficient but incomplete on infinite graphs.

    Requires: successors, goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 5.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 5

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        stack: list[tuple[Any, list[Any]]] = [(initial, [])]
        visited: set[Any] = set()
        expanded = 0

        while stack:
            state, path = stack.pop()
            if state in visited:
                continue
            visited.add(state)
            if self.space._goal(state):
                return SearchResult(
                    state, path, float(len(path)),
                    "DFS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                if child not in visited:
                    stack.append((child, path + [(action, child)]))

        return SearchResult.not_found("DFS", expanded, time.perf_counter() - t0)


@register
class IDDFS(Algorithm):
    """Iterative Deepening DFS — optimal memory with BFS-like completeness.

    Requires: successors, goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 8.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 8

    def _dls(self, state: Any, path: list[Any], depth: int, visited: set[Any]) -> tuple[Any, list[Any]] | None:
        if self.space._goal(state):
            return state, path
        if depth == 0:
            return None
        visited.add(state)
        for action, child in self.space._successors(state):
            if child not in visited:
                result = self._dls(child, path + [(action, child)], depth - 1, visited)
                if result is not None:
                    return result
        visited.discard(state)
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        initial = self.space._initial
        expanded = 0
        for depth in range(1000):
            result = self._dls(initial, [], depth, set())
            if result is not None:
                state, path = result
                return SearchResult(
                    state, path, float(len(path)),
                    "IDDFS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
        return SearchResult.not_found("IDDFS", expanded, time.perf_counter() - t0)


@register
class UCS(Algorithm):
    """Uniform-Cost Search (Dijkstra) — optimal for weighted graphs.

    Requires: successors, goal, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 12.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.EVALUATE})
    power_rank = 12

    def solve(self) -> SearchResult:
        import heapq
        t0 = time.perf_counter()
        initial = self.space._initial
        frontier: list[Any] = [(0.0, 0, initial, [])]
        visited: dict[Any, float] = {}
        counter = 1
        expanded = 0

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost
            if self.space._goal(state):
                return SearchResult(
                    state, path, cost, "UCS", expanded, time.perf_counter() - t0, True,
                )
            expanded += 1
            for action, child in self.space._successors(state):
                edge = self.space._evaluate(child)
                new_cost = cost + edge
                if child not in visited or visited[child] > new_cost:
                    heapq.heappush(
                        frontier,
                        (new_cost, counter, child, path + [(action, child)])
                    )
                    counter += 1

        return SearchResult.not_found("UCS", expanded, time.perf_counter() - t0)
