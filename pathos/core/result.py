from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Result returned by every PATHOS algorithm after a solve() call.

    Attributes:
        solution: The goal state (or best state found for local search).
        path: List of (action, state) tuples from initial to solution,
            or None for algorithms that don't track paths.
        cost: Accumulated cost to reach the solution, or None if not applicable.
        algorithm: Name of the algorithm that produced this result.
        nodes_expanded: Number of nodes expanded during search.
        elapsed: Wall-clock seconds taken by solve().
        found: True if a solution was found, False if search exhausted.
        epsilon: Suboptimality bound on `cost`. 1.0 = proven optimal,
            >1.0 = cost ≤ ε × optimal (bounded suboptimal),
            inf = unbounded suboptimal (e.g. greedy),
            None = not applicable (e.g. metaheuristics with no quality bound,
            or not_found results).
    """
    solution: Any
    path: list[tuple[Any, Any]] | None
    cost: float | None
    algorithm: str
    nodes_expanded: int
    elapsed: float
    found: bool
    epsilon: float | None = None

    @property
    def optimal(self) -> bool:
        """True iff the result is proven optimal (epsilon == 1.0)."""
        return self.epsilon == 1.0

    @classmethod
    def not_found(
        cls, algorithm: str, nodes_expanded: int, elapsed: float
    ) -> SearchResult:
        return cls(
            solution=None,
            path=None,
            cost=None,
            algorithm=algorithm,
            nodes_expanded=nodes_expanded,
            elapsed=elapsed,
            found=False,
            epsilon=None,
        )
