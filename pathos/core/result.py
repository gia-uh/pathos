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
    """
    solution: Any
    path: list[tuple[Any, Any]] | None
    cost: float | None
    algorithm: str
    nodes_expanded: int
    elapsed: float
    found: bool

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
        )
