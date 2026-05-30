from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
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
