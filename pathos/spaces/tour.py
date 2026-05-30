from __future__ import annotations
import random
from typing import Any
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class TourSpace(Space):
    """
    Space for tour/routing problems (TSP and variants).
    Auto-provides @successors as 2-opt neighborhood.
    User provides @evaluate for tour cost.
    """

    def __init__(self, nodes: list[Any], distances: dict[Any, Any] | None = None) -> None:
        super().__init__()
        self._nodes = nodes
        self._distances = distances
        # initial = random tour
        self._initial_factory = lambda: random.sample(nodes, len(nodes))
        self._setup_successors()

    def _setup_successors(self) -> None:
        def _two_opt(tour: list[Any]) -> Any:
            n = len(tour)
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_tour = tour[:i] + list(reversed(tour[i:j])) + tour[j:]
                    yield f"2opt_{i}_{j}", new_tour

        self._successors = _two_opt
        self.capabilities.add(Capability.SUCCESSORS)
