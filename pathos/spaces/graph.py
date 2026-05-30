from __future__ import annotations
from typing import Any
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class GraphSpace(Space):
    """
    Space for problems defined on explicit graphs.
    Automatically provides successors and evaluate from the adjacency structure.

    graph: dict mapping node -> list of (neighbor, edge_cost) tuples
    """

    def __init__(self, graph: dict[Any, list[tuple[Any, float]]]) -> None:
        super().__init__()
        self._graph = graph
        self._pending_edge_cost: dict[Any, float] = {}
        self._setup_successors()
        self._setup_evaluate()

    def _setup_successors(self) -> None:
        graph = self._graph
        pending = self._pending_edge_cost

        def _successors(state: Any) -> Any:
            for neighbor, cost in graph.get(state, []):
                pending[neighbor] = cost
                yield neighbor, neighbor

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)

    def _setup_evaluate(self) -> None:
        pending = self._pending_edge_cost

        def _evaluate(state: Any) -> float:
            return pending.get(state, 1.0)

        self._evaluate = _evaluate
        self.capabilities.add(Capability.EVALUATE)
