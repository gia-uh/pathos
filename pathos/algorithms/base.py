from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult

if TYPE_CHECKING:
    from pathos.core.space import Space


class Algorithm(ABC):
    requires: frozenset[Capability] = frozenset()
    power_rank: int = 0  # higher = preferred when multiple are compatible

    def __init__(self, space: Space) -> None:
        missing = self.requires - space.capabilities
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires capabilities: "
                f"{', '.join(c.name for c in missing)}"
            )
        self.space = space
        self._n_workers: int = space._n_workers

    @abstractmethod
    def solve(self) -> SearchResult:
        ...

    @classmethod
    def compatible_with(cls, space: Space) -> bool:
        return cls.requires <= space.capabilities

    @classmethod
    def score_for(cls, space: Space) -> float:
        """Context-aware preference score for `space`. Higher = preferred.

        The auto-solver picks the compatible algorithm with the highest
        `score_for(space)`. Defaults to `float(power_rank)`, so any
        algorithm that doesn't care about problem context keeps its
        existing fixed-rank behavior.

        Override to encode self-knowledge that isn't captured by the
        coarse static rank — for example, "I work well on small problems
        but lose to a sibling on large ones", or "without `@successors`
        I degenerate to a no-op, so cede to siblings that don't need it".

        Returning a value below `power_rank` lets the algorithm cede to
        better-suited siblings without changing its global ordering.
        """
        return float(cls.power_rank)

    def _goal_reached(self, state: Any) -> bool:
        """Whether `state` satisfies the space's declared goal predicate.

        If the space has no GOAL capability, returns True — the algorithm
        terminated by its own stopping rule and there's no goal to check.
        Otherwise consults `space._goal(state)`.

        Used by local-search / metaheuristic algorithms (HC, Tabu, Beam, SA,
        GA, DE) so they can't report `found=True` on a goal-bearing problem
        when the best state they reached doesn't actually satisfy the goal.
        """
        if Capability.GOAL in self.space.capabilities:
            return bool(self.space._goal(state))
        return True
