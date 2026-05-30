from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
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
