from __future__ import annotations
import warnings
import time
from typing import TYPE_CHECKING
from pathos.core.result import SearchResult

if TYPE_CHECKING:
    from pathos.core.space import Space
    from pathos.algorithms.base import Algorithm


_REGISTRY: list[type[Algorithm]] = []


def register(cls: type[Algorithm]) -> type[Algorithm]:
    _REGISTRY.append(cls)
    return cls


class Solver:
    def __init__(
        self,
        space: Space,
        candidates: list[type[Algorithm]] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.space = space
        self.candidates = candidates
        self.timeout = timeout

    def _select(self) -> type[Algorithm]:
        pool = self.candidates if self.candidates is not None else _REGISTRY
        compatible = [cls for cls in pool if cls.compatible_with(self.space)]
        if not compatible:
            raise RuntimeError(
                f"No compatible algorithm for capabilities: "
                f"{', '.join(c.name for c in self.space.capabilities)}"
            )
        best = max(compatible, key=lambda cls: cls.power_rank)
        # warn about unused capabilities
        used = best.requires
        declared = self.space.capabilities
        unused = declared - used
        if unused:
            warnings.warn(
                f"Capabilities declared but not used by {best.__name__}: "
                f"{', '.join(c.name.lower() for c in unused)}",
                UserWarning,
                stacklevel=3,
            )
        return best

    def solve(self) -> SearchResult:
        cls = self._select()
        return cls(self.space).solve()
