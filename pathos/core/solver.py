from __future__ import annotations
import signal
import time
import warnings
from typing import TYPE_CHECKING
from pathos.core.capabilities import Capability
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
        # When the space declares a goal, prefer algorithms that honour it
        # — local-search / metaheuristic algorithms optimise @evaluate and
        # ignore @goal, so they should not outrank goal-seeking algorithms
        # on a goal-bearing problem just because their power_rank is higher.
        # Falls back to the full pool if no goal-honouring algorithm fits.
        if Capability.GOAL in self.space.capabilities:
            goal_honoring = [
                c for c in compatible if Capability.GOAL in c.requires
            ]
            if goal_honoring:
                compatible = goal_honoring
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
        if self.timeout is None:
            return cls(self.space).solve()

        # Wall-clock guard via SIGALRM. Unix-only and main-thread-only;
        # multiprocessing workers spawned by .parallel(n) are out of scope
        # because solve() is the top-level entry, always on the main thread.
        def _handler(signum, frame):  # type: ignore[no-untyped-def]
            raise TimeoutError(f"solver timeout exceeded ({self.timeout}s)")

        prev_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        t0 = time.perf_counter()
        try:
            return cls(self.space).solve()
        except TimeoutError:
            return SearchResult.not_found(
                cls.__name__, 0, time.perf_counter() - t0,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev_handler)
