from __future__ import annotations
import signal
import time
import warnings
from typing import TYPE_CHECKING
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult

if TYPE_CHECKING:
    from pathos.core.space import Space, Mode
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
        mode: Mode | None = None,
    ) -> None:
        self.space = space
        self.candidates = candidates
        self.timeout = timeout
        # `None` means "inherit from space"; an explicit value overrides
        # the space's setting for this solver call only (mirrors timeout).
        self.mode: Mode | None = mode

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
        # Per-solver mode overrides the space setting for this call.
        prev_mode = self.space._mode
        if self.mode is not None and self.mode != prev_mode:
            self.space._mode = self.mode
            try:
                best = max(compatible, key=lambda cls: cls.score_for(self.space))
            finally:
                self.space._mode = prev_mode
        else:
            best = max(compatible, key=lambda cls: cls.score_for(self.space))
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

        # Cooperative-cancel + watchdog backstop.
        #
        # On primary SIGALRM (ITIMER_REAL), set the cancel token.
        # Cooperating algorithms check space._cancel_requested() and
        # return best-so-far cleanly. WATCHDOG_GRACE seconds later, a
        # follow-up SIGVTALRM (ITIMER_VIRTUAL) raises TimeoutError as a
        # backstop for any algorithm that doesn't yet check the token
        # (e.g. IDA*, CSP algorithms in v1).
        WATCHDOG_GRACE = 2.0

        def _on_primary(signum, frame):  # type: ignore[no-untyped-def]
            self.space._request_cancel()
            signal.signal(signal.SIGVTALRM, _on_watchdog)
            signal.setitimer(signal.ITIMER_VIRTUAL, WATCHDOG_GRACE)

        def _on_watchdog(signum, frame):  # type: ignore[no-untyped-def]
            raise TimeoutError(
                f"solver timeout exceeded ({self.timeout}s) and "
                f"cancel-token grace ({WATCHDOG_GRACE}s) lapsed",
            )

        prev_primary = signal.signal(signal.SIGALRM, _on_primary)
        prev_secondary = signal.signal(signal.SIGVTALRM, signal.SIG_IGN)
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
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            signal.signal(signal.SIGALRM, prev_primary)
            signal.signal(signal.SIGVTALRM, prev_secondary)
            # Reset cancel token for any future solver call on this space.
            from pathos.core.cancel import CancelToken
            self.space._cancel_token = CancelToken()
