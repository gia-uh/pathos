from __future__ import annotations
from typing import Any, Callable, Iterable
from pathos.core.capabilities import Capability


class Space:
    """Problem definition container for PATHOS search algorithms.

    Declare *what your problem can do* using decorator hooks.
    The auto-solver selects the most powerful compatible algorithm.

    Example:
        >>> space = Space().initial("A")
        >>> @space.successors
        ... def expand(state): yield "go_B", "B"
        >>> @space.goal
        ... def is_goal(state): return state == "B"
        >>> result = space.solver().solve()
    """

    def __init__(self) -> None:
        self.capabilities: set[Capability] = set()
        self._initial_value: Any = None
        self._initial_factory: Callable[[], Any] | None = None
        self._timeout: float | None = None
        self._n_workers: int = 1
        self._adversarial: bool = False
        self._players: int = 2
        self._maximizing_player: int = 0

        self._successors: Callable | None = None
        self._goal: Callable | None = None
        self._heuristic: Callable | None = None
        self._evaluate: Callable | None = None
        self._terminal: Callable | None = None
        self._utility: Callable | None = None
        self._reverse_successors: Callable | None = None

    # --- fluent builder ---

    def initial(self, state: Any) -> Space:
        if callable(state):
            self._initial_factory = state
        else:
            self._initial_value = state
        return self

    def adversarial(self, players: int = 2, maximizing_player: int = 0) -> Space:
        self._adversarial = True
        self._players = players
        self._maximizing_player = maximizing_player
        self.capabilities.add(Capability.TERMINAL)  # structural flag
        return self

    def timeout(self, seconds: float) -> Space:
        self._timeout = seconds
        return self

    def parallel(self, workers: int) -> Space:
        self._n_workers = workers
        return self

    @property
    def _initial(self) -> Any:
        if self._initial_factory is not None:
            return self._initial_factory()
        return self._initial_value

    # --- decorator hooks ---

    def _make_hook(self, attr: str, cap: Capability) -> Callable:
        def decorator(fn: Callable) -> Callable:
            setattr(self, attr, fn)
            self.capabilities.add(cap)
            return fn
        return decorator

    @property
    def successors(self) -> Callable:
        return self._make_hook("_successors", Capability.SUCCESSORS)

    @property
    def goal(self) -> Callable:
        return self._make_hook("_goal", Capability.GOAL)

    @property
    def heuristic(self) -> Callable:
        return self._make_hook("_heuristic", Capability.HEURISTIC)

    @property
    def evaluate(self) -> Callable:
        return self._make_hook("_evaluate", Capability.EVALUATE)

    @property
    def terminal(self) -> Callable:
        return self._make_hook("_terminal", Capability.TERMINAL)

    @property
    def utility(self) -> Callable:
        return self._make_hook("_utility", Capability.UTILITY)

    @property
    def reverse_successors(self) -> Callable:
        return self._make_hook("_reverse_successors", Capability.REVERSE_SUCCESSORS)

    # --- solver factory ---

    def solver(
        self,
        candidates: list | None = None,
        timeout: float | None = None,
    ) -> "Solver":
        from pathos.core.solver import Solver
        return Solver(self, candidates=candidates, timeout=timeout or self._timeout)
