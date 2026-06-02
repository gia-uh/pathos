from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal
from pathos.core.capabilities import Capability

if TYPE_CHECKING:
    from pathos.core.solver import Solver


Optimality = Literal["exact", "approximate"]
_VALID_OPTIMALITY: frozenset[str] = frozenset({"exact", "approximate"})


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
        self._optimality: Optimality = "exact"
        self._n_workers: int = 1
        self._adversarial: bool = False
        self._players: int = 2
        self._maximizing_player: int = 0

        # These are set via decorator hooks before any algorithm uses them;
        # typed as Any to avoid false-positive "None not callable" mypy errors.
        self._successors: Any = None
        self._goal: Any = None
        self._heuristic: Any = None
        self._evaluate: Any = None
        self._terminal: Any = None
        self._utility: Any = None
        self._reverse_successors: Any = None

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

    def optimality(self, mode: Optimality) -> Space:
        """Declare optimality preference for the auto-solver.

        - "exact" (default): prefer admissible algorithms that guarantee
          the optimal solution (e.g. A*, IDA*, BFS, UCS).
        - "approximate": prefer bounded-suboptimal variants that trade
          a small quality bound for speed (e.g. WeightedA*). Useful when
          A*'s admissibility bill is too expensive and an ε-optimal
          answer is acceptable.

        Mirrors the .timeout() pattern: the value is read by
        Algorithm.score_for(space) at selection time, so any
        capability-compatible algorithm can consult it.
        """
        if mode not in _VALID_OPTIMALITY:
            raise ValueError(
                f"optimality must be one of {sorted(_VALID_OPTIMALITY)}, "
                f"got {mode!r}"
            )
        self._optimality = mode
        return self

    def parallel(self, workers: int) -> Space:
        # evaluate/successors functions must be picklable (module-level) when workers > 1
        self._n_workers = workers
        return self

    @property
    def _initial(self) -> Any:
        if self._initial_factory is not None:
            return self._initial_factory()
        return self._initial_value

    # --- decorator hooks ---

    def _make_hook(self, attr: str, cap: Capability) -> Callable[..., Any]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            setattr(self, attr, fn)
            self.capabilities.add(cap)
            return fn
        return decorator

    @property
    def successors(self) -> Callable[..., Any]:
        return self._make_hook("_successors", Capability.SUCCESSORS)

    @property
    def goal(self) -> Callable[..., Any]:
        return self._make_hook("_goal", Capability.GOAL)

    @property
    def heuristic(self) -> Callable[..., Any]:
        return self._make_hook("_heuristic", Capability.HEURISTIC)

    @property
    def evaluate(self) -> Callable[..., Any]:
        return self._make_hook("_evaluate", Capability.EVALUATE)

    @property
    def terminal(self) -> Callable[..., Any]:
        return self._make_hook("_terminal", Capability.TERMINAL)

    @property
    def utility(self) -> Callable[..., Any]:
        return self._make_hook("_utility", Capability.UTILITY)

    @property
    def reverse_successors(self) -> Callable[..., Any]:
        return self._make_hook("_reverse_successors", Capability.REVERSE_SUCCESSORS)

    # --- solver factory ---

    def solver(
        self,
        candidates: list[Any] | None = None,
        timeout: float | None = None,
        optimality: Optimality | None = None,
    ) -> "Solver":
        from pathos.core.solver import Solver
        if optimality is not None and optimality not in _VALID_OPTIMALITY:
            raise ValueError(
                f"optimality must be one of {sorted(_VALID_OPTIMALITY)}, "
                f"got {optimality!r}"
            )
        return Solver(
            self,
            candidates=candidates,
            timeout=timeout or self._timeout,
            optimality=optimality,
        )
