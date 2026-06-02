from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal
from pathos.core.capabilities import Capability
from pathos.core.cancel import CancelToken

if TYPE_CHECKING:
    from pathos.core.solver import Solver


Mode = Literal["exact", "approximate", "auto"]
_VALID_MODES: frozenset[str] = frozenset({"exact", "approximate", "auto"})


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
        self._mode: Mode = "auto"
        self._cancel_token: CancelToken = CancelToken()
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

    def mode(self, mode: Mode) -> Space:
        """Declare the selection mode for the auto-solver.

        - "exact" (current default — flips to "auto" once AnytimeAStar
          is registered): admissible algorithms preferred.
        - "approximate": bounded-suboptimal A* variants outrank exact
          ones — useful when A*'s admissibility bill is too expensive.
        - "auto": cascade meta-algorithm wins selection (anytime
          delivery: best incumbent at any point in time).

        Mirrors the .timeout() pattern: the value is read by
        Algorithm.score_for(space) at selection time.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        self._mode = mode
        return self

    def parallel(self, workers: int) -> Space:
        # evaluate/successors functions must be picklable (module-level) when workers > 1
        self._n_workers = workers
        return self

    # --- cancel-token wire (used by Solver + algorithm solve loops) ---

    def _cancel_requested(self) -> bool:
        return self._cancel_token.is_set()

    def _request_cancel(self) -> None:
        self._cancel_token.request_cancel()

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
        mode: Mode | None = None,
    ) -> "Solver":
        from pathos.core.solver import Solver
        if mode is not None and mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        effective_mode = mode if mode is not None else self._mode
        effective_timeout = timeout or self._timeout
        # The cascade is meaningless without a budget — running every
        # phase to completion defeats the anytime contract. Inject a 1h
        # default when mode=auto and neither layer supplied one.
        if effective_mode == "auto" and effective_timeout is None:
            effective_timeout = 3600.0
        return Solver(
            self,
            candidates=candidates,
            timeout=effective_timeout,
            mode=mode,
        )
