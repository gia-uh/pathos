from __future__ import annotations
from typing import Any, Callable, Hashable, Iterable, Sequence
from pathos.core.space import Space


class ScheduleSpace(Space):
    """Space for discrete-time scheduling problems with capacity constraints
    and fairness objectives.

    Each (slot, entity) cell is a binary decision: on or off. State is a
    frozenset of (slot_index, entity_index) tuples for the ON cells.
    Capacity violations are folded into _evaluate as a penalty so the
    existing AnytimeLocal cascade (HC -> SA -> Tabu) solves it directly.

    See `docs/superpowers/specs/2026-06-09-schedule-space-power-grid-design.md`.
    """

    def __init__(
        self,
        entities: Sequence[Hashable],
        slots: int,
        downstream: Callable[[Hashable], Iterable[Hashable]] | None = None,
        penalty: float = 1e3,
    ) -> None:
        super().__init__()
        entities_tuple = tuple(entities)
        if not entities_tuple:
            raise ValueError("entities must be non-empty")
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        self._entities: tuple[Hashable, ...] = entities_tuple
        self._slots: int = slots
        self._downstream: Callable[[Hashable], Iterable[Hashable]] = (
            downstream if downstream is not None else (lambda e: (e,))
        )
        self._penalty: float = penalty
        self._initial_value = frozenset()
        # User-attached decorators populated in Task 3.
        self._demand_fn: Callable[[Hashable, int], float] | None = None
        self._capacity_fn: Callable[[int], float] | None = None
        self._fairness_fn: Callable[[tuple[tuple[bool, ...], ...]], float] | None = None
        # Target band, set via .target(). Default: upper bound only.
        self._tolerance: float = 0.0

    # --- decorator hooks ---

    @property
    def demand(self) -> Callable[..., Any]:
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._demand_fn is not None:
                raise RuntimeError("@demand already defined on this space")
            self._demand_fn = fn
            self._maybe_finalize()
            return fn
        return decorator

    @property
    def capacity(self) -> Callable[..., Any]:
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._capacity_fn is not None:
                raise RuntimeError("@capacity already defined on this space")
            self._capacity_fn = fn
            self._maybe_finalize()
            return fn
        return decorator

    @property
    def fairness(self) -> Callable[..., Any]:
        from pathos.core.capabilities import Capability
        def decorator(fn: Callable[..., float]) -> Callable[..., float]:
            if self._fairness_fn is not None:
                raise RuntimeError("@fairness already defined on this space")
            self._fairness_fn = fn

            def _evaluate(state: frozenset[tuple[int, int]]) -> float:
                matrix = self._to_matrix(state)
                fairness_score = self._fairness_fn(matrix)
                overshoot = self._overshoot(state)
                return -float(fairness_score) + self._penalty * overshoot

            self._evaluate = _evaluate
            self.capabilities.add(Capability.EVALUATE)
            self._maybe_finalize()
            return fn
        return decorator

    def _overshoot(self, state: frozenset[tuple[int, int]]) -> float:
        """Total capacity-violation overshoot summed across slots.

        For each slot, compute (load - capacity) and clip below at 0.
        Lower band from .target() is layered in by Task 6.
        """
        if self._demand_fn is None or self._capacity_fn is None:
            return 0.0
        total = 0.0
        for t in range(self._slots):
            cap_t = self._capacity_fn(t)
            if cap_t < 0:
                raise ValueError(
                    f"@capacity returned negative value {cap_t} at slot {t}",
                )
            load_t = sum(
                self._demand_fn(self._entities[e], t)
                for tt, e in state if tt == t
            )
            total += max(0.0, load_t - cap_t)
        return total

    # --- internal helpers ---

    def _to_matrix(self, state: frozenset[tuple[int, int]]) -> tuple[tuple[bool, ...], ...]:
        """Convert a frozenset of (slot, entity_idx) ON cells to a (T, N) matrix."""
        n = len(self._entities)
        return tuple(
            tuple((t, e) in state for e in range(n))
            for t in range(self._slots)
        )

    def _setup_successors(self) -> None:
        from pathos.core.capabilities import Capability
        T, N = self._slots, len(self._entities)

        def _successors(state: frozenset[tuple[int, int]]) -> Any:
            for t in range(T):
                for e in range(N):
                    cell = (t, e)
                    if cell in state:
                        yield f"off({t},{e})", state - {cell}
                    else:
                        yield f"on({t},{e})", state | {cell}

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)

    def _maybe_finalize(self) -> None:
        if (
            self._demand_fn is not None
            and self._capacity_fn is not None
            and self._fairness_fn is not None
            and self._successors is None
        ):
            self._setup_successors()
