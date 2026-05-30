from __future__ import annotations
from typing import Any, Callable
from pathos.core.space import Space
from pathos.core.capabilities import Capability


class CSPSpace(Space):
    """
    Space for Constraint Satisfaction Problems.
    Auto-provides @successors (partial-assignment expansion) and @goal.
    User provides @domain and @constraint decorators.

    Also exposes _variables(), _domain(var), and _constraints(assignment)
    accessors so that AC3 and other capability-aware algorithms can use
    VARIABLES / DOMAINS / CONSTRAINTS capabilities directly.
    """

    def __init__(self, variables: list[Any]) -> None:
        super().__init__()
        self._variables_list = variables
        self._domain_fn: Callable | None = None
        self._constraint_fn: Callable | None = None
        self._initial_value = {}  # empty assignment
        self._setup_goal()

    def _setup_goal(self) -> None:
        n = len(self._variables_list)

        def _goal(assignment: dict) -> bool:
            return len(assignment) == n

        self._goal = _goal
        self.capabilities.add(Capability.GOAL)

    def _setup_successors(self) -> None:
        variables = self._variables_list
        domain_fn = self._domain_fn
        constraint_fn = self._constraint_fn

        def _successors(assignment: dict):
            col = len(assignment)
            if col >= len(variables):
                return
            var = variables[col]
            for val in domain_fn(var):  # type: ignore
                new_assign = dict(assignment)
                new_assign[var] = val
                if constraint_fn(new_assign):  # type: ignore
                    yield f"{var}={val}", new_assign

        self._successors = _successors
        self.capabilities.add(Capability.SUCCESSORS)

    # --- accessor methods for VARIABLES / DOMAINS / CONSTRAINTS capabilities ---

    def _variables(self) -> list[Any]:
        return self._variables_list

    def _domain(self, var: Any):
        return self._domain_fn(var)  # type: ignore

    def _constraints(self, assignment: dict) -> bool:
        return self._constraint_fn(assignment)  # type: ignore

    # --- decorator hooks ---

    @property
    def domain(self) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._domain_fn = fn
            self.capabilities.add(Capability.VARIABLES)
            self.capabilities.add(Capability.DOMAINS)
            self._maybe_finalize()
            return fn
        return decorator

    @property
    def constraint(self) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._constraint_fn = fn
            self.capabilities.add(Capability.CONSTRAINTS)
            self._maybe_finalize()
            return fn
        return decorator

    def _maybe_finalize(self) -> None:
        if self._domain_fn is not None and self._constraint_fn is not None:
            self._setup_successors()
