from __future__ import annotations
import time
import random
from typing import Any, cast
from pathos.algorithms.base import Algorithm
from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult
from pathos.core.solver import register


def _is_csp_shaped(space: Any) -> bool:
    """True if state is a partial-assignment dict — the precondition for the
    incremental-extension recursion used by Backtracking / ForwardChecking /
    MinConflicts. Without this guard they're offered for any
    successors+goal space and recurse forever (8-puzzle: RecursionError) or
    silent-fail (MinConflicts on plain Space)."""
    return isinstance(space._initial, dict)


@register
class Backtracking(Algorithm):
    """Backtracking search — systematic recursive CSP solver.

    Uses the space's successors as CSP expansion (typically via CSPSpace).

    Requires: successors, goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 9.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 9

    @classmethod
    def compatible_with(cls, space: Any) -> bool:
        return super().compatible_with(space) and _is_csp_shaped(space)

    def _bt(self, state: Any, expanded: list[int]) -> Any | None:
        if self.space._goal(state):
            return state
        for _, child in self.space._successors(state):
            expanded[0] += 1
            result = self._bt(child, expanded)
            if result is not None:
                return result
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        expanded = [0]
        result = self._bt(self.space._initial, expanded)
        elapsed = time.perf_counter() - t0
        if result is not None:
            return SearchResult(result, None, None, "Backtracking", expanded[0], elapsed, True)
        return SearchResult.not_found("Backtracking", expanded[0], elapsed)


@register
class ForwardChecking(Algorithm):
    """Forward Checking — Backtracking with look-ahead pruning.

    Before recursing, checks that each child has at least one valid successor.

    The current look-ahead is shallow (a child is pruned only when it has no
    successors and is not itself a goal), so node counts match plain
    Backtracking and the constant factor is higher. Until real domain-level
    pruning is implemented, FC is ranked below Backtracking so the auto-
    solver picks the faster of the two.

    Requires: successors, goal.

    Attributes:
        requires: Capability set needed.
        power_rank: 8 (below Backtracking's 9).
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL})
    power_rank = 8

    @classmethod
    def compatible_with(cls, space: Any) -> bool:
        return super().compatible_with(space) and _is_csp_shaped(space)

    def _fc(self, state: Any, expanded: list[int]) -> Any | None:
        if self.space._goal(state):
            return state
        children = list(self.space._successors(state))
        if not children:
            return None
        for _, child in children:
            expanded[0] += 1
            # check if any successor exists from child (look-ahead)
            future = list(self.space._successors(child))
            if future or self.space._goal(child):
                result = self._fc(child, expanded)
                if result is not None:
                    return result
        return None

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        expanded = [0]
        result = self._fc(self.space._initial, expanded)
        elapsed = time.perf_counter() - t0
        if result is not None:
            return SearchResult(result, None, None, "ForwardChecking", expanded[0], elapsed, True)
        return SearchResult.not_found("ForwardChecking", expanded[0], elapsed)


class AC3(Algorithm):
    """AC-3 arc consistency — domain-pruning preprocessor for CSPs.

    AC-3 is not a stand-alone CSP solver: it iterates over arcs and prunes
    domain values that can't satisfy any binary constraint, then hands off
    a (possibly smaller) problem to Backtracking / ForwardChecking. It is
    intentionally NOT @register'd so the auto-selector won't pick it as a
    terminal solver — its `solve()` returns the pruned domains, not an
    assignment, so power_rank=22 would mis-rank it above MinConflicts.

    Use directly: `AC3(csp_space).solve()` returns a SearchResult whose
    `.solution` is the pruned-domain dict. Production code typically
    composes AC-3 with a Backtracking-style solver — see issue tracker
    for the planned `pathos.algorithms.preprocessors` API.
    """
    requires = frozenset({Capability.VARIABLES, Capability.DOMAINS, Capability.CONSTRAINTS})
    power_rank = 22

    def solve(self) -> SearchResult:
        from pathos.spaces.csp import CSPSpace
        t0 = time.perf_counter()
        # This is invoked by CSPSpace which sets up the variables/domains/constraints
        csp = cast(CSPSpace, self.space)
        variables = csp._variables()
        domains: dict[Any, list[Any]] = {v: list(csp._domain(v)) for v in variables}
        arcs = [(xi, xj) for xi in variables for xj in variables if xi != xj]
        queue = list(arcs)

        while queue:
            xi, xj = queue.pop(0)
            if self._revise(domains, xi, xj, csp):
                if not domains[xi]:
                    return SearchResult.not_found("AC3", 0, time.perf_counter() - t0)
                for xk in variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return SearchResult(domains, None, None, "AC3", 0, time.perf_counter() - t0, True)

    def _revise(self, domains: dict[Any, list[Any]], xi: Any, xj: Any, csp: Any) -> bool:
        revised = False
        for x in domains[xi][:]:
            if not any(csp._constraints({xi: x, xj: y}) for y in domains[xj]):
                domains[xi].remove(x)
                revised = True
        return revised


@register
class MinConflicts(Algorithm):
    """Min-Conflicts heuristic — local repair CSP solver.

    Repeatedly selects the neighbor that minimizes constraint violations.

    Requires: successors, goal, evaluate.

    Attributes:
        requires: Capability set needed.
        power_rank: 19.
    """

    requires = frozenset({Capability.SUCCESSORS, Capability.GOAL, Capability.EVALUATE})
    power_rank = 19

    @classmethod
    def compatible_with(cls, space: Any) -> bool:
        return super().compatible_with(space) and _is_csp_shaped(space)

    def __init__(self, space: Any, max_iter: int = 1000) -> None:
        super().__init__(space)
        self.max_iter = max_iter

    def solve(self) -> SearchResult:
        t0 = time.perf_counter()
        current = self.space._initial

        for i in range(self.max_iter):
            if self.space._goal(current):
                return SearchResult(current, None, 0.0, "MinConflicts", i, time.perf_counter() - t0, True)
            neighbors = list(self.space._successors(current))
            if not neighbors:
                break
            _, current = min(neighbors, key=lambda ac: self.space._evaluate(ac[1]))

        return SearchResult.not_found("MinConflicts", self.max_iter, time.perf_counter() - t0)
