"""Regression tests for FINDINGS §3a.

Local-search/metaheuristic algorithms used to return `found=True`
whenever they terminated by their own stopping rule, ignoring the
@goal predicate even when the space declared one. On problems like
the 8-puzzle this surfaced as `found=True, cost=1.0` without the
algorithm actually reaching the goal.

After the fix, `Algorithm._goal_reached(state)` consults
`space._goal(state)` whenever `Capability.GOAL` is declared and
gates the `found` flag on it. When no GOAL is declared (pure
optimization), behaviour is unchanged.
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.evolutionary import (
    DifferentialEvolution,
    GeneticAlgorithm,
    SimulatedAnnealing,
)
from pathos.algorithms.local import (
    HillClimbing,
    LocalBeamSearch,
    TabuSearch,
)


def _unreachable_goal_space() -> Space:
    """Space whose goal is never reachable; @evaluate is monotonic but
    local search can only ever wander among non-goal states."""
    space = Space().initial((0,))

    @space.successors
    def expand(state):
        # Each step adds 1 to the head; never reaches the goal sentinel.
        yield "inc", (state[0] + 1,)

    @space.goal
    def is_goal(state):
        return state == ("GOAL",)  # never matches an int tuple

    @space.evaluate
    def step_cost(state):
        return 1.0

    return space


def _no_goal_optimization_space() -> Space:
    """Space without a GOAL capability — pure optimization. The algorithm's
    own stopping rule should still set found=True (degrades gracefully)."""
    space = Space().initial((0,))

    @space.successors
    def expand(state):
        yield "step", (state[0] + 1,)

    @space.evaluate
    def step_cost(state):
        return 1.0

    return space


LOCAL_AND_METAHEURISTIC = [
    HillClimbing,
    TabuSearch,
    LocalBeamSearch,
    SimulatedAnnealing,
    GeneticAlgorithm,
    DifferentialEvolution,
]


def test_goal_aware_reporting_on_unreachable_goal():
    """When @goal is declared but unreachable, every local-search algo must
    report found=False after running. Previously they all returned True."""
    space = _unreachable_goal_space()
    for cls in LOCAL_AND_METAHEURISTIC:
        if not cls.compatible_with(space):
            continue
        result = cls(space).solve()
        assert result.found is False, (
            f"{cls.__name__} reported found=True on a space whose goal is "
            f"unreachable — see FINDINGS §3a"
        )


def test_no_goal_means_found_true_unchanged():
    """When no @goal is declared, local search degrades to pure optimization
    and reports found=True after terminating by its own stopping rule."""
    space = _no_goal_optimization_space()
    for cls in LOCAL_AND_METAHEURISTIC:
        if not cls.compatible_with(space):
            continue
        result = cls(space).solve()
        assert result.found is True, (
            f"{cls.__name__} reported found=False on a no-goal space — "
            f"the goal-aware reporting fix overreached"
        )
