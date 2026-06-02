"""Tests for mode="auto" behaviour:
  - default timeout is 3600s when neither space.timeout() nor
    solver(timeout=…) supplies one.
  - explicit timeout (kwarg or fluent) overrides the default.
  - mode="exact" / "approximate" still have no implicit timeout.

AnytimeAStar selection + cascade tests come in Task 7-9.
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space


def _trivial_goal_space() -> Space:
    space = Space().initial("a")

    @space.successors
    def expand(s):
        yield "go", "b"

    @space.goal
    def is_goal(s):
        return s == "b"

    @space.heuristic
    def h(s):
        return 0.0

    @space.evaluate
    def cost(s):
        return 1.0

    return space


def test_auto_mode_sets_default_timeout_3600():
    space = _trivial_goal_space()
    solver = space.solver(mode="auto")
    assert solver.timeout == 3600.0


def test_auto_mode_respects_explicit_kwarg_timeout():
    space = _trivial_goal_space()
    solver = space.solver(mode="auto", timeout=10.0)
    assert solver.timeout == 10.0


def test_auto_mode_respects_fluent_timeout():
    space = _trivial_goal_space().timeout(20.0)
    solver = space.solver(mode="auto")
    assert solver.timeout == 20.0


def test_exact_mode_has_no_implicit_timeout():
    space = _trivial_goal_space()
    solver = space.solver(mode="exact")
    assert solver.timeout is None


def test_approximate_mode_has_no_implicit_timeout():
    space = _trivial_goal_space()
    solver = space.solver(mode="approximate")
    assert solver.timeout is None
