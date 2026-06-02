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


# ---------------------------------------------------------------------------
# AnytimeAStar selection (added in T7)
# ---------------------------------------------------------------------------

from pathos.algorithms.informed import AnytimeAStar  # noqa: E402


def test_anytime_astar_wins_selection_when_mode_auto_on_puzzle_capabilities():
    space = _trivial_goal_space()  # mode defaults to "auto" after this task
    picked = space.solver()._select()
    assert picked is AnytimeAStar


def test_anytime_astar_does_not_win_under_exact_mode():
    space = _trivial_goal_space()
    from pathos.algorithms.informed import AStar
    picked = space.solver(mode="exact")._select()
    assert picked is AStar  # admissible — AnytimeAStar's score_for returns -inf


def test_anytime_astar_does_not_win_under_approximate_mode():
    space = _trivial_goal_space()
    from pathos.algorithms.informed import WeightedAStar
    picked = space.solver(mode="approximate")._select()
    assert picked is WeightedAStar


def test_anytime_astar_capability_set_matches_astar():
    """AnytimeAStar runs WeightedAStar phases — needs everything A* needs."""
    from pathos.algorithms.informed import AStar
    assert AnytimeAStar.requires == AStar.requires


def test_anytime_astar_score_in_auto_exceeds_all_siblings():
    space = _trivial_goal_space()  # default mode auto
    from pathos.algorithms.informed import (
        AStar, WeightedAStar, GreedyBestFirst, IDAstar,
    )
    s_anytime = AnytimeAStar.score_for(space)
    assert s_anytime > AStar.score_for(space)
    assert s_anytime > WeightedAStar.score_for(space)
    assert s_anytime > GreedyBestFirst.score_for(space)
    assert s_anytime > IDAstar.score_for(space)
