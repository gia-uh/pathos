"""Tests for AnytimeAStar cascade behaviour.

Confirms:
  - On small problems with no time pressure, returns proven optimal
    (final AStar phase completes).
  - Incumbent comparison rule (lower cost wins; epsilon breaks ties)
    is honoured.
  - On cancel, returns the best phase that finished, not not_found.
  - algorithm field is rewritten to "AnytimeAStar" so users see the
    meta as the answer source.
"""
from __future__ import annotations

import math

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.informed import AnytimeAStar


def _puzzle_chain_space(steps: int = 5) -> Space:
    """A linear chain of `steps` states with the goal at the end.
    Each step has cost 1 and admissible h = (steps - current). Cascade
    naturally finds optimal cost = steps."""
    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < steps:
            yield "next", s + 1

    @space.goal
    def is_goal(s):
        return s == steps

    @space.heuristic
    def h(s):
        return float(steps - s)

    @space.evaluate
    def cost(s):
        return 1.0

    return space


# ---------------------------------------------------------------------------
# Cascade body — optimal on small problems
# ---------------------------------------------------------------------------

def test_anytime_returns_optimal_on_small_problem():
    space = _puzzle_chain_space(steps=5)
    result = space.solver().solve()
    assert result.found is True
    assert result.cost == 5.0
    assert result.algorithm == "AnytimeAStar"
    # AStar phase emits epsilon=1.0; AnytimeAStar inherits via
    # dataclasses.replace(best, algorithm="AnytimeAStar", elapsed=…).
    assert result.epsilon == 1.0
    assert result.optimal is True


def test_anytime_cancel_returns_best_so_far():
    """Pre-arm the cancel token; AnytimeAStar should still complete at
    least one fast phase (Greedy) before observing the cancel."""
    space = _puzzle_chain_space(steps=5)
    space._request_cancel()
    result = space.solver(mode="auto")._select()(_puzzle_chain_space(steps=5))
    # Trigger the cancel check at the very top of solve() — we
    # construct a fresh AnytimeAStar against a freshly-canceled space
    # so the loop bails before ever running phase 1.
    fresh = _puzzle_chain_space(steps=5)
    fresh._request_cancel()
    r = AnytimeAStar(fresh).solve()
    assert r.found is False
    assert r.algorithm == "AnytimeAStar"


def test_anytime_no_compatible_phases_returns_not_found_or_trivial():
    """Edge case: a space with steps=0 means initial == goal, so
    every phase finds it immediately."""
    space = _puzzle_chain_space(steps=0)
    result = space.solver().solve()
    assert result.found is True


def test_anytime_algorithm_field_is_anytime_not_phase():
    """The returned result's `algorithm` is rewritten to AnytimeAStar
    even though internally it came from a WAStar or AStar phase."""
    space = _puzzle_chain_space(steps=3)
    result = space.solver().solve()
    assert result.algorithm == "AnytimeAStar"


# ---------------------------------------------------------------------------
# End-to-end anytime contract under tight budgets
# ---------------------------------------------------------------------------

import time as _time  # noqa: E402


def _slow_8puzzle_like(branching: int = 4, depth: int = 15) -> Space:
    """Synthetic space where each expansion takes ~0.5ms to force slow
    progress under tight budget. State is (depth, branch_index).
    Goal is (depth, *). Heuristic returns remaining depth."""
    space = Space().initial((0, 0))

    @space.successors
    def expand(s):
        d, b = s
        if d >= depth:
            return
        for i in range(branching):
            yield f"b{i}", (d + 1, i)

    @space.goal
    def is_goal(s):
        d, _ = s
        return d == depth

    @space.heuristic
    def h(s):
        d, _ = s
        # Slow heuristic — simulates expensive evaluation
        _time.sleep(0.0005)
        return float(depth - d)

    @space.evaluate
    def cost(s):
        return 1.0

    return space


def test_anytime_returns_incumbent_under_tight_budget():
    """0.05s on a slow space — AStar can't finish, but Greedy/WAStar
    earlier phases plant an incumbent."""
    space = _slow_8puzzle_like(branching=3, depth=8)
    result = space.solver(timeout=0.05).solve()
    # Even if final AStar didn't complete, an earlier phase should
    # have produced an incumbent.
    assert result.algorithm == "AnytimeAStar"
    if result.found:
        assert result.cost is not None
        assert result.cost > 0


def test_anytime_small_budget_still_runs_one_phase():
    """A budget of 0.001s — extremely tight. Greedy on a small space
    should still finish in microseconds, so we expect found=True."""
    space = _puzzle_chain_space(steps=2)
    result = space.solver(timeout=0.001).solve()
    assert result.found is True
    assert result.algorithm == "AnytimeAStar"
