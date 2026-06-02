"""Tests for the optimality preference knob.

Closes FINDINGS §2c: when several goal+heuristic algorithms are compatible
(e.g. AStar + WeightedAStar + GreedyBestFirst on 8-puzzle), the auto-pick
has no way to express "ε-optimal is fine, give me speed". The
`optimality` knob exposes that axis at the Space + Solver level.

Two modes:
  - "exact"       — default. Admissible algorithms preferred. Behaviour
                    is identical to pre-optimality-knob.
  - "approximate" — bounded-suboptimal A* variants outrank exact ones;
                    greedy variants get a smaller bump (still below
                    bounded-suboptimal because the quality bound matters).
"""
from __future__ import annotations

import pytest

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.informed import (
    AStar,
    GreedyBestFirst,
    IDAstar,
    WeightedAStar,
)


def _puzzle_like_space() -> Space:
    """A {SUCCESSORS, GOAL, HEURISTIC, EVALUATE} space — same capability
    set the 8-puzzle declares. State shape is irrelevant for selection."""
    space = Space().initial("start")

    @space.successors
    def expand(s):
        yield "noop", s

    @space.goal
    def is_goal(s):
        return s == "end"

    @space.heuristic
    def h(s):
        return 0.0

    @space.evaluate
    def cost(s):
        return 1.0

    return space


# ---------------------------------------------------------------------------
# Default behavior: exact mode preserves pre-knob auto-pick
# ---------------------------------------------------------------------------

def test_default_optimality_is_exact():
    space = _puzzle_like_space()
    assert space._optimality == "exact"


def test_exact_mode_picks_astar_on_puzzle_capabilities():
    """Default (exact) on {SUCC, GOAL, HEUR, EVAL} → AStar wins."""
    space = _puzzle_like_space()
    picked = space.solver()._select()
    assert picked is AStar


# ---------------------------------------------------------------------------
# Approximate mode flips the auto-pick
# ---------------------------------------------------------------------------

def test_approximate_mode_via_fluent_picks_weighted_astar():
    space = _puzzle_like_space().optimality("approximate")
    picked = space.solver()._select()
    assert picked is WeightedAStar


def test_approximate_mode_via_solver_kwarg_picks_weighted_astar():
    space = _puzzle_like_space()
    picked = space.solver(optimality="approximate")._select()
    assert picked is WeightedAStar


def test_approximate_mode_ranks_wastar_above_astar_above_greedy():
    """Under approximate, WAStar wins, but Greedy doesn't leap past it —
    the bounded-suboptimal A* variant is still safer than pure-greedy.
    Direct score_for comparison, independent of pool ordering."""
    space = _puzzle_like_space().optimality("approximate")
    s_w = WeightedAStar.score_for(space)
    s_a = AStar.score_for(space)
    s_g = GreedyBestFirst.score_for(space)
    assert s_w > s_a, f"WAStar {s_w} should outrank AStar {s_a} in approximate"
    assert s_w > s_g, f"WAStar {s_w} should outrank Greedy {s_g} in approximate"


def test_exact_mode_ranks_astar_above_wastar():
    """Sanity for the inverse: under exact, AStar must stay above WAStar."""
    space = _puzzle_like_space()  # exact by default
    assert AStar.score_for(space) > WeightedAStar.score_for(space)
    assert AStar.score_for(space) > GreedyBestFirst.score_for(space)


def test_idastar_demoted_in_approximate():
    """IDAstar is admissible like AStar; both demoted in approximate mode."""
    space = _puzzle_like_space()
    exact_score = IDAstar.score_for(space)
    space_approx = _puzzle_like_space().optimality("approximate")
    approx_score = IDAstar.score_for(space_approx)
    assert approx_score < exact_score


# ---------------------------------------------------------------------------
# Validation + explicit overrides still work
# ---------------------------------------------------------------------------

def test_invalid_optimality_value_raises():
    space = _puzzle_like_space()
    with pytest.raises(ValueError, match="optimality"):
        space.optimality("loose")  # type: ignore[arg-type]


def test_invalid_optimality_via_solver_kwarg_raises():
    space = _puzzle_like_space()
    with pytest.raises(ValueError, match="optimality"):
        space.solver(optimality="loose")  # type: ignore[arg-type]


def test_explicit_candidates_override_optimality_pick():
    """When the user pins candidates, optimality shouldn't second-guess."""
    space = _puzzle_like_space().optimality("approximate")
    picked = space.solver(candidates=[AStar])._select()
    assert picked is AStar


def test_solver_kwarg_overrides_space_setting():
    """Mirrors the timeout pattern: kwarg on solver() takes precedence
    over the value set via the fluent builder."""
    space = _puzzle_like_space().optimality("approximate")
    picked = space.solver(optimality="exact")._select()
    assert picked is AStar


# ---------------------------------------------------------------------------
# End-to-end: both modes still solve correctly
# ---------------------------------------------------------------------------

def _two_step_space() -> Space:
    space = Space().initial("a")

    @space.successors
    def expand(s):
        if s == "a":
            yield "to_b", "b"
        elif s == "b":
            yield "to_c", "c"

    @space.goal
    def is_goal(s):
        return s == "c"

    @space.heuristic
    def h(s):
        return 0.0

    @space.evaluate
    def cost(s):
        return 1.0

    return space


def test_solve_runs_in_exact_mode():
    space = _two_step_space()
    result = space.solver().solve()
    assert result.found
    assert result.algorithm == "AStar"


def test_solve_runs_in_approximate_mode():
    space = _two_step_space().optimality("approximate")
    result = space.solver().solve()
    assert result.found
    assert result.algorithm == "WeightedAStar"
