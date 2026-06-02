"""Tests for the `mode` selection axis.

Three modes for v1:
  - "exact" (default for now; flips to "auto" in a later commit):
    admissible algorithms preferred (current pre-knob behaviour).
  - "approximate": bounded-suboptimal A* variants outrank exact.
  - "auto": cascade meta-algorithm wins selection (becomes default
    once AnytimeAStar is registered — covered by separate tests).

This file covers the rename `optimality` → `mode` plus the addition
of "auto" as a third valid value. The cascade behaviour itself is
covered in tests/test_anytime_astar.py (Task 7-9).
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
# Default and exact-mode behaviour
# ---------------------------------------------------------------------------

def test_default_mode_is_exact_until_auto_arrives():
    """The default flips to 'auto' once AnytimeAStar is registered.
    Until then (Task 4 commit), the default stays 'exact' for backward
    compatibility with existing tests."""
    space = _puzzle_like_space()
    assert space._mode == "exact"


def test_exact_mode_picks_astar_on_puzzle_capabilities():
    space = _puzzle_like_space()
    picked = space.solver()._select()
    assert picked is AStar


def test_exact_mode_ranks_astar_above_wastar_and_greedy():
    space = _puzzle_like_space()
    assert AStar.score_for(space) > WeightedAStar.score_for(space)
    assert AStar.score_for(space) > GreedyBestFirst.score_for(space)


# ---------------------------------------------------------------------------
# Approximate-mode behaviour (preserved from previous optimality knob)
# ---------------------------------------------------------------------------

def test_approximate_mode_via_fluent_picks_weighted_astar():
    space = _puzzle_like_space().mode("approximate")
    picked = space.solver()._select()
    assert picked is WeightedAStar


def test_approximate_mode_via_solver_kwarg_picks_weighted_astar():
    space = _puzzle_like_space()
    picked = space.solver(mode="approximate")._select()
    assert picked is WeightedAStar


def test_approximate_ranks_wastar_above_astar_above_greedy():
    space = _puzzle_like_space().mode("approximate")
    s_w = WeightedAStar.score_for(space)
    s_a = AStar.score_for(space)
    s_g = GreedyBestFirst.score_for(space)
    assert s_w > s_a
    assert s_w > s_g


def test_idastar_demoted_in_approximate():
    space_exact = _puzzle_like_space()
    space_approx = _puzzle_like_space().mode("approximate")
    assert IDAstar.score_for(space_approx) < IDAstar.score_for(space_exact)


# ---------------------------------------------------------------------------
# Validation + explicit overrides
# ---------------------------------------------------------------------------

def test_invalid_mode_value_raises():
    space = _puzzle_like_space()
    with pytest.raises(ValueError, match="mode"):
        space.mode("loose")  # type: ignore[arg-type]


def test_invalid_mode_via_solver_kwarg_raises():
    space = _puzzle_like_space()
    with pytest.raises(ValueError, match="mode"):
        space.solver(mode="loose")  # type: ignore[arg-type]


def test_explicit_candidates_override_mode_pick():
    space = _puzzle_like_space().mode("approximate")
    picked = space.solver(candidates=[AStar])._select()
    assert picked is AStar


def test_solver_kwarg_overrides_space_setting():
    space = _puzzle_like_space().mode("approximate")
    picked = space.solver(mode="exact")._select()
    assert picked is AStar


def test_solver_kwarg_does_not_mutate_space_setting():
    """Per-call kwarg is non-mutating (mirrors timeout)."""
    space = _puzzle_like_space().mode("approximate")
    space.solver(mode="exact")._select()
    assert space._mode == "approximate"


# ---------------------------------------------------------------------------
# "auto" is accepted but its selection behaviour is covered elsewhere
# ---------------------------------------------------------------------------

def test_auto_is_a_valid_mode_value():
    space = _puzzle_like_space()
    space.mode("auto")
    assert space._mode == "auto"


def test_solver_accepts_auto_kwarg():
    space = _puzzle_like_space()
    # Until AnytimeAStar is registered (Task 7), auto behaves identically
    # to exact — no algorithm overrides score_for on "auto" yet.
    solver = space.solver(mode="auto")
    assert solver is not None
