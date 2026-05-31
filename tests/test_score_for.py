"""Tests for context-aware score_for(space) selection.

Algorithm.score_for(space) lets each algorithm declare a problem-aware
preference instead of relying on the coarse static `power_rank`. Default
returns float(power_rank); overrides on GA and HC encode self-knowledge
that empirically improves auto-pick quality.
"""
from __future__ import annotations

import random

import pathos.algorithms  # noqa: F401
from pathos import Space, TourSpace
from pathos.algorithms.evolutionary import (
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarm,
    SimulatedAnnealing,
)
from pathos.algorithms.local import (
    HillClimbing,
    LocalBeamSearch,
    TabuSearch,
)


def _tsp(n: int = 10) -> TourSpace:
    cities = list(range(n))
    distances = {(i, j): 1.0 for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def cost(tour):
        return 1.0

    return space


def _continuous_no_successors(dim: int = 5) -> Space:
    """Pure-{evaluate} continuous optimization, no neighborhood."""
    rng = random.Random(0)
    space = Space().initial(lambda: [rng.uniform(-5, 5) for _ in range(dim)])

    @space.evaluate
    def f(x):
        return sum(xi * xi for xi in x)

    return space


def test_score_for_default_equals_power_rank():
    """An algorithm that doesn't override score_for must return its
    power_rank — the default keeps existing rankings backward compatible."""
    # Any algorithm that hasn't overridden score_for. SimulatedAnnealing
    # is a stable pick — power_rank=17 since v0.1.
    space = _tsp()
    assert SimulatedAnnealing.score_for(space) == float(
        SimulatedAnnealing.power_rank
    )


def test_hill_climbing_bumps_above_tabu_on_tsp():
    """HC at default rank 15 < TS rank 18 — yet on TSP HC matches or beats
    TS on quality and is 4-5× faster. HC.score_for bumps to 20 when the
    space is pure-optimization (no GOAL) and state is list/tuple."""
    space = _tsp()
    assert HillClimbing.score_for(space) > TabuSearch.score_for(space)
    # Confirm the auto-pick lands on HC, not TS.
    assert space.solver()._select() is HillClimbing


def test_hill_climbing_does_not_bump_on_string_state():
    """The bump only triggers on list/tuple states — string state shouldn't
    get the +5 (no meaningful gradient to climb)."""
    space = Space().initial("hello")

    @space.successors
    def expand(s):
        yield "noop", s

    @space.evaluate
    def f(s):
        return 1.0

    assert HillClimbing.score_for(space) == float(HillClimbing.power_rank)


def test_genetic_algorithm_penalty_without_successors():
    """GA degenerates to deepcopy without @successors. score_for must
    demote it so DE/PSO win the auto-pick on pure-{evaluate} continuous
    problems."""
    space = _continuous_no_successors()
    ga_score = GeneticAlgorithm.score_for(space)
    de_score = DifferentialEvolution.score_for(space)
    pso_score = ParticleSwarm.score_for(space)
    assert ga_score < de_score
    assert ga_score < pso_score
    # End-to-end: auto-pick on a pure-continuous space should land on DE.
    picked = space.solver()._select()
    assert picked in {DifferentialEvolution, ParticleSwarm}


def test_genetic_algorithm_unchanged_when_successors_present():
    """When @successors is declared, GA gets its meaningful default
    operator (random-neighbor) — no penalty applies."""
    space = _tsp()  # TourSpace declares SUCCESSORS via 2-opt
    assert GeneticAlgorithm.score_for(space) == float(
        GeneticAlgorithm.power_rank
    )


def test_score_for_is_called_through_solver_path():
    """The Solver picks via score_for, not raw power_rank — verify by
    asserting the pick differs from what raw power_rank would have given."""
    space = _continuous_no_successors()
    pool = [GeneticAlgorithm, DifferentialEvolution, ParticleSwarm]
    if any(not c.compatible_with(space) for c in pool):
        # ParticleSwarm requires _is_numeric_vector; both DE and PSO
        # should pass on this space (init is a list of floats).
        pass
    picked = space.solver(candidates=pool)._select()
    assert picked is not GeneticAlgorithm  # would win by power_rank=14
    assert picked is DifferentialEvolution  # power_rank=13, unbeaten here
