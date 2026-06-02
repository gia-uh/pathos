"""Tests confirming metaheuristics return best-so-far on cancel.

The cancel-token protocol makes population/iteration-based algorithms
naturally anytime. They check space._cancel_requested() at the top of
their main loop and return SearchResult(best_seen, …, found=True) when
the token is set.
"""
from __future__ import annotations

import random

import pytest

import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import Space
from pathos.algorithms.local import HillClimbing


def _descent_chain_space(length: int = 1000) -> Space:
    """A long chain of states with strictly decreasing evaluate cost.

    Initial state 0; successor of state i is state (i+1) with cost
    (length - i). The optimum is state `length` with cost 0. A full
    descent visits `length` states; cancelling partway through should
    return whichever state HC has reached so far.
    """
    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < length:
            yield "down", s + 1

    @space.evaluate
    def cost(s):
        return float(length - s)

    return space


def test_hill_climbing_returns_best_so_far_when_cancelled_mid_descent():
    space = _descent_chain_space(length=1000)
    space._request_cancel()  # cancel before solve even starts

    hc = HillClimbing(space, max_restarts=1)
    result = hc.solve()

    # With token already set, HC should bail at the top of the first
    # iteration and return the initial state as best-so-far.
    assert result.solution == 0
    assert result.cost == 1000.0
    assert result.algorithm == "HillClimbing"


def test_hill_climbing_progresses_when_token_set_partway():
    """Set the token after HC has descended a few steps and confirm
    the returned state has cost < initial cost (some progress made)."""
    space = _descent_chain_space(length=10000)

    # Pre-arm the token after N evaluations using a counter wrapped
    # around the evaluate function.
    counter = {"n": 0}
    raw_cost = space._evaluate

    def counted_cost(s):
        counter["n"] += 1
        if counter["n"] >= 50:
            space._request_cancel()
        return raw_cost(s)

    space._evaluate = counted_cost

    hc = HillClimbing(space, max_restarts=1)
    result = hc.solve()

    # We should have made some progress (cost < initial 10000) but not
    # have run to completion (state < 10000).
    assert result.cost is not None
    assert result.cost < 10000.0
    assert isinstance(result.solution, int)
    assert result.solution < 10000


def test_hill_climbing_no_cancel_still_runs_normally():
    """Sanity: when no cancel, HC behaves exactly as before — runs to
    its natural completion."""
    space = _descent_chain_space(length=100)
    hc = HillClimbing(space, max_restarts=1)
    result = hc.solve()
    # On this chain, HC descends to the end.
    assert result.solution == 100
    assert result.cost == 0.0


# ---------------------------------------------------------------------------
# Other iteration-based metaheuristics
# ---------------------------------------------------------------------------

from pathos.algorithms.local import TabuSearch, LocalBeamSearch  # noqa: E402
from pathos.algorithms.evolutionary import (  # noqa: E402
    SimulatedAnnealing,
    GeneticAlgorithm,
    DifferentialEvolution,
    ParticleSwarm,
)


def _cycle_space(size: int = 1000) -> Space:
    """A cycle of `size` states; every state has a single successor
    (next-mod-size). Cost is the state index. Loop-based algorithms
    will keep iterating forever (or until max_iter) without naturally
    exiting — perfect to exercise the cancel-token check."""
    space = Space().initial(0)

    @space.successors
    def expand(s):
        yield "next", (s + 1) % size

    @space.evaluate
    def cost(s):
        return float(s)

    return space


@pytest.mark.parametrize("alg_cls,kwargs", [
    (TabuSearch, {"max_iter": 1_000_000, "tabu_size": 0}),
    (LocalBeamSearch, {"max_iter": 1_000_000, "k": 3}),
    (SimulatedAnnealing, {"max_iter": 1_000_000}),
])
def test_metaheuristic_returns_best_so_far_on_cancel(alg_cls, kwargs):
    """Each iteration-based metaheuristic should bail out cleanly when
    the cancel token is set. The cycle space won't let them exit
    naturally; without the cancel-check they'd loop 1M times.
    Verify by wall-time bound: bail must happen in <0.5s."""
    import time as _time_mod

    space = _cycle_space(size=1000)
    space._request_cancel()

    alg = alg_cls(space, **kwargs)
    t0 = _time_mod.perf_counter()
    result = alg.solve()
    wall = _time_mod.perf_counter() - t0

    assert result.algorithm == alg_cls.__name__
    assert wall < 0.5, (
        f"{alg_cls.__name__} ran {wall:.3f}s — cancel check missing"
    )


def _quadratic_bowl_space(dim: int = 5) -> Space:
    """Continuous optimization: minimize sum(xi**2)."""
    space = Space().initial(lambda: [random.uniform(-5, 5) for _ in range(dim)])

    @space.evaluate
    def f(x):
        return sum(xi * xi for xi in x)

    return space


@pytest.mark.parametrize("alg_cls,kwargs", [
    (GeneticAlgorithm, {"pop_size": 30, "generations": 100_000}),
    (DifferentialEvolution, {"pop_size": 30, "generations": 100_000}),
    (ParticleSwarm, {"pop_size": 30, "generations": 100_000}),
])
def test_population_algorithm_returns_best_so_far_on_cancel(alg_cls, kwargs):
    """Population-based algorithms with absurd generation count would
    take seconds-to-minutes without a cancel check. Pre-arm the token
    and verify wall time stays bounded."""
    import time as _time_mod

    space = _quadratic_bowl_space()
    space._request_cancel()

    alg = alg_cls(space, **kwargs)
    t0 = _time_mod.perf_counter()
    result = alg.solve()
    wall = _time_mod.perf_counter() - t0

    assert result.algorithm == alg_cls.__name__
    assert wall < 1.0, (
        f"{alg_cls.__name__} ran {wall:.3f}s — cancel check missing"
    )
