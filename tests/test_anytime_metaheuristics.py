"""Tests confirming metaheuristics return best-so-far on cancel.

The cancel-token protocol makes population/iteration-based algorithms
naturally anytime. They check space._cancel_requested() at the top of
their main loop and return SearchResult(best_seen, …, found=True) when
the token is set.
"""
from __future__ import annotations

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
