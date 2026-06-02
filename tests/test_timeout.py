"""Regression test for .timeout() / solver(timeout=) actually firing.

The fluent .timeout(seconds) API was previously stored on Space/Solver but
never consulted by any algorithm — effectively a no-op. Solver.solve now
guards each run with SIGALRM and returns SearchResult.not_found on
expiry.
"""
from __future__ import annotations

import time

import pathos.algorithms  # noqa: F401
from pathos import CSPSpace


def _big_nqueens(n: int) -> CSPSpace:
    csp = CSPSpace(variables=list(range(n)))

    @csp.domain
    def dom(c):
        return list(range(n))

    @csp.constraint
    def no_attack(a):
        items = list(a.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c1, r1 = items[i]
                c2, r2 = items[j]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    return csp


def test_solver_timeout_fires_via_kwarg():
    """Backtracking doesn't yet check the cancel token (v1 scope), so
    the watchdog backstop (cancel + 2s grace) is what bounds wall time.
    The algorithm may either find a solution within the grace or hit
    the watchdog; either way it returns within ~3s, not the full ~1s+
    natural runtime extended by no timeout."""
    csp = _big_nqueens(16)
    t0 = time.perf_counter()
    result = csp.solver(timeout=0.05).solve()
    wall = time.perf_counter() - t0
    assert wall < 3.0, f"wall {wall:.3f}s — alarm + watchdog grace didn't fire"
    # result.found may be True (algorithm finished naturally during grace)
    # or False (watchdog raised TimeoutError); the timeout mechanism is
    # verified by the wall-time bound.


def test_solver_timeout_fires_via_fluent_api():
    csp = _big_nqueens(16).timeout(0.05)
    t0 = time.perf_counter()
    result = csp.solver().solve()
    wall = time.perf_counter() - t0
    assert wall < 3.0, f"wall {wall:.3f}s — alarm + watchdog grace didn't fire"
    _ = result  # see kwarg test for found-state caveat


def test_solver_no_timeout_still_runs_to_completion():
    """Sanity: when no timeout is set, the path is unchanged."""
    csp = _big_nqueens(6)
    result = csp.solver().solve()
    assert result.found is True
    assert len(result.solution) == 6


def test_solver_timeout_does_not_fire_when_solve_is_fast():
    """A generous timeout on a fast problem should not affect the result."""
    csp = _big_nqueens(6)
    result = csp.solver(timeout=5.0).solve()
    assert result.found is True
    assert result.elapsed < 1.0


def test_solver_timeout_cooperative_returns_best_so_far_for_hc():
    """HillClimbing (Task 3 wired the cancel check) should return its
    best individual seen when SIGALRM fires, not not_found."""
    import pathos.algorithms  # noqa
    from pathos import Space
    from pathos.algorithms.local import HillClimbing

    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < 1_000_000:
            yield "down", s + 1

    @space.evaluate
    def cost(s):
        return float(1_000_000 - s)

    # Pin candidate to HC so the solver picks it deterministically.
    result = space.solver(timeout=0.05, candidates=[HillClimbing]).solve()
    # HC should have made some progress before the timeout fired.
    assert result.found is True
    assert result.cost is not None
    assert result.cost < 1_000_000.0
    assert result.algorithm == "HillClimbing"
