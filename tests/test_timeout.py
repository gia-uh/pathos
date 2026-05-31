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
    csp = _big_nqueens(16)  # ~1.2s on the reference machine
    t0 = time.perf_counter()
    result = csp.solver(timeout=0.05).solve()
    wall = time.perf_counter() - t0
    assert result.found is False
    # Should return within a hair of the timeout, not the full ~1s.
    assert wall < 0.5, f"wall {wall:.3f}s — alarm didn't fire"


def test_solver_timeout_fires_via_fluent_api():
    csp = _big_nqueens(16).timeout(0.05)
    t0 = time.perf_counter()
    result = csp.solver().solve()
    wall = time.perf_counter() - t0
    assert result.found is False
    assert wall < 0.5, f"wall {wall:.3f}s — alarm didn't fire"


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
