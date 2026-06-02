"""Tests for AnytimeCSP cascade behaviour.

Confirms:
  - On CSP-shaped spaces under mode=auto, AnytimeCSP wins selection and
    finds a solution.
  - On non-CSP spaces, AnytimeCSP is NOT picked (score_for returns -inf).
  - Under mode="exact" / "approximate", a CSP-shaped space falls back to
    Backtracking / MinConflicts rather than AnytimeCSP.
  - The cascade adapts to declared capabilities: MinConflicts is included
    only when EVALUATE is present.
  - On pre-armed cancel, returns not_found cleanly.
  - algorithm field is rewritten to "AnytimeCSP".
"""
from __future__ import annotations

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.csp import AnytimeCSP, Backtracking


def _nqueens_space(n: int = 4) -> Space:
    """N-Queens as a CSP-shaped plain Space (matches the convention used
    in tests/test_csp.py — initial is an empty dict, successors extend
    the partial assignment column by column)."""
    space = Space().initial({})

    @space.successors
    def expand(assignment):
        col = len(assignment)
        if col >= n:
            return
        for row in range(n):
            consistent = all(
                assignment[c] != row
                and abs(assignment[c] - row) != abs(c - col)
                for c in assignment
            )
            if consistent:
                new_assign = dict(assignment)
                new_assign[col] = row
                yield f"col{col}={row}", new_assign

    @space.goal
    def is_complete(assignment):
        return len(assignment) == n

    return space


def _nqueens_with_evaluate(n: int = 4) -> Space:
    """N-Queens with an @evaluate counting attacking pairs — enables
    MinConflicts in the cascade."""
    space = _nqueens_space(n)

    @space.evaluate
    def attacks(assignment):
        # Count pairs of queens attacking each other in this (partial)
        # assignment. MinConflicts uses this to greedily pick the lower-
        # violation child.
        cols = list(assignment.keys())
        count = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ci, cj = cols[i], cols[j]
                if assignment[ci] == assignment[cj]:
                    count += 1
                elif abs(assignment[ci] - assignment[cj]) == abs(ci - cj):
                    count += 1
        return float(count)

    return space


def _linear_chain(steps: int = 5) -> Space:
    """A non-CSP-shaped chain (initial is int, not dict) — AnytimeCSP
    must NOT be picked here."""
    space = Space().initial(0)

    @space.successors
    def expand(s):
        if s < steps:
            yield "next", s + 1

    @space.goal
    def is_goal(s):
        return s == steps

    return space


# ---------------------------------------------------------------------------
# Selection: AnytimeCSP wins on CSP-shaped spaces under mode=auto
# ---------------------------------------------------------------------------

def test_anytime_csp_wins_selection_on_csp_shaped_space():
    space = _nqueens_space(4)
    result = space.solver().solve()
    assert result.found is True
    assert result.algorithm == "AnytimeCSP"


def test_anytime_csp_solves_4queens_correctly():
    space = _nqueens_space(4)
    result = space.solver().solve()
    assert result.found is True
    sol = result.solution
    assert len(sol) == 4
    cols = list(sol.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            assert sol[cols[i]] != sol[cols[j]]
            assert abs(sol[cols[i]] - sol[cols[j]]) != abs(cols[i] - cols[j])


def test_anytime_csp_not_picked_on_non_csp_space():
    """A linear chain (initial=int) is NOT CSP-shaped. AnytimeCSP's
    score_for must return -inf so another algorithm wins."""
    space = _linear_chain(3)
    result = space.solver().solve()
    assert result.algorithm != "AnytimeCSP"


def test_anytime_csp_not_picked_under_mode_exact():
    """mode=exact: a CSP space falls back to plain Backtracking."""
    space = _nqueens_space(4)
    result = space.solver(mode="exact").solve()
    assert result.found is True
    assert result.algorithm == "Backtracking"


# ---------------------------------------------------------------------------
# Cascade body
# ---------------------------------------------------------------------------

def test_anytime_csp_cascade_without_evaluate_is_backtracking_only():
    """No EVALUATE → schedule is [Backtracking]. Verify by inspecting
    the schedule the algorithm constructs."""
    space = _nqueens_space(4)
    alg = AnytimeCSP(space)
    schedule = alg._build_schedule()
    assert len(schedule) == 1
    assert schedule[0][0] is Backtracking


def test_anytime_csp_cascade_with_evaluate_includes_minconflicts():
    """EVALUATE present → cascade is [MinConflicts, Backtracking]."""
    from pathos.algorithms.csp import MinConflicts
    space = _nqueens_with_evaluate(4)
    alg = AnytimeCSP(space)
    schedule = alg._build_schedule()
    assert len(schedule) == 2
    assert schedule[0][0] is MinConflicts
    assert schedule[1][0] is Backtracking


def test_anytime_csp_cancel_returns_not_found():
    """Pre-arm the cancel; AnytimeCSP bails before running any phase."""
    space = _nqueens_space(4)
    space._request_cancel()
    result = AnytimeCSP(space).solve()
    assert result.found is False
    assert result.algorithm == "AnytimeCSP"


def test_anytime_csp_algorithm_field_is_meta_not_phase():
    """Returned result's algorithm is 'AnytimeCSP' even though the
    underlying solver was Backtracking."""
    space = _nqueens_space(5)
    result = space.solver().solve()
    assert result.algorithm == "AnytimeCSP"


def test_anytime_csp_with_evaluate_still_finds_4queens():
    """End-to-end with EVALUATE: cascade succeeds (either phase)."""
    space = _nqueens_with_evaluate(4)
    result = space.solver().solve()
    assert result.found is True
    assert result.algorithm == "AnytimeCSP"
