from pathos.core.space import Space
from pathos.algorithms.csp import Backtracking, ForwardChecking, AC3, MinConflicts


def _nqueens_space(n=4):
    """N-Queens as a CSP: variables=cols, domain=rows, constraint=no attack."""
    space = Space().initial({})
    space._n = n

    @space.successors
    def expand(assignment):
        col = len(assignment)
        if col >= n:
            return
        for row in range(n):
            consistent = all(
                assignment[c] != row and
                abs(assignment[c] - row) != abs(c - col)
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


def test_backtracking_4queens():
    space = _nqueens_space(4)
    result = Backtracking(space).solve()
    assert result.found
    sol = result.solution
    assert len(sol) == 4
    # verify no queens attack each other
    cols = list(sol.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            assert sol[cols[i]] != sol[cols[j]]
            assert abs(sol[cols[i]] - sol[cols[j]]) != abs(cols[i] - cols[j])


def test_forward_checking_4queens():
    space = _nqueens_space(4)
    result = ForwardChecking(space).solve()
    assert result.found
