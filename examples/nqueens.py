"""N-Queens via CSPSpace."""
import pathos.algorithms  # ensure all algorithms are registered
from pathos import CSPSpace

N = 8
csp = CSPSpace(variables=list(range(N)))


@csp.domain
def dom(col): return list(range(N))


@csp.constraint
def no_attack(assignment):
    items = list(assignment.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            c1, r1 = items[i]; c2, r2 = items[j]
            if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                return False
    return True


result = csp.solver().solve()
if result.found:
    board = [
        "." * result.solution[col] + "Q" + "." * (N - result.solution[col] - 1)
        for col in range(N)
    ]
    print("\n".join(board))
    print(f"\nAlgorithm: {result.algorithm} | Nodes expanded: {result.nodes_expanded}")
