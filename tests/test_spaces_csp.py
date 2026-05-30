import pathos.algorithms  # ensure all algorithms are registered
from pathos.spaces.csp import CSPSpace


def test_csp_nqueens():
    n = 4
    csp = CSPSpace(variables=list(range(n)))

    @csp.domain
    def dom(var): return list(range(n))

    @csp.constraint
    def no_attack(assignment):
        items = list(assignment.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c1, r1 = items[i]
                c2, r2 = items[j]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    result = csp.solver().solve()
    assert result.found
    assert len(result.solution) == n


def test_csp_graph_coloring():
    # 3-color a triangle: must use different colors for adjacent nodes
    csp = CSPSpace(variables=["A", "B", "C"])

    @csp.domain
    def dom(var): return ["red", "green", "blue"]

    edges = {("A", "B"), ("B", "C"), ("A", "C")}

    @csp.constraint
    def different_colors(assignment):
        for (u, v) in edges:
            if u in assignment and v in assignment:
                if assignment[u] == assignment[v]:
                    return False
        return True

    result = csp.solver().solve()
    assert result.found
    sol = result.solution
    assert sol["A"] != sol["B"]
    assert sol["B"] != sol["C"]
    assert sol["A"] != sol["C"]
