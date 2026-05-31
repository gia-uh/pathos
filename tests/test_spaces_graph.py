import pathos.algorithms  # ensure all algorithms are registered
from pathos.algorithms.uninformed import BFS
from pathos.spaces.graph import GraphSpace


def test_graphspace_astar():
    # Simple weighted graph
    graph = {
        "A": [("B", 1.0), ("C", 4.0)],
        "B": [("C", 2.0), ("D", 5.0)],
        "C": [("D", 1.0)],
        "D": [],
    }
    space = GraphSpace(graph=graph).initial("A")

    @space.goal
    def reached(n): return n == "D"

    @space.heuristic
    def h(n): return {"A": 5.0, "B": 3.0, "C": 1.0, "D": 0.0}.get(n, 0.0)

    result = space.solver().solve()
    assert result.found
    assert result.solution == "D"
    assert result.cost == 4.0  # A->B->C->D = 1+2+1


def test_graphspace_bfs_without_heuristic():
    graph = {"a": [("b", 1)], "b": [("c", 1)], "c": []}
    space = GraphSpace(graph=graph).initial("a")

    @space.goal
    def reached(n): return n == "c"

    # Pin BFS explicitly — the bare auto-pick lands on TabuSearch because
    # GraphSpace auto-declares EVALUATE from edge weights, and TabuSearch
    # outranks BFS. The auto-pick mismatch is tracked in FINDINGS §2b;
    # this test's intent (per its name) is "BFS works on a goal-bearing
    # GraphSpace without a heuristic."
    result = space.solver(candidates=[BFS]).solve()
    assert result.found
    assert result.solution == "c"
