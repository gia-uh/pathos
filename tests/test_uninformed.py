from pathos.core.space import Space
from pathos.algorithms.uninformed import BFS, DFS, IDDFS, UCS

def _graph_space(graph: dict, start: str, goal: str):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for action, (neighbor, cost) in graph.get(s, {}).items():
            yield action, neighbor

    @space.goal
    def is_goal(s): return s == goal
    return space

def _weighted_space(graph: dict, start: str, goal: str):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for neighbor, cost in graph.get(s, []):
            yield neighbor, neighbor  # action = neighbor name

    @space.goal
    def is_goal(s): return s == goal

    @space.evaluate
    def edge_cost(s): return 1.0
    return space

SIMPLE = {"a": {"ab": ("b", 1)}, "b": {"bc": ("c", 1)}}

def test_bfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = BFS(space).solve()
    assert result.found
    assert result.solution == "c"
    assert result.algorithm == "BFS"

def test_dfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = DFS(space).solve()
    assert result.found

def test_iddfs_finds_goal():
    space = _graph_space(SIMPLE, "a", "c")
    result = IDDFS(space).solve()
    assert result.found
    assert result.solution == "c"

def test_bfs_optimal_path():
    # BFS finds shortest path in unweighted graph
    graph = {
        "a": {"ab": ("b", 1), "ac": ("c", 1)},
        "b": {"bd": ("d", 1)},
        "c": {"cd": ("d", 1)},
    }
    space = _graph_space(graph, "a", "d")
    result = BFS(space).solve()
    assert result.found
    assert len(result.path) == 2  # a->b->d or a->c->d

def test_bfs_no_solution():
    space = Space().initial("a")

    @space.successors
    def expand(s): return iter([])

    @space.goal
    def is_goal(s): return s == "b"

    result = BFS(space).solve()
    assert not result.found
