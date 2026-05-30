from pathos.core.space import Space
from pathos.algorithms.informed import AStar, IDAstar, GreedyBestFirst, WeightedAStar, BidirectionalAStar

def _heuristic_space(graph, costs, start, goal, h_fn):
    space = Space().initial(start)

    @space.successors
    def expand(s):
        for neighbor in graph.get(s, []):
            yield neighbor, neighbor

    @space.goal
    def is_goal(s): return s == goal

    @space.heuristic
    def h(s): return h_fn(s)

    @space.evaluate
    def cost(s): return costs.get(s, 1.0)

    return space

# Simple graph: a -> b -> c, h decreases toward goal
GRAPH = {"a": ["b"], "b": ["c"]}
COSTS = {"a": 1.0, "b": 1.0, "c": 0.0}
H = {"a": 2.0, "b": 1.0, "c": 0.0}

def test_astar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = AStar(space).solve()
    assert result.found
    assert result.solution == "c"
    assert result.algorithm == "AStar"

def test_idastar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = IDAstar(space).solve()
    assert result.found
    assert result.solution == "c"

def test_greedy_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = GreedyBestFirst(space).solve()
    assert result.found

def test_weighted_astar_finds_goal():
    space = _heuristic_space(GRAPH, COSTS, "a", "c", lambda s: H[s])
    result = WeightedAStar(space, weight=1.5).solve()
    assert result.found

def test_bidirectional_astar_finds_goal():
    bidir_graph = {"a": ["b"], "b": ["c"], "c": [], "rev_c": ["b"], "rev_b": ["a"]}
    space = Space().initial("a")

    @space.successors
    def fwd(s): yield from ((n, n) for n in GRAPH.get(s, []))

    @space.reverse_successors
    def bwd(s): yield from (("rev_" + s, "a") if s == "b" else [])

    @space.goal
    def is_goal(s): return s == "c"

    @space.heuristic
    def h(s): return H.get(s, 0.0)

    @space.evaluate
    def cost(s): return 1.0

    result = BidirectionalAStar(space).solve()
    assert result.found
