from pathos.core.capabilities import Capability
from pathos.core.result import SearchResult


def test_capability_enum_members():
    assert Capability.SUCCESSORS in Capability
    assert Capability.GOAL in Capability
    assert Capability.HEURISTIC in Capability
    assert Capability.EVALUATE in Capability
    assert Capability.TERMINAL in Capability
    assert Capability.UTILITY in Capability
    assert Capability.REVERSE_SUCCESSORS in Capability
    assert Capability.VARIABLES in Capability
    assert Capability.DOMAINS in Capability
    assert Capability.CONSTRAINTS in Capability


def test_capability_set_operations():
    required = {Capability.SUCCESSORS, Capability.GOAL}
    available = {Capability.SUCCESSORS, Capability.GOAL, Capability.HEURISTIC}
    assert required <= available


def test_search_result_found():
    r = SearchResult(
        solution="goal",
        path=[("move", "goal")],
        cost=1.0,
        algorithm="BFS",
        nodes_expanded=5,
        elapsed=0.01,
        found=True,
    )
    assert r.found
    assert r.solution == "goal"
    assert r.algorithm == "BFS"


def test_search_result_not_found():
    r = SearchResult.not_found(algorithm="AStar", nodes_expanded=100, elapsed=0.5)
    assert not r.found
    assert r.solution is None
    assert r.cost is None


from pathos.core.space import Space


def test_space_successors_decorator():
    space = Space().initial("a")

    @space.successors
    def expand(state):
        yield "go_b", "b"

    assert Capability.SUCCESSORS in space.capabilities
    results = list(space._successors("a"))
    assert results == [("go_b", "b")]

def test_space_goal_decorator():
    space = Space().initial("a")

    @space.goal
    def is_goal(state):
        return state == "b"

    assert Capability.GOAL in space.capabilities
    assert space._goal("b")
    assert not space._goal("a")

def test_space_heuristic_decorator():
    space = Space().initial("a")

    @space.heuristic
    def h(state):
        return 0.0

    assert Capability.HEURISTIC in space.capabilities

def test_space_evaluate_decorator():
    space = Space().initial("a")

    @space.evaluate
    def cost(state):
        return 1.0

    assert Capability.EVALUATE in space.capabilities

def test_space_initial_value():
    space = Space().initial("start")
    assert space._initial == "start"

def test_space_initial_callable():
    space = Space().initial(lambda: "start")
    assert space._initial == "start"
