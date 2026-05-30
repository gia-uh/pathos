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
