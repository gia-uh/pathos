from pathos.core.capabilities import Capability


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
