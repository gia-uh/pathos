import pytest
from pathos.spaces.schedule import ScheduleSpace
from pathos.fairness import weighted_minmax


def test_weighted_minmax_returns_a_callable():
    s = ScheduleSpace(entities=["a"], slots=1)
    helper = weighted_minmax(weights={}, space=s)
    assert callable(helper)


def test_weighted_minmax_all_on_returns_min_weight():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    # downstream defaults to identity: leaves = entities
    helper = weighted_minmax(weights={"a": 0.5, "b": 1.0}, space=s)
    # All ON for both entities -> uptime_fraction = 1.0 each
    # term[a] = 0.5 * 1.0 = 0.5; term[b] = 1.0 * 1.0 = 1.0; min = 0.5
    matrix = ((True, True), (True, True))
    assert helper(matrix) == pytest.approx(0.5)


def test_weighted_minmax_partial_uptime():
    s = ScheduleSpace(entities=["a", "b"], slots=4)
    helper = weighted_minmax(weights={"a": 1.0, "b": 1.0}, space=s)
    # a ON for slot 0 only (uptime 0.25); b ON every slot (uptime 1.0)
    matrix = (
        (True, True),
        (False, True),
        (False, True),
        (False, True),
    )
    # term[a] = 0.25, term[b] = 1.0; min = 0.25
    assert helper(matrix) == pytest.approx(0.25)


def test_weighted_minmax_zero_weight_leaf_is_skipped():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    helper = weighted_minmax(weights={"a": 0.0, "b": 1.0}, space=s)
    # a always OFF (uptime 0), but weight 0 -> skipped
    # b always OFF (uptime 0), weight 1 -> term = 0
    matrix = ((False, False), (False, False))
    assert helper(matrix) == pytest.approx(0.0)
    # a always OFF (skipped); b ON every slot -> term = 1.0; min = 1.0
    matrix2 = ((False, True), (False, True))
    assert helper(matrix2) == pytest.approx(1.0)


def test_weighted_minmax_uses_downstream():
    leaves = {"sub1": ["x", "y"], "sub2": ["z"]}
    s = ScheduleSpace(entities=["sub1", "sub2"], slots=2,
                       downstream=lambda e: leaves[e])
    helper = weighted_minmax(weights={"x": 1.0, "y": 1.0, "z": 1.0}, space=s)
    # sub1 ON every slot, sub2 OFF every slot
    matrix = ((True, False), (True, False))
    # x and y feed from sub1 -> uptime 1.0 each; z feeds from sub2 -> uptime 0.0
    # terms: x=1.0, y=1.0, z=0.0; min = 0.0
    assert helper(matrix) == pytest.approx(0.0)


def test_weighted_minmax_no_weighted_leaves_returns_inf():
    """Edge case: weights map is empty. Min over empty set = +inf
    so the fairness term doesn't influence _evaluate."""
    s = ScheduleSpace(entities=["a"], slots=1)
    helper = weighted_minmax(weights={}, space=s)
    matrix = ((True,),)
    assert helper(matrix) == float("inf")
