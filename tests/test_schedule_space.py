import pytest
from pathos.spaces.schedule import ScheduleSpace


def test_constructor_accepts_entities_and_slots():
    s = ScheduleSpace(entities=["a", "b", "c"], slots=4)
    assert s._entities == ("a", "b", "c")
    assert s._slots == 4


def test_constructor_rejects_empty_entities():
    with pytest.raises(ValueError, match="entities"):
        ScheduleSpace(entities=[], slots=4)


def test_constructor_rejects_zero_or_negative_slots():
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=0)
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=-1)


def test_constructor_default_downstream_is_identity():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    assert list(s._downstream("a")) == ["a"]
    assert list(s._downstream("b")) == ["b"]


def test_constructor_custom_downstream():
    leaves = {"sub1": ["x", "y"], "sub2": ["z"]}
    s = ScheduleSpace(entities=["sub1", "sub2"], slots=2,
                       downstream=lambda e: leaves[e])
    assert list(s._downstream("sub1")) == ["x", "y"]
    assert list(s._downstream("sub2")) == ["z"]


def test_constructor_default_penalty_is_1000():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._penalty == 1e3


def test_constructor_custom_penalty():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=42.0)
    assert s._penalty == 42.0


def test_initial_state_is_empty_frozenset():
    s = ScheduleSpace(entities=["a", "b"], slots=3)
    assert s._initial == frozenset()


def test_capabilities_empty_before_any_decorator():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.capabilities == set()


from pathos.core.capabilities import Capability


def _attach_demand_capacity_fairness(s):
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 1.0
    @s.fairness
    def _f(schedule): return 0.5
    return _d, _c, _f


def test_demand_decorator_stores_function():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.demand
    def my_demand(entity, slot): return 7.0
    assert s._demand_fn is my_demand


def test_capacity_decorator_stores_function():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.capacity
    def my_capacity(slot): return 99.0
    assert s._capacity_fn is my_capacity


def test_fairness_decorator_stores_function_and_emits_evaluate():
    s = ScheduleSpace(entities=["a"], slots=2)
    assert Capability.EVALUATE not in s.capabilities
    @s.fairness
    def my_fairness(schedule): return 0.3
    assert s._fairness_fn is my_fairness
    assert Capability.EVALUATE in s.capabilities


def test_demand_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.demand
    def _d1(e, t): return 1.0
    with pytest.raises(RuntimeError, match="demand"):
        @s.demand
        def _d2(e, t): return 2.0


def test_capacity_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.capacity
    def _c1(t): return 1.0
    with pytest.raises(RuntimeError, match="capacity"):
        @s.capacity
        def _c2(t): return 2.0


def test_fairness_decorator_raises_on_reattach():
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.fairness
    def _f1(schedule): return 0.5
    with pytest.raises(RuntimeError, match="fairness"):
        @s.fairness
        def _f2(schedule): return 0.6


def test_fairness_decorator_returns_the_callable_unchanged():
    """Standard pathos pattern — decorator returns fn so user can also call it."""
    s = ScheduleSpace(entities=["a"], slots=2)
    @s.fairness
    def f(schedule): return 0.5
    assert f is s._fairness_fn
