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


def test_to_matrix_empty_state_is_all_false():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    m = s._to_matrix(frozenset())
    assert m == ((False, False), (False, False))


def test_to_matrix_partial_state():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    # ON cells: (slot=0, entity_idx=1), (slot=1, entity_idx=0)
    m = s._to_matrix(frozenset({(0, 1), (1, 0)}))
    assert m == ((False, True), (True, False))


def test_successors_count_is_T_times_N():
    s = ScheduleSpace(entities=["a", "b", "c"], slots=4)
    _attach_demand_capacity_fairness(s)
    state = frozenset()
    children = list(s._successors(state))
    assert len(children) == 12   # 4 * 3


def test_successors_each_action_toggles_one_cell():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    _attach_demand_capacity_fairness(s)
    state = frozenset({(0, 0)})
    children = dict(s._successors(state))
    # action label is "(slot, entity_idx)" for the toggled cell
    assert frozenset({(0, 0), (0, 1)}) in children.values()  # turn (0,1) ON
    assert frozenset() in children.values()                    # turn (0,0) OFF
    assert frozenset({(0, 0), (1, 0)}) in children.values()  # turn (1,0) ON
    assert frozenset({(0, 0), (1, 1)}) in children.values()  # turn (1,1) ON


def test_successors_capability_emitted_after_demand_capacity_fairness():
    s = ScheduleSpace(entities=["a"], slots=2)
    _attach_demand_capacity_fairness(s)
    assert Capability.SUCCESSORS in s.capabilities


def test_evaluate_pure_fairness_when_feasible():
    s = ScheduleSpace(entities=["a", "b"], slots=2, penalty=1e3)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 10.0    # huge, never violated
    @s.fairness
    def _f(schedule): return 0.42
    state = frozenset({(0, 0), (1, 1)})
    # fairness = 0.42, no violation -> _evaluate = -0.42
    assert s._evaluate(state) == pytest.approx(-0.42)


def test_evaluate_penalises_capacity_overshoot():
    s = ScheduleSpace(entities=["a", "b"], slots=1, penalty=1e3)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0     # any single entity ON overshoots by 2
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset({(0, 0)})   # one entity ON, demand=5, cap=3 -> overshoot=2
    # _evaluate = -0 + 1000*2 = 2000
    assert s._evaluate(state) == pytest.approx(2000.0)


def test_evaluate_overshoot_summed_across_slots():
    s = ScheduleSpace(entities=["a"], slots=3, penalty=10.0)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0    # overshoot=2 per slot with entity ON
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset({(0, 0), (1, 0), (2, 0)})  # ON every slot
    # overshoot = 2 + 2 + 2 = 6; _evaluate = 60.0
    assert s._evaluate(state) == pytest.approx(60.0)


def test_evaluate_off_slot_contributes_zero_overshoot():
    s = ScheduleSpace(entities=["a"], slots=2, penalty=10.0)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 3.0
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset()  # all off, no demand, no overshoot
    assert s._evaluate(state) == pytest.approx(0.0)


def test_evaluate_rejects_negative_capacity():
    s = ScheduleSpace(entities=["a"], slots=1)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return -1.0
    @s.fairness
    def _f(schedule): return 0.0
    with pytest.raises(ValueError, match="capacity"):
        s._evaluate(frozenset())
