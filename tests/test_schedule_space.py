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


def test_target_default_is_zero_tolerance():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._tolerance == 0.0


def test_target_sets_tolerance():
    s = ScheduleSpace(entities=["a"], slots=1).target(tolerance=0.1)
    assert s._tolerance == 0.1


def test_target_returns_self_for_chaining():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.target(tolerance=0.05) is s


def test_target_rejects_tolerance_below_0():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="tolerance"):
        s.target(tolerance=-0.1)


def test_target_rejects_tolerance_above_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="tolerance"):
        s.target(tolerance=1.5)


def test_target_lower_band_penalises_under_use():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=10.0).target(tolerance=0.5)
    @s.demand
    def _d(e, t): return 5.0
    @s.capacity
    def _c(t): return 10.0    # tolerance=0.5 -> lower band = 5.0
    @s.fairness
    def _f(schedule): return 0.0
    state = frozenset()  # load=0, lower band undershoots by 5
    # _evaluate = -0 + 10 * 5 = 50
    assert s._evaluate(state) == pytest.approx(50.0)


def test_neighborhood_default_is_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._k == 1


def test_neighborhood_sets_k():
    s = ScheduleSpace(entities=["a"], slots=1).neighborhood(k=2)
    assert s._k == 2


def test_neighborhood_returns_self_for_chaining():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.neighborhood(k=2) is s


def test_neighborhood_rejects_k_below_1():
    s = ScheduleSpace(entities=["a"], slots=1)
    with pytest.raises(ValueError, match="k"):
        s.neighborhood(k=0)


import dataclasses
import pathos.algorithms  # noqa: F401 — ensure registry populated
from pathos import ScheduleSpace as _PublicScheduleSpace  # re-export check
from pathos.spaces.schedule import ScheduleSpace as _PrivateScheduleSpace


def test_schedule_space_re_exported_at_top_level():
    assert _PublicScheduleSpace is _PrivateScheduleSpace


def test_solver_returns_matrix_solution_and_slack_for_feasible_problem():
    s = ScheduleSpace(entities=["a"], slots=2, penalty=1e3)
    @s.demand
    def _d(e, t): return 1.0
    @s.capacity
    def _c(t): return 5.0   # huge headroom
    @s.fairness
    def _f(schedule): return 0.5
    result = s.solver(timeout=2).solve()
    assert result.found
    # solution is a (T, N) tuple-of-tuples, not a frozenset
    assert isinstance(result.solution, tuple)
    assert len(result.solution) == 2
    assert all(isinstance(row, tuple) for row in result.solution)
    # slack is a per-slot list, one entry per slot
    assert result.slack is not None
    assert len(result.slack) == 2


def test_solver_slack_reflects_capacity_minus_load():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=1e3)
    @s.demand
    def _d(e, t): return 2.0
    @s.capacity
    def _c(t): return 10.0
    @s.fairness
    def _f(schedule):
        # ON => uptime fraction 1; prefer ON
        return float(schedule[0][0])
    result = s.solver(timeout=2).solve()
    assert result.found
    # If the algorithm chose ON: slack = 10 - 2 = 8; if OFF: slack = 10
    assert result.slack[0] in (pytest.approx(8.0), pytest.approx(10.0))


def test_example_power_grid_runs_end_to_end():
    """Smoke: the example runs, picks AnytimeLocal, returns a feasible
    schedule. Locks no exact float values; catches API drift only."""
    import importlib.util
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "power_grid.py"
    assert example_path.exists(), f"Missing example at {example_path}"
    spec = importlib.util.spec_from_file_location("power_grid_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run()
    assert result.found
    assert result.algorithm == "AnytimeLocal"
    # All slots should be feasible (slack >= 0) after the AnytimeLocal cascade.
    assert result.slack is not None
    assert all(s >= -1e-6 for s in result.slack), (
        f"Infeasible schedule returned: slack min = {min(result.slack)}"
    )
