import pytest
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.project_scheduling  # noqa: E402,F401
from benchmarks.suites import SUITE_REGISTRY


def test_r5_registered():
    assert "R5" in SUITE_REGISTRY


def test_r5_marked_missing_capability():
    spec = SUITE_REGISTRY["R5"]
    assert spec.missing_capability == "precedence"
    assert spec.expressibility == "gap"
    assert "ii" in spec.constraint_classes


def test_r5_instance_has_precedence_edges():
    spec = SUITE_REGISTRY["R5"]
    inst = spec.generate(size=(10, 24), seed=1)
    assert hasattr(inst, "precedence")
    # Precedence edges are (predecessor, successor) tuples.
    for u, v in inst.precedence:
        assert u in inst.tasks
        assert v in inst.tasks


def test_r5_runs_with_penalty_fold():
    spec = SUITE_REGISTRY["R5"]
    inst = spec.generate(size=(8, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None


def test_r5_empty_schedule_is_not_optimal():
    """The previous fairness function returned 1.0 on the empty schedule
    (no precedence violations possible when nothing is on), so
    `_evaluate = -fairness + penalty * overshoot = -1.0` and the cascade
    found 'do nothing' as the optimum. After the coverage*precedence
    multiplicative fix, empty schedules have fairness=0 → cost=0, so
    any non-trivial schedule beats them."""
    spec = SUITE_REGISTRY["R5"]
    inst = spec.generate(size=(8, 24), seed=1)
    space = spec.express(inst)
    empty = frozenset()  # ScheduleSpace state shape — no cells ON
    empty_cost = space._evaluate(empty)
    assert empty_cost == 0.0, (
        f"empty schedule cost = {empty_cost}; the multiplicative "
        f"coverage*precedence formulation should give it 0 (not -1)."
    )
