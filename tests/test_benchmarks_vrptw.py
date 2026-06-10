import benchmarks.suites.vrp  # noqa: F401 — registers R3 + R4
from benchmarks.suites import SUITE_REGISTRY


def test_r4_registered():
    assert "R4" in SUITE_REGISTRY


def test_r4_marked_missing_capability():
    spec = SUITE_REGISTRY["R4"]
    assert spec.missing_capability == "time_windows"
    assert spec.expressibility == "gap"
    assert spec.constraint_classes == frozenset({"i", "iii", "v"})


def test_r4_runs_with_penalty_fold_workaround():
    spec = SUITE_REGISTRY["R4"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None


def test_r4_instance_has_time_windows():
    spec = SUITE_REGISTRY["R4"]
    inst = spec.generate(size=10, seed=1)
    assert hasattr(inst, "time_windows")
    assert len(inst.time_windows) == 10
