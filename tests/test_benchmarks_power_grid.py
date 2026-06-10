import pytest

# Skip the whole module if ScheduleSpace hasn't shipped yet.
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.power_grid  # noqa: E402,F401 — registers R1
from benchmarks.suites import SUITE_REGISTRY


def test_r1_registered():
    assert "R1" in SUITE_REGISTRY


def test_r1_generator_deterministic():
    spec = SUITE_REGISTRY["R1"]
    sub_a, slots_a = 20, 72
    a = spec.generate(size=(sub_a, slots_a), seed=7)
    b = spec.generate(size=(sub_a, slots_a), seed=7)
    assert a == b


def test_r1_express_yields_schedulespace_that_solves_default():
    spec = SUITE_REGISTRY["R1"]
    inst = spec.generate(size=(10, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None


def test_r1_constraint_classes_include_iii_iv_v():
    assert SUITE_REGISTRY["R1"].constraint_classes == frozenset({"iii", "iv", "v"})
    assert SUITE_REGISTRY["R1"].expressibility == "partial"
