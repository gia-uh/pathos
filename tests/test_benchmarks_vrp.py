import benchmarks.suites.vrp  # noqa: F401 — registers R3 (R4 added in Task 11)
from benchmarks.suites import SUITE_REGISTRY


def test_r3_registered():
    assert "R3" in SUITE_REGISTRY


def test_r3_generator_deterministic():
    spec = SUITE_REGISTRY["R3"]
    a = spec.generate(size=20, seed=7)
    b = spec.generate(size=20, seed=7)
    c = spec.generate(size=20, seed=8)
    assert a == b
    assert a != c


def test_r3_generator_size_is_customer_count():
    spec = SUITE_REGISTRY["R3"]
    inst = spec.generate(size=30, seed=1)
    assert len(inst.customers) == 30


def test_r3_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R3"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None


def test_r3_constraint_classes_include_iii_and_v():
    assert SUITE_REGISTRY["R3"].constraint_classes == frozenset({"iii", "v"})
