import benchmarks.suites.bin_packing  # noqa: F401 — registers R6
from benchmarks.suites import SUITE_REGISTRY


def test_r6_registered():
    assert "R6" in SUITE_REGISTRY


def test_r6_generator_is_deterministic_per_seed():
    spec = SUITE_REGISTRY["R6"]
    a = spec.generate(size=50, seed=7)
    b = spec.generate(size=50, seed=7)
    c = spec.generate(size=50, seed=8)
    assert a == b
    assert a != c


def test_r6_generator_respects_size():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=50, seed=1)
    assert len(inst.items) == 50


def test_r6_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=20, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    # bin packing is pure optimization; "found" is true when local
    # search terminates (no goal predicate). Cost must be finite.
    assert res.cost is not None
    assert res.cost > 0


def test_r6_lower_bound_callable():
    spec = SUITE_REGISTRY["R6"]
    inst = spec.generate(size=20, seed=1)
    assert spec.lower_bound is not None
    lb = spec.lower_bound(inst)
    assert lb >= 1
