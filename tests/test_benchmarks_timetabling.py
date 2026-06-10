import benchmarks.suites.timetabling  # noqa: F401 — registers R7
from benchmarks.suites import SUITE_REGISTRY


def test_r7_registered():
    assert "R7" in SUITE_REGISTRY


def test_r7_generator_deterministic():
    spec = SUITE_REGISTRY["R7"]
    a = spec.generate(size=20, seed=1)
    b = spec.generate(size=20, seed=1)
    assert a == b


def test_r7_express_yields_cspspace_that_solves_default():
    spec = SUITE_REGISTRY["R7"]
    inst = spec.generate(size=15, seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None or res.found


def test_r7_partial_gap_recorded():
    spec = SUITE_REGISTRY["R7"]
    assert spec.expressibility == "partial"
    assert "soft_preferences" in spec.notes
