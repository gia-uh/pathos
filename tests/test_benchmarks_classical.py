import pytest
import benchmarks.suites.classical  # noqa: F401 — registers C1/C2/C3
from benchmarks.suites import SUITE_REGISTRY


@pytest.mark.parametrize("sid", ["C1", "C2", "C3"])
def test_classical_suite_registered(sid):
    assert sid in SUITE_REGISTRY


def test_c1_generates_nqueens_csp_and_solves_default():
    spec = SUITE_REGISTRY["C1"]
    inst = spec.generate(8, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.found, "C1 N-Queens N=8 must solve in 2s"


def test_c2_generates_tsp_tourspace_and_solves_default():
    spec = SUITE_REGISTRY["C2"]
    inst = spec.generate(8, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.cost is not None and res.cost > 0


def test_c3_generates_puzzle_and_solves_default():
    spec = SUITE_REGISTRY["C3"]
    inst = spec.generate(10, seed=42)
    space = spec.express(inst)
    res = space.solver(timeout=2).solve()
    assert res.found


def test_classical_suites_have_no_constraint_classes():
    for sid in ("C1", "C2", "C3"):
        assert SUITE_REGISTRY[sid].constraint_classes == frozenset()
        assert SUITE_REGISTRY[sid].expressibility == "full"
        assert SUITE_REGISTRY[sid].missing_capability is None
