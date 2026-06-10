import pytest
pytest.importorskip("pathos.spaces.schedule")

import benchmarks.suites.rostering  # noqa: E402,F401 — registers R2
from benchmarks.suites import SUITE_REGISTRY


def test_r2_registered():
    assert "R2" in SUITE_REGISTRY


def test_r2_generator_deterministic():
    spec = SUITE_REGISTRY["R2"]
    a = spec.generate(size=(20, 72), seed=7)
    b = spec.generate(size=(20, 72), seed=7)
    assert a == b


def test_r2_express_yields_space_that_solves_default():
    spec = SUITE_REGISTRY["R2"]
    inst = spec.generate(size=(10, 24), seed=1)
    space = spec.express(inst)
    res = space.solver(timeout=3).solve()
    assert res.cost is not None


def test_r2_constraint_classes_include_iii_iv_v_and_expressibility_partial():
    spec = SUITE_REGISTRY["R2"]
    assert spec.constraint_classes == frozenset({"iii", "iv", "v"})
    assert spec.expressibility == "partial"
    assert "soft_preferences" in spec.notes  # the partial-gap flag
