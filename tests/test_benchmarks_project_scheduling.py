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
