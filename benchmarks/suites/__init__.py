"""Suite registry for the realistic benchmark sweep.

Every suite module registers a `SuiteSpec` into `SUITE_REGISTRY` at
import time. `benchmarks.realistic.runner` iterates this dict.
"""
from __future__ import annotations
from benchmarks.suites._spec import SuiteSpec

SUITE_REGISTRY: dict[str, SuiteSpec] = {}


def register(spec: SuiteSpec) -> SuiteSpec:
    if spec.id in SUITE_REGISTRY:
        raise RuntimeError(f"suite id {spec.id!r} already registered")
    SUITE_REGISTRY[spec.id] = spec
    return spec
