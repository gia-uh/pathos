"""Classical bench suites C1/C2/C3 — N-Queens, TSP, 8-puzzle.

These wrap the existing builders in `benchmarks.bench` so the
classical generators carry over into the unified ladder without
duplication. They occupy the bottom of the difficulty ladder; the
realistic suites R1-R7 sit above.
"""
from __future__ import annotations
import pathos.algorithms  # noqa: F401 — register algorithms
from benchmarks.bench import build_nqueens, build_tsp, build_puzzle
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


CLASSICAL_BUDGETS = {"S": 5.0, "M": 5.0, "L": 5.0}


def _c1_generate(size: int, seed: int) -> int:
    return size  # N-Queens needs only N; seed unused.


def _c1_express(n: int):
    return build_nqueens(n)


register(SuiteSpec(
    id="C1",
    family="csp",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (6, 8), "M": (10, 12), "L": (14, 16)},
    generate=_c1_generate,
    express=_c1_express,
    budgets=CLASSICAL_BUDGETS,
    notes="N-Queens via CSPSpace; classical bottom-of-ladder.",
))


def _c2_generate(size: int, seed: int) -> tuple[int, int]:
    return (size, seed)


def _c2_express(payload):
    n, seed = payload
    return build_tsp(n, seed)


register(SuiteSpec(
    id="C2",
    family="tour",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (5, 8), "M": (12, 16), "L": (20, 25)},
    generate=_c2_generate,
    express=_c2_express,
    budgets=CLASSICAL_BUDGETS,
    notes="TSP uniform-random on TourSpace; classical bottom-of-ladder.",
))


def _c3_generate(size: int, seed: int) -> tuple[int, int]:
    return (size, seed)


def _c3_express(payload):
    depth, seed = payload
    return build_puzzle(depth, seed)


register(SuiteSpec(
    id="C3",
    family="informed",
    constraint_classes=frozenset(),
    expressibility="full",
    sizes={"S": (10, 20), "M": (30, 40), "L": (50,)},
    generate=_c3_generate,
    express=_c3_express,
    budgets=CLASSICAL_BUDGETS,
    notes="8-puzzle with Manhattan heuristic; classical bottom-of-ladder.",
))
