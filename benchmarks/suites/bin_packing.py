"""R6 — Bin packing with variable items, single bin class."""
from __future__ import annotations
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import Space
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.lower_bounds import bin_packing_waste_lb


@dataclass(frozen=True)
class BinPackingInstance:
    items: tuple[float, ...]
    capacity: float


def _generate(size: int, seed: int) -> BinPackingInstance:
    rng = random.Random(seed)
    capacity = 1.0
    items = tuple(rng.uniform(0.1, 0.9) for _ in range(size))
    return BinPackingInstance(items=items, capacity=capacity)


def _express(inst: BinPackingInstance) -> Space:
    n = len(inst.items)
    initial = tuple(range(n))
    space = Space().initial(initial)

    @space.successors
    def neighbors(state: tuple[int, ...]):
        used = set(state)
        for i in range(n):
            for target in list(used) + [max(used) + 1]:
                if target == state[i]:
                    continue
                new = list(state)
                new[i] = target
                yield (i, target), tuple(new)

    @space.evaluate
    def cost(state: tuple[int, ...]) -> float:
        load: dict[int, float] = {}
        for idx, b in enumerate(state):
            load[b] = load.get(b, 0.0) + inst.items[idx]
        bins_used = len(load)
        overshoot = sum(max(0.0, l - inst.capacity) for l in load.values())
        return float(bins_used) + 1e3 * overshoot

    return space


def _lb(inst: BinPackingInstance) -> float:
    return float(bin_packing_waste_lb(items=inst.items, capacity=inst.capacity))


register(SuiteSpec(
    id="R6",
    family="evaluate",
    constraint_classes=frozenset({"v"}),
    expressibility="full",
    sizes={"S": (50,), "M": (200,), "L": (1000,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    lower_bound=_lb,
    notes="Bin packing — uniform items, single bin class; penalty-fold capacity.",
))
