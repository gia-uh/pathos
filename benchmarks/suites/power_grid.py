"""R1 — Power-grid blackout scheduling on ScheduleSpace.

Generator parameters mirror the ScheduleSpace worked example:
- `size` is (n_substations, n_slots).
- Each substation has 2-4 downstream leaves, each weighted by importance
  (0.1 residential / 0.5 industrial / 1.0 critical).
- Base load is uniform-random per (substation, slot); supply is sized
  at 70% of total demand so the solver must shed load.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import ScheduleSpace
from pathos.fairness import weighted_minmax
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class PowerGridInstance:
    substations: tuple[str, ...]
    leaves_per_sub: dict[str, tuple[str, ...]]
    weights: dict[str, float]
    base_load: dict[tuple[str, int], float]
    supply: tuple[float, ...]
    n_slots: int


def _generate(size, seed: int) -> PowerGridInstance:
    n_sub, n_slots = size
    rng = random.Random(seed)
    substations = tuple(f"sub_{i}" for i in range(n_sub))
    leaves_per_sub = {
        s: tuple(f"{s}_leaf_{j}" for j in range(rng.randint(2, 4)))
        for s in substations
    }
    all_leaves = [l for ls in leaves_per_sub.values() for l in ls]
    weights = {l: rng.choice([1.0, 0.5, 0.1]) for l in all_leaves}
    base_load = {
        (s, t): rng.uniform(50.0, 150.0)
        for s in substations for t in range(n_slots)
    }
    supply = tuple(
        sum(base_load[s, t] for s in substations) * 0.7
        for t in range(n_slots)
    )
    return PowerGridInstance(
        substations=substations,
        leaves_per_sub=dict(leaves_per_sub),
        weights=weights,
        base_load=base_load,
        supply=supply,
        n_slots=n_slots,
    )


def _express(inst: PowerGridInstance):
    space = (
        ScheduleSpace(
            entities=list(inst.substations),
            slots=inst.n_slots,
            downstream=lambda s: inst.leaves_per_sub[s],
            penalty=1e3,
        )
        .target(tolerance=0.05)
    )

    @space.demand
    def demand(sub, slot):
        return inst.base_load[sub, slot]

    @space.capacity
    def capacity(slot):
        return inst.supply[slot]

    @space.fairness
    def fairness(schedule):
        return weighted_minmax(inst.weights, space)(schedule)

    return space


register(SuiteSpec(
    id="R1",
    family="schedule",
    constraint_classes=frozenset({"iii", "iv", "v"}),
    expressibility="partial",
    sizes={"S": ((20, 72),), "M": ((50, 168),), "L": ((100, 336),)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes="Power-grid blackout scheduling; multi-resource folded into penalty.",
))
