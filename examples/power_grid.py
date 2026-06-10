"""Power-grid blackout scheduling — ScheduleSpace worked example.

Run directly:
    python examples/power_grid.py
"""
from __future__ import annotations
import random
import pathos.algorithms  # ensure all algorithms are registered
from pathos import ScheduleSpace
from pathos.fairness import weighted_minmax


def run():
    rng = random.Random(42)
    n_substations = 20
    n_slots = 168

    substations = [f"sub_{i}" for i in range(n_substations)]
    leaves_per_sub = {
        s: [f"{s}_leaf_{j}" for j in range(rng.randint(2, 4))]
        for s in substations
    }
    all_leaves = [
        leaf for leaves in leaves_per_sub.values() for leaf in leaves
    ]
    # Higher importance = critical (hospital). Lower = tolerant (residential).
    weights = {leaf: rng.choice([1.0, 0.5, 0.1]) for leaf in all_leaves}

    base_load = {
        (s, t): rng.uniform(50, 150)
        for s in substations for t in range(n_slots)
    }
    supply = [
        sum(base_load[s, t] for s in substations) * 0.7
        for t in range(n_slots)
    ]

    space = (
        ScheduleSpace(
            entities=substations,
            slots=n_slots,
            downstream=lambda s: leaves_per_sub[s],
            penalty=1e4,
        )
        .target(tolerance=0.05)
        .mode("auto")
    )

    @space.demand
    def demand(sub, slot):
        return base_load[sub, slot]

    @space.capacity
    def capacity(slot):
        return supply[slot]

    _fair = weighted_minmax(weights, space)

    @space.fairness
    def fairness(schedule):
        return _fair(schedule)

    # 30s is generous for the 20x168 instance — the example's purpose is
    # API-drift smoke, not a perf benchmark, so give slow CI runners
    # comfortable headroom for the AnytimeLocal cascade to land on an
    # incumbent.
    return space.solver(timeout=30).solve()


if __name__ == "__main__":
    result = run()
    print(f"Algorithm: {result.algorithm}")
    print(f"Found feasible schedule: {result.found}")
    print(f"Objective (-fairness + lambda*violations): {result.cost:.3f}")
    print(f"Mean slack: {sum(result.slack) / len(result.slack):.1f}")
    print(f"Min slack:  {min(result.slack):+.1f}  (negative = capacity overshoot)")
