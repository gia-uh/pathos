"""R3 — Capacitated VRP (delivery routes, vehicle loads)."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import TourSpace
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec


@dataclass(frozen=True)
class VRPInstance:
    customers: tuple[tuple[float, float], ...]   # coords
    demands: tuple[float, ...]
    n_vehicles: int
    vehicle_capacity: float
    depot: tuple[float, float]


def _generate(size: int, seed: int) -> VRPInstance:
    rng = random.Random(seed)
    customers = tuple(
        (rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(size)
    )
    demands = tuple(rng.uniform(1.0, 10.0) for _ in range(size))
    n_vehicles = max(2, size // 20)
    vehicle_capacity = float(sum(demands) / n_vehicles) * 1.25
    depot = (50.0, 50.0)
    return VRPInstance(
        customers=customers,
        demands=demands,
        n_vehicles=n_vehicles,
        vehicle_capacity=vehicle_capacity,
        depot=depot,
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_tour_space(inst: VRPInstance) -> TourSpace:
    n_cust = len(inst.customers)
    depot_ids = list(range(inst.n_vehicles))
    cust_ids = list(range(inst.n_vehicles, inst.n_vehicles + n_cust))
    nodes = depot_ids + cust_ids

    def coord(node_id: int) -> tuple[float, float]:
        if node_id < inst.n_vehicles:
            return inst.depot
        return inst.customers[node_id - inst.n_vehicles]

    distances = {
        (i, j): _dist(coord(i), coord(j))
        for i in nodes for j in nodes if i != j
    }
    space = TourSpace(nodes=nodes, distances=distances)

    @space.evaluate
    def tour_cost(tour: tuple[int, ...]) -> float:
        base = sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )
        loads: list[float] = []
        cur = 0.0
        for node in tour:
            if node < inst.n_vehicles:
                loads.append(cur)
                cur = 0.0
            else:
                cur += inst.demands[node - inst.n_vehicles]
        loads.append(cur)
        overshoot = sum(max(0.0, l - inst.vehicle_capacity) for l in loads)
        return base + 1e3 * overshoot

    return space


def _express(inst: VRPInstance) -> TourSpace:
    return _build_tour_space(inst)


register(SuiteSpec(
    id="R3",
    family="tour",
    constraint_classes=frozenset({"iii", "v"}),
    expressibility="full",
    sizes={"S": (100,), "M": (300,), "L": (1000,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    notes="Capacitated VRP via tour with depot-copy boundaries; penalty-fold cap.",
))
