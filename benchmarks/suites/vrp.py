"""R3 — Capacitated VRP (delivery routes, vehicle loads)."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
import pathos.algorithms  # noqa: F401 — register algorithms
from pathos import TourSpace
from benchmarks.suites import register
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.lower_bounds import tsp_one_tree_lb


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


def _r3_lb(inst: VRPInstance) -> float:
    """1-tree LB on the underlying TSP over customers + depot."""
    nodes = list(range(len(inst.customers) + 1))  # 0 = depot
    coords = [inst.depot] + list(inst.customers)
    distances = {
        (i, j): _dist(coords[i], coords[j])
        for i in nodes for j in nodes if i != j
    }
    return float(tsp_one_tree_lb(nodes=nodes, distances=distances))


register(SuiteSpec(
    id="R3",
    family="tour",
    constraint_classes=frozenset({"iii", "v"}),
    expressibility="full",
    sizes={"S": (100,), "M": (300,), "L": (1000,)},
    generate=_generate,
    express=_express,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    lower_bound=_r3_lb,
    notes="Capacitated VRP via tour with depot-copy boundaries; penalty-fold cap.",
))


# ---------------------------------------------------------------------------
# R4 — VRP with time windows (missing-capability suite)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VRPTWInstance:
    customers: tuple[tuple[float, float], ...]
    demands: tuple[float, ...]
    time_windows: tuple[tuple[float, float], ...]  # per customer
    service_time: float
    n_vehicles: int
    vehicle_capacity: float
    depot: tuple[float, float]


def _generate_tw(size: int, seed: int) -> VRPTWInstance:
    rng = random.Random(seed)
    base = _generate(size, seed)
    # Synthesize windows on top of the base capacitated instance.
    horizon = 24.0 * 60.0  # 24h in minutes
    windows: list[tuple[float, float]] = []
    for _ in range(size):
        start = rng.uniform(0.0, horizon * 0.75)
        width = rng.uniform(horizon * 0.05, horizon * 0.25)
        windows.append((start, min(horizon, start + width)))
    return VRPTWInstance(
        customers=base.customers,
        demands=base.demands,
        time_windows=tuple(windows),
        service_time=5.0,
        n_vehicles=base.n_vehicles,
        vehicle_capacity=base.vehicle_capacity,
        depot=base.depot,
    )


def _express_tw(inst: VRPTWInstance) -> TourSpace:
    # Build the same TourSpace as R3, then layer a window penalty on top
    # of the existing @evaluate. Because @evaluate is a single function,
    # we rebuild from scratch and combine both penalties inline.
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
    speed = 1.0  # 1 distance-unit per minute (fast/synthetic)

    @space.evaluate
    def cost(tour: tuple[int, ...]) -> float:
        base = sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )
        # Walk + accumulate vehicle load + per-vehicle time clock.
        loads: list[float] = []
        cur_load = 0.0
        clock = 0.0
        window_overshoot = 0.0
        prev = tour[0]
        for node in tour:
            if node != prev:
                clock += distances[(prev, node)] / speed
            if node < inst.n_vehicles:
                loads.append(cur_load)
                cur_load = 0.0
                clock = 0.0
            else:
                idx = node - inst.n_vehicles
                a, b = inst.time_windows[idx]
                if clock < a:
                    clock = a  # wait at door, no penalty
                elif clock > b:
                    window_overshoot += clock - b
                clock += inst.service_time
                cur_load += inst.demands[idx]
            prev = node
        loads.append(cur_load)
        cap_overshoot = sum(max(0.0, l - inst.vehicle_capacity) for l in loads)
        return base + 1e3 * cap_overshoot + 1e3 * window_overshoot

    return space


register(SuiteSpec(
    id="R4",
    family="tour",
    constraint_classes=frozenset({"i", "iii", "v"}),
    expressibility="gap",
    sizes={"S": (100,), "M": (300,), "L": (1000,)},
    generate=_generate_tw,
    express=_express_tw,
    budgets={"S": 10.0, "M": 30.0, "L": 150.0},
    missing_capability="time_windows",
    notes=(
        "VRP with time windows — TourSpace lacks first-class window "
        "modeling; both capacity and window violations are penalty-folded."
    ),
))
