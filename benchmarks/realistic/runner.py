"""Realistic benchmark runner.

Iterates (suite, size, seed) and emits RunRows. Reuses
benchmarks.bench._wall_clock_limit as a safety net around solver
timeouts; pathos's anytime cascade already honours the timeout
cooperatively in most cases.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import pathos.algorithms  # noqa: F401 — register algorithms
from pathos.core.solver import _REGISTRY
from pathos.core.result import SearchResult
from benchmarks.bench import _wall_clock_limit
from benchmarks.suites import SUITE_REGISTRY
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.records import RunRow


@dataclass
class RunConfig:
    seeds: int = 5
    oracle: bool = True
    stress_multiplier: float = 5.0
    base_seed: int = 42
    budget_scale: float = 1.0


def _solve_and_record(
    suite_id: str, size_payload, seed: int, mode: str, algorithm_name: str | None,
    space, timeout: float, missing_capability: str | None,
) -> RunRow:
    """Run one solve and wrap into a RunRow. Handles crash + timeout."""
    candidates = None
    algo_cls = None
    if algorithm_name is not None:
        algo_cls = next(
            (c for c in _REGISTRY if c.__name__ == algorithm_name), None,
        )
        if algo_cls is None:
            return RunRow(
                suite=suite_id, size=_size_to_int(size_payload), seed=seed,
                mode=mode, algorithm=algorithm_name, cost=None, feasible=False,
                elapsed=0.0, nodes_expanded=-2,
                missing_capability=missing_capability,
            )
        candidates = [algo_cls]

    t0 = time.perf_counter()
    try:
        with _wall_clock_limit(timeout + 2.0):  # safety net beyond cooperative timeout
            res: SearchResult = space.solver(
                candidates=candidates, timeout=timeout,
            ).solve()
    except TimeoutError:
        return RunRow(
            suite=suite_id, size=_size_to_int(size_payload), seed=seed,
            mode=mode, algorithm=algorithm_name or "?",
            cost=None, feasible=False, elapsed=timeout + 2.0,
            nodes_expanded=-1, missing_capability=missing_capability,
        )
    except Exception as e:
        print(
            f"    ! {algorithm_name or '?'} raised {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return RunRow(
            suite=suite_id, size=_size_to_int(size_payload), seed=seed,
            mode=mode, algorithm=algorithm_name or "?",
            cost=None, feasible=False, elapsed=time.perf_counter() - t0,
            nodes_expanded=-2, missing_capability=missing_capability,
        )
    return RunRow(
        suite=suite_id, size=_size_to_int(size_payload), seed=seed,
        mode=mode, algorithm=res.algorithm,
        cost=res.cost, feasible=res.found, elapsed=res.elapsed,
        nodes_expanded=res.nodes_expanded,
        missing_capability=missing_capability, epsilon=res.epsilon,
    )


def _size_to_int(size_payload) -> int:
    """Best-effort integer key for a size payload.

    Tuples like (n_substations, n_slots) collapse to their product;
    plain ints stay as-is. Used only for the RunRow.size field for
    table rendering — the original payload is what's passed to
    generate()/express().
    """
    if isinstance(size_payload, int):
        return size_payload
    if isinstance(size_payload, tuple):
        prod = 1
        for v in size_payload:
            if isinstance(v, int):
                prod *= v
            elif isinstance(v, tuple):
                # (n_sub, n_slots) etc.
                for vv in v:
                    prod *= vv if isinstance(vv, int) else 1
        return prod
    return 0


def run_one(suite_id: str, tier: str, cfg: RunConfig) -> list[RunRow]:
    """Run one (suite, tier) for `cfg.seeds` seeds. Returns all rows
    (auto_headline + auto_stress + optionally oracle_* per compatible algorithm)."""
    spec: SuiteSpec = SUITE_REGISTRY[suite_id]
    if tier not in spec.sizes:
        return []
    rows: list[RunRow] = []
    budget = spec.budgets[tier] * cfg.budget_scale
    for size_payload in spec.sizes[tier]:
        for r in range(cfg.seeds):
            seed = cfg.base_seed + r
            inst = spec.generate(size_payload, seed)
            # Build once; reused for headline+stress (oracle gets a fresh build
            # per algorithm so cancellation state doesn't leak).
            space = spec.express(inst)
            rows.append(_solve_and_record(
                suite_id, size_payload, seed, "auto_headline", None,
                space, budget, spec.missing_capability,
            ))
            stress_space = spec.express(inst)
            rows.append(_solve_and_record(
                suite_id, size_payload, seed, "auto_stress", None,
                stress_space, budget * cfg.stress_multiplier,
                spec.missing_capability,
            ))
            if cfg.oracle:
                probe = spec.express(inst)
                compatible = [
                    c.__name__ for c in _REGISTRY if c.compatible_with(probe)
                ]
                for algo_name in compatible:
                    oracle_space = spec.express(inst)
                    rows.append(_solve_and_record(
                        suite_id, size_payload, seed,
                        f"oracle_{algo_name}", algo_name,
                        oracle_space, budget, spec.missing_capability,
                    ))
            if spec.lower_bound is not None:
                lb_value = spec.lower_bound(inst)
                rows.append(RunRow(
                    suite=suite_id,
                    size=_size_to_int(size_payload),
                    seed=seed, mode="lower_bound",
                    algorithm="lb",
                    cost=float(lb_value), feasible=True,
                    elapsed=0.0, nodes_expanded=0,
                    missing_capability=spec.missing_capability,
                ))
    return rows


def run_suite_tier(suite_id: str, tier: str, cfg: RunConfig) -> list[RunRow]:
    """Alias for run_one — kept for API symmetry with later batch helpers."""
    return run_one(suite_id, tier, cfg)


def run_sweep(
    suite_ids: Iterable[str], tiers: Iterable[str], cfg: RunConfig,
) -> list[RunRow]:
    """Full sweep across (suite_id, tier) pairs."""
    rows: list[RunRow] = []
    for sid in suite_ids:
        for tier in tiers:
            print(f"\n→ {sid} / {tier}", file=sys.stderr)
            rows.extend(run_one(sid, tier, cfg))
    return rows
