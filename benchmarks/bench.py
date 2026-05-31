"""Speed benchmarks for PATHOS across problem sizes.

Run:   python -m benchmarks.bench
       python -m benchmarks.bench --repeat 5 --json results.json
       python -m benchmarks.bench --suites nqueens,tsp
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable

import pathos.algorithms  # noqa: F401  — register algorithms
from pathos import CSPSpace, Space, TourSpace
from pathos.core.result import SearchResult


# ---------------------------------------------------------------------------
# Problem builders — each returns a configured Space ready to .solver().solve()
# ---------------------------------------------------------------------------

def build_nqueens(n: int) -> CSPSpace:
    csp = CSPSpace(variables=list(range(n)))

    @csp.domain
    def dom(col):
        return list(range(n))

    @csp.constraint
    def no_attack(assignment):
        items = list(assignment.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c1, r1 = items[i]
                c2, r2 = items[j]
                if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    return csp


def build_tsp(n_cities: int, seed: int) -> TourSpace:
    rng = random.Random(seed)
    cities = list(range(n_cities))
    coords = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in cities}

    def dist(a, b):
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    distances = {(i, j): dist(i, j) for i in cities for j in cities if i != j}
    space = TourSpace(nodes=cities, distances=distances)

    @space.evaluate
    def tour_cost(tour):
        return sum(
            distances[(tour[i], tour[(i + 1) % len(tour)])]
            for i in range(len(tour))
        )

    return space


def build_puzzle(scramble_depth: int, seed: int) -> Space:
    """Standard 3x3 8-puzzle scrambled `scramble_depth` random moves from GOAL."""
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def neighbors(board):
        i = board.index(0)
        row, col = divmod(i, 3)
        out = []
        for dr, dc, name in (
            (-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")
        ):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                j = nr * 3 + nc
                lst = list(board)
                lst[i], lst[j] = lst[j], lst[i]
                out.append((name, tuple(lst)))
        return out

    def manhattan(board):
        total = 0
        for idx, val in enumerate(board):
            if val == 0:
                continue
            goal_i = GOAL.index(val)
            total += abs(idx // 3 - goal_i // 3) + abs(idx % 3 - goal_i % 3)
        return total

    REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}
    rng = random.Random(seed)
    state = GOAL
    last = None
    for _ in range(scramble_depth):
        forbid = REVERSE.get(last)
        moves = [m for m in neighbors(state) if m[0] != forbid]
        last, state = rng.choice(moves)

    space = Space().initial(state)

    @space.successors
    def expand(b):
        yield from neighbors(b)

    @space.goal
    def solved(b):
        return b == GOAL

    @space.heuristic
    def h(b):
        return manhattan(b)

    @space.evaluate
    def cost(b):
        return 1.0

    return space


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

@dataclass
class Suite:
    name: str
    sizes: list[int]
    size_label: str
    build: Callable[[int, int], object]  # (size, seed) -> Space-like
    timeout: float = 30.0


SUITES: dict[str, Suite] = {
    "nqueens": Suite(
        name="nqueens",
        sizes=[6, 8, 10, 12, 14, 16],
        size_label="N",
        build=lambda size, seed: build_nqueens(size),
        timeout=30.0,
    ),
    "tsp": Suite(
        name="tsp",
        sizes=[5, 8, 12, 16, 20, 25],
        size_label="cities",
        build=lambda size, seed: build_tsp(size, seed),
        timeout=30.0,
    ),
    "puzzle8": Suite(
        name="puzzle8",
        sizes=[10, 20, 30, 40, 50],
        size_label="scramble",
        build=lambda size, seed: build_puzzle(size, seed),
        timeout=30.0,
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    suite: str
    size: int
    repeat: int
    seed: int
    algorithm: str | None
    found: bool
    cost: float | None
    nodes_expanded: int
    elapsed: float


@dataclass
class SizeStats:
    suite: str
    size: int
    runs: int
    algorithm: str
    found_rate: float
    elapsed_median: float
    elapsed_min: float
    elapsed_max: float
    nodes_median: float
    cost_median: float | None
    records: list[RunRecord] = field(default_factory=list)


def run_one(suite: Suite, size: int, seed: int) -> RunRecord:
    space = suite.build(size, seed)
    try:
        result: SearchResult = space.solver(timeout=suite.timeout).solve()
    except TypeError:
        # Some Space subclasses may not accept `timeout` in solver(); fall back.
        result = space.solver().solve()
    return RunRecord(
        suite=suite.name,
        size=size,
        repeat=-1,  # filled by caller
        seed=seed,
        algorithm=result.algorithm,
        found=result.found,
        cost=result.cost,
        nodes_expanded=result.nodes_expanded,
        elapsed=result.elapsed,
    )


def run_suite(suite: Suite, repeat: int, base_seed: int) -> list[SizeStats]:
    out: list[SizeStats] = []
    for size in suite.sizes:
        records: list[RunRecord] = []
        for r in range(repeat):
            rec = run_one(suite, size, seed=base_seed + r)
            rec.repeat = r
            records.append(rec)
            print(
                f"  {suite.name} {suite.size_label}={size} run {r + 1}/{repeat}: "
                f"{rec.algorithm} elapsed={rec.elapsed:.4f}s "
                f"nodes={rec.nodes_expanded} found={rec.found}",
                file=sys.stderr,
            )
        elapsed = [r.elapsed for r in records]
        nodes = [r.nodes_expanded for r in records]
        costs = [r.cost for r in records if r.cost is not None]
        algos = {r.algorithm for r in records}
        out.append(
            SizeStats(
                suite=suite.name,
                size=size,
                runs=repeat,
                algorithm=", ".join(sorted(a or "?" for a in algos)),
                found_rate=sum(1 for r in records if r.found) / len(records),
                elapsed_median=statistics.median(elapsed),
                elapsed_min=min(elapsed),
                elapsed_max=max(elapsed),
                nodes_median=statistics.median(nodes),
                cost_median=statistics.median(costs) if costs else None,
                records=records,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown(all_stats: dict[str, list[SizeStats]]) -> str:
    lines: list[str] = ["# PATHOS benchmark results\n"]
    for suite_name, stats in all_stats.items():
        size_label = SUITES[suite_name].size_label
        lines.append(f"## {suite_name}\n")
        lines.append(
            f"| {size_label} | algorithm | found | elapsed median (s) | "
            f"elapsed min–max | nodes median | cost median |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for s in stats:
            cost = f"{s.cost_median:.3f}" if s.cost_median is not None else "—"
            lines.append(
                f"| {s.size} | {s.algorithm} | {s.found_rate:.0%} | "
                f"{s.elapsed_median:.4f} | "
                f"{s.elapsed_min:.4f}–{s.elapsed_max:.4f} | "
                f"{int(s.nodes_median)} | {cost} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3,
                        help="repetitions per (suite,size) pair (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base RNG seed (default: 42)")
    parser.add_argument("--suites", type=str, default=",".join(SUITES),
                        help=f"comma-separated suite names (default: all = {','.join(SUITES)})")
    parser.add_argument("--json", type=str, default=None,
                        help="also write raw records to this JSON path")
    args = parser.parse_args()

    selected = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SUITES]
    if unknown:
        print(f"unknown suites: {unknown}. known: {list(SUITES)}", file=sys.stderr)
        return 2

    all_stats: dict[str, list[SizeStats]] = {}
    for name in selected:
        print(f"\n→ suite: {name}", file=sys.stderr)
        all_stats[name] = run_suite(SUITES[name], args.repeat, args.seed)

    print(render_markdown(all_stats))

    if args.json:
        payload = {
            name: [asdict(s) for s in stats]
            for name, stats in all_stats.items()
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote raw records to {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
