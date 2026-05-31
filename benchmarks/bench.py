"""Speed benchmarks for PATHOS across problem sizes.

Run:   python -m benchmarks.bench
       python -m benchmarks.bench --repeat 5 --json results.json
       python -m benchmarks.bench --suites nqueens,tsp
       python -m benchmarks.bench --all-algorithms --timeout 10
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import signal
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable

import pathos.algorithms  # noqa: F401  — register algorithms
from pathos import CSPSpace, Space, TourSpace
from pathos.algorithms.base import Algorithm
from pathos.core.result import SearchResult
from pathos.core.solver import _REGISTRY


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


@contextlib.contextmanager
def _wall_clock_limit(seconds: float):
    """SIGALRM-based wall-clock guard. Unix only; one timer at a time."""
    def handler(signum, frame):
        raise TimeoutError(f"exceeded {seconds:.1f}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_one(
    suite: Suite,
    size: int,
    seed: int,
    algo_cls: type[Algorithm] | None = None,
    timeout: float | None = None,
) -> RunRecord | None:
    """Run one (suite, size) trial.

    If `algo_cls` is given, force-select that algorithm (returns None if
    incompatible with the space). Otherwise let the solver pick.
    """
    space = suite.build(size, seed)
    if algo_cls is not None and not algo_cls.compatible_with(space):
        return None

    algo_name = algo_cls.__name__ if algo_cls else None
    limit = timeout if timeout is not None else suite.timeout
    try:
        with _wall_clock_limit(limit):
            result: SearchResult = space.solver(
                candidates=[algo_cls] if algo_cls else None
            ).solve()
    except TimeoutError:
        return RunRecord(
            suite=suite.name,
            size=size,
            repeat=-1,
            seed=seed,
            algorithm=algo_name or "?",
            found=False,
            cost=None,
            nodes_expanded=-1,
            elapsed=limit,
        )
    except Exception as e:
        # Algorithm declared compatibility but blew up at runtime
        # (e.g. BFS/DFS on CSPSpace: unhashable dict state).
        print(
            f"    ! {algo_name or '?'} raised {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return RunRecord(
            suite=suite.name,
            size=size,
            repeat=-1,
            seed=seed,
            algorithm=algo_name or "?",
            found=False,
            cost=None,
            nodes_expanded=-2,  # sentinel for "raised"
            elapsed=0.0,
        )
    return RunRecord(
        suite=suite.name,
        size=size,
        repeat=-1,
        seed=seed,
        algorithm=result.algorithm,
        found=result.found,
        cost=result.cost,
        nodes_expanded=result.nodes_expanded,
        elapsed=result.elapsed,
    )


def run_suite(suite: Suite, repeat: int, base_seed: int, timeout: float | None = None) -> list[SizeStats]:
    out: list[SizeStats] = []
    for size in suite.sizes:
        records: list[RunRecord] = []
        for r in range(repeat):
            rec = run_one(suite, size, seed=base_seed + r, timeout=timeout)
            assert rec is not None  # only None when algo_cls given
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
# Head-to-head: every compatible algorithm per (suite, size)
# ---------------------------------------------------------------------------

@dataclass
class HeadToHeadCell:
    algorithm: str
    size: int
    runs: int
    found_rate: float
    timeouts: int
    errors: int
    elapsed_median: float | None  # None if all runs timed out / failed
    nodes_median: int | None
    cost_median: float | None


def run_head_to_head(
    suite: Suite, repeat: int, base_seed: int, timeout: float
) -> dict[str, list[HeadToHeadCell]]:
    """For each size, run every algorithm in _REGISTRY compatible with the
    space built at that size. Returns {algorithm_name: [cell_per_size]}."""
    # Use first-seed instance to enumerate compatibility at each size.
    out: dict[str, list[HeadToHeadCell]] = {}
    for size in suite.sizes:
        probe = suite.build(size, base_seed)
        compatible = [c for c in _REGISTRY if c.compatible_with(probe)]
        for algo_cls in compatible:
            records: list[RunRecord] = []
            for r in range(repeat):
                rec = run_one(
                    suite, size,
                    seed=base_seed + r,
                    algo_cls=algo_cls,
                    timeout=timeout,
                )
                if rec is None:
                    continue
                rec.repeat = r
                records.append(rec)
                if rec.nodes_expanded == -1:
                    tag = "T/O"
                elif rec.nodes_expanded == -2:
                    tag = "ERR"
                else:
                    tag = f"{rec.elapsed:.4f}s"
                found = "✓" if rec.found else "✗"
                print(
                    f"  {suite.name} {suite.size_label}={size} "
                    f"{algo_cls.__name__:22s} run {r + 1}/{repeat}: "
                    f"{tag:>10s} {found} nodes={rec.nodes_expanded}",
                    file=sys.stderr,
                )

            if not records:
                continue

            timeouts = sum(1 for r in records if r.nodes_expanded == -1)
            errors = sum(1 for r in records if r.nodes_expanded == -2)
            successful = [r for r in records if r.found]
            elapsed_med = (
                statistics.median(r.elapsed for r in successful)
                if successful else None
            )
            nodes_med = (
                int(statistics.median(r.nodes_expanded for r in successful))
                if successful else None
            )
            costs = [r.cost for r in successful if r.cost is not None]
            cost_med = statistics.median(costs) if costs else None

            cell = HeadToHeadCell(
                algorithm=algo_cls.__name__,
                size=size,
                runs=len(records),
                found_rate=sum(1 for r in records if r.found) / len(records),
                timeouts=timeouts,
                errors=errors,
                elapsed_median=elapsed_med,
                nodes_median=nodes_med,
                cost_median=cost_med,
            )
            out.setdefault(algo_cls.__name__, []).append(cell)
    return out


def render_head_to_head_markdown(
    all_h2h: dict[str, dict[str, list[HeadToHeadCell]]],
) -> str:
    """all_h2h: {suite_name: {algorithm_name: [cell_per_size]}}"""
    lines: list[str] = ["# PATHOS head-to-head benchmark\n"]
    for suite_name, by_algo in all_h2h.items():
        suite = SUITES[suite_name]
        sizes = suite.sizes
        lines.append(f"## {suite_name} — elapsed seconds (median)\n")
        header = f"| algorithm | " + " | ".join(
            f"{suite.size_label}={s}" for s in sizes
        ) + " |"
        sep = "|---|" + "|".join("---:" for _ in sizes) + "|"
        lines.append(header)
        lines.append(sep)

        # Order algorithms by registry order (deterministic).
        ordered_names = [c.__name__ for c in _REGISTRY if c.__name__ in by_algo]
        for name in ordered_names:
            cells_by_size = {c.size: c for c in by_algo[name]}
            row = [name]
            for s in sizes:
                cell = cells_by_size.get(s)
                if cell is None:
                    row.append("—")
                elif cell.errors == cell.runs:
                    row.append("ERR")
                elif cell.timeouts == cell.runs:
                    row.append("T/O")
                elif cell.elapsed_median is None:
                    row.append("fail")
                else:
                    suffix = ""
                    if cell.timeouts:
                        suffix = f" ({cell.timeouts}T/O)"
                    elif cell.errors:
                        suffix = f" ({cell.errors}ERR)"
                    elif cell.found_rate < 1.0:
                        suffix = f" ({int((1 - cell.found_rate) * cell.runs)}✗)"
                    row.append(f"{cell.elapsed_median:.4f}{suffix}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Cost-quality sub-table (only when at least one cell has cost data)
        has_cost = any(
            any(c.cost_median is not None for c in cells)
            for cells in by_algo.values()
        )
        if has_cost:
            lines.append(f"### {suite_name} — solution cost (median)\n")
            lines.append(header)
            lines.append(sep)
            for name in ordered_names:
                cells_by_size = {c.size: c for c in by_algo[name]}
                row = [name]
                for s in sizes:
                    cell = cells_by_size.get(s)
                    if cell is None or cell.cost_median is None:
                        row.append("—")
                    else:
                        row.append(f"{cell.cost_median:.2f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--all-algorithms", action="store_true",
                        help="head-to-head: run every compatible algorithm "
                             "per (suite, size), not just the solver's pick")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="per-run wall-clock timeout in seconds "
                             "(default: 10, used in --all-algorithms mode "
                             "and as the cap in normal mode)")
    args = parser.parse_args()

    selected = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SUITES]
    if unknown:
        print(f"unknown suites: {unknown}. known: {list(SUITES)}", file=sys.stderr)
        return 2

    if args.all_algorithms:
        all_h2h: dict[str, dict[str, list[HeadToHeadCell]]] = {}
        for name in selected:
            print(f"\n→ suite: {name} (head-to-head)", file=sys.stderr)
            all_h2h[name] = run_head_to_head(
                SUITES[name], args.repeat, args.seed, args.timeout
            )
        print(render_head_to_head_markdown(all_h2h))

        if args.json:
            payload = {
                suite_name: {
                    algo: [asdict(c) for c in cells]
                    for algo, cells in by_algo.items()
                }
                for suite_name, by_algo in all_h2h.items()
            }
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nwrote raw records to {args.json}", file=sys.stderr)
        return 0

    all_stats: dict[str, list[SizeStats]] = {}
    for name in selected:
        print(f"\n→ suite: {name}", file=sys.stderr)
        all_stats[name] = run_suite(
            SUITES[name], args.repeat, args.seed, args.timeout
        )

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
