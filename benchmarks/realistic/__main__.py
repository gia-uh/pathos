"""CLI: python -m benchmarks.realistic [options]"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Import all suite modules so SUITE_REGISTRY is populated.
import benchmarks.suites.classical       # noqa: F401
import benchmarks.suites.bin_packing     # noqa: F401
import benchmarks.suites.vrp             # noqa: F401
try:
    import benchmarks.suites.power_grid  # noqa: F401
    import benchmarks.suites.rostering   # noqa: F401
except Exception as e:
    print(
        f"warning: ScheduleSpace-dependent suites not loaded ({e})",
        file=sys.stderr,
    )
import benchmarks.suites.timetabling          # noqa: F401
import benchmarks.suites.project_scheduling   # noqa: F401

from benchmarks.suites import SUITE_REGISTRY
from benchmarks.realistic.runner import RunConfig, run_sweep
from benchmarks.realistic.report import build_report, render_report_md


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Realistic benchmark sweep")
    parser.add_argument("--suites", type=str, default=",".join(SUITE_REGISTRY),
                        help=f"comma-separated suite ids (default: all = {','.join(SUITE_REGISTRY)})")
    parser.add_argument("--tiers", type=str, default="S,M,L",
                        help="comma-separated tier names (default: S,M,L)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="seeds per (suite, size) (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (default: 42)")
    parser.add_argument("--oracle", type=str, default="auto",
                        choices=("on", "off", "auto"),
                        help="run oracle row? auto = on for S/M, off for L (default)")
    parser.add_argument("--quick", action="store_true",
                        help="trimmed run: M tier only, 3 seeds, oracle off")
    parser.add_argument("--json", type=str, default=None,
                        help="path for results JSON (default: results-<TS>.json)")
    parser.add_argument("--report", type=str, default=None,
                        help="path for REPORT.md (default: REPORT-<TS>.md)")
    args = parser.parse_args()

    if args.quick:
        args.tiers = "M"
        args.repeat = 3
        args.oracle = "off"

    suite_ids = [s.strip() for s in args.suites.split(",") if s.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [s for s in suite_ids if s not in SUITE_REGISTRY]
    if unknown:
        print(f"unknown suites: {unknown}. known: {sorted(SUITE_REGISTRY)}",
              file=sys.stderr)
        return 2

    # Per-tier oracle policy when --oracle=auto.
    def _oracle_for(tier: str) -> bool:
        if args.oracle == "on":
            return True
        if args.oracle == "off":
            return False
        return tier in ("S", "M")

    # Quick mode trims wall-clock so the smoke test fits in ~60s:
    # budgets scaled to 1/10, stress only 2x headline.
    budget_scale = 0.1 if args.quick else 1.0
    stress_mult = 2.0 if args.quick else 5.0

    rows = []
    for tier in tiers:
        cfg = RunConfig(
            seeds=args.repeat, oracle=_oracle_for(tier),
            stress_multiplier=stress_mult, base_seed=args.seed,
            budget_scale=budget_scale,
        )
        rows.extend(run_sweep(suite_ids, [tier], cfg))

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "pathos_sha": _git_sha(),
        "run_started": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cli": " ".join(sys.argv[1:]),
    }
    rep = build_report(rows, meta=meta)

    here = Path(__file__).parent
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    json_path = Path(args.json) if args.json else here / f"results-{ts}.json"
    report_path = Path(args.report) if args.report else here / f"REPORT-{ts}.md"

    with open(json_path, "w") as f:
        json.dump(asdict(rep), f, indent=2, default=str)
    with open(report_path, "w") as f:
        f.write(render_report_md(rep))

    print(f"\nwrote: {json_path}", file=sys.stderr)
    print(f"wrote: {report_path}", file=sys.stderr)
    print(f"no-cliff failures: {len(rep.no_cliff_failures)}", file=sys.stderr)
    return 1 if rep.no_cliff_failures else 0


if __name__ == "__main__":
    sys.exit(main())
