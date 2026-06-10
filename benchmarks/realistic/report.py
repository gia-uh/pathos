from __future__ import annotations
from collections import defaultdict
from benchmarks.realistic.records import (
    BENCH_SCHEMA, NoCliffFailure, Report, RunRow, SuiteRow,
)
from benchmarks.realistic.metrics import (
    gap_pct, is_no_cliff_failure, median_cost, oracle_row, tunability_gain,
)
from benchmarks.suites import SUITE_REGISTRY


def build_report(rows: list[RunRow], meta: dict[str, str]) -> Report:
    bucket: dict[tuple[str, int], list[RunRow]] = defaultdict(list)
    for r in rows:
        bucket[(r.suite, r.size)].append(r)

    suite_rows: list[SuiteRow] = []
    for (suite, size), group in sorted(bucket.items()):
        head = [r for r in group if r.mode == "auto_headline"]
        stress = [r for r in group if r.mode == "auto_stress"]
        head_cost = median_cost(head)
        stress_cost = median_cost(stress)
        head_feas = any(r.feasible for r in head)
        stress_feas = any(r.feasible for r in stress)
        head_alg = (head[0].algorithm if head else "—")
        oracle_alg, oracle_cost = oracle_row(group)
        gap = gap_pct(head_cost, oracle_cost)
        tg, tg_alg = tunability_gain(group)
        spec = SUITE_REGISTRY.get(suite)
        mc = spec.missing_capability if spec is not None else None
        suite_rows.append(SuiteRow(
            suite=suite, size=size,
            auto_headline_cost=head_cost,
            auto_headline_feasible=head_feas,
            auto_headline_algorithm=head_alg,
            auto_stress_cost=stress_cost,
            auto_stress_feasible=stress_feas,
            oracle_cost=oracle_cost,
            oracle_algorithm=oracle_alg,
            gap_to_oracle_pct=gap,
            gap_to_lb_pct=None,  # TODO Task 13 hook LB into runner output
            tunability_gain_pct=tg,
            tunability_algorithm=tg_alg,
            missing_capability=mc,
        ))

    no_cliff: list[NoCliffFailure] = []
    for r in rows:
        spec = SUITE_REGISTRY.get(r.suite)
        tol = spec.tolerance if spec is not None else 0.0
        if not is_no_cliff_failure(r, tol):
            continue
        kind = (
            "crash" if r.nodes_expanded == -2
            else "not_found" if r.cost is None and not r.feasible
            else "infeasible"
        )
        no_cliff.append(NoCliffFailure(
            suite=r.suite, size=r.size, seed=r.seed,
            failure_kind=kind, detail=f"algo={r.algorithm} elapsed={r.elapsed:.2f}s",
            reproduction_args=f"--suites {r.suite} --tiers <tier> --seed {r.seed}",
        ))

    return Report(
        bench_schema=BENCH_SCHEMA,
        meta=meta,
        suite_rows=suite_rows,
        raw_rows=rows,
        no_cliff_failures=no_cliff,
    )


def render_report_md(rep: Report) -> str:
    lines: list[str] = [
        "# Realistic benchmark report",
        "",
        f"- `bench_schema`: {rep.bench_schema}",
    ]
    for k, v in rep.meta.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Regression table",
        "",
        "| suite | size | auto cost | auto feasible | auto alg | oracle cost | gap % | tunability % | tunability alg | missing-cap |",
        "|---|---:|---:|:---:|---|---:|---:|---:|---|---|",
    ]
    for sr in rep.suite_rows:
        lines.append(
            f"| {sr.suite} | {sr.size} | "
            f"{_fmt(sr.auto_headline_cost)} | "
            f"{'Y' if sr.auto_headline_feasible else 'N'} | "
            f"{sr.auto_headline_algorithm} | "
            f"{_fmt(sr.oracle_cost)} | "
            f"{_fmt(sr.gap_to_oracle_pct)} | "
            f"{_fmt(sr.tunability_gain_pct)} | "
            f"{sr.tunability_algorithm or '—'} | "
            f"{sr.missing_capability or '—'} |"
        )

    lines += ["", "## No-cliff failures", ""]
    if not rep.no_cliff_failures:
        lines.append("(none — gate clean)")
    else:
        lines.append("| suite | size | seed | kind | detail | repro |")
        lines.append("|---|---:|---:|---|---|---|")
        for f in rep.no_cliff_failures:
            lines.append(
                f"| {f.suite} | {f.size} | {f.seed} | {f.failure_kind} | "
                f"{f.detail} | `{f.reproduction_args}` |"
            )

    lines += ["", "## Selection-suboptimal", ""]
    sub = [sr for sr in rep.suite_rows
           if sr.gap_to_oracle_pct is not None and sr.gap_to_oracle_pct > 5.0
           and sr.missing_capability is None]
    if not sub:
        lines.append("(none above 5% gap)")
    else:
        lines.append("| suite | size | auto alg | oracle alg | gap % |")
        lines.append("|---|---:|---|---|---:|")
        for sr in sub:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.auto_headline_algorithm} | "
                f"{sr.oracle_algorithm or '—'} | "
                f"{_fmt(sr.gap_to_oracle_pct)} |"
            )

    lines += ["", "## Tunability gains", ""]
    tg = [sr for sr in rep.suite_rows
          if sr.tunability_gain_pct is not None and sr.tunability_gain_pct >= 5.0]
    if not tg:
        lines.append("(no candidate beat default by ≥ 5%)")
    else:
        lines.append("| suite | size | candidates=[X] | gain % |")
        lines.append("|---|---:|---|---:|")
        for sr in tg:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.tunability_algorithm} | "
                f"{_fmt(sr.tunability_gain_pct)} |"
            )

    lines += ["", "## Missing-capability gaps", ""]
    mc = [sr for sr in rep.suite_rows if sr.missing_capability is not None]
    if not mc:
        lines.append("(no missing-capability suites in this run)")
    else:
        lines.append("| suite | size | constraint class | auto cost (penalty-folded) |")
        lines.append("|---|---:|---|---:|")
        for sr in mc:
            lines.append(
                f"| {sr.suite} | {sr.size} | "
                f"{sr.missing_capability} | "
                f"{_fmt(sr.auto_headline_cost)} |"
            )

    return "\n".join(lines) + "\n"


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)
