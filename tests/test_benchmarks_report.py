from dataclasses import asdict
from benchmarks.realistic.records import (
    RunRow, Report, BENCH_SCHEMA,
)
from benchmarks.realistic.report import build_report, render_report_md


def _row(suite, size, seed, mode, cost, feasible=True, alg="A"):
    return RunRow(
        suite=suite, size=size, seed=seed, mode=mode, algorithm=alg,
        cost=cost, feasible=feasible, elapsed=0.1, nodes_expanded=10,
    )


def test_build_report_collapses_seeds_per_suite_size():
    rows = [
        _row("C1", 8, 1, "auto_headline", 1.0),
        _row("C1", 8, 2, "auto_headline", 1.0),
        _row("C1", 8, 1, "auto_stress", 1.0),
        _row("C1", 8, 1, "oracle_A", 1.0),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    suite_rows = [sr for sr in rep.suite_rows if sr.suite == "C1" and sr.size == 8]
    assert len(suite_rows) == 1
    sr = suite_rows[0]
    assert sr.auto_headline_cost == 1.0
    assert sr.auto_stress_cost == 1.0
    assert sr.oracle_cost == 1.0
    assert sr.gap_to_oracle_pct == 0.0


def test_build_report_records_no_cliff_when_auto_not_feasible():
    rows = [
        _row("R3", 100, 1, "auto_headline", None, feasible=False),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    assert len(rep.no_cliff_failures) == 1
    assert rep.no_cliff_failures[0].failure_kind in (
        "not_found", "infeasible", "crash",
    )


def test_build_report_excludes_missing_capability_from_no_cliff():
    rows = [
        RunRow(
            suite="R4", size=100, seed=1, mode="auto_headline",
            algorithm="TabuSearch", cost=None, feasible=False,
            elapsed=10.0, nodes_expanded=0,
            missing_capability="time_windows",
        ),
    ]
    rep = build_report(rows, meta={"cpu": "test"})
    assert rep.no_cliff_failures == []


def test_render_report_md_has_required_sections():
    rows = [_row("C1", 8, 1, "auto_headline", 1.0)]
    rep = build_report(rows, meta={"cpu": "test"})
    md = render_report_md(rep)
    assert "# Realistic benchmark report" in md
    assert "## Regression table" in md
    assert "## No-cliff failures" in md
    assert "## Selection-suboptimal" in md
    assert "## Tunability gains" in md
    assert "## Missing-capability gaps" in md


def test_report_json_round_trip():
    import json
    rows = [_row("C1", 8, 1, "auto_headline", 1.0)]
    rep = build_report(rows, meta={"cpu": "t"})
    payload = json.loads(json.dumps(asdict(rep)))
    assert payload["bench_schema"] == BENCH_SCHEMA
    assert any(sr["suite"] == "C1" for sr in payload["suite_rows"])
