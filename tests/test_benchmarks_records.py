import json
from dataclasses import asdict
from benchmarks.suites._spec import SuiteSpec
from benchmarks.realistic.records import (
    RunRow, SuiteRow, Report, BENCH_SCHEMA,
)


def test_suitespec_holds_generate_and_express_callables():
    spec = SuiteSpec(
        id="X1",
        family="example",
        constraint_classes=frozenset({"v"}),
        expressibility="full",
        sizes={"S": (10,), "M": (50,), "L": (200,)},
        generate=lambda size, seed: {"size": size, "seed": seed},
        express=lambda instance: instance,
        budgets={"S": 1.0, "M": 5.0, "L": 30.0},
    )
    assert spec.id == "X1"
    inst = spec.generate(10, 42)
    assert inst == {"size": 10, "seed": 42}
    assert spec.express(inst) is inst


def test_runrow_serializes_to_json():
    row = RunRow(
        suite="X1", size=10, seed=42, mode="auto_headline",
        algorithm="AStar", cost=12.5, feasible=True, elapsed=0.42,
        nodes_expanded=137, missing_capability=None,
    )
    j = json.dumps(asdict(row))
    assert json.loads(j)["algorithm"] == "AStar"


def test_runrow_missing_capability_is_optional_string():
    row = RunRow(
        suite="R4", size=100, seed=1, mode="auto_headline",
        algorithm="TabuSearch", cost=None, feasible=False,
        elapsed=10.0, nodes_expanded=0,
        missing_capability="time_windows",
    )
    assert row.missing_capability == "time_windows"


def test_suiterow_aggregates_per_size():
    sr = SuiteRow(
        suite="C1", size=8,
        auto_headline_cost=1.0, auto_headline_feasible=True,
        auto_headline_algorithm="Backtracking",
        auto_stress_cost=1.0, auto_stress_feasible=True,
        oracle_cost=1.0, oracle_algorithm="Backtracking",
        gap_to_oracle_pct=0.0, gap_to_lb_pct=None,
        tunability_gain_pct=0.0, tunability_algorithm=None,
        missing_capability=None,
    )
    assert sr.gap_to_oracle_pct == 0.0


def test_report_round_trips_to_json():
    rep = Report(
        bench_schema=BENCH_SCHEMA,
        meta={"cpu": "test", "python": "3.13", "pathos_sha": "abc"},
        suite_rows=[],
        raw_rows=[],
        no_cliff_failures=[],
    )
    j = json.dumps(asdict(rep))
    loaded = json.loads(j)
    assert loaded["bench_schema"] == BENCH_SCHEMA


def test_bench_schema_is_pinned_integer():
    assert isinstance(BENCH_SCHEMA, int)
    assert BENCH_SCHEMA >= 1
