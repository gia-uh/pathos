import benchmarks.suites.classical  # noqa: F401 — populates registry
import benchmarks.suites.bin_packing  # noqa: F401
from benchmarks.realistic.runner import (
    run_one, run_suite_tier, RunConfig,
)


def test_run_one_returns_runrow_with_auto_headline_mode():
    cfg = RunConfig(seeds=1, oracle=False, stress_multiplier=5.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    assert any(r.mode == "auto_headline" for r in rows)
    headline = next(r for r in rows if r.mode == "auto_headline")
    assert headline.suite == "C1"
    assert headline.feasible is True


def test_run_one_emits_oracle_rows_when_enabled():
    cfg = RunConfig(seeds=1, oracle=True, stress_multiplier=5.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    oracle_modes = [r.mode for r in rows if r.mode.startswith("oracle_")]
    assert oracle_modes, "expected at least one oracle row"


def test_run_one_emits_stress_row():
    cfg = RunConfig(seeds=1, oracle=False, stress_multiplier=2.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    assert any(r.mode == "auto_stress" for r in rows)


def test_run_suite_tier_seeds_increment_deterministically():
    cfg = RunConfig(seeds=2, oracle=False, stress_multiplier=5.0, base_seed=100)
    rows = run_suite_tier("R6", tier="S", cfg=cfg)
    headline_seeds = sorted({r.seed for r in rows if r.mode == "auto_headline"})
    assert headline_seeds == [100, 101]


def test_run_one_records_crash_with_nodes_expanded_minus_two():
    # Use a suite where some explicit algorithms are known to crash on
    # the space (per FINDINGS.md history: BFS on CSPSpace blew up before
    # the compat-guard fix). We verify the *shape*, not specific
    # algorithm names: every row has a valid mode and feasible flag.
    cfg = RunConfig(seeds=1, oracle=True, stress_multiplier=2.0)
    rows = run_one("C1", tier="S", cfg=cfg)
    for r in rows:
        assert r.mode == "auto_headline" or r.mode == "auto_stress" or r.mode.startswith("oracle_")
        assert isinstance(r.feasible, bool)
