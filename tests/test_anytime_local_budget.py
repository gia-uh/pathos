"""Mechanism tests for AnytimeLocal's deadline-aware cascade.

Two guarantees:

1. **Per-phase budget split.** When `space._deadline_at` is set by the
   Solver, AnytimeLocal installs `space._phase_deadline_at` before each
   phase, dividing remaining time across remaining phases. This makes
   `space._cancel_requested()` return True once a phase exhausts its
   slice, so cooperating phases (HC/SA/Tabu) terminate cleanly without
   the watchdog needing to fire.

2. **Incumbent survives watchdog TimeoutError.** When the watchdog
   raises mid-phase (because the phase didn't honour the cancel token
   in time), the cascade catches it and returns the best incumbent
   from earlier phases instead of vaporising to not_found.

These two together fix R1/R2 in the realistic bench — HC was
monopolising the global budget on big ScheduleSpace problems, the
watchdog was vaporising any incumbent, and SA (which works on these
problems standalone) never ran.
"""
from __future__ import annotations

import time

import pathos.algorithms  # noqa: F401
from pathos import Space
from pathos.algorithms.local import AnytimeLocal, HillClimbing


def _trivial_space() -> Space:
    """Minimal pure-optimization space: single state, no neighbors."""
    space = Space().initial((0,))

    @space.successors
    def neighbors(state):
        return []

    @space.evaluate
    def cost(state):
        return float(state[0] ** 2)

    return space


def test_anytime_local_installs_phase_deadline_during_phase():
    """During each phase, `space._phase_deadline_at` is set to a value
    that splits the remaining budget across remaining phases. Verified
    by spying on the deadline from inside a fake phase."""
    space = _trivial_space()
    # Pretend the Solver set this — the Anytime cascade reads it.
    space._deadline_at = time.perf_counter() + 0.9  # ~0.3s per phase × 3 phases

    seen_deadlines: list[float | None] = []

    class _SpyPhase:
        def __init__(self, sp, **kwargs):
            self.space = sp

        def solve(self):
            from pathos.core.result import SearchResult
            seen_deadlines.append(self.space._phase_deadline_at)
            return SearchResult(
                solution=(0,), path=None, cost=0.0, algorithm="Spy",
                nodes_expanded=0, elapsed=0.0, found=True,
            )

    # Substitute the cascade with three identical spy phases.
    original = AnytimeLocal.SCHEDULE
    AnytimeLocal.SCHEDULE = [(_SpyPhase, {}), (_SpyPhase, {}), (_SpyPhase, {})]
    try:
        AnytimeLocal(space).solve()
    finally:
        AnytimeLocal.SCHEDULE = original
        space._deadline_at = None
        space._phase_deadline_at = None

    assert len(seen_deadlines) == 3, "expected all three phases to run"
    # Each phase saw a non-None deadline.
    assert all(d is not None for d in seen_deadlines), seen_deadlines


def test_anytime_local_catches_watchdog_timeout_and_returns_incumbent():
    """If a later phase raises TimeoutError (the watchdog), the cascade
    catches it and returns the incumbent from earlier successful phases."""
    space = _trivial_space()
    space._deadline_at = time.perf_counter() + 10.0  # generous; we control timing

    from pathos.core.result import SearchResult

    class _GoodPhase:
        def __init__(self, sp, **kwargs):
            self.space = sp

        def solve(self):
            return SearchResult(
                solution=(0,), path=None, cost=42.0, algorithm="Good",
                nodes_expanded=1, elapsed=0.01, found=True,
            )

    class _ExplodePhase:
        def __init__(self, sp, **kwargs):
            self.space = sp

        def solve(self):
            raise TimeoutError("simulated watchdog mid-phase")

    original = AnytimeLocal.SCHEDULE
    AnytimeLocal.SCHEDULE = [(_GoodPhase, {}), (_ExplodePhase, {})]
    try:
        result = AnytimeLocal(space).solve()
    finally:
        AnytimeLocal.SCHEDULE = original
        space._deadline_at = None
        space._phase_deadline_at = None

    assert result.found is True
    assert result.cost == 42.0
    assert result.algorithm == "AnytimeLocal"


def test_anytime_local_no_incumbent_returns_not_found_after_watchdog():
    """If watchdog fires during the FIRST phase with no incumbent yet,
    the cascade still returns not_found cleanly (no exception leak)."""
    space = _trivial_space()
    space._deadline_at = time.perf_counter() + 10.0

    class _ExplodePhase:
        def __init__(self, sp, **kwargs):
            self.space = sp

        def solve(self):
            raise TimeoutError("simulated watchdog on first phase")

    original = AnytimeLocal.SCHEDULE
    AnytimeLocal.SCHEDULE = [(_ExplodePhase, {})]
    try:
        result = AnytimeLocal(space).solve()
    finally:
        AnytimeLocal.SCHEDULE = original
        space._deadline_at = None
        space._phase_deadline_at = None

    assert result.found is False
    assert result.algorithm == "AnytimeLocal"


def test_cancel_requested_honours_phase_deadline():
    """When `_phase_deadline_at` is in the past, `_cancel_requested()`
    returns True even though the cancel token itself isn't set. This is
    how the cascade hands phases a deadline they can cooperate with."""
    space = _trivial_space()
    assert space._cancel_requested() is False

    space._phase_deadline_at = time.perf_counter() - 0.1  # already past
    try:
        assert space._cancel_requested() is True
    finally:
        space._phase_deadline_at = None

    space._phase_deadline_at = time.perf_counter() + 10.0  # far future
    try:
        assert space._cancel_requested() is False
    finally:
        space._phase_deadline_at = None
