"""Cooperative cancellation token shared by Solver, meta-algorithms,
and any cooperating algorithm's solve() loop.

Algorithms check `space._cancel_requested()` at well-defined points
inside their main loop and return their best-so-far cleanly when the
token is set. This is the single primitive that powers both the global
solver timeout (Solver sets the token on SIGALRM) and meta-algorithms
that compose other algorithms (e.g. AnytimeAStar).
"""
from __future__ import annotations


class CancelToken:
    """Cooperative cancellation signal.

    Set by the Solver when the wall-clock deadline approaches; read by
    Algorithm.solve() loops at well-defined check points. Algorithms
    that observe a set token are expected to return their best-so-far
    cleanly (not raise).
    """

    __slots__ = ("_set",)

    def __init__(self) -> None:
        self._set: bool = False

    def request_cancel(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set

    def __bool__(self) -> bool:
        return self._set
