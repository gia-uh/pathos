from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


def batch_map(fn: Callable[[Any], Any], items: list[Any], n_workers: int) -> list[Any]:
    if n_workers <= 1 or len(items) <= 1:
        return list(map(fn, items))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, items))
