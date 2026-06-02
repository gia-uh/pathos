"""Tests for SearchResult.epsilon and the derived .optimal property.

epsilon semantics:
  - 1.0  → proven optimal (admissible algorithm finished)
  - >1.0 → ε-bounded suboptimal (cost ≤ ε × optimal); e.g. WeightedAStar(weight)
  - inf  → unbounded suboptimal; e.g. GreedyBestFirst
  - None → not applicable (e.g. metaheuristics with no quality bound,
           or not_found results)
"""
from __future__ import annotations

import math

from pathos.core.result import SearchResult


def test_epsilon_default_is_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Foo", nodes_expanded=0, elapsed=0.0, found=True,
    )
    assert r.epsilon is None


def test_optimal_is_true_when_epsilon_one():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="AStar", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=1.0,
    )
    assert r.optimal is True


def test_optimal_is_false_when_epsilon_above_one():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="WeightedAStar", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=2.0,
    )
    assert r.optimal is False


def test_optimal_is_false_when_epsilon_inf():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="GreedyBestFirst", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=math.inf,
    )
    assert r.optimal is False


def test_optimal_is_false_when_epsilon_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="HillClimbing", nodes_expanded=0, elapsed=0.0, found=True,
        epsilon=None,
    )
    assert r.optimal is False


def test_not_found_has_epsilon_none():
    r = SearchResult.not_found("AStar", 0, 0.0)
    assert r.epsilon is None
    assert r.optimal is False
