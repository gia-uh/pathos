# Algorithms

PATHOS algorithms are organized by family. The auto-solver picks the most powerful compatible
algorithm via [`Algorithm.score_for`][pathos.algorithms.base.Algorithm.score_for] — see the
[Modes & Anytime delivery](../guides/modes-and-anytime.md) guide for how `mode="auto"`
makes [`AnytimeAStar`][pathos.algorithms.informed.AnytimeAStar] the default on A\*-family
problems.

## Base

::: pathos.algorithms.base.Algorithm

## Meta-algorithms

Meta-algorithms compose base algorithms. Four ship in v0.2.0 — one per
family — and each wins selection under `mode="auto"` on its capability
shape. See the [Modes & Anytime delivery](../guides/modes-and-anytime.md)
guide for the cascade tables.

::: pathos.algorithms.informed.AnytimeAStar
::: pathos.algorithms.local.AnytimeLocal
::: pathos.algorithms.csp.AnytimeCSP
::: pathos.algorithms.adversarial.AnytimeAdversarial

## Uninformed

::: pathos.algorithms.uninformed.BFS
::: pathos.algorithms.uninformed.DFS
::: pathos.algorithms.uninformed.IDDFS
::: pathos.algorithms.uninformed.UCS

## Informed

::: pathos.algorithms.informed.AStar
::: pathos.algorithms.informed.GreedyBestFirst
::: pathos.algorithms.informed.WeightedAStar
::: pathos.algorithms.informed.IDAstar
::: pathos.algorithms.informed.BidirectionalAStar

## Local Search

::: pathos.algorithms.local.HillClimbing
::: pathos.algorithms.local.TabuSearch
::: pathos.algorithms.local.LocalBeamSearch

## Evolutionary / Metaheuristic

::: pathos.algorithms.evolutionary.SimulatedAnnealing
::: pathos.algorithms.evolutionary.GeneticAlgorithm
::: pathos.algorithms.evolutionary.DifferentialEvolution
::: pathos.algorithms.evolutionary.ParticleSwarm

## Adversarial

::: pathos.algorithms.adversarial.Minimax
::: pathos.algorithms.adversarial.AlphaBeta
::: pathos.algorithms.adversarial.Negamax
::: pathos.algorithms.adversarial.MCTS

## CSP

::: pathos.algorithms.csp.Backtracking
::: pathos.algorithms.csp.ForwardChecking
::: pathos.algorithms.csp.MinConflicts
