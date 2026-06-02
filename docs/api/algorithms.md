# Algorithms

PATHOS algorithms are organized by family. The auto-solver picks the most powerful compatible
algorithm via [`Algorithm.score_for`][pathos.algorithms.base.Algorithm.score_for] — see the
[Modes & Anytime delivery](../guides/modes-and-anytime.md) guide for how `mode="auto"`
makes [`AnytimeAStar`][pathos.algorithms.informed.AnytimeAStar] the default on A\*-family
problems.

## Base

::: pathos.algorithms.base.Algorithm

## Meta-algorithms

Meta-algorithms compose base algorithms. The first one shipped is the
anytime A\* cascade — see the [Modes & Anytime delivery](../guides/modes-and-anytime.md) guide.

::: pathos.algorithms.informed.AnytimeAStar

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
