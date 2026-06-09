"""Split optimizers package.

All stochastic algorithms share the same max_evals FFE budget and cost function
for direct comparison.
"""

from .base import (
    N_SPLITS,
    SPLIT_NAMES,
    Optimizer,
    SplitResult,
    evaluate_assignment,
)
from .random_search import RandomSearch
from .sa import SimulatedAnnealing
from .dee import DifferentialEvolutionOptimizer
from .de import DifferentialEvolution2VecOptimizer

__all__ = [
    "Optimizer",
    "SplitResult",
    "SimulatedAnnealing",
    "RandomSearch",
    "DifferentialEvolutionOptimizer",
    "DifferentialEvolution2VecOptimizer",
    "SPLIT_NAMES",
    "N_SPLITS",
    "evaluate_assignment",
]
