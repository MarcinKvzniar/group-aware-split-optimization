import time
import numpy as np

from .base import N_SPLITS, Optimizer, SplitResult


class DifferentialEvolution2VecOptimizer(Optimizer):
    """Optimizer utilizing the Differential Evolution (DE) algorithm.

    Supports strategies with TWO difference vectors (2-vec):
    - DE/rand/2/bin
    - DE/best/2/bin
    - DE/rand/2/exp
    - DE/best/2/exp
    
    Discrete version tailored to split indices {0, 1, 2} using rounding and modulo operations.
    """

    def __init__(
        self,
        data,
        ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
        max_evals: int = 500_000,
        pop_size: int = 50,
        f_weight: float = 0.5,
        crossover_prob: float = 0.7,
        strategy: str = "DE/rand/2/bin",
        seed: int | None = None,
    ) -> None:
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

        valid_strategies = {"DE/rand/2/bin", "DE/best/2/bin", "DE/rand/2/exp", "DE/best/2/exp"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown DE 2-vec strategy. Choose from: {valid_strategies}")

        # Requires at least 5 individuals (base + 4 for difference pairs for 'rand', or leader + 4 for 'best')
        if pop_size < 5:
            raise ValueError("Population size for DE with 2 difference pairs must be at least 5.")
        if not 0.0 < f_weight <= 2.0:
            raise ValueError("Mutation factor f_weight must be in the range (0, 2].")
        if not 0.0 <= crossover_prob <= 1.0:
            raise ValueError("Crossover probability crossover_prob must be in the range [0, 1].")

        self.pop_size = pop_size
        self.f_weight = f_weight
        self.crossover_prob = crossover_prob
        self.strategy = strategy

    def optimize(
        self,
        verbose: bool = True,
        log_interval: int = 10_000,
    ) -> SplitResult:
        """Runs the selected 2-vec Differential Evolution strategy and returns the best split."""
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        n_evals = 0
        iteration = 0
        cost_history: list[tuple[int, float]] = []

        # Parse strategy name: "DE/rand/2/bin" -> base_type="rand", crossover_type="bin"
        _, base_type, _, crossover_type = self.strategy.split("/")

        # ---------------------------------------------------------------------
        # 1. Initialize Initial Population
        # ---------------------------------------------------------------------
        population = rng.choice(
            N_SPLITS, 
            size=(self.pop_size, self.data.n_groups), 
            p=self.ratios
        )
        
        costs = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            costs[i] = self.evaluate(population[i])
            n_evals += 1

        best_idx = np.argmin(costs)
        best_assignment = population[best_idx].copy()
        best_cost = costs[best_idx]

        if verbose:
            print(
                f"[{self.strategy}] {self.data.dataset_name}"
                f"  groups={self.data.n_groups}"
                f"  budget={self.max_evals:,} FFEs"
                f"  pop_size={self.pop_size}"
                f"  initial_best_cost={best_cost:.4f}"
            )

        # ---------------------------------------------------------------------
        # 2. Main DE Loop
        # ---------------------------------------------------------------------
        while n_evals < self.max_evals:
            iteration += 1

            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                target_mutant = population[i]

                # --- Mutation (2 Difference Vectors) ---
                available_idxs = [idx for idx in range(self.pop_size) if idx != i]
                
                if base_type == "best":
                    # Base vector is the current best individual
                    current_best_idx = np.argmin(costs)
                    base_vector = population[current_best_idx]
                    
                    # Pick 4 unique individuals for the two difference pairs
                    r1, r2, r3, r4 = rng.choice(available_idxs, size=4, replace=False)
                else:
                    # Base vector is r1, we need 4 more individuals: r2, r3, r4, r5
                    r1, r2, r3, r4, r5 = rng.choice(available_idxs, size=5, replace=False)
                    base_vector = population[r1]

                # Retrieve appropriate vectors to compute differences
                p1 = population[r1] if base_type == "best" else population[r2]
                p2 = population[r2] if base_type == "best" else population[r3]
                p3 = population[r3] if base_type == "best" else population[r4]
                p4 = population[r4] if base_type == "best" else population[r5]

                # Compute two difference vectors
                diff1 = p1.astype(np.float64) - p2.astype(np.float64)
                diff2 = p3.astype(np.float64) - p4.astype(np.float64)

                # Continuous mutation step
                mutant_continuous = base_vector.astype(np.float64) + self.f_weight * diff1 + self.f_weight * diff2
                
                # Project back to the discrete space {0, 1, 2}
                mutant = np.mod(np.round(mutant_continuous).astype(np.int64), N_SPLITS)

                # --- Crossover ---
                trial = target_mutant.copy()

                if crossover_type == "bin":
                    # Binomial Crossover
                    forced_change_idx = rng.integers(0, self.data.n_groups)
                    crossover_mask = rng.random(self.data.n_groups) < self.crossover_prob
                    crossover_mask[forced_change_idx] = True
                    trial[crossover_mask] = mutant[crossover_mask]
                else:
                    # Exponential Crossover
                    start_idx = rng.integers(0, self.data.n_groups)
                    curr_idx = start_idx
                    while True:
                        trial[curr_idx] = mutant[curr_idx]
                        curr_idx = (curr_idx + 1) % self.data.n_groups
                        if rng.random() >= self.crossover_prob or curr_idx == start_idx:
                            break

                # --- Evaluation and Selection ---
                trial_cost = self.evaluate(trial)
                n_evals += 1

                if trial_cost <= costs[i]:
                    population[i] = trial
                    costs[i] = trial_cost

                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_assignment = trial.copy()

            if n_evals % log_interval < self.pop_size:
                elapsed = time.perf_counter() - t_start
                cost_history.append((n_evals, best_cost))
                if verbose:
                    print(f"  evals {n_evals:>7,}  best_cost={best_cost:.4f}  elapsed={elapsed:.1f}s")

        elapsed = time.perf_counter() - t_start
        best_actual = self._count_matrix(best_assignment)

        return SplitResult(
            assignment=best_assignment,
            cost=best_cost,
            n_evals=n_evals,
            n_iterations=iteration,
            converged=False,
            elapsed_time=elapsed,
            cost_history=cost_history,
            target_counts=self._target.copy(),
            actual_counts=best_actual,
        )