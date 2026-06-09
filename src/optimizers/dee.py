import time
import numpy as np

from .base import N_SPLITS, Optimizer, SplitResult


class DifferentialEvolutionOptimizer(Optimizer):
    """Optimizer utilizing the Differential Evolution (DE) algorithm.

    Supports strategies:
    - DE/rand/1/bin
    - DE/best/1/bin
    - DE/rand/1/exp
    - DE/best/1/exp
    
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
        strategy: str = "DE/rand/1/bin",
        seed: int | None = None,
    ) -> None:
        super().__init__(data, ratios, max_evals=max_evals, seed=seed)

        valid_strategies = {"DE/rand/1/bin", "DE/best/1/bin", "DE/rand/1/exp", "DE/best/1/exp"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown DE strategy. Choose from: {valid_strategies}")

        if pop_size < 4:
            raise ValueError("Population size for DE with 1 difference pair must be at least 4.")
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
        """Runs the selected Differential Evolution strategy and returns the best split."""
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        n_evals = 0
        iteration = 0
        cost_history: list[tuple[int, float]] = []

        # Parse strategy name into components for faster conditional checks in the loop
        # e.g., "DE/rand/1/bin" -> base_type="rand", crossover_type="bin"
        _, base_type, _, crossover_type = self.strategy.split("/")

        # ---------------------------------------------------------------------
        # 1. Initialize Initial Population (size: pop_size)
        # ---------------------------------------------------------------------
        population = rng.choice(
            N_SPLITS, 
            size=(self.pop_size, self.data.n_groups), 
            p=self.ratios
        )
        
        # Evaluate initial costs
        costs = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            costs[i] = self.evaluate(population[i])
            n_evals += 1

        # Determine the initial best individual
        best_idx = np.argmin(costs)
        best_assignment = population[best_idx].copy()
        best_cost = costs[best_idx]

        if verbose:
            print(
                f"[{self.strategy}] {self.data.dataset_name}"
                f"  groups={self.data.n_groups}"
                f"  classes={self.data.n_classes}"
                f"  budget={self.max_evals:,} FFEs"
                f"  pop_size={self.pop_size}"
                f"  initial_best_cost={best_cost:.4f}"
            )

        # ---------------------------------------------------------------------
        # 2. Main DE Loop
        # ---------------------------------------------------------------------
        while n_evals < self.max_evals:
            iteration += 1

            # In each generation, evaluate every individual as a target vector
            for i in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                target_mutant = population[i]

                # --- Mutation (Base Vector Selection Based on Strategy) ---
                available_idxs = [idx for idx in range(self.pop_size) if idx != i]
                
                if base_type == "best":
                    # Base vector is the current best individual in the population
                    current_best_idx = np.argmin(costs)
                    base_vector = population[current_best_idx]
                    
                    # Randomly pick 2 other distinct individuals for the difference pair
                    r1, r2 = rng.choice(available_idxs, size=2, replace=False)
                else:
                    # base_type == "rand" (classic random base individual)
                    # Randomly pick 3 distinct individuals: r1 (base), r2 and r3 (difference pair)
                    r1, r2, r3 = rng.choice(available_idxs, size=3, replace=False)
                    base_vector = population[r1]

                # Assign appropriate vectors to calculate the difference vector
                if base_type == "best":
                    diff = population[r1].astype(np.float64) - population[r2].astype(np.float64)
                else:
                    diff = population[r2].astype(np.float64) - population[r3].astype(np.float64)

                mutant_continuous = base_vector.astype(np.float64) + self.f_weight * diff
                
                # Project continuous values back into valid discrete split indices {0, 1, 2}
                mutant = np.mod(np.round(mutant_continuous).astype(np.int64), N_SPLITS)

                # --- Crossover ---
                trial = target_mutant.copy()

                if crossover_type == "bin":
                    # --- Binomial Crossover ---
                    forced_change_idx = rng.integers(0, self.data.n_groups)
                    crossover_mask = rng.random(self.data.n_groups) < self.crossover_prob
                    crossover_mask[forced_change_idx] = True
                    trial[crossover_mask] = mutant[crossover_mask]
                    
                else:
                    # --- Exponential Crossover ---
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

                # If the trial vector is better or equal, replace the target vector in the population
                if trial_cost <= costs[i]:
                    population[i] = trial
                    costs[i] = trial_cost

                    # Update global best record
                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_assignment = trial.copy()

            # --- Status Logging ---
            if n_evals % log_interval < self.pop_size:
                elapsed = time.perf_counter() - t_start
                cost_history.append((n_evals, best_cost))
                if verbose:
                    print(
                        f"  evals {n_evals:>7,}"
                        f"  best_cost={best_cost:.4f}"
                        f"  generations={iteration}"
                        f"  elapsed={elapsed:.1f}s"
                    )

        # ---------------------------------------------------------------------
        # 3. Final Results
        # ---------------------------------------------------------------------
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