"""Hyperparameter grid search tuning for Differential Evolution.

TESTS ONLY: synth_mild_imbalance (All 108 parameter combinations)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimizers import DifferentialEvolutionOptimizer
from src.preprocessing.common import load_dataset

# Restricted strictly to the required dataset
_DATASET_PATHS = {
    "synth_mild_imbalance": "datasets/synthetic/preprocessed/synth_mild_imbalance.pkl",
}

# --- EXPERIMENT CONFIGURATION ---
RATIOS = (0.70, 0.15, 0.15)
MAX_EVALS = 50_000  # Change to 500_000 for full, definitive tuning runs
SEED = 42

# --- HYPERPARAMETER GRID ---
STRATEGIES = ["DE/rand/1/bin", "DE/best/1/bin", "DE/rand/1/exp", "DE/best/1/exp"]
POP_SIZES = [20, 50, 100]
F_WEIGHTS = [0.2, 0.5, 0.9]
CROSSOVER_PROBS = [0.3, 0.5, 0.9]


def run_full_tuning():
    dataset_name = "synth_mild_imbalance"
    
    total_combinations = len(STRATEGIES) * len(POP_SIZES) * len(F_WEIGHTS) * len(CROSSOVER_PROBS)
    
    print(f"=== STARTING FULL DE 1-VECTOR GRID SEARCH ===")
    print(f"Dataset: {dataset_name}")
    print(f"FFE Budget per run: {MAX_EVALS:,}")
    print(f"Total configurations to evaluate: {total_combinations}\n")

    data = load_dataset(_DATASET_PATHS[dataset_name])
    results_list = []
    
    counter = 0

    # Main Grid Search loop
    for strategy in STRATEGIES:
        for pop_size in POP_SIZES:
            for f_weight in F_WEIGHTS:
                for cr_prob in CROSSOVER_PROBS:
                    counter += 1
                    
                    print(
                        f"[{counter}/{total_combinations}] "
                        f"Strat: {strategy:<13} | "
                        f"NP: {pop_size:<3} | "
                        f"F: {f_weight:<3} | "
                        f"CR: {cr_prob:<3} ... ", 
                        end="", 
                        flush=True
                    )
                    
                    # Initializing your algorithm instance with the current parameters
                    optimizer = DifferentialEvolutionOptimizer(
                        data=data,
                        ratios=RATIOS,
                        max_evals=MAX_EVALS,
                        pop_size=pop_size,
                        f_weight=f_weight,
                        crossover_prob=cr_prob,
                        strategy=strategy,
                        seed=SEED
                    )
                    
                    t_start = time.perf_counter()
                    res = optimizer.optimize(verbose=False)
                    t_end = time.perf_counter()
                    
                    duration = t_end - t_start
                    print(f"OK (Cost: {res.cost:.4f})")
                    
                    # Saving evaluation metrics
                    results_list.append({
                        "strategy": strategy,
                        "pop_size": pop_size,
                        "f_weight": f_weight,
                        "crossover_prob": cr_prob,
                        "cost": res.cost,
                        "time": duration,
                        "iterations": res.n_iterations
                    })

    # Sort results from best (lowest cost) to worst
    results_list.sort(key=lambda x: x["cost"])

    # --- GENERATING PERFORMANCE REPORT ---
    os.makedirs("results", exist_ok=True)
    report_path = "results/de_tuning_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== DE HYPERPARAMETER TUNING REPORT ===\n")
        f.write(f"Dataset:      {dataset_name}\n")
        f.write(f"FFE Budget:   {MAX_EVALS:,}\n")
        f.write(f"Total configurations evaluated: {total_combinations}\n")
        f.write("-" * 95 + "\n")
        f.write(
            f" {'Rank':<4} | {'Strategy':<15} | {'Pop (NP)':<8} | {'Weight (F)':<10} | {'Cross (CR)':<10} | "
            f"{'Cost (MAPE)':<12} | {'Time':<8}\n"
        )
        f.write("-" * 95 + "\n")
        
        for idx, r in enumerate(results_list, 1):
            f.write(
                f" #{idx:<3} | {r['strategy']:<15} | {r['pop_size']:<8} | {r['f_weight']:<10.1f} | {r['crossover_prob']:<10.1f} | "
                f"{r['cost']:<12.4f} | {r['time']:>6.2f}s\n"
            )
        f.write("-" * 95 + "\n")

    print("\n" + "="*60)
    print(f"Grid Search Finished Successfully!")
    best = results_list[0]
    print(f"BEST CONFIGURATION FOUND:")
    print(f"  -> Strategy:        {best['strategy']}")
    print(f"  -> Pop Size (NP):   {best['pop_size']}")
    print(f"  -> Weight (F):      {best['f_weight']}")
    print(f"  -> Crossover (CR):  {best['crossover_prob']}")
    print(f"  -> Cost (MAPE):     {best['cost']:.4f}")
    print(f"Full rankings saved to: {report_path}")
    print("="*60)


if __name__ == "__main__":
    run_full_tuning()