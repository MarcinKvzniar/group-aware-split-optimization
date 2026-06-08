import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimizers import SimulatedAnnealing
from src.preprocessing.common import load_dataset

DATASET_PATH = "datasets/synthetic/preprocessed/synth_mild_imbalance.pkl"
RATIOS = (0.70, 0.15, 0.15)
MAX_EVALS = 500_000
SEED = 42

GRID = {
    "initial_temp": [1.0, 10.0, 100.0],
    "cooling_rate": [0.99, 0.999, 0.9999],
    "min_temp": [1e-2, 1e-3, 1e-4],
}


def _unpack_history(result) -> tuple[np.ndarray, np.ndarray]:
    if not result.cost_history:
        return np.array([result.n_evals], dtype=float), np.array([result.cost], dtype=float)

    filtered_evals, filtered_costs = [], []
    best_so_far = float('inf')

    for e, c in result.cost_history:
        if c < best_so_far:
            best_so_far = c
            filtered_evals.append(e)
            filtered_costs.append(c)

    last_e = max(result.n_evals, result.cost_history[-1][0])
    if filtered_evals[-1] < last_e:
        filtered_evals.append(last_e)
        filtered_costs.append(best_so_far)

    return np.asarray(filtered_evals, dtype=float), np.asarray(filtered_costs, dtype=float)


if __name__ == "__main__":
    outdir = "results/tuning"
    os.makedirs(outdir, exist_ok=True)
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log(f"Loading {DATASET_PATH}...")
    data = load_dataset(DATASET_PATH)

    keys = list(GRID.keys())
    combinations = list(itertools.product(*(GRID[k] for k in keys)))

    log(f"Starting Grid Search: {len(combinations)} combinations ({MAX_EVALS:,} max_evals each)")
    log("-" * 75)

    results = {}

    t_start_all = time.time()
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        opt = SimulatedAnnealing(
            data=data, ratios=RATIOS, max_evals=MAX_EVALS, seed=SEED, **params
        )
        res = opt.optimize(verbose=False)
        results[values] = res
        log(f"[{i + 1:2d}/{len(combinations)}] T0={params['initial_temp']:>5.1f} | CR={params['cooling_rate']:.4f} | Tmin={params['min_temp']:.4f}  -> Cost: {res.cost:.4f}")

    log("-" * 75)
    log(f"Grid search completed in {time.time() - t_start_all:.1f}s")

    best_params = min(results.keys(), key=lambda k: results[k].cost)
    log(f"\nBEST PARAMS: T0={best_params[0]}, CR={best_params[1]}, Tmin={best_params[2]} -> Cost: {results[best_params].cost:.4f}")

    txt_path = os.path.join(outdir, "sa_grid_search_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved text report to {txt_path}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("Simulated Annealing Hyperparameter Grid Search", fontsize=16, fontweight="bold")

    styles = {
        GRID["min_temp"][0]: {"color": "#1f77b4", "linewidth": 4.5, "linestyle": "-", "alpha": 0.6},
        GRID["min_temp"][1]: {"color": "#ff7f0e", "linewidth": 2.5, "linestyle": "--", "alpha": 0.9},
        GRID["min_temp"][2]: {"color": "#2ca02c", "linewidth": 1.2, "linestyle": "-", "alpha": 1.0}
    }

    for row, cr in enumerate(GRID["cooling_rate"]):
        for col, t0 in enumerate(GRID["initial_temp"]):
            ax = axes[row, col]

            for tmin in GRID["min_temp"]:
                res = results[(t0, cr, tmin)]
                evals, costs = _unpack_history(res)
                sty = styles[tmin]

                ax.step(evals, costs, label=f"min_temp={tmin}", color=sty["color"],
                        linewidth=sty["linewidth"], linestyle=sty["linestyle"], alpha=sty["alpha"], where='post')

            ax.set_title(f"T0 = {t0} | CR = {cr}")
            ax.grid(True, alpha=0.3, linestyle=":")

            if row == 2:
                ax.set_xlabel("Function Evaluations")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))
            if col == 0:
                ax.set_ylabel("Cost")
            if row == 0 and col == 2:
                ax.legend(title="Min Temp")

    plt.tight_layout()
    png_path = os.path.join(outdir, "sa_grid_search.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved visualization to {png_path}")
