"""Benchmark runner: compare split optimizers from src/optimizers.

Usage: uv run python compare.py [bcss|celeba|isic|all] (default: all)
Results saved to results/<name>_report.txt and results/summary.txt.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.optimizers import (
    N_SPLITS,
    SPLIT_NAMES,
    RandomSearch,
    SimulatedAnnealing,
    SplitResult,
)
from src.preprocessing.common import DatasetGroups, load_dataset

# Dataset paths
_DATASET_PATHS = {
    "bcss": "datasets/bcss/preprocessed/groups.pkl",
    "celeba": "datasets/celeb-faces/preprocessed/groups.pkl",
    "isic": "datasets/isic2020/preprocessed/groups.pkl",
}

# Config
RATIOS = (0.70, 0.15, 0.15)
TARGET_COST = 0.5
MAX_EVALS = 500_000
INITIAL_TEMP = 10.0
COOLING_RATE = 0.9999
MIN_TEMP = 1e-4
SEED = 42


# Optimizer registry - add entries here to compare additional algorithms.
_OPTIMIZERS: list[tuple[str, type, dict]] = [
    (
        "SA",
        SimulatedAnnealing,
        dict(
            target_cost=TARGET_COST,
            max_evals=MAX_EVALS,
            initial_temp=INITIAL_TEMP,
            cooling_rate=COOLING_RATE,
            min_temp=MIN_TEMP,
            seed=SEED,
        ),
    ),
    (
        "RS",
        RandomSearch,
        dict(
            target_cost=TARGET_COST,
            max_evals=MAX_EVALS,
            seed=SEED,
        ),
    ),
]


# Runner
def run_one(name: str) -> tuple[DatasetGroups, dict[str, SplitResult]]:
    data = load_dataset(_DATASET_PATHS[name])
    results: dict[str, SplitResult] = {}
    for alg_name, cls, kwargs in _OPTIMIZERS:
        opt = cls(data=data, ratios=RATIOS, **kwargs)
        results[alg_name] = opt.optimize(verbose=False)
    return data, results


# Reporting
def build_report(
    data: DatasetGroups,
    results: dict[str, SplitResult],
    ratios: tuple[float, ...] = RATIOS,
) -> str:
    """Build a text comparison report for a single dataset."""
    buf = io.StringIO()
    ratios_ = np.asarray(ratios, dtype=np.float64)
    names = list(results)

    buf.write(f"\n{'=' * 100}\n")
    buf.write(
        f"  {data.dataset_name}  groups={data.n_groups}  classes={data.n_classes}"
        f"  FFE budget={MAX_EVALS:,}  ratios={tuple(f'{r:.2f}' for r in ratios)}\n"
    )
    buf.write(f"{'=' * 100}\n")

    header = (
        f"  {'Method':<20}"
        + "  ".join(f"{n:>22}" for n in SPLIT_NAMES)
        + f"  {'Cost':>10}  {'Evals':>10}  {'Time':>8}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    costs: dict[str, float] = {}
    for alg_name, res in results.items():
        row = f"  {alg_name:<20}"
        for s in range(N_SPLITS):
            n_g = int((res.assignment == s).sum())
            n_i = int(data.group_sizes[res.assignment == s].sum())
            row += f"  {n_g:>5}g/{n_i:>7}i ({n_i / data.total_items * 100:.1f}%)"
        row += f"  {res.cost:>10.4f}  {res.n_evals:>10,}  {res.elapsed_time:>7.2f}s"
        buf.write(row + "\n")
        costs[alg_name] = res.cost

    col_w = 9
    buf.write("\n  Per-class |actual - target| deviation (% of class total):\n")
    h2 = f"  {'Class':<42}" + "".join(
        f"  {n + ' tr':>{col_w}}  {n + ' va':>{col_w}}  {n + ' te':>{col_w}}"
        for n in names
    )
    buf.write(h2 + "\n")
    buf.write("  " + "-" * (len(h2) - 2) + "\n")

    total_counts = data.global_class_counts.astype(float)
    for c in range(data.n_classes):
        total = total_counts[c]
        if total == 0:
            continue
        row = f"  {data.class_names[c]:<42}"
        for res in results.values():
            for s in range(N_SPLITS):
                dev = abs(res.actual_counts[s, c] / total * 100.0 - ratios_[s] * 100.0)
                row += f"  {dev:>{col_w}.2f}%"
        buf.write(row + "\n")

    best = min(costs, key=costs.__getitem__)
    buf.write("\n")
    for alg_name, cost in costs.items():
        tag = " <- best" if alg_name == best else ""
        buf.write(f"  {alg_name:<20}  cost={cost:.4f}{tag}\n")
    buf.write("=" * 100 + "\n")
    return buf.getvalue()


# Main
if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    names = list(_DATASET_PATHS) if target == "all" else [target]

    if target not in list(_DATASET_PATHS) + ["all"]:
        print(f"Unknown dataset '{target}'. Choose from: {list(_DATASET_PATHS) + ['all']}")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    summary_rows = []

    for name in names:
        print(f"[{name}] running...", flush=True)
        data, results = run_one(name)

        report_path = f"results/{name}_report.txt"
        with open(report_path, "w") as f:
            f.write(build_report(data, results))

        costs = {k: v.cost for k, v in results.items()}
        best = min(costs, key=costs.__getitem__)
        cost_str = "  ".join(f"{k}={v:.4f}" for k, v in costs.items())
        print(f"[{name}] {cost_str}  best={best}  -> {report_path}")
        summary_rows.append((name, data.n_groups, data.n_classes, results, costs))

    # Summary table
    buf = io.StringIO()
    buf.write("\n" + "=" * 90 + "\n")
    buf.write("  SUMMARY: Weighted MAPE cost (lower is better)\n")
    buf.write(f"  FFE budget = {MAX_EVALS:,} for all stochastic algorithms\n")
    buf.write("=" * 90 + "\n")
    all_algs = list(summary_rows[0][3].keys()) if summary_rows else []
    col = 12
    header = f"  {'Dataset':<10}  {'Groups':>7}  {'Classes':>7}"
    for a in all_algs:
        header += f"  {a[:col]:>{col}}"
    header += f"  {'Best':>10}  {'SA evals':>10}  {'RS evals':>10}  {'SA time':>8}  {'RS time':>8}"
    buf.write(header + "\n")
    buf.write("  " + "-" * (len(header) - 2) + "\n")

    for name, n_groups, n_classes, results, costs in summary_rows:
        row = f"  {name:<10}  {n_groups:>7}  {n_classes:>7}"
        for a in all_algs:
            row += f"  {costs[a]:>{col}.4f}"
        best = min(costs, key=costs.__getitem__)
        sa_t = results["SA"].elapsed_time
        rs_t = results["RS"].elapsed_time if "RS" in results else float("nan")
        sa_ev = results["SA"].n_evals
        rs_ev = results["RS"].n_evals if "RS" in results else 0
        row += f"  {best:>10}  {sa_ev:>10,}  {rs_ev:>10,}  {sa_t:>7.2f}s  {rs_t:>7.2f}s"
        buf.write(row + "\n")

    buf.write("=" * 90 + "\n")
    summary_text = buf.getvalue()
    print(summary_text)

    summary_path = "results/summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {summary_path}")
