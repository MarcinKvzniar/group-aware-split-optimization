"""Benchmark runner: compare split optimizers from src/optimizers.

Usage: uv run python compare.py [bcss|celeba|isic|synth_*|all] (default: all)
Results saved to results/<type>/<name>_report.txt and results/summary.txt.
"""

import glob
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

for _pkl in sorted(glob.glob("datasets/synthetic/preprocessed/*.pkl")):
    _key = os.path.splitext(os.path.basename(_pkl))[0]
    _DATASET_PATHS[_key] = _pkl


def _result_folder(name: str) -> str:
    """Map a dataset name to its results subfolder."""
    return "synthetic" if name.startswith("synth_") else name

# Config
RATIOS = (0.70, 0.15, 0.15)
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
    """Build a per-dataset report showing the exact class distribution per split."""
    buf = io.StringIO()

    buf.write(f"\n{'=' * 100}\n")
    buf.write(
        f"  {data.dataset_name}  groups={data.n_groups}  classes={data.n_classes}"
        f"  FFE budget={MAX_EVALS:,}  ratios={tuple(f'{r:.2f}' for r in ratios)}\n"
    )
    buf.write(f"{'=' * 100}\n")

    cell_w = 20
    total_counts = data.global_class_counts.astype(float)

    for alg_name, res in results.items():
        buf.write(f"\n  {alg_name}\n")
        header = f"  {'Class':<42}" + "".join(
            f"  {name:>{cell_w}}" for name in SPLIT_NAMES
        )
        buf.write(header + "\n")
        buf.write("  " + "-" * (len(header) - 2) + "\n")

        for c in range(data.n_classes):
            total = total_counts[c]
            if total == 0:
                continue
            row = f"  {data.class_names[c]:<42}"
            for s in range(N_SPLITS):
                count = int(res.actual_counts[s, c])
                pct = count / total * 100.0
                cell = f"{count:,} ({pct:.1f}%)"
                row += f"  {cell:>{cell_w}}"
            buf.write(row + "\n")

        # Totals row
        buf.write("  " + "-" * (len(header) - 2) + "\n")
        row = f"  {'TOTAL':<42}"
        for s in range(N_SPLITS):
            n_items = int(data.group_sizes[res.assignment == s].sum())
            pct = n_items / data.total_items * 100.0
            cell = f"{n_items:,} ({pct:.1f}%)"
            row += f"  {cell:>{cell_w}}"
        buf.write(row + "\n")

    buf.write("=" * 100 + "\n")
    return buf.getvalue()


# Main
if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]] if len(sys.argv) > 1 else ["all"]

    if "all" in args:
        names = list(_DATASET_PATHS)
    else:
        unknown = [a for a in args if a not in _DATASET_PATHS]
        if unknown:
            print(f"Unknown dataset(s): {unknown}. Choose from: {list(_DATASET_PATHS) + ['all']}")
            sys.exit(1)
        names = args

    os.makedirs("results", exist_ok=True)
    summary_rows = []

    for name in names:
        print(f"[{name}] running...", flush=True)
        data, results = run_one(name)

        outdir = os.path.join("results", _result_folder(name))
        os.makedirs(outdir, exist_ok=True)
        report_path = os.path.join(outdir, f"{name}_report.txt")
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

    summary_path = os.path.join("results", "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {summary_path}")
