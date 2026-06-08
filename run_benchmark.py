"""Benchmark runner: compare split optimizers from src/optimizers.
Creates visualizations of cost-convergence curves for split optimizers.

Usage: uv run python run_benchmark.py [bcss|celeba|isic|synth_*]
Results saved to results/<type>/<name>_report.txt, results/<type>/convergence_<name>.[png|csv]. and results/summary.txt.
"""

import csv
import glob
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimizers import N_SPLITS, SPLIT_NAMES, RandomSearch, SimulatedAnnealing, SplitResult
from src.optimizers.stratified_group_k_fold import SGKFBaseline
from src.preprocessing.common import DatasetGroups, load_dataset

_DATASET_PATHS = {
    "bcss": "datasets/bcss/preprocessed/groups.pkl",
    "celeba": "datasets/celeb-faces/preprocessed/groups.pkl",
    "isic": "datasets/isic2020/preprocessed/groups.pkl",
}

for _pkl in sorted(glob.glob("datasets/synthetic/preprocessed/*.pkl")):
    _DATASET_PATHS[os.path.splitext(os.path.basename(_pkl))[0]] = _pkl


def _result_folder(name: str) -> str:
    return "synthetic" if name.startswith("synth_") else name


RATIOS = (0.70, 0.15, 0.15)
MAX_EVALS = 500_000
INITIAL_TEMP = 100.0
COOLING_RATE = 0.9999
MIN_TEMP = 1e-4
SEED = 42

_OPTIMIZERS = [
    ("SA", SimulatedAnnealing, dict(max_evals=MAX_EVALS, initial_temp=INITIAL_TEMP, cooling_rate=COOLING_RATE, min_temp=MIN_TEMP, seed=SEED)),
    ("RS", RandomSearch, dict(max_evals=MAX_EVALS, seed=SEED)),
    ("SGKF", SGKFBaseline, {}),
]

_STYLE = {
    "SA": dict(color="#1f77b4", linestyle="-", linewidth=1.8, marker="o", markersize=4),
    "RS": dict(color="#ff7f0e", linestyle="--", linewidth=1.8, marker="s", markersize=4),
    "SGKF": dict(color="#2ca02c", linestyle=":", linewidth=2.0, marker="", markersize=0),
}
_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_style(alg_name: str, idx: int) -> dict:
    if alg_name in _STYLE:
        return dict(_STYLE[alg_name])
    return dict(color=_FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)], linestyle="-", linewidth=1.8, marker="x", markersize=4)


def run_one(name: str) -> tuple[DatasetGroups, dict[str, SplitResult]]:
    data = load_dataset(_DATASET_PATHS[name])
    results = {}
    for alg_name, cls, kwargs in _OPTIMIZERS:
        results[alg_name] = cls(data=data, ratios=RATIOS, **kwargs).optimize(verbose=False)
    return data, results


def build_report(data: DatasetGroups, results: dict[str, SplitResult], ratios: tuple[float, ...] = RATIOS) -> str:
    buf = io.StringIO()
    buf.write(f"\n{data.dataset_name} | Groups: {data.n_groups} | Classes: {data.n_classes} | Budget: {MAX_EVALS}\n\n")
    cell_w = 20
    total_counts = data.global_class_counts.astype(float)

    for alg_name, res in results.items():
        buf.write(f"[{alg_name}]\n")
        header = f"{'Class':<42}" + "".join(f"{name:>{cell_w}}" for name in SPLIT_NAMES)
        buf.write(header + "\n" + "-" * len(header) + "\n")

        for c in range(data.n_classes):
            total = total_counts[c]
            if total == 0:
                continue
            row = f"{data.class_names[c]:<42}"
            for s in range(N_SPLITS):
                count = int(res.actual_counts[s, c])
                pct = count / total * 100.0
                row += f"{f'{count:,} ({pct:.1f}%)':>{cell_w}}"
            buf.write(row + "\n")

        buf.write("-" * len(header) + "\n")
        row = f"{'TOTAL':<42}"
        for s in range(N_SPLITS):
            n_items = int(data.group_sizes[res.assignment == s].sum())
            pct = n_items / data.total_items * 100.0
            row += f"{f'{n_items:,} ({pct:.1f}%)':>{cell_w}}"
        buf.write(row + "\n\n")
    return buf.getvalue()


def _unpack_history(result: SplitResult) -> tuple[np.ndarray, np.ndarray]:
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


def save_convergence_data(name: str, results: dict[str, SplitResult], outdir: str) -> str:
    out_path = os.path.join(outdir, f"convergence_{name}.csv")
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "FFE", "Cost", "Time_s"])
        for alg_name, res in results.items():
            evals, costs = _unpack_history(res)
            times = np.minimum(evals / max(1, res.n_evals), 1.0) * res.elapsed_time
            best_so_far = float('inf')
            for e, c, t in zip(evals, costs, times):
                if c < best_so_far:
                    writer.writerow([alg_name, int(e), float(c), float(t)])
                    best_so_far = c
    return out_path


def plot_convergence(name: str, results: dict[str, SplitResult], outdir: str) -> str:
    fig, (ax_ffe, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{name}", fontsize=14, fontweight="bold")

    budget = max(max((r.cost_history[-1][0] if r.cost_history else r.n_evals) for r in results.values()), 1)

    for idx, (alg_name, res) in enumerate(results.items()):
        style = _get_style(alg_name, idx)
        evals, costs = _unpack_history(res)

        if len(evals) == 1:
            ax_ffe.axhline(y=costs[0], label=alg_name, **style)
            ax_time.axhline(y=costs[0], label=alg_name, **style)
        else:
            ax_ffe.step(evals, costs, label=alg_name, **style, where='post')
            times = np.minimum(evals / max(1, res.n_evals), 1.0) * res.elapsed_time
            ax_time.step(times, costs, label=alg_name, **style, where='post')

    ax_ffe.set(title="Cost vs FFEs", xlabel=f"FFEs (max {budget:,})", ylabel="Cost")
    ax_ffe.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"))

    ax_time.set(title="Cost vs Time", xlabel="Time (s)", ylabel="Cost")

    for ax in (ax_ffe, ax_time):
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = os.path.join(outdir, f"convergence_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    pos = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a.lstrip("-").lower() for a in sys.argv[1:] if a.startswith("-")}
    names = list(_DATASET_PATHS) if not pos or "all" in pos else pos

    if "fast" in flags:
        MAX_EVALS = 50_000
        for _label, _cls, kwargs in _OPTIMIZERS:
            if "max_evals" in kwargs:
                kwargs["max_evals"] = MAX_EVALS

    os.makedirs("results", exist_ok=True)
    summary_rows = []

    for name in names:
        print(f"-> {name:<22} ", end="", flush=True)
        data, results = run_one(name)
        outdir = os.path.join("results", _result_folder(name))
        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, f"{name}_report.txt"), "w") as f:
            f.write(build_report(data, results))

        plot_convergence(name, results, outdir)
        save_convergence_data(name, results, outdir)

        costs = {k: v.cost for k, v in results.items()}
        best = min(costs, key=costs.__getitem__)
        print(f"Done. Best: {best} ({costs[best]:.4f})")

        summary_rows.append((name, data.n_groups, data.n_classes, results, costs))

    buf = io.StringIO()
    algs = list(summary_rows[0][3].keys()) if summary_rows else []
    header = f"{'Dataset':<18} {'Groups':>8} {'Classes':>8} " + "".join(f"{a:>10}" for a in algs) + f" {'Best':>8}"

    buf.write(f"\n{header}\n{'-' * len(header)}\n")
    for name, grp, cls, res, costs in summary_rows:
        row = f"{name:<18} {grp:>8} {cls:>8} " + "".join(f"{costs[a]:>10.4f}" for a in algs)
        best = min(costs, key=costs.__getitem__)
        buf.write(f"{row} {best:>8}\n")

    print(buf.getvalue())
    with open("results/summary.txt", "w") as f:
        f.write(buf.getvalue())
