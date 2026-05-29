"""Visualize cost-convergence curves for split optimizers.

Usage: uv run python visualize.py [bcss|celeba|isic|all] [--fast] - quick preview (50k cap)
Figures are saved to results/convergence_<name>.png.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import compare as _cmp
from src.optimizers import SplitResult

_STYLE: dict[str, dict] = {
    "SA": dict(color="#1f77b4", linestyle="-", linewidth=1.8, marker="o", markersize=4),
    "RS": dict(color="#ff7f0e", linestyle="--", linewidth=1.8, marker="s", markersize=4),
}
_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_style(alg_name: str, idx: int) -> dict:
    """Return the visual style for alg_name, with a fallback for unknown names."""
    if alg_name in _STYLE:
        return dict(_STYLE[alg_name])
    return dict(
        color=_FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
        linestyle="-",
        linewidth=1.8,
        marker="x",
        markersize=4,
    )


def _unpack_history(result: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (evals, best_costs) arrays extracted from result.cost_history."""
    if not result.cost_history:
        return (
            np.array([result.n_evals], dtype=float),
            np.array([result.cost], dtype=float),
        )
    evals, costs = zip(*result.cost_history)
    return np.asarray(evals, dtype=float), np.asarray(costs, dtype=float)


def plot_convergence(
    name: str,
    results: dict[str, SplitResult],
    outdir: str = "results",
) -> str:
    """Create and save a two-panel convergence figure for one dataset. """
    fig, (ax_ffe, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Cost convergence  —  {name}",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (alg_name, res) in enumerate(results.items()):
        style = _get_style(alg_name, idx)
        evals, costs = _unpack_history(res)

        markevery = max(1, len(evals) // 8)

        ax_ffe.plot(evals, costs, label=alg_name, markevery=markevery, **style)

        times = evals / res.n_evals * res.elapsed_time
        ax_time.plot(times, costs, label=alg_name, markevery=markevery, **style)

    budget = max(r.n_evals for r in results.values())
    ax_ffe.set_title("Cost vs. FFEs")
    ax_ffe.set_xlabel(f"Function evaluations  (budget = {budget:,})")
    ax_ffe.set_ylabel("Best cost  (weighted MAPE min)")
    ax_ffe.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"
        )
    )

    ax_time.set_title("Cost vs. wall-clock time")
    ax_time.set_xlabel("Elapsed time (s)  [linearly interpolated from total]")
    ax_time.set_ylabel("Best cost  (weighted MAPE min)")

    for ax in (ax_ffe, ax_time):
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"convergence_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a.lstrip("-").lower() for a in sys.argv[1:] if a.startswith("-")}

    target = positional[0].lower() if positional else "all"

    if target not in list(_cmp._DATASET_PATHS) + ["all"]:
        print(
            f"Unknown dataset '{target}'. "
            f"Choose from: {list(_cmp._DATASET_PATHS) + ['all']}"
        )
        sys.exit(1)

    names = list(_cmp._DATASET_PATHS) if target == "all" else [target]

    if "fast" in flags:
        fast_budget = 50_000
        for _label, _cls, kwargs in _cmp._OPTIMIZERS:
            kwargs["max_evals"] = fast_budget
        print(f"[fast mode] max_evals capped at {fast_budget:,}\n")

    for name in names:
        print(f"[{name}] running optimizers …", flush=True)
        _, results = _cmp.run_one(name)
        out = plot_convergence(name, results)
        costs_str = "  ".join(f"{k}={v.cost:.4f}" for k, v in results.items())
        print(f"[{name}] {costs_str}  -> {out}")
