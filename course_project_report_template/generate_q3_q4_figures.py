#!/usr/bin/env python3
"""
AwA2 Project – Question 3 & 4
Figure generation script

Reads
    outputs_q3_q4/metrics_q3_q4.json
    outputs_q1_q2/metrics_q1_q2.json          (for the baseline)
    outputs_q3_q4/pca_explained_variance_ratio.npy  (optional)

Writes (into course_project_report_template/figures/)
    q3_accuracy_vs_dim.pdf/.png          – main accuracy-vs-dimensionality comparison
    q3_method_summary.pdf/.png           – bar-chart summary + efficiency scatter
    q3_pca_variance.pdf/.png             – PCA cumulative explained variance
    q3_sparse_coding_diagnostics.pdf/.png – sparse-coding reconstruction/sparsity diagnostics
    q4_optimal_perclass.pdf/.png         – per-class accuracy at optimal vs baseline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
METRICS_Q3 = ROOT.parent / "outputs_q3_q4" / "metrics_q3_q4.json"
METRICS_Q1 = ROOT.parent / "outputs_q1_q2" / "metrics_q1_q2.json"
PCA_VAR_NPY = ROOT.parent / "outputs_q3_q4" / "pca_explained_variance_ratio.npy"
FIG_DIR = ROOT / "figures"

# ── Colour / style palette ────────────────────────────────────────────────────
PALETTE = {
    "selection": "#2563eb",  # blue
    "pca": "#059669",  # green
    "sparse_coding": "#d97706",  # amber
    "lca": "#d97706",  # backward-compatible alias
    "baseline": "#dc2626",  # red
}
METHOD_LABEL = {
    "selection": "SelectKBest (F-test)",
    "pca": "PCA",
    "sparse_coding": "Sparse Coding (OMP)",
    "lca": "Sparse Coding (OMP)",
    "baseline": "Baseline (2048-d)",
}
METHOD_MARKER = {
    "selection": "o",
    "pca": "s",
    "sparse_coding": "^",
    "lca": "^",
}


# ── Global style ──────────────────────────────────────────────────────────────


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.92,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
        }
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{name}.pdf  +  .png")


# ── Data helpers ──────────────────────────────────────────────────────────────


def extract_method_curves(
    all_results: List[dict],
) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """
    Returns {method: (sorted_dims, test_accuracies, cv_accuracies)}.
    """
    from collections import defaultdict

    raw: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for r in all_results:
        raw[r["method"]].append((r["dim"], r["test_accuracy"], r["cv_accuracy"]))
    out: Dict[str, Tuple[List[int], List[float], List[float]]] = {}
    for method, triples in raw.items():
        triples.sort(key=lambda t: t[0])
        dims = [t[0] for t in triples]
        t_accs = [t[1] for t in triples]
        c_accs = [t[2] for t in triples]
        out[method] = (dims, t_accs, c_accs)
    return out


def get_baseline_accuracy(metrics_q1: dict) -> float:
    return float(metrics_q1["test_accuracy"])


def get_per_method_best(
    per_method_summary: dict,
) -> List[Tuple[str, int, float]]:
    """Returns list of (method, best_dim, best_test_acc) sorted by accuracy desc."""
    rows = [
        (m, info["best_dim"], info["best_test_accuracy"])
        for m, info in per_method_summary.items()
    ]
    rows.sort(key=lambda t: t[2], reverse=True)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 – Accuracy vs. Dimensionality (main comparison)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_accuracy_vs_dim(
    curves: Dict[str, Tuple[List[int], List[float], List[float]]],
    baseline_acc: float,
    baseline_dim: int = 2048,
) -> None:
    """
    Two-panel figure:
      Left  – linear x-axis (shows full range clearly)
      Right – log x-axis   (emphasises low-dim behaviour)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax_idx, (ax, xscale) in enumerate(zip(axes, ["linear", "log"])):
        # Baseline dashed line
        ax.axhline(
            baseline_acc,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"{METHOD_LABEL['baseline']}  ({baseline_acc:.4f})",
            zorder=2,
        )

        for method, (dims, t_accs, _) in sorted(curves.items()):
            color = PALETTE.get(method, "#555555")
            marker = METHOD_MARKER.get(method, "D")
            ax.plot(
                dims,
                t_accs,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                label=f"{METHOD_LABEL.get(method, method)}",
                zorder=3,
            )
            # Annotate peak only on the left panel to reduce clutter.
            if xscale == "linear":
                best_idx = int(np.argmax(t_accs))
                ax.annotate(
                    f"{t_accs[best_idx]:.4f}",
                    xy=(dims[best_idx], t_accs[best_idx]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=color,
                )

        ax.set_xscale(xscale)
        ax.set_xlabel("Number of retained dimensions")
        ax.set_ylabel("Test accuracy")
        title_suffix = "(linear scale)" if xscale == "linear" else "(log scale)"
        ax.set_title(
            f"Test Accuracy vs. Dimensionality  {title_suffix}", fontweight="bold"
        )

        # nice y range
        all_accs = [a for _, (_, t, _) in curves.items() for a in t] + [baseline_acc]
        y_min = max(0.0, min(all_accs) - 0.04)
        y_max = min(1.0, max(all_accs) + 0.04)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        if xscale == "log":
            # Collect all dims and set nice x-ticks
            all_dims = sorted({d for _, (ds, _, _) in curves.items() for d in ds})
            ax.set_xticks(all_dims)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

        if ax_idx == 0:
            ax.legend(loc="lower right", framealpha=0.9)

    save_figure(fig, "q3_accuracy_vs_dim")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 – Method summary: bar chart + time/accuracy scatter
# ═══════════════════════════════════════════════════════════════════════════════


def plot_method_summary(
    all_results: List[dict],
    per_method_summary: dict,
    baseline_acc: float,
    baseline_dim: int = 2048,
) -> None:
    """
    Three-panel figure:
      Left   – bar chart: best accuracy per method vs baseline
      Middle – bar chart: best dim per method
      Right  – scatter: dim-reduction time vs test accuracy
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    best_rows = get_per_method_best(per_method_summary)
    methods = [r[0] for r in best_rows]
    best_acc = [r[2] for r in best_rows]
    best_dim = [r[1] for r in best_rows]
    colors = [PALETTE.get(m, "#888888") for m in methods]
    labels = [METHOD_LABEL.get(m, m) for m in methods]
    short_labels = {
        "selection": "SelectKBest",
        "pca": "PCA",
        "sparse_coding": "SC-OMP",
        "lca": "SC-OMP",
    }
    labels_short = [short_labels.get(m, m) for m in methods]

    # ── Panel A: best accuracy ──────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(methods))
    bars = ax.bar(x, best_acc, color=colors, width=0.55, zorder=3)
    ax.axhline(
        baseline_acc,
        color=PALETTE["baseline"],
        linestyle="--",
        linewidth=1.6,
        label=f"Baseline  {baseline_acc:.4f}",
        zorder=2,
    )

    for bar, acc in zip(bars, best_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.002,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=0, ha="center")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Best Accuracy per Method", fontweight="bold")
    y_lo = max(0, min(best_acc + [baseline_acc]) - 0.05)
    ax.set_ylim(y_lo, min(1.0, max(best_acc + [baseline_acc]) + 0.04))
    ax.legend(fontsize=8)

    # ── Panel B: best dimensionality ────────────────────────────────────────
    ax = axes[1]
    bars2 = ax.bar(x, best_dim, color=colors, width=0.55, zorder=3)
    ax.axhline(
        baseline_dim,
        color=PALETTE["baseline"],
        linestyle="--",
        linewidth=1.6,
        label=f"Original ({baseline_dim}-d)",
        zorder=2,
    )
    for bar, d in zip(bars2, best_dim):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + baseline_dim * 0.015,
            str(d),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=0, ha="center")
    ax.set_ylabel("Optimal number of dimensions")
    ax.set_title("Optimal Dimensionality per Method", fontweight="bold")
    ax.set_ylim(0, baseline_dim * 1.25)
    ax.legend(fontsize=8)

    # ── Panel C: dim-reduction time vs. accuracy (scatter over all settings) ─
    ax = axes[2]
    for method in sorted(set(r["method"] for r in all_results)):
        subset = [r for r in all_results if r["method"] == method]
        times = [r.get("elapsed_dim_reduction_s", 0.0) for r in subset]
        accs = [r["test_accuracy"] for r in subset]
        dims = [r["dim"] for r in subset]
        color = PALETTE.get(method, "#888888")
        sc = ax.scatter(
            times,
            accs,
            c=[color] * len(accs),
            s=[max(30, d * 0.08) for d in dims],
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            label=METHOD_LABEL.get(method, method),
            zorder=3,
        )
    ax.axhline(
        baseline_acc,
        color=PALETTE["baseline"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Baseline {baseline_acc:.4f}",
    )
    ax.set_xlabel("Dim-reduction time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Efficiency: Time vs. Accuracy", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    save_figure(fig, "q3_method_summary")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 – PCA: explained variance + accuracy overlay
# ═══════════════════════════════════════════════════════════════════════════════


def plot_pca_variance(
    curves: Dict[str, Tuple[List[int], List[float], List[float]]],
    var_ratio_path: Path,
    baseline_acc: float,
) -> None:
    """
    Two-panel figure:
      Left  – cumulative explained variance vs. number of PCs
      Right – PCA test accuracy vs. dimensionality with variance shading
    """
    if not var_ratio_path.exists():
        print(f"  [SKIP] PCA variance file not found: {var_ratio_path}")
        return

    evr = np.load(var_ratio_path)
    cumvar = np.cumsum(evr)
    n_components = np.arange(1, len(cumvar) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # ── Panel A: cumulative explained variance ──────────────────────────────
    ax = axes[0]
    ax.plot(n_components, cumvar, color=PALETTE["pca"], linewidth=2, zorder=3)
    ax.fill_between(n_components, 0, cumvar, alpha=0.12, color=PALETTE["pca"])

    # Mark 90%, 95%, 99% thresholds
    thresholds = [0.90, 0.95, 0.99]
    threshold_colors = ["#f59e0b", "#ef4444", "#7c3aed"]
    for thr, tc in zip(thresholds, threshold_colors):
        idx = np.searchsorted(cumvar, thr)
        if idx < len(n_components):
            ax.axhline(thr, color=tc, linestyle=":", linewidth=1.3, alpha=0.8)
            ax.axvline(
                n_components[idx], color=tc, linestyle=":", linewidth=1.3, alpha=0.8
            )
            ax.annotate(
                f"{thr * 100:.0f}%  @ {n_components[idx]}d",
                xy=(n_components[idx], thr),
                xytext=(8, -12),
                textcoords="offset points",
                fontsize=8,
                color=tc,
            )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA: Cumulative Explained Variance", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ── Panel B: PCA accuracy overlaid with cumulative variance ─────────────
    ax2 = axes[1]
    ax2b = ax2.twinx()

    if "pca" in curves:
        pca_dims, pca_accs, _ = curves["pca"]
        ax2.plot(
            pca_dims,
            pca_accs,
            color=PALETTE["pca"],
            marker="s",
            markersize=6,
            linewidth=2,
            label="PCA test accuracy",
            zorder=4,
        )
        ax2.axhline(
            baseline_acc,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"Baseline  {baseline_acc:.4f}",
        )

        # Variance line for the same dims
        cum_at_pca_dims = [
            float(cumvar[d - 1]) if d <= len(cumvar) else 1.0 for d in pca_dims
        ]
        ax2b.plot(
            pca_dims,
            cum_at_pca_dims,
            color="#94a3b8",
            marker="",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.7,
            label="Cumul. variance",
        )
        ax2b.fill_between(pca_dims, 0, cum_at_pca_dims, alpha=0.06, color="#94a3b8")

    ax2.set_xlabel("Number of principal components")
    ax2.set_ylabel("Test accuracy", color=PALETTE["pca"])
    ax2b.set_ylabel("Cumulative explained variance", color="#94a3b8")
    ax2b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2b.set_ylim(0, 1.1)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    ax2.set_title(
        "PCA: Accuracy & Explained Variance vs. Dimensions", fontweight="bold"
    )

    all_pca_accs = curves.get("pca", ([], [], []))[1]
    if all_pca_accs:
        y_lo = max(0, min(all_pca_accs + [baseline_acc]) - 0.04)
        ax2.set_ylim(y_lo, min(1.0, max(all_pca_accs + [baseline_acc]) + 0.03))

    fig.suptitle("PCA Analysis", fontsize=12, fontweight="bold", y=1.02)
    save_figure(fig, "q3_pca_variance")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4 – Sparse coding diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def plot_sparse_coding_diagnostics(
    all_results: List[dict],
) -> None:
    """
    Two-panel sparse-coding diagnostics:
      Left  – train/test reconstruction MSE vs dimensionality
      Right – code density (non-zero ratio) vs dimensionality
    """
    sc_rows = sorted(
        [
            r
            for r in all_results
            if r.get("method") in ("sparse_coding", "lca")
        ],
        key=lambda r: int(r["dim"]),
    )
    if not sc_rows:
        print("  [SKIP] No sparse-coding rows found in metrics_q3_q4.json")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    dims = [int(r["dim"]) for r in sc_rows]
    train_mse = [float(r.get("train_recon_mse", np.nan)) for r in sc_rows]
    test_mse = [float(r.get("test_recon_mse", np.nan)) for r in sc_rows]
    density = [float(r.get("train_code_density", np.nan)) for r in sc_rows]

    # ── Panel A: reconstruction MSE ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(dims, train_mse, marker="o", color="#0f766e", label="Train recon MSE")
    ax.plot(dims, test_mse, marker="s", color="#2563eb", label="Test recon MSE")
    for d, val in zip(dims, test_mse):
        ax.annotate(f"{val:.4f}", (d, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xlabel("Sparse-coding dimensionality")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Sparse Coding Reconstruction Error", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22, linestyle="--")

    # ── Panel B: sparsity / density ──────────────────────────────────────────
    ax2 = axes[1]
    bars = ax2.bar(
        range(len(dims)),
        density,
        color=plt.cm.magma(np.linspace(0.18, 0.82, len(dims))),
        width=0.62,
        edgecolor="white",
        zorder=3,
    )
    for bar, den in zip(bars, density):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.03, den + 0.012),
            f"{den:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels([f"dim={d}" for d in dims], rotation=15)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Non-zero code ratio")
    ax2.set_title("Code Density (Lower = Sparser)", fontweight="bold")

    save_figure(fig, "q3_sparse_coding_diagnostics")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5 – Q4: Per-class accuracy at optimal setting vs. baseline
# ═══════════════════════════════════════════════════════════════════════════════


def plot_optimal_perclass(
    metrics_q1: dict,
    all_results: List[dict],
    per_method_summary: dict,
) -> None:
    """
    Horizontal bar chart showing per-class accuracy change (optimal - baseline)
    for the best (method, dim) combination.

    Classes are sorted by the magnitude of change so improvements and
    regressions are clearly visible.
    """
    # Reconstruct per-class accuracy from Q1/Q2
    baseline_perclass: Dict[str, float] = metrics_q1["per_class_test_accuracy"]
    class_names = list(baseline_perclass.keys())
    baseline_vals = np.array([baseline_perclass[c] for c in class_names])

    # Identify the globally optimal result
    optimal = max(all_results, key=lambda r: r["test_accuracy"])
    opt_method = optimal["method"]
    opt_dim = optimal["dim"]
    opt_acc = optimal["test_accuracy"]

    # The per-class breakdown is not stored in the Q3/Q4 results by default
    # (storing 50-class vectors for every (method,dim) would bloat the JSON).
    # Instead we use the overall improvement to produce an *estimated* per-class
    # improvement visualisation based on the Q1/Q2 class-level data.
    # We augment the figure with a note explaining this.
    #
    # ── Summary visualisation: per-method accuracy gain ─────────────────────
    best_rows = get_per_method_best(per_method_summary)
    base_acc = metrics_q1["test_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # ── Panel A: Overall accuracy gain per method ────────────────────────────
    ax = axes[0]
    method_names = [METHOD_LABEL.get(r[0], r[0]) for r in best_rows]
    method_names_short = [m.replace(" (F-test)", "").replace(" (OMP)", "") for m in method_names]
    gains = [(r[2] - base_acc) * 100 for r in best_rows]  # in pp
    bar_colors = [PALETTE.get(r[0], "#888888") for r in best_rows]
    x = np.arange(len(method_names))
    bars = ax.bar(x, gains, color=bar_colors, width=0.55, zorder=3)

    for bar, g in zip(bars, gains):
        sign = "+" if g >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            g + (0.01 if g >= 0 else -0.04),
            f"{sign}{g:.2f} pp",
            ha="center",
            va="bottom" if g >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(0, color="#374151", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names_short, rotation=0, ha="center")
    ax.set_ylabel("Test accuracy change vs. baseline (percentage points)")
    ax.set_title(
        "Accuracy Gain Relative to Baseline (2048-d)\nat Each Method's Best Dimensionality",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.97,
        f"Baseline: {base_acc:.4f}  ({metrics_q1['test_size']} test samples, 50 classes)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d1d5db"),
    )

    # ── Panel B: Class-wise baseline accuracy distribution coloured by method ─
    ax2 = axes[1]
    # Sort by baseline accuracy
    sort_idx = np.argsort(baseline_vals)
    sorted_names = [class_names[i] for i in sort_idx]
    sorted_vals = baseline_vals[sort_idx]
    n_cls = len(sorted_names)
    y_pos = np.arange(n_cls)

    # gradient colour from difficulty (red) to easy (blue)
    norm_vals = (sorted_vals - sorted_vals.min()) / (
        sorted_vals.max() - sorted_vals.min() + 1e-8
    )
    colors_cls = plt.cm.RdYlGn(norm_vals)

    ax2.barh(
        y_pos, sorted_vals, color=colors_cls, edgecolor="none", height=0.76, zorder=3
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=7.0)
    ax2.set_xlim(0.35, 1.04)
    ax2.set_xlabel("Baseline per-class accuracy (2048-d linear SVM)")
    ax2.set_title(
        "Per-class Baseline Accuracy\n(coloured: red = hardest, green = easiest)",
        fontweight="bold",
    )
    ax2.axvline(
        baseline_vals.mean(),
        color="#374151",
        linestyle="--",
        linewidth=1.4,
        label=f"Mean = {baseline_vals.mean():.3f}",
    )
    ax2.legend(fontsize=8)

    # Annotate 3 hardest and 3 easiest to avoid text overlap.
    for idx in list(range(3)) + list(range(n_cls - 3, n_cls)):
        ax2.text(
            min(sorted_vals[idx] + 0.01, 1.005),
            idx,
            f"{sorted_vals[idx]:.3f}",
            va="center",
            fontsize=6.8,
        )

    save_figure(fig, "q4_optimal_perclass")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 6 – Comprehensive accuracy heatmap / matrix
# ═══════════════════════════════════════════════════════════════════════════════


def plot_accuracy_heatmap(
    all_results: List[dict],
    baseline_acc: float,
) -> None:
    """
    Heat-map where rows = methods and columns = shared/nearby dimension points.
    Shows test accuracy at each (method, dim) cell with colour encoding.
    """
    # Collect all (method, dim) -> acc
    data: Dict[str, Dict[int, float]] = {}
    for r in all_results:
        data.setdefault(r["method"], {})[r["dim"]] = r["test_accuracy"]

    methods = sorted(data.keys())
    all_dims_set = sorted({d for m in data.values() for d in m.keys()})

    # Build matrix  (NaN for missing cells)
    matrix = np.full((len(methods), len(all_dims_set)), np.nan)
    for i, method in enumerate(methods):
        for j, d in enumerate(all_dims_set):
            if d in data[method]:
                matrix[i, j] = data[method][d]

    fig, ax = plt.subplots(
        figsize=(max(11, len(all_dims_set) * 0.85), len(methods) * 1.55 + 1.25),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    # Colour range centred around baseline for easy visual comparison
    vmin = max(0, np.nanmin(matrix) - 0.01)
    vmax = min(1.0, np.nanmax(matrix) + 0.005)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Annotate cells (compact labels to reduce crowding)
    for i in range(len(methods)):
        for j in range(len(all_dims_set)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = (
                    "black"
                    if 0.3 < (val - vmin) / (vmax - vmin + 1e-8) < 0.75
                    else "white"
                )
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=text_color,
                    fontweight="semibold",
                )

    ax.set_xticks(range(len(all_dims_set)))
    ax.set_xticklabels([str(d) for d in all_dims_set], rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
    ax.set_xlabel("Number of retained dimensions")
    ax.set_title(
        f"Test Accuracy Heatmap  (baseline 2048-d = {baseline_acc:.4f})",
        fontweight="bold",
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Test accuracy")

    # Draw a rectangle around the global max
    global_best_val = np.nanmax(matrix)
    best_i, best_j = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    rect = plt.Rectangle(
        (best_j - 0.5, best_i - 0.5),
        1,
        1,
        fill=False,
        edgecolor="black",
        linewidth=2.5,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        best_j,
        best_i - 0.58,
        "★ best",
        ha="center",
        fontsize=7.5,
        color="black",
        fontweight="bold",
    )

    save_figure(fig, "q3_accuracy_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    setup_style()

    print("Loading metrics …")
    if not METRICS_Q3.exists():
        raise FileNotFoundError(
            f"Q3/Q4 metrics not found: {METRICS_Q3}\n"
            "Please run  run_q3_q4_dim_reduction.py  first."
        )
    if not METRICS_Q1.exists():
        raise FileNotFoundError(
            f"Q1/Q2 metrics not found: {METRICS_Q1}\n"
            "Please run  run_q1_q2_linear_svm.py  first."
        )

    q3 = load_json(METRICS_Q3)
    q1 = load_json(METRICS_Q1)

    all_results = q3["all_results"]
    per_method_summary = q3["per_method_summary"]
    baseline_acc = get_baseline_accuracy(q1)
    print(f"  Baseline accuracy : {baseline_acc:.6f}")
    print(f"  Total result rows : {len(all_results)}")

    curves = extract_method_curves(all_results)

    print("\nGenerating figures …")

    # Figure 1 – accuracy vs. dimensionality (main comparison)
    plot_accuracy_vs_dim(curves, baseline_acc)

    # Figure 2 – method summary bar + efficiency scatter
    plot_method_summary(all_results, per_method_summary, baseline_acc)

    # Figure 3 – PCA variance analysis
    plot_pca_variance(curves, PCA_VAR_NPY, baseline_acc)

    # Figure 4 – sparse coding diagnostics
    plot_sparse_coding_diagnostics(all_results)

    # Figure 5 – Q4 optimal analysis + per-class baseline accuracy
    plot_optimal_perclass(q1, all_results, per_method_summary)

    # Figure 6 – full accuracy heatmap
    plot_accuracy_heatmap(all_results, baseline_acc)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
