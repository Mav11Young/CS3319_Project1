from __future__ import annotations

"""
AwA2 Q1/Q2 — Report Figure Generator
=====================================
Generates three publication-quality figures:

  q1_q2_overview.pdf        — full-width 3-panel summary figure (NEW design)
  q1_q2_ranked_accuracy.pdf — per-class accuracy ranked bar chart
  q1_q2_support_vs_accuracy.pdf — support vs. accuracy scatter plot

Run from the course_project_report_template/ directory:
    python generate_report_figures.py
"""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT.parent / "outputs_q1_q2" / "metrics_q1_q2.json"
FIG_DIR = ROOT / "figures"

# ── Unified colour palette ─────────────────────────────────────────────────────
C_TRAIN = "#2563EB"  # vivid blue   – training set
C_TEST = "#D97706"  # amber        – test set
C_TESTBAR = "#1D4ED8"  # dark blue    – test accuracy bar
C_CVBAR = "#0F766E"  # teal         – cv accuracy bar
C_WF1 = "#0369A1"  # steel blue   – weighted F1 bar
C_MF1 = "#7C3AED"  # violet       – macro F1 bar
C_MEAN = "#1D4ED8"  # blue         – mean line
C_MEDIAN = "#7C3AED"  # violet       – median line
C_KDE = "#1E293B"  # near-black   – KDE curve

# Performance-zone colours (histogram bars + shading)
ZONES = [
    (0.40, 0.70, "#EF4444", "Struggling"),
    (0.70, 0.85, "#F59E0B", "Moderate"),
    (0.85, 0.95, "#0D9488", "Strong"),
    (0.95, 1.02, "#16A34A", "Excellent"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Global style
# ═══════════════════════════════════════════════════════════════════════════════


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.framealpha": 0.93,
            "legend.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def class_records(metrics: dict) -> list:
    per_class = metrics["per_class_test_accuracy"]
    report = metrics["classification_report"]
    records = []
    for idx, name in enumerate(per_class.keys(), start=1):
        s = report[str(idx)]
        records.append(
            {
                "name": name,
                "accuracy": float(per_class[name]),
                "support": int(s["support"]),
                "precision": float(s["precision"]),
                "recall": float(s["recall"]),
                "f1": float(s["f1-score"]),
            }
        )
    return records


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{name}.pdf + .png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: draw a rounded rectangle using a Polygon
#  (works on any matplotlib version without FancyBboxPatch quirks)
# ═══════════════════════════════════════════════════════════════════════════════


def _rounded_rect(
    ax, x, y, w, h, r, *, fc, ec="none", lw=0.8, transform=None, zorder=2
):
    """Draw a rounded rectangle as a Polygon patch."""
    from matplotlib.patches import FancyBboxPatch

    t = transform if transform is not None else ax.transData
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        transform=t,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(patch)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 – Overview  (3-panel, full-width)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_overview(metrics: dict, records: list) -> None:
    """
    Full-width three-panel overview figure:
      Panel A  – Dataset split bar + key stat cards
      Panel B  – Horizontal metric bars with reference lines
      Panel C  – Zone-coloured per-class accuracy histogram + KDE + mean/median
    """
    # ── Extract data ──────────────────────────────────────────────────────────
    train_size = metrics["train_size"]
    test_size = metrics["test_size"]
    n_total = metrics["n_samples_total"]
    n_cls = metrics["n_classes"]
    best_c = metrics["selected_C"]
    test_acc = metrics["test_accuracy"]
    cv_acc = metrics["cv_best_accuracy"]
    report = metrics["classification_report"]
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    values = np.array([r["accuracy"] for r in records], dtype=float)
    mean_acc = float(values.mean())
    med_acc = float(np.median(values))

    # ── Figure & GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 4.4), facecolor="white")
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.15, 1.25, 1.60],
        left=0.04,
        right=0.97,
        top=0.87,
        bottom=0.13,
        wspace=0.38,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Global suptitle ───────────────────────────────────────────────────────
    fig.text(
        0.505,
        0.975,
        f"Baseline Experiment: Linear SVM (C = {best_c}) on AwA2 ResNet-101 Features  "
        f"—  Test Accuracy = {test_acc:.4f}",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#111827",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A – Dataset overview
    # ══════════════════════════════════════════════════════════════════════════
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Dataset: AwA2", fontweight="bold", fontsize=11, pad=6)

    # ── Proportional split bar ────────────────────────────────────────────────
    bar_y = 0.76
    bar_h = 0.15
    t_frac = train_size / n_total  # 0.60

    # train block
    _rounded_rect(
        ax1, 0.0, bar_y, t_frac, bar_h, r=0.02, fc=C_TRAIN, transform=ax1.transAxes
    )
    # test block
    _rounded_rect(
        ax1,
        t_frac,
        bar_y,
        1 - t_frac,
        bar_h,
        r=0.02,
        fc=C_TEST,
        transform=ax1.transAxes,
    )

    # labels inside bar
    ax1.text(
        t_frac / 2,
        bar_y + bar_h / 2,
        "Train  60 %",
        ha="center",
        va="center",
        transform=ax1.transAxes,
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=5,
    )
    ax1.text(
        t_frac + (1 - t_frac) / 2,
        bar_y + bar_h / 2,
        "Test  40 %",
        ha="center",
        va="center",
        transform=ax1.transAxes,
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=5,
    )

    # ── Stat cards (2 × 2 grid) ───────────────────────────────────────────────
    stat_data = [
        (f"{n_total:,}", "Total Images", C_TRAIN),
        (f"{n_cls}", "Animal Classes", "#6D28D9"),
        (f"{train_size:,}", "Training Samples", C_TRAIN),
        (f"{test_size:,}", "Testing Samples", C_TEST),
    ]

    col_xs = [0.245, 0.755]
    row_ys = [0.54, 0.24]
    card_w, card_h = 0.46, 0.21

    for i, (val, label, color) in enumerate(stat_data):
        cx = col_xs[i % 2]
        cy = row_ys[i // 2]
        x0 = cx - card_w / 2
        y0 = cy - card_h / 2

        # card background
        _rounded_rect(
            ax1,
            x0,
            y0,
            card_w,
            card_h,
            r=0.01,
            fc="#F8FAFC",
            ec="#E2E8F0",
            lw=0.9,
            transform=ax1.transAxes,
        )
        # value
        ax1.text(
            cx,
            cy + 0.032,
            val,
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        # label
        ax1.text(
            cx,
            cy - 0.052,
            label,
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=7.5,
            color="#64748B",
        )

    # feature-dim note at bottom
    ax1.text(
        0.5,
        0.025,
        "Feature: 2,048-d ResNet-101  |  Seed: 42",
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
        fontsize=7.8,
        color="#94A3B8",
        style="italic",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B – Performance metrics  (horizontal bars)
    # ══════════════════════════════════════════════════════════════════════════
    metric_labels = ["Macro F1", "Weighted F1", "CV Accuracy", "Test Accuracy"]
    metric_values = [macro_f1, weighted_f1, cv_acc, test_acc]
    bar_colors = [C_MF1, C_WF1, C_CVBAR, C_TESTBAR]

    y_pos = np.arange(len(metric_labels))

    # grey background track
    ax2.barh(y_pos, [1.0] * 4, height=0.54, color="#F1F5F9", zorder=1, left=0.0)

    # coloured value bar
    bars = ax2.barh(
        y_pos, metric_values, height=0.54, color=bar_colors, alpha=0.92, zorder=3
    )

    # value labels to the right of each bar
    for bar, val, col in zip(bars, metric_values, bar_colors):
        ax2.text(
            min(val + 0.0015, 0.977),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=col,
        )

    # vertical reference lines
    for ref in (0.90, 0.95):
        ax2.axvline(ref, color="#CBD5E1", linestyle="--", linewidth=1.0, zorder=2)
        ax2.text(
            ref,
            len(metric_labels) - 0.42,
            f"{int(ref * 100)} %",
            ha="center",
            fontsize=7.5,
            color="#94A3B8",
        )

    ax2.set_xlim(0.83, 0.975)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metric_labels, fontsize=9.5)
    ax2.set_xlabel("Score", fontsize=9.5)
    ax2.set_title("Classification Performance", fontweight="bold", fontsize=11, pad=6)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.grid(axis="x", alpha=0.13, linestyle="--", zorder=0)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel C – Per-class accuracy distribution
    # ══════════════════════════════════════════════════════════════════════════
    bins = np.linspace(0.38, 1.02, 17)
    bw = bins[1] - bins[0]
    n_counts, edges = np.histogram(values, bins=bins)
    max_count = int(n_counts.max())

    # ── Zone shading ──────────────────────────────────────────────────────────
    for z_lo, z_hi, z_col, z_name in ZONES:
        ax3.axvspan(z_lo, min(z_hi, 1.02), alpha=0.07, color=z_col, zorder=1, lw=0)
        mid = (z_lo + min(z_hi, 1.01)) / 2
        ax3.text(
            mid,
            max_count * 1.065,
            z_name,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=z_col,
        )

    # ── Zone-coloured histogram bars ──────────────────────────────────────────
    for left, count in zip(edges[:-1], n_counts):
        if count == 0:
            continue
        mid = left + bw / 2
        bar_color = "#94A3B8"
        for z_lo, z_hi, z_col, _ in ZONES:
            if z_lo <= mid < z_hi:
                bar_color = z_col
                break
        ax3.bar(
            mid,
            count,
            width=bw * 0.84,
            color=bar_color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax3.text(
            mid,
            count + 0.12,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#374151",
            zorder=4,
        )

    # ── KDE overlay (scipy optional) ──────────────────────────────────────────
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values, bw_method=0.22)
        x_kde = np.linspace(0.38, 1.02, 400)
        y_kde = kde(x_kde)
        scale = max_count / y_kde.max() * 0.80
        ax3.plot(
            x_kde,
            y_kde * scale,
            color=C_KDE,
            linewidth=2.0,
            zorder=6,
            label="KDE",
            alpha=0.85,
        )
    except ImportError:
        pass

    # ── Mean & median lines ───────────────────────────────────────────────────
    ax3.axvline(
        mean_acc,
        color=C_MEAN,
        linewidth=2.0,
        linestyle="--",
        zorder=7,
        label=f"Mean = {mean_acc:.3f}",
    )
    ax3.axvline(
        med_acc,
        color=C_MEDIAN,
        linewidth=2.0,
        linestyle=":",
        zorder=7,
        label=f"Median = {med_acc:.3f}",
    )

    ax3.set_xlim(0.36, 1.04)
    ax3.set_ylim(0, max_count * 1.18)
    ax3.set_xlabel("Per-class test accuracy", fontsize=9.5)
    ax3.set_ylabel("Number of classes", fontsize=9.5)
    ax3.set_title(
        "Per-class Accuracy Distribution  (50 classes)",
        fontweight="bold",
        fontsize=11,
        pad=6,
    )
    ax3.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.93, edgecolor="#CBD5E1")
    ax3.grid(axis="y", alpha=0.15, linestyle="--", zorder=0)

    save_figure(fig, "q1_q2_overview")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 – Per-class accuracy ranked (redesigned)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_ranked_accuracy(records: list) -> None:
    """
    Horizontal bar chart of per-class accuracy, ranked worst → best.
    Bars are coloured by the same four performance zones used in Figure 1.
    Annotations show exact accuracy and test-set size for the 6 hardest
    and 5 easiest classes.
    """
    ranked = sorted(records, key=lambda r: r["accuracy"])
    names = [r["name"] for r in ranked]
    values = np.array([r["accuracy"] for r in ranked], dtype=float)
    supports = np.array([r["support"] for r in ranked], dtype=int)
    n = len(names)
    y = np.arange(n)

    # Per-bar zone colour
    bar_colors = []
    for v in values:
        col = "#94A3B8"
        for z_lo, z_hi, z_col, _ in ZONES:
            if z_lo <= v < z_hi:
                col = z_col
                break
        bar_colors.append(col)

    fig, ax = plt.subplots(figsize=(11, 10.5), facecolor="white")
    ax.set_facecolor("white")

    ax.barh(y, values, color=bar_colors, edgecolor="none", height=0.76, zorder=3)

    # Subtle zone shading in background
    for z_lo, z_hi, z_col, z_name in ZONES:
        ax.axvspan(z_lo, min(z_hi, 1.02), alpha=0.06, color=z_col, zorder=1, lw=0)
        ax.text(
            z_lo + 0.005,
            n + 0.1,
            z_name,
            va="bottom",
            fontsize=8,
            color=z_col,
            fontweight="bold",
        )

    # Mean line
    ax.axvline(
        values.mean(),
        color="#374151",
        linestyle="--",
        linewidth=1.6,
        zorder=5,
        label=f"Mean = {values.mean():.3f}",
    )

    # Annotations: 6 hardest + 5 easiest
    highlight = list(range(6)) + list(range(n - 5, n))
    for idx in highlight:
        x_pos = min(values[idx] + 0.008, 1.002)
        ax.text(
            x_pos,
            idx,
            f"{values[idx]:.3f}  (n = {supports[idx]})",
            va="center",
            fontsize=7.8,
            color="#1E293B",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0.35, 1.06)
    ax.set_xlabel("Per-class test accuracy", fontsize=10)
    ax.set_title(
        "Class-wise Test Accuracy: Ranked from Hardest to Easiest",
        fontweight="bold",
        fontsize=12,
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.15, linestyle="--", zorder=0)

    # Legend patches for zones
    zone_patches = [
        mpatches.Patch(
            facecolor=z_col, alpha=0.75, label=f"{z_name} ({z_lo:.0%}–{z_hi:.0%})"
        )
        for z_lo, z_hi, z_col, z_name in ZONES
        if z_hi <= 1.02
    ]
    leg2 = ax.legend(
        handles=zone_patches,
        loc="lower center",
        bbox_to_anchor=(0.72, 0.01),
        fontsize=8,
        framealpha=0.9,
        title="Performance Zone",
        title_fontsize=8,
    )
    ax.add_artist(leg2)
    # Re-add mean legend
    ax.legend(
        [
            mpatches.Patch(
                facecolor="none",
                edgecolor="#374151",
                linestyle="--",
                label=f"Mean = {values.mean():.3f}",
            )
        ],
        [f"Mean = {values.mean():.3f}"],
        loc="lower right",
        fontsize=9,
    )

    ax.text(
        0.36,
        -1.8,
        "Lower scores concentrate in visually ambiguous or low-support categories.",
        fontsize=8.5,
        color="#64748B",
        style="italic",
    )

    save_figure(fig, "q1_q2_ranked_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 – Test-set support vs. accuracy scatter
# ═══════════════════════════════════════════════════════════════════════════════


def plot_support_vs_accuracy(records: list) -> None:
    """
    Bubble scatter plot: x = test-set support, y = per-class accuracy.
    Bubble size scales with support; colour encodes per-class F1.
    Annotates the 6 weakest and 4 strongest classes.
    """
    supports = np.array([r["support"] for r in records], dtype=float)
    accuracies = np.array([r["accuracy"] for r in records], dtype=float)
    f1_scores = np.array([r["f1"] for r in records], dtype=float)
    names = [r["name"] for r in records]

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        supports,
        accuracies,
        c=f1_scores,
        s=45 + supports * 0.32,
        cmap="plasma",
        vmin=f1_scores.min() - 0.02,
        vmax=f1_scores.max() + 0.01,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.7,
        zorder=4,
    )

    # Horizontal zone bands (background)
    for z_lo, z_hi, z_col, _ in ZONES:
        ax.axhspan(z_lo, min(z_hi, 1.02), alpha=0.05, color=z_col, zorder=1, lw=0)

    # Trend line (linear regression)
    m, b = np.polyfit(supports, accuracies, 1)
    x_line = np.linspace(supports.min(), supports.max(), 200)
    ax.plot(
        x_line,
        m * x_line + b,
        color="#64748B",
        linewidth=1.6,
        linestyle="--",
        zorder=3,
        alpha=0.7,
        label=f"Linear fit  (slope = {m:.5f})",
    )

    # Pearson correlation annotation
    r = float(np.corrcoef(supports, accuracies)[0, 1])
    ax.text(
        0.02,
        0.07,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CBD5E1", lw=0.9),
    )

    # Annotate extremes
    worst_idx = list(np.argsort(accuracies)[:6])
    best_idx = list(np.argsort(accuracies)[-4:])
    for idx in worst_idx + best_idx:
        ax.annotate(
            names[idx],
            xy=(supports[idx], accuracies[idx]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#1E293B",
        )

    ax.set_ylim(0.38, 1.04)
    ax.set_xlabel("Number of test samples in class", fontsize=10)
    ax.set_ylabel("Per-class test accuracy", fontsize=10)
    ax.set_title(
        "Relationship Between Test-set Support and Per-class Accuracy",
        fontweight="bold",
        fontsize=12,
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.13, linestyle="--", zorder=0)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.030)
    cbar.set_label("Per-class F1 score", fontsize=9.5)

    save_figure(fig, "q1_q2_support_vs_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    setup_style()

    print("Loading metrics …")
    metrics = load_metrics()
    records = class_records(metrics)
    print(
        f"  {metrics['n_samples_total']} samples, "
        f"{metrics['n_classes']} classes, "
        f"test_acc = {metrics['test_accuracy']:.4f}"
    )

    print("\nGenerating figures …")
    plot_overview(metrics, records)
    plot_ranked_accuracy(records)
    plot_support_vs_accuracy(records)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
