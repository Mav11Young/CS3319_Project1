from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT.parent / "outputs_q1_q2" / "metrics_q1_q2.json"
FIG_DIR = ROOT / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def class_records(metrics: dict) -> list[dict]:
    per_class = metrics["per_class_test_accuracy"]
    report = metrics["classification_report"]
    records = []
    class_names = list(per_class.keys())
    for idx, name in enumerate(class_names, start=1):
        stats = report[str(idx)]
        records.append(
            {
                "name": name,
                "accuracy": float(per_class[name]),
                "support": int(stats["support"]),
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "f1": float(stats["f1-score"]),
            }
        )
    return records


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_overview(metrics: dict, records: list[dict]) -> None:
    train_size = metrics["train_size"]
    test_size = metrics["test_size"]
    total = metrics["n_samples_total"]
    values = np.array([r["accuracy"] for r in records], dtype=float)
    summary_values = [
        metrics["cv_best_accuracy"],
        metrics["test_accuracy"],
        metrics["classification_report"]["macro avg"]["f1-score"],
        metrics["classification_report"]["weighted avg"]["f1-score"],
    ]
    summary_labels = ["CV Acc.", "Test Acc.", "Macro F1", "Weighted F1"]
    summary_colors = ["#7c3aed", "#0f766e", "#2563eb", "#dc2626"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].bar(
        ["Train", "Test"],
        [train_size, test_size],
        color=["#2563eb", "#f59e0b"],
        width=0.56,
    )
    axes[0].set_title("Dataset Split")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_ylim(0, max(train_size, test_size) * 1.2)
    axes[0].text(0, train_size + total * 0.015, f"{train_size}\n(60%)", ha="center")
    axes[0].text(1, test_size + total * 0.015, f"{test_size}\n(40%)", ha="center")

    axes[1].bar(summary_labels, summary_values, color=summary_colors, width=0.58)
    axes[1].set_title("Core Metrics")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.84, 1.0)
    for i, val in enumerate(summary_values):
        axes[1].text(i, val + 0.004, f"{val:.3f}", ha="center", va="bottom")

    bins = np.linspace(0.4, 1.0, 13)
    axes[2].hist(values, bins=bins, color="#0891b2", edgecolor="white")
    axes[2].axvline(values.mean(), color="#dc2626", linewidth=2, linestyle="--")
    axes[2].axvline(np.median(values), color="#7c3aed", linewidth=2, linestyle=":")
    axes[2].set_title("Distribution of Per-class Accuracy")
    axes[2].set_xlabel("Per-class accuracy")
    axes[2].set_ylabel("Number of classes")
    axes[2].text(values.mean() + 0.005, 8.6, f"mean={values.mean():.3f}", color="#dc2626")
    axes[2].text(np.median(values) + 0.005, 7.2, f"median={np.median(values):.3f}", color="#7c3aed")

    fig.suptitle(
        f"Linear SVM on AwA2 Deep Features  |  selected C = {metrics['selected_C']:.2f}  |  runtime = {metrics['elapsed_seconds'] / 60:.1f} min",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, "q1_q2_overview")


def plot_ranked_accuracy(records: list[dict]) -> None:
    ranked = sorted(records, key=lambda r: r["accuracy"])
    names = [r["name"] for r in ranked]
    values = np.array([r["accuracy"] for r in ranked], dtype=float)
    supports = np.array([r["support"] for r in ranked], dtype=int)
    colors = plt.cm.viridis((values - values.min()) / (values.max() - values.min() + 1e-8))

    fig, ax = plt.subplots(figsize=(11, 10.5))
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, edgecolor="none", height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("Per-class accuracy")
    ax.set_title("Class-wise Test Accuracy Ranked from Hardest to Easiest", fontweight="bold")
    ax.axvline(values.mean(), color="#dc2626", linestyle="--", linewidth=1.7, label=f"Mean = {values.mean():.3f}")
    ax.legend(loc="lower right")

    for idx in list(range(5)) + list(range(len(names) - 5, len(names))):
        ax.text(
            min(values[idx] + 0.01, 1.005),
            idx,
            f"{values[idx]:.3f} (n={supports[idx]})",
            va="center",
            fontsize=8,
        )

    ax.text(0.36, -1.7, "Lower scores often occur on fine-grained or visually ambiguous animal categories.", fontsize=9, color="#374151")
    save_figure(fig, "q1_q2_ranked_accuracy")


def plot_support_vs_accuracy(records: list[dict]) -> None:
    supports = np.array([r["support"] for r in records], dtype=float)
    accuracies = np.array([r["accuracy"] for r in records], dtype=float)
    f1_scores = np.array([r["f1"] for r in records], dtype=float)
    names = [r["name"] for r in records]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    sc = ax.scatter(
        supports,
        accuracies,
        c=f1_scores,
        s=55 + supports * 0.35,
        cmap="plasma",
        alpha=0.85,
        edgecolors="white",
        linewidth=0.6,
    )
    ax.set_title("Relationship Between Test-set Support and Class Accuracy", fontweight="bold")
    ax.set_xlabel("Number of test samples in class")
    ax.set_ylabel("Per-class accuracy")
    ax.set_ylim(0.4, 1.02)

    coeff = np.corrcoef(supports, accuracies)[0, 1]
    ax.text(
        0.02,
        0.06,
        f"Pearson correlation = {coeff:.3f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db"},
    )

    worst_idx = np.argsort(accuracies)[:6]
    best_idx = np.argsort(accuracies)[-4:]
    for idx in list(worst_idx) + list(best_idx):
        ax.annotate(
            names[idx],
            (supports[idx], accuracies[idx]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Per-class F1 score")
    save_figure(fig, "q1_q2_support_vs_accuracy")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    setup_style()
    metrics = load_metrics()
    records = class_records(metrics)
    plot_overview(metrics, records)
    plot_ranked_accuracy(records)
    plot_support_vs_accuracy(records)
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
