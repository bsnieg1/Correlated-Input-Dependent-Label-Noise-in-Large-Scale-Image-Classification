"""Figures saved for every run (PNG, 150 dpi). Uses a non-interactive backend."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(epochs_csv, path, best_epoch=None):
    """Loss, noisy-val top-1, and train-set agreement with given vs true labels (memorization)."""
    df = pd.read_csv(epochs_csv)
    noisy = df["memorized"].notna().any()
    fig, axes = plt.subplots(1, 4 if noisy else 3, figsize=(17 if noisy else 13, 3.6))
    panels = [
        ("Train loss", [("train_loss", "train loss")]),
        ("Val top-1 (noisy val labels)", [("val_top1", "val top-1")]),
        ("Train set: prediction = label", [("train_acc_given", "given (noisy) label"),
                                           ("train_acc_clean", "true label")]),
    ]
    if noisy:
        panels.append(("Mislabeled train examples", [("memorized", "predicts wrong given label"),
                                                    ("corrected", "predicts true label")]))
    for ax, (title, cols) in zip(axes, panels):
        for col, label in cols:
            ax.plot(df["epoch"], df[col], marker="o", ms=3, label=label)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch ({best_epoch})")
        ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    if noisy:
        axes[3].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(reliability, path, title):
    """Reliability diagram; reliability: {mode: [(confidence, accuracy, count), ...]}."""
    labels = {"mc": "MC mean (default)", "mu": "softmax(mu) (HET, no noise)"}
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfect calibration")
    for mode, rows in reliability.items():
        conf = np.array([r[0] for r in rows], dtype=float)
        acc = np.array([r[1] for r in rows], dtype=float)
        ok = ~np.isnan(conf)
        ax.plot(conf[ok], acc[ok], marker="o", ms=4, label=labels.get(mode, mode))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_scores(score, is_noisy, path, title):
    """Histograms of -log p(given label | x) for correctly and wrongly labeled training examples."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, np.quantile(score, 0.999), 60)
    ax.hist(score[~is_noisy], bins=bins, alpha=0.6, density=True, label="correct label")
    ax.hist(score[is_noisy], bins=bins, alpha=0.6, density=True, label="wrong label")
    ax.set_yscale("log")
    ax.set_xlabel("-log p(given label | x)")
    ax.set_ylabel("density (log)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transition(T, path, title, class_names=None):
    _matrix_plot(np.asarray(T, dtype=float), path, title, "Oranges", 0.0, 1.0, ".2f", class_names,
                 "given label", "true class")


def _matrix_plot(m, path, title, cmap, vmin, vmax, fmt, class_names=None, xlabel="", ylabel="",
                 show_all=False):
    k = m.shape[0]
    names = class_names or [str(i) for i in range(k)]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(k), names, rotation=45 if class_names else 0, ha="right" if class_names else "center")
    ax.set_yticks(range(k), names)
    for i in range(k):
        for j in range(k):
            v = m[i, j]
            if show_all or abs(v) >= 0.005 or i == j:
                # dark cells: far from the colormap's light end (vmin for sequential, 0 for diverging)
                light = 0.0 if vmin < 0 else vmin
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="white" if abs(v - light) > 0.55 * (vmax - light) else "black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title, class_names=None):
    """Row-normalized confusion matrix (rows: true class, columns: predicted)."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    _matrix_plot(norm, path, title, "Blues", 0.0, 1.0, ".2f", class_names, "predicted", "true")


def plot_correlation(corr, path, title, class_names=None):
    """Average class correlation from HET covariance; strongly negative = classes swap."""
    corr = np.asarray(corr, dtype=float)
    _matrix_plot(corr, path, title, "RdBu_r", -1.0, 1.0, ".2f", class_names, show_all=True)
