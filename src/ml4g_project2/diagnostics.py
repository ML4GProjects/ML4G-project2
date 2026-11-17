"""Error diagnostics for the patient-grouped CV in sc_classify.py: per-class Brier contribution + confusion matrix.

Run from the repo root: uv run python -m ml4g_project2.diagnostics
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ml4g_project2.sc_classify import run_cv

PLOTS = Path("plots")


def plot_errors(y, groups, classes, oof, patient="P2"):
    """Per-class Brier contribution per held-out patient + confusion matrix for one patient."""
    onehot = (y[:, None] == classes).astype(float)
    sq = pd.DataFrame((oof - onehot) ** 2, columns=classes)
    contrib = sq.groupby(groups).mean()  # rows sum to each patient's Brier score

    m = groups == patient
    cm = confusion_matrix(y[m], classes[oof[m].argmax(axis=1)], labels=classes)
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm_frac = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)  # row-normalized; empty rows stay 0

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.heatmap(contrib, annot=True, fmt=".3f", cmap="Reds", ax=axes[0])
    axes[0].set(title="Brier contribution per cell type (rows sum to patient's Brier)",
                xlabel="cell type", ylabel="held-out patient")
    sns.heatmap(cm_frac, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, ax=axes[1])
    axes[1].set(title=f"{patient} confusion matrix (color = row fraction, number = cells)",
                xlabel="predicted", ylabel="true")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "cv_rf_errors.png", dpi=150, bbox_inches="tight")
    print(f"saved {PLOTS / 'cv_rf_errors.png'}")


if __name__ == "__main__":
    plot_errors(*run_cv())
