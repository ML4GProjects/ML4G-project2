"""Diagnostics for sc_classify.py: per-class Brier contribution + confusion matrix, and canonical marker dot plots.

Run from the repo root: uv run python -m ml4g_project2.diagnostics
"""
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ml4g_project2.predict import N_MARKERS, make_model
from ml4g_project2.sc_classify import CANONICAL_MARKERS, DATA, lognorm, run_cv

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
    plt.savefig(PLOTS / "cv_errors.png", dpi=150, bbox_inches="tight")
    print(f"saved {PLOTS / 'cv_errors.png'}")


def plot_markers():
    """Canonical marker expression per cell type, and for myofibroblasts split by patient."""
    adata = ad.read_h5ad(DATA / "train_sc.h5ad")
    adata.X = lognorm(adata.X)
    PLOTS.mkdir(exist_ok=True)

    sc.pl.dotplot(adata, CANONICAL_MARKERS, groupby="cell_type", standard_scale="var", show=False)
    plt.savefig(PLOTS / "markers_by_cell_type.png", dpi=150, bbox_inches="tight")

    # are the "myofibroblasts" smooth-muscle-like or T-cell-like, and does that differ by patient?
    genes = {k: CANONICAL_MARKERS[k] for k in ["Myofibroblasts", "Fibroblasts", "T_cells", "NK_cells"]}
    myo = adata[adata.obs["cell_type"] == "Myofibroblasts"]
    sc.pl.dotplot(myo, genes, groupby="patient", show=False)
    plt.savefig(PLOTS / "markers_myofibroblasts_by_patient.png", dpi=150, bbox_inches="tight")
    print(f"saved {PLOTS / 'markers_by_cell_type.png'}, {PLOTS / 'markers_myofibroblasts_by_patient.png'}")


if __name__ == "__main__":
    plot_errors(*run_cv(N_MARKERS, make_model=make_model))  # current predict.py model
    plot_markers()
