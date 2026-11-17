"""Patient-grouped stratified k-fold CV of a random forest cell-type classifier, scored by Brier score.

Run from the repo root: uv run python -m ml4g_project2.sc_classify
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

DATA = Path("data")
N_SPLITS = 5  # 5 train patients -> each fold holds out exactly one patient
N_MARKERS = 20  # top marker genes per cell type, picked on the training fold only

# textbook markers (PanglaoDB / CellMarker style); only genes present in the panel.
# missing from panel: KLRF1 NCAM1 CD19 PAX5 SDC1 MYH11 CNN1 DES CDH5 KDR
CANONICAL_MARKERS = {
    "T_cells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "CD8A", "CD4", "IL7R"],
    "NK_cells": ["NKG7", "GNLY", "KLRD1", "FCGR3A", "PRF1", "GZMB"],
    "B_cells": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "Plasmablasts": ["JCHAIN", "MZB1", "IGHG1", "XBP1", "PRDM1"],
    "Myeloid_cells": ["CD14", "LYZ", "CD68", "CSF1R", "ITGAX", "C1QA"],
    "Mast_cells": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC"],
    "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP"],
    "Myofibroblasts": ["ACTA2", "TAGLN", "PDGFRB"],
    "Endothelial_cells": ["PECAM1", "VWF", "CLDN5", "EMCN"],
}


def lognorm(X, target_sum=1e4):
    """log1p(counts per 10k), same as normalize_sc in data_exploration.ipynb."""
    lib = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)
    return (sp.diags(target_sum / lib) @ X).tocsr().log1p()


def marker_genes(X, y, n=N_MARKERS):
    """Column indices of the top-n ranked genes per cell type (scanpy t-test, one-vs-rest)."""
    a = ad.AnnData(X, obs=pd.DataFrame({"y": pd.Categorical(y)}))
    sc.tl.rank_genes_groups(a, "y")
    names = pd.DataFrame(a.uns["rank_genes_groups"]["names"]).head(n)
    return np.unique(names.to_numpy().astype(int))


def run_cv(n_markers=N_MARKERS, fixed_genes=None):
    """Print per-fold Brier scores; return labels, patient groups, classes and out-of-fold probabilities.

    fixed_genes: list of gene names to use in every fold instead of per-fold marker selection.
    """
    adata = ad.read_h5ad(DATA / "train_sc.h5ad")
    X = lognorm(adata.X)
    y = adata.obs["cell_type"].astype(str).to_numpy()
    groups = adata.obs["patient"].astype(str).to_numpy()
    classes = np.unique(y)
    if fixed_genes is not None:
        fixed_idx = adata.var_names.get_indexer(fixed_genes)
        assert (fixed_idx >= 0).all(), "some fixed_genes are not in the panel"

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    scores = []
    oof = np.zeros((len(y), len(classes)))  # out-of-fold probabilities
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        if fixed_genes is not None:
            genes = fixed_idx
        else:
            genes = marker_genes(X[tr], y[tr], n_markers)  # no peeking at the held-out patient
        rf.fit(X[tr][:, genes], y[tr])

        # a rare cell type can be missing from a training fold -> align columns to all classes
        proba = np.zeros((len(te), len(classes)))
        proba[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(X[te][:, genes])
        oof[te] = proba

        score = brier_score_loss(y[te], proba, labels=classes)
        scores.append(score)
        print(f"fold {fold} | held-out patient(s) {sorted(set(groups[te]))} | "
              f"n={len(te)} | genes={len(genes)} | brier={score:.4f}")

    print(f"\nBrier score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return y, groups, classes, oof


def main():
    run_cv()


if __name__ == "__main__":
    main()
