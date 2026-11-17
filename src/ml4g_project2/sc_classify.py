"""Patient-grouped stratified k-fold CV of a random forest cell-type classifier, scored by Brier score.

Run from the repo root: uv run python -m ml4g_project2.sc_classify
"""
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

DATA = Path("data")
N_SPLITS = 5  # 5 train patients -> each fold holds out exactly one patient


def lognorm(X, target_sum=1e4):
    """log1p(counts per 10k), same as normalize_sc in data_exploration.ipynb."""
    lib = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)
    return (sp.diags(target_sum / lib) @ X).tocsr().log1p()


def run_cv():
    """Print per-fold Brier scores; return labels, patient groups, classes and out-of-fold probabilities."""
    adata = ad.read_h5ad(DATA / "train_sc.h5ad")
    X = lognorm(adata.X)
    y = adata.obs["cell_type"].astype(str).to_numpy()
    groups = adata.obs["patient"].astype(str).to_numpy()
    classes = np.unique(y)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    scores = []
    oof = np.zeros((len(y), len(classes)))  # out-of-fold probabilities
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        rf.fit(X[tr], y[tr])

        # a rare cell type can be missing from a training fold -> align columns to all classes
        proba = np.zeros((len(te), len(classes)))
        proba[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(X[te])
        oof[te] = proba

        score = brier_score_loss(y[te], proba, labels=classes)
        scores.append(score)
        print(f"fold {fold} | held-out patient(s) {sorted(set(groups[te]))} | "
              f"n={len(te)} | brier={score:.4f}")

    print(f"\nBrier score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return y, groups, classes, oof


def main():
    run_cv()


if __name__ == "__main__":
    main()
