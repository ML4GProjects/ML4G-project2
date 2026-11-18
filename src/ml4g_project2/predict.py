"""Full pipeline: train on all training cells, predict test cells and test bulk samples, write the submission.

1. Normalize counts: log1p(counts per 10k) per cell.
2. Pick the top N_MARKERS marker genes per cell type on the training cells (scanpy rank_genes_groups).
3. Train a multinomial logistic regression (C=C_LR) on those genes; predict a probability
   distribution for every test cell.
4. Bulk deconvolution, per bulk replicate: build a signature matrix from the cells of its own sample
   (mean raw counts per cell of each type, cells weighted by predicted probabilities), shrunk towards
   the training-cell signature by ALPHA pseudo-cells; NNLS of the bulk counts on it (marker genes,
   each gene scaled by its mean signature value); coefficients ~ cell numbers -> proportions.
5. Write results/submission.csv with the same ids and columns as sample_submission.csv.

Before predicting, cv_score() reports the competition score S on leave-one-patient-out CV.
N_MARKERS and C_LR were chosen by a leave-one-patient-out sweep of single-cell Brier, ALPHA by the bulk
experiment in experiments/bulk_nnls.py (both with the paired one-SE rule; see journal.md).
Run from the repo root: uv run python -m ml4g_project2.predict
"""
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ml4g_project2.sc_classify import DATA, lognorm, make_lr, marker_genes, run_cv

RESULTS = Path("results")
N_MARKERS = 100  # marker genes per cell type
C_LR = 0.003  # inverse regularization strength of the logistic regression
make_model = partial(make_lr, C=C_LR)
ALPHA = 50  # shrinkage of sample-specific signatures towards the training signature, in pseudo-cells


def brier(pred, target):
    """Per-row Brier score as in the task: sum over cell types of squared differences."""
    return ((np.asarray(pred) - np.asarray(target)) ** 2).sum(axis=1)


def signature(counts, W):
    """Genes x types: mean raw counts per cell of each type, cells weighted by W; plus effective cell numbers."""
    n = W.sum(axis=0)
    return np.asarray(counts.T @ W) / np.maximum(n, 1e-9), n


def deconvolve_bulk(counts, W, cell_sample, bulk_counts, rep_sample, ref, alpha=ALPHA):
    """Proportions (replicates x types) by NNLS on sample-specific signatures shrunk towards ref.

    counts: cells x marker genes (raw); W: cells x types (predicted probabilities);
    bulk_counts: marker genes x replicates (raw); ref: genes x types reference signature.
    """
    out = np.zeros((bulk_counts.shape[1], W.shape[1]))
    for r in range(bulk_counts.shape[1]):
        cells = cell_sample == rep_sample[r]
        S, n = signature(counts[cells], W[cells])
        S = (S * n + ref * alpha) / (n + alpha)
        w = 1 / np.maximum(S.mean(axis=1), 1e-9)  # relative error per gene
        b = bulk_counts[:, r].astype(float)
        x, _ = nnls(S * w[:, None], b / b.sum() * S.sum() / S.shape[1] * w)
        out[r] = x / x.sum() if x.sum() > 0 else 1 / len(x)
    return out


def cv_score(make_model=make_model, n_markers=N_MARKERS):
    """Competition score S = 1/2 Bcell/B0cell + 1/2 Bbulk/B0bulk, on leave-one-patient-out CV.

    The references mimic the task's: B0cell predicts the training-fold cell-type frequencies for every
    held-out cell, B0bulk predicts the mean bulk proportions of the other patients.
    """
    y, groups, classes, oof = run_cv(n_markers, make_model=make_model)
    train = ad.read_h5ad(DATA / "train_sc.h5ad")
    counts, X = train.X.tocsr().astype(float), lognorm(train.X)
    sample = train.obs["sample"].astype(str).to_numpy()
    onehot = (y[:, None] == classes).astype(float)
    bulk = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
    meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[bulk.index]
    bulk_counts = pd.read_csv(DATA / "train_bulk.csv", index_col="gene")[bulk.index]
    assert (bulk_counts.index == train.var_names).all(), "bulk and sc genes differ"

    rows, cell_b, cell_b0, bulk_b, bulk_b0 = {}, [], [], [], []
    for g in np.unique(groups):
        te, rep_ = groups == g, (meta["patient"] == g).to_numpy()
        cb, cb0 = brier(oof[te], onehot[te]), brier(onehot[~te].mean(axis=0)[None], onehot[te])
        genes = marker_genes(X[~te], y[~te], n_markers)  # same markers as run_cv's fold (deterministic)
        ref, _ = signature(counts[~te][:, genes], onehot[~te])  # other patients, true labels
        pred = deconvolve_bulk(counts[te][:, genes], oof[te], sample[te],
                               bulk_counts.to_numpy()[genes][:, rep_], meta["sample"].to_numpy()[rep_], ref)
        bb = brier(pred, bulk[rep_])
        bb0 = brier(bulk[~rep_].mean(axis=0).to_numpy()[None], bulk[rep_])
        cell_b.append(cb); cell_b0.append(cb0); bulk_b.append(bb); bulk_b0.append(bb0)
        rows[g] = {"B_cell": cb.mean(), "B0_cell": cb0.mean(), "B_bulk": bb.mean(), "B0_bulk": bb0.mean()}

    rows["pooled"] = {"B_cell": np.concatenate(cell_b).mean(), "B0_cell": np.concatenate(cell_b0).mean(),
                      "B_bulk": np.concatenate(bulk_b).mean(), "B0_bulk": np.concatenate(bulk_b0).mean()}
    res = pd.DataFrame(rows).T
    res["S"] = 0.5 * res["B_cell"] / res["B0_cell"] + 0.5 * res["B_bulk"] / res["B0_bulk"]
    print("\nLeave-one-patient-out CV, competition metric (lower is better; pass: S <= 0.30144):")
    print(res.round(4).to_string())
    return res


def main():
    cv_score()

    train = ad.read_h5ad(DATA / "train_sc.h5ad")
    test = ad.read_h5ad(DATA / "test_sc.h5ad")
    assert (train.var_names == test.var_names).all(), "train and test genes differ"
    template = pd.read_csv(DATA / "sample_submission.csv", index_col="id")

    # 1-3: cell-type annotation
    X_train, X_test = lognorm(train.X), lognorm(test.X)
    y = train.obs["cell_type"].astype(str).to_numpy()
    genes = marker_genes(X_train, y, N_MARKERS)
    print(f"{len(genes)} marker genes ({N_MARKERS} per cell type)")

    model = make_model()
    model.fit(X_train[:, genes], y)
    cells = pd.DataFrame(model.predict_proba(X_test[:, genes]), index=test.obs_names, columns=model.classes_)

    # 4: bulk deconvolution with sample-specific signatures from the test cells
    classes = model.classes_
    onehot = (y[:, None] == classes).astype(float)
    ref, _ = signature(train.X.tocsr().astype(float)[:, genes], onehot)  # all training cells, true labels
    bulk_meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id")
    bulk_ids = bulk_meta.index[bulk_meta["split"] == "test"]
    test_bulk = pd.read_csv(DATA / "test_bulk.csv", index_col="gene")[bulk_ids]
    assert (test_bulk.index == test.var_names).all(), "bulk and sc genes differ"
    props = deconvolve_bulk(test.X.tocsr().astype(float)[:, genes], cells.to_numpy(),
                            test.obs["sample"].astype(str).to_numpy(), test_bulk.to_numpy()[genes],
                            bulk_meta.loc[bulk_ids, "sample"].to_numpy(), ref)
    bulk = pd.DataFrame(props, index=bulk_ids, columns=classes)

    # 5: match the template by id, not by position
    sub = pd.concat([cells, bulk]).reindex(index=template.index, columns=template.columns)
    assert not sub.isna().any().any(), "missing predictions for some ids or cell types"
    assert (sub.sum(axis=1) > 0).all(), "a row sums to zero"

    RESULTS.mkdir(exist_ok=True)
    sub.index.name = "id"
    sub.to_csv(RESULTS / "submission.csv")
    print(f"saved {RESULTS / 'submission.csv'}: {len(cells)} cells + {len(bulk)} bulk rows")
    print("\npredicted bulk proportions per test replicate:")
    print(bulk[template.columns].round(3).to_string())


if __name__ == "__main__":
    main()
