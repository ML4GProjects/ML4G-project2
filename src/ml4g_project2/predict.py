"""Full pipeline: train on all training cells, predict test cells and test bulk samples, write the submission.

1. Normalize counts: log1p(counts per 10k) per cell.
2. Pick the top N_MARKERS marker genes per cell type on the training cells (scanpy rank_genes_groups).
3. Train a multinomial logistic regression (C=C_LR) on those genes; predict a probability
   distribution for every test cell.
4. Bulk deconvolution: each test bulk sample gets the mean predicted distribution of the test cells
   from its single-cell sample (all replicates of a sample get the same prediction).
5. Write results/submission.csv with the same ids and columns as sample_submission.csv.

Before predicting, cv_score() reports the competition score S on leave-one-patient-out CV.
N_MARKERS and C_LR were chosen by a leave-one-patient-out sweep of single-cell Brier (one-SE rule, paired).
Run from the repo root: uv run python -m ml4g_project2.predict
"""
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ml4g_project2.sc_classify import DATA, lognorm, make_lr, marker_genes, run_cv

RESULTS = Path("results")
N_MARKERS = 100  # marker genes per cell type
C_LR = 0.003  # inverse regularization strength of the logistic regression
make_model = partial(make_lr, C=C_LR)


def brier(pred, target):
    """Per-row Brier score as in the task: sum over cell types of squared differences."""
    return ((np.asarray(pred) - np.asarray(target)) ** 2).sum(axis=1)


def cv_score(make_model=make_model, n_markers=N_MARKERS):
    """Competition score S = 1/2 Bcell/B0cell + 1/2 Bbulk/B0bulk, on leave-one-patient-out CV.

    The references mimic the task's: B0cell predicts the training-fold cell-type frequencies for every
    held-out cell, B0bulk predicts the mean bulk proportions of the other patients.
    """
    y, groups, classes, oof = run_cv(n_markers, make_model=make_model)
    sample = ad.read_h5ad(DATA / "train_sc.h5ad", backed="r").obs["sample"].astype(str).to_numpy()
    onehot = (y[:, None] == classes).astype(float)
    bulk = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
    meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[bulk.index]
    sample_pred = pd.DataFrame(oof, columns=classes).groupby(sample).mean()

    rows, cell_b, cell_b0, bulk_b, bulk_b0 = {}, [], [], [], []
    for g in np.unique(groups):
        te, rep_ = groups == g, (meta["patient"] == g).to_numpy()
        cb, cb0 = brier(oof[te], onehot[te]), brier(onehot[~te].mean(axis=0)[None], onehot[te])
        bb = brier(sample_pred.loc[meta.loc[rep_, "sample"]], bulk[rep_])
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

    # 4: bulk deconvolution from the test cells of the same sample
    per_sample = cells.groupby(test.obs["sample"].astype(str).to_numpy()).mean()
    bulk_meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id")
    bulk_ids = bulk_meta.index[bulk_meta["split"] == "test"]
    bulk = per_sample.loc[bulk_meta.loc[bulk_ids, "sample"]].set_axis(bulk_ids)

    # 5: match the template by id, not by position
    sub = pd.concat([cells, bulk]).reindex(index=template.index, columns=template.columns)
    assert not sub.isna().any().any(), "missing predictions for some ids or cell types"
    assert (sub.sum(axis=1) > 0).all(), "a row sums to zero"

    RESULTS.mkdir(exist_ok=True)
    sub.index.name = "id"
    sub.to_csv(RESULTS / "submission.csv")
    print(f"saved {RESULTS / 'submission.csv'}: {len(cells)} cells + {len(bulk)} bulk rows")
    print("\npredicted bulk proportions per test sample:")
    print(per_sample[template.columns].round(3).to_string())


if __name__ == "__main__":
    main()
