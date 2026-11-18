"""Scratch: sample-specific signature matrices + NNLS deconvolution of each bulk replicate.

Per held-out patient (nested leave-one-patient-out):
  genes    = top-100 markers per type, picked on the other patients' cells
  profiles = per sample, mean raw counts per cell of each type, cells weighted by out-of-fold predicted
             probabilities (LR 100/0.003); shrunk towards the other patients' profile with strength alpha
  global   = one signature for every sample: other patients' cells, true labels (alpha -> infinity)
  fit      = NNLS of the bulk counts on the profiles, genes scaled by their mean profile value;
             coefficients ~ cell numbers, normalized to proportions
Compared with the current estimate (mean predicted probabilities per sample) and blends.

Run from the repo root: uv run python experiments/bulk_nnls.py   (needs experiments/out/lr100_oof.npy)
"""
import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import sem

from ml4g_project2.sc_classify import DATA, lognorm, marker_genes

OUT = "experiments/out"
ALPHAS = [0, 10, 50]
BLENDS = [0.25, 0.5, 0.75]

a = ad.read_h5ad(DATA / "train_sc.h5ad")
counts = a.X.tocsr().astype(float)
X = lognorm(a.X)
y = a.obs["cell_type"].astype(str).to_numpy()
sample = a.obs["sample"].astype(str).to_numpy()
cell_pat = a.obs["patient"].astype(str).to_numpy()
classes = np.unique(y)
oof = np.load(f"{OUT}/lr100_oof.npy")
onehot = (y[:, None] == classes).astype(float)

props = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[props.index]
bulk = pd.read_csv(DATA / "train_bulk.csv", index_col="gene")[props.index]
assert (bulk.index == a.var_names).all(), "bulk and sc genes differ"
Y, rep_pat = props.to_numpy(), meta["patient"].to_numpy()


def brier(p, t):
    return ((np.asarray(p) - np.asarray(t)) ** 2).sum(axis=1)


def profiles(cells, W, genes):
    """Weighted mean raw counts per cell, per type (genes x types); also effective cell numbers."""
    C = counts[cells][:, genes]
    n = W.sum(axis=0)
    return np.asarray(C.T @ W) / np.maximum(n, 1e-9), n


def deconvolve(b, S):
    w = 1 / np.maximum(S.mean(axis=1), 1e-9)  # relative error per gene
    x, _ = nnls(S * w[:, None], b / b.sum() * S.sum() / S.shape[1] * w)
    return x / x.sum() if x.sum() > 0 else np.full(len(x), 1 / len(x))


preds = {f"nnls a={al}": np.zeros_like(Y) for al in ALPHAS}
preds["nnls oracle labels"] = np.zeros_like(Y)
preds["nnls global signature"] = np.zeros_like(Y)
preds["current (mean proba)"] = np.zeros_like(Y)
for g in np.unique(rep_pat):
    tr = cell_pat != g
    genes = marker_genes(X[tr], y[tr], 100)
    glob, _ = profiles(np.where(tr)[0], onehot[tr], genes)  # other patients, true labels
    for r in np.where(rep_pat == g)[0]:
        cells = np.where(sample == meta["sample"].iloc[r])[0]
        b = bulk.iloc[:, r].to_numpy()[genes].astype(float)
        S, n = profiles(cells, oof[cells], genes)
        for al in ALPHAS:
            S_al = (S * n + glob * al) / (n + al) if al else S
            preds[f"nnls a={al}"][r] = deconvolve(b, S_al)
        S_or, n_or = profiles(cells, onehot[cells], genes)
        S_or = (S_or * n_or + glob * 10) / (n_or + 10)
        preds["nnls oracle labels"][r] = deconvolve(b, S_or)
        preds["current (mean proba)"][r] = oof[cells].mean(axis=0)
        preds["nnls global signature"][r] = deconvolve(b, glob)  # one signature from all other patients' cells
for bl in BLENDS:
    preds[f"blend {bl} nnls a=10"] = bl * preds["nnls a=10"] + (1 - bl) * preds["current (mean proba)"]

P = sorted(set(rep_pat))
base = pd.Series(brier(preds["current (mean proba)"], Y)).groupby(rep_pat).mean()
rows = []
for name, p in preds.items():
    per = pd.Series(brier(p, Y)).groupby(rep_pat).mean()
    d = per - base
    rows.append({"method": name, "B_bulk": brier(p, Y).mean(), "paired_change": d.mean(),
                 "sem": sem(d) if d.abs().sum() else 0, "n_better": int((d < 0).sum()), **per.round(4).to_dict()})
df = pd.DataFrame(rows).set_index("method")
print(df.round(4).to_string())
df.to_csv(f"{OUT}/bulk_nnls.csv")
