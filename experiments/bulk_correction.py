"""Per-cell-type correction of sc-derived bulk proportions, nested leave-one-patient-out.

sc proportions per sample = mean out-of-fold predicted probabilities of its cells.
  multiplicative: bulk ~ normalize(exp(theta) * sc), theta fitted by Brier + lam * ||theta||^2
  additive:       bulk ~ normalize(clip(sc + mean(bulk - sc), 0))
For each held-out patient the correction is fitted on the other patients' bulk replicates only.

Run from the repo root: uv run python experiments/bulk_correction.py
"""
import contextlib
import io
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import sem

from ml4g_project2.sc_classify import DATA, make_lr, make_rf, run_cv

OUT = "experiments/out"
LAMS = [0.0, 0.01, 0.1, 1.0]


def brier(p, t):
    return ((np.asarray(p) - np.asarray(t)) ** 2).sum(axis=1)


def normalize(p):
    p = np.clip(p, 0, None)
    return p / p.sum(axis=1, keepdims=True)


def fit_mult(X, Y, lam):
    k = X.shape[1]
    loss = lambda th: brier(normalize(np.exp(th) * X), Y).mean() + lam * (th ** 2).sum() / k
    return np.exp(minimize(loss, np.zeros(k), method="L-BFGS-B").x)


def oof_probs():
    """Out-of-fold probabilities for the two base models (LR cached by temperature.py if present)."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            lr = np.load(f"{OUT}/lr100_oof.npy")
            y, groups, classes, rf = run_cv(10, make_model=make_rf)
        except FileNotFoundError:
            y, groups, classes, lr = run_cv(100, make_model=partial(make_lr, C=0.003))
            np.save(f"{OUT}/lr100_oof.npy", lr)
            y, groups, classes, rf = run_cv(10, make_model=make_rf)
    return classes, {"LR 100/0.003": lr, "RF 10": rf}


def main():
    classes, bases = oof_probs()
    sample = ad.read_h5ad(DATA / "train_sc.h5ad", backed="r").obs["sample"].astype(str).to_numpy()
    bulk = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
    meta = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[bulk.index]
    Y, pat = bulk.to_numpy(), meta["patient"].to_numpy()
    b0 = np.concatenate([brier(bulk[pat != g].mean().to_numpy()[None], Y[pat == g])
                         for g in np.unique(pat)]).mean()  # pooled task reference, as in cv_score

    rows, weights = [], {}
    for base, oof in bases.items():
        sc = pd.DataFrame(oof, columns=classes).groupby(sample).mean().loc[meta["sample"]].to_numpy()
        preds = {"none": sc.copy(), "additive": np.zeros_like(sc)}
        preds.update({f"mult lam={lam}": np.zeros_like(sc) for lam in LAMS})
        for g in np.unique(pat):
            te, tr = pat == g, pat != g
            preds["additive"][te] = normalize(sc[te] + (Y[tr] - sc[tr]).mean(axis=0))
            for lam in LAMS:
                preds[f"mult lam={lam}"][te] = normalize(fit_mult(sc[tr], Y[tr], lam) * sc[te])
        weights[base] = fit_mult(sc, Y, 0.1)  # all 12 replicates: what would go into predict.py
        for name, p in preds.items():
            per = pd.Series(brier(p, Y)).groupby(pat).mean()
            rows.append({"base": base, "correction": name, "B_bulk": brier(p, Y).mean(),
                         "bulk_term": 0.5 * brier(p, Y).mean() / b0, **per.round(4).to_dict()})

    df = pd.DataFrame(rows)
    P = sorted(set(pat))
    for base in bases:
        d = df[df.base == base].set_index("correction")
        diff = d[P].sub(d.loc["none", P], axis=1)
        d["paired_change"], d["sem"] = diff.mean(axis=1), diff.apply(lambda r: sem(r) if r.abs().sum() else 0, axis=1)
        d["n_better"] = (diff < 0).sum(axis=1)
        print(f"\n{base} (pooled B0_bulk = {b0:.4f})")
        print(d[["B_bulk", "bulk_term", "paired_change", "sem", "n_better"] + P].round(4).to_string())
    print("\nmultiplicative weights fitted on all replicates (lam=0.1):")
    print(pd.DataFrame(weights, index=classes).round(2).T.to_string())
    df.to_csv(f"{OUT}/bulk_correction.csv", index=False)


if __name__ == "__main__":
    main()
