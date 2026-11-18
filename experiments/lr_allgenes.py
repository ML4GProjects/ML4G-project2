"""LR on all genes, (C x fold) jobs in parallel; same folds as sc_classify.run_cv."""
import time, anndata as ad, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from ml4g_project2.sc_classify import DATA, N_SPLITS, lognorm, make_lr

a = ad.read_h5ad(DATA / "train_sc.h5ad")
X = lognorm(a.X)
y = a.obs["cell_type"].astype(str).to_numpy()
groups = a.obs["patient"].astype(str).to_numpy()
classes = np.unique(y)
folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(X, y, groups))
Cs = [0.0001, 0.0003, 0.001, 0.003, 0.01]

def fit(C, tr, te):
    t = time.time()
    m = make_lr(C).fit(X[tr], y[tr])
    p = np.zeros((len(te), len(classes)))
    p[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[te])
    print(f"C={C} held-out {groups[te][0]} done in {time.time()-t:.0f}s", flush=True)
    return C, te, p

out = Parallel(n_jobs=10, verbose=0)(delayed(fit)(C, tr, te) for C in Cs for tr, te in folds)
rows = []
for C in Cs:
    oof = np.zeros((len(y), len(classes)))
    for c, te, p in out:
        if c == C: oof[te] = p
    b = ((oof - (y[:, None] == classes)) ** 2).sum(1)
    rows.append({"n_markers": "all", "C": C, "pooled": b.mean(), **pd.Series(b).groupby(groups).mean().to_dict()})
df = pd.DataFrame(rows); df.to_csv("experiments/out/lr_allgenes.csv", index=False)
print(df.round(4).to_string(index=False))
