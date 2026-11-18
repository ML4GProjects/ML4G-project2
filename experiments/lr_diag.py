"""LR along the n_markers/C diagonal, (config x fold) jobs in parallel; same folds as sc_classify.run_cv."""
import warnings; warnings.filterwarnings("ignore")
import time, anndata as ad, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from ml4g_project2.sc_classify import DATA, N_SPLITS, lognorm, make_lr, marker_genes

a = ad.read_h5ad(DATA / "train_sc.h5ad")
X = lognorm(a.X)
y = a.obs["cell_type"].astype(str).to_numpy()
groups = a.obs["patient"].astype(str).to_numpy()
classes = np.unique(y)
folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(X, y, groups))
configs = [(200, 0.001), (200, 0.003), (500, 0.001), (500, 0.003)]

def fit(n, C, tr, te):
    import warnings; warnings.filterwarnings("ignore")
    t = time.time()
    genes = marker_genes(X[tr], y[tr], n)  # picked on the training fold only
    m = make_lr(C).fit(X[tr][:, genes], y[tr])
    p = np.zeros((len(te), len(classes)))
    p[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[te][:, genes])
    print(f"n={n} C={C} held-out {groups[te][0]} genes={len(genes)} done in {time.time()-t:.0f}s", flush=True)
    return n, C, te, p, len(genes)

out = Parallel(n_jobs=10)(delayed(fit)(n, C, tr, te) for n, C in configs for tr, te in folds)
rows = []
for n, C in configs:
    oof = np.zeros((len(y), len(classes)))
    ng = []
    for n_, c_, te, p, g in out:
        if (n_, c_) == (n, C): oof[te] = p; ng.append(g)
    b = ((oof - (y[:, None] == classes)) ** 2).sum(1)
    rows.append({"n_markers": n, "C": C, "genes": int(np.mean(ng)), "pooled": b.mean(), **pd.Series(b).groupby(groups).mean().to_dict()})
df = pd.DataFrame(rows); df.to_csv("experiments/out/lr_diag.csv", index=False)
print(df.round(4).to_string(index=False))
