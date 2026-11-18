"""Scratch: does Harmony help the LR cell classifier? Same folds/markers as predict.py (100/type)."""
import warnings; warnings.filterwarnings("ignore")
S = "experiments/out"
import time, anndata as ad, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from ml4g_project2.sc_classify import DATA, N_SPLITS, lognorm, make_lr, marker_genes

a = ad.read_h5ad(DATA / "train_sc.h5ad")
X = lognorm(a.X)
y = a.obs["cell_type"].astype(str).to_numpy()
obs = a.obs[["sample", "patient"]].astype(str).reset_index(drop=True)
groups = obs["patient"].to_numpy()
classes = np.unique(y)
folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(X, y, groups))
Cs = [0.01, 0.1, 1.0]
VARIANTS = ["pca", "harmony_sample", "harmony_patient"]

def fold(tr, te):
    import warnings; warnings.filterwarnings("ignore")
    import harmonypy
    t = time.time()
    genes = marker_genes(X[tr], y[tr], 100)
    idx = np.concatenate([tr, te])  # held-out cells enter the (unsupervised) correction, unlabeled
    cache = f"{S}/cache/harmony_{groups[te][0]}.npz"
    try:
        c = np.load(cache)
        assert (c["tr"] == tr).all() and (c["te"] == te).all() and (c["genes"] == genes).all()
        emb = {v: c[v] for v in VARIANTS}
    except (FileNotFoundError, AssertionError):
        Z = StandardScaler().fit_transform(X[idx][:, genes].toarray())
        pcs = PCA(50, random_state=0).fit_transform(Z)
        emb = {"pca": pcs}
        for key in ["sample", "patient"]:
            ho = harmonypy.run_harmony(pcs, obs.iloc[idx], [key], verbose=False, random_state=0, ncores=1)
            Zc = np.asarray(ho.Z_corr)
            emb[f"harmony_{key}"] = Zc.T if Zc.shape[0] != len(idx) else Zc
        np.savez_compressed(cache, tr=tr, te=te, genes=genes, **emb)
    ntr = len(tr)
    out = {}
    for v, E in emb.items():
        for C in Cs:
            m = make_lr(C).fit(E[:ntr], y[tr])
            p = np.zeros((len(te), len(classes)))
            p[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(E[ntr:])
            out[(v, C)] = p
        rf = RandomForestClassifier(n_estimators=200, n_jobs=3, random_state=0).fit(E[:ntr], y[tr])  # = make_rf, fewer threads
        p = np.zeros((len(te), len(classes)))
        p[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(E[ntr:])
        out[(v, "rf")] = p
    print(f"held-out {groups[te][0]} done in {time.time()-t:.0f}s", flush=True)
    return te, out

res = Parallel(n_jobs=5)(delayed(fold)(tr, te) for tr, te in folds)
rows = []
probs = {}
for v in VARIANTS:
    for C in Cs + ["rf"]:
        oof = np.zeros((len(y), len(classes)))
        for te, out in res: oof[te] = out[(v, C)]
        probs[f"{v}|{C}"] = oof
        b = ((oof - (y[:, None] == classes)) ** 2).sum(1)
        rows.append({"variant": v, "C": C, "pooled": b.mean(), **pd.Series(b).groupby(groups).mean().to_dict()})
np.savez_compressed(f"{S}/harmony_oof.npz", **probs)
df = pd.DataFrame(rows); df.to_csv("experiments/out/harmony_exp.csv", index=False)
print(df.round(4).to_string(index=False))
