"""Scratch: scANVI label transfer per leave-one-patient-out fold (markers 100/type, batch=sample)."""
import warnings; warnings.filterwarnings("ignore")
S = "experiments/out"
import time, anndata as ad, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from ml4g_project2.sc_classify import DATA, N_SPLITS, lognorm, make_lr, marker_genes

a = ad.read_h5ad(DATA / "train_sc.h5ad")
counts = a.X.tocsr()
X = lognorm(counts)
y = a.obs["cell_type"].astype(str).to_numpy()
sample = a.obs["sample"].astype(str).to_numpy()
groups = a.obs["patient"].astype(str).to_numpy()
classes = np.unique(y)
folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(X, y, groups))
Cs = [0.01, 0.1, 1.0]

def fold(tr, te):
    import warnings, logging; warnings.filterwarnings("ignore")
    import torch, scvi
    torch.set_num_threads(2)
    scvi.settings.seed = 0
    scvi.settings.verbosity = logging.ERROR
    t = time.time()
    cache = f"{S}/cache/scanvi_markers_{groups[te][0]}.npz"
    genes = marker_genes(X[tr], y[tr], 100)
    ntr = len(tr)
    try:
        c = np.load(cache)
        assert (c["tr"] == tr).all() and (c["te"] == te).all() and (c["genes"] == genes).all()
        lat, scanvi_p = c["latent"], c["scanvi_proba"]
        print(f"held-out {groups[te][0]}: loaded cached latent", flush=True)
    except (FileNotFoundError, AssertionError):
        lat, scanvi_p = train_scanvi(tr, te, genes)
        np.savez_compressed(cache, tr=tr, te=te, genes=genes, latent=lat, scanvi_proba=scanvi_p)
    out = {"scanvi": scanvi_p}
    for C in Cs:
        m = make_lr(C).fit(lat[:ntr], y[tr])
        p = np.zeros((len(te), len(classes)))
        p[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(lat[ntr:])
        out[f"lr_latent C={C}"] = p
    rf = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0).fit(lat[:ntr], y[tr])  # = make_rf, fewer threads
    p = np.zeros((len(te), len(classes)))
    p[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(lat[ntr:])
    out["rf_latent"] = p
    print(f"held-out {groups[te][0]} done in {time.time()-t:.0f}s", flush=True)
    return te, out


def train_scanvi(tr, te, genes):
    import torch, scvi
    torch.set_num_threads(2)
    idx = np.concatenate([tr, te])
    ntr = len(tr)
    labels = np.concatenate([y[tr], np.full(len(te), "Unknown")])
    sub = ad.AnnData(counts[idx][:, genes].astype(np.float32),
                     obs=pd.DataFrame({"sample": sample[idx], "label": labels}))
    scvi.model.SCVI.setup_anndata(sub, batch_key="sample", labels_key="label")
    vae = scvi.model.SCVI(sub, n_latent=20)
    vae.train(max_epochs=100, accelerator="mps", enable_progress_bar=False)
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key="label")
    lvae.train(max_epochs=20, n_samples_per_label=100, accelerator="mps", enable_progress_bar=False)

    pred = lvae.predict(sub[ntr:], soft=True)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return lvae.get_latent_representation(sub), pred.reindex(columns=classes, fill_value=0).to_numpy()


res = Parallel(n_jobs=5)(delayed(fold)(tr, te) for tr, te in folds)
rows, probs = [], {}
for v in res[0][1]:
    oof = np.zeros((len(y), len(classes)))
    for te, out in res: oof[te] = out[v]
    probs[v] = oof
    b = ((oof - (y[:, None] == classes)) ** 2).sum(1)
    rows.append({"variant": v, "pooled": b.mean(), **pd.Series(b).groupby(groups).mean().to_dict()})
np.savez_compressed("experiments/out/scanvi_oof.npz", **probs)
df = pd.DataFrame(rows); df.to_csv("experiments/out/scanvi_exp.csv", index=False)
print(df.round(4).to_string(index=False))
