"""Scratch: temperature scaling of the current best model (LR, 100 markers/type, C=0.003), nested by patient."""
import io, contextlib, anndata as ad, numpy as np, pandas as pd
from functools import partial
from scipy.optimize import minimize_scalar
from scipy.stats import sem
from ml4g_project2.sc_classify import DATA, run_cv, make_lr

with contextlib.redirect_stdout(io.StringIO()):
    y, groups, classes, oof = run_cv(100, make_model=partial(make_lr, C=0.003))
np.save("experiments/out/lr100_oof.npy", oof)  # cache
onehot = (y[:, None] == classes).astype(float)

def scale(p, T):
    z = np.log(np.clip(p, 1e-12, None)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

brier = lambda p, t: ((p - t) ** 2).sum(axis=1)
fit_T = lambda p, t: np.exp(minimize_scalar(lambda lt: brier(scale(p, np.exp(lt)), t).mean(),
                                            bounds=(-3, 3), method="bounded").x)

# nested: T for patient g is fitted on the other patients' out-of-fold predictions
scaled, Ts = oof.copy(), {}
for g in np.unique(groups):
    m = groups == g
    Ts[g] = fit_T(oof[~m], onehot[~m])
    scaled[m] = scale(oof[m], Ts[g])
print("T per held-out patient (fitted on the others):", {k: round(v, 3) for k, v in Ts.items()})
print("T fitted on all 5 patients (for the submission):", round(fit_T(oof, onehot), 3))

b0, b1 = brier(oof, onehot), brier(scaled, onehot)
per0, per1 = pd.Series(b0).groupby(groups).mean(), pd.Series(b1).groupby(groups).mean()
d = per1 - per0
print("\nsingle-cell Brier, pooled: before %.4f  after %.4f" % (b0.mean(), b1.mean()))
print("paired change (after - before): %.4f ± %.4f SEM, better for %d/5 patients" % (d.mean(), sem(d), (d < 0).sum()))
print(pd.DataFrame({"before": per0, "after": per1}).round(4).to_string())

# bulk: mean probabilities per sample vs true bulk proportions
sample = ad.read_h5ad(DATA / "train_sc.h5ad", backed="r").obs["sample"].astype(str).to_numpy()
bulk = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
bs = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[bulk.index, "sample"]
for name, p in [("before", oof), ("after", scaled)]:
    est = pd.DataFrame(p, columns=classes).groupby(sample).mean().loc[bs].to_numpy()
    print(f"bulk Brier {name}: {brier(est, bulk.to_numpy()).mean():.4f}")
