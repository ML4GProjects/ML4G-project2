import io, contextlib, numpy as np, pandas as pd
from functools import partial
from scipy.stats import sem
from ml4g_project2.sc_classify import run_cv, make_lr
rows = []
for n in [3, 5, 10, 20, 50, 100]:
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        with contextlib.redirect_stdout(io.StringIO()):
            y, groups, classes, oof = run_cv(n, make_model=partial(make_lr, C=C))
        b = ((oof - (y[:, None] == classes)) ** 2).sum(1)
        per = pd.Series(b).groupby(groups).mean()
        rows.append({"n_markers": n, "C": C, "pooled": b.mean(), "sem": sem(per), **per.to_dict()})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv("experiments/out/lr_grid.csv", index=False)
print("\nPOOLED B_cell (rows n_markers, cols C)")
print(df.pivot(index="n_markers", columns="C", values="pooled").round(4).to_string())
