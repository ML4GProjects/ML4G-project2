"""Heatmap of the LR markers x C sweep (pooled single-cell Brier) from the cached sweep CSVs.

Needs experiments/out/{lr_grid,lr_diag,lr_allgenes}.csv (from lr_grid.py, lr_diag.py, lr_allgenes.py).
Run from the repo root: uv run python experiments/plot_lr_sweep.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = "experiments/out"
df = pd.concat([pd.read_csv(f"{OUT}/{f}.csv") for f in ["lr_grid", "lr_diag", "lr_allgenes"]])
df["n_markers"] = df["n_markers"].astype(str)
grid = df.pivot_table(index="n_markers", columns="C", values="pooled")
order = sorted([i for i in grid.index if i != "all"], key=lambda s: int(float(s))) + ["all"]
grid = grid.loc[order]
grid.index = [i if i == "all" else str(int(float(i))) for i in grid.index]

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.heatmap(grid, annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "pooled Brier (lower is better)"}, ax=ax)
r, c = list(grid.index).index("100"), list(grid.columns).index(0.003)
ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="red", lw=2))  # chosen setting
ax.set(xlabel="C (inverse regularization)", ylabel="marker genes per cell type",
       title="Logistic regression, leave-one-patient-out CV (red: chosen, 100 markers, C=0.003)")
plt.tight_layout()
plt.savefig("plots/sweep_lr.png", dpi=150, bbox_inches="tight")
print("saved plots/sweep_lr.png")
