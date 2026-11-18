"""How close do per-sample cell-type counts from the sc classifier get to the true bulk proportions?

Uses out-of-fold predictions from sc_classify.run_cv (the model never saw the sample's patient),
aggregates them per sample, and compares against every bulk replicate of that sample (MAE over cell types).

Run from the repo root: uv run python -m ml4g_project2.bulk_check
"""
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml4g_project2.diagnostics import PLOTS
from ml4g_project2.predict import N_MARKERS, make_model
from ml4g_project2.sc_classify import DATA, run_cv


def main():
    y, groups, classes, oof = run_cv(N_MARKERS, make_model=make_model)  # current predict.py model
    sample = ad.read_h5ad(DATA / "train_sc.h5ad", backed="r").obs["sample"].astype(str).to_numpy()

    bulk = pd.read_csv(DATA / "train_bulk_props.csv", index_col="id")[classes]
    bulk_sample = pd.read_csv(DATA / "bulk_metadata.csv", index_col="id").loc[bulk.index, "sample"]

    onehot = pd.DataFrame(y[:, None] == classes, columns=classes).astype(float)
    hard = pd.DataFrame(oof.argmax(axis=1)[:, None] == np.arange(len(classes)), columns=classes).astype(float)
    estimates = {
        "true sc labels (ceiling)": onehot.groupby(sample).mean(),
        "classifier counts (argmax)": hard.groupby(sample).mean(),
        "classifier mean proba": pd.DataFrame(oof, columns=classes).groupby(sample).mean(),
    }
    # ponytail: constant baseline uses all train bulk incl. the evaluated sample (slightly optimistic)
    const = pd.DataFrame([bulk.mean()] * len(estimates["true sc labels (ceiling)"]),
                         index=estimates["true sc labels (ceiling)"].index)
    estimates["constant (mean train bulk)"] = const

    rows = {name: (est.loc[bulk_sample].to_numpy() - bulk.to_numpy()).__abs__().mean(axis=1)
            for name, est in estimates.items()}
    mae = pd.DataFrame(rows, index=bulk.index)

    # noise floor: replicate vs the other replicate of the same sample
    other = bulk.groupby(bulk_sample).transform(lambda c: c[::-1].to_numpy())
    mae["replicate vs replicate (noise)"] = (bulk - other).abs().mean(axis=1)

    pd.set_option("display.width", 200)
    print("\nMAE between estimated and bulk proportions (mean over 9 cell types), per bulk replicate:")
    print(mae.round(4).to_string())
    print("\nmean:")
    print(mae.mean().round(4).to_string())

    # scatter: mean predicted probabilities (what predict.py submits) vs bulk, one point per (replicate, cell type)
    pred = estimates["classifier mean proba"].loc[bulk_sample].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    for j, ct in enumerate(classes):
        ax.scatter(bulk.iloc[:, j], pred[:, j], label=ct, s=30)
    lim = max(bulk.to_numpy().max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="bulk proportion (truth)",
           ylabel="mean predicted probability", title="Per-sample sc classifier estimate vs bulk")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "bulk_check.png", dpi=150, bbox_inches="tight")
    print(f"saved {PLOTS / 'bulk_check.png'}")


if __name__ == "__main__":
    main()
