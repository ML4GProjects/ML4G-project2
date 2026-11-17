"""Sweep the number of marker genes per cell type; plot CV Brier score (mean ± SEM over held-out patients).

Run from the repo root: uv run python -m ml4g_project2.sweep_markers
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.metrics import brier_score_loss

from ml4g_project2.sc_classify import run_cv

PLOTS = Path("plots")
N_MARKERS = [3, 5, 10, 20, 50, 100]


def main():
    means, sems = [], []
    for n in N_MARKERS:
        print(f"===== n_markers={n}")
        y, groups, classes, oof = run_cv(n)
        per_patient = [brier_score_loss(y[groups == g], oof[groups == g], labels=classes)
                       for g in np.unique(groups)]
        means.append(np.mean(per_patient))
        sems.append(sem(per_patient))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(N_MARKERS, means, yerr=sems, marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xticks(N_MARKERS, [str(n) for n in N_MARKERS])
    ax.set(xlabel="marker genes per cell type", ylabel="Brier score (lower is better)",
           title="RF on marker genes: leave-one-patient-out CV (mean ± SEM over patients)")
    plt.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    plt.savefig(PLOTS / "sweep_markers.png", dpi=150, bbox_inches="tight")
    print(f"saved {PLOTS / 'sweep_markers.png'}")


if __name__ == "__main__":
    main()
