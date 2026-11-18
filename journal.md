# Journal

Findings and how to reproduce them. All commands run from the repo root; data in `data/`.

## Setup

- **Validation:** leave-one-patient-out CV over the 5 training patients (P1–P4, P7; `StratifiedGroupKFold` grouped by `patient`). Test patients (P5, P6, P8) are unseen, so random cell splits overestimate.
- **Metric:** task Brier score (sum over 9 cell types of squared errors), pooled over all cells (single-cell) or bulk replicates. Competition score S = ½ B_cell/B0_cell + ½ B_bulk/B0_bulk; pass at S ≤ 0.30144.
- **Comparing settings:** paired per-patient differences to the best setting (mean ± SEM). The unpaired SEM (~0.015) mostly reflects P2 being hard.
- **Features:** log1p(counts per 10k). Marker genes = top n per cell type from scanpy `rank_genes_groups`, picked inside each fold on training cells only.

## Reproduce

```
uv sync
uv run python -m ml4g_project2.predict        # local CV score S + results/submission.csv
uv run python -m ml4g_project2.sc_classify    # RF CV (single-cell Brier)
uv run python -m ml4g_project2.diagnostics    # plots/cv_rf_errors.png, marker dot plots
uv run python -m ml4g_project2.sweep_markers  # RF marker sweep, plots/sweep_markers.png
uv run python -m ml4g_project2.bulk_check     # sc counts vs bulk proportions

mkdir -p experiments/out/cache                # experiment outputs (git-ignored)
uv run python experiments/lr_grid.py          # LR markers x C grid (~15 min)
uv run python experiments/lr_diag.py          # LR 200/500 markers
uv run python experiments/lr_allgenes.py      # LR on all genes
uv run python experiments/temperature.py      # temperature scaling
uv run --with harmonypy==2.1.0 python experiments/harmony_exp.py
uv run --with scvi-tools==1.5.2 python experiments/scanvi_exp.py       # MPS, ~10 min
uv run --with scvi-tools==1.5.2 python experiments/scanvi_allgenes.py  # MPS
```

Embeddings and per-cell CV probabilities are cached in `experiments/out/` and reused when folds and genes match. On Apple GPU (MPS) run only one scANVI script at a time: two concurrent runs deadlocked.

## Data findings

- 21,198 train / 12,264 test cells, 7,725 genes; 83% zeros. Test bulk samples come from the same samples as the test cells.
- UMAP: cell types form clean clusters, but split by patient within types. Patient and chemo status are confounded (each patient is all-pre or all-post).
- Cell-cycle markers are mostly missing from the panel (16/43 S, 20/54 G2M genes) → phase calls are noise.
- **"Myofibroblasts" are mislabelled-looking:** in P2, P4, P7 they express T cell markers (CD3D, CD2, TRAC, IL7R in ~50–57% of cells) and ACTA2 in only 11% (fibroblasts: 34%). In P3 they look like fibroblasts. P2 holds 172 of 343 myofibroblasts and 523 of 734 mast cells, which makes it the hardest patient.
- Low-depth cells err slightly more (lowest quintile 6.4% vs ~4%), mostly because T/NK/myofibroblasts have the lowest counts.

## Cell-type classifier

### Random forest
- All genes: pooled Brier 0.135. P2 worst (0.218): 163 of 172 myofibroblasts → T cells, half the NK cells → T cells.
- `class_weight="balanced"`: worse for every patient (mean over folds 0.143 → 0.180). Not used.
- Textbook markers (48 genes, `canonical_markers.yaml`): 0.105. Data-driven markers are better because they separate the labels as given.
- Marker count sweep (pooled Brier):

| Markers per type | 3 | 5 | **10** | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|
| RF | 0.101 | 0.092 | **0.090** | 0.095 | 0.100 | 0.104 | 0.135 |

### Logistic regression (multinomial, per-gene scaling; `make_lr`)

| Markers \ C | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|---|---|
| 3 | 0.152 | 0.125 | 0.109 | 0.100 | 0.095 | 0.095 | 0.097 |
| 5 | 0.130 | 0.107 | 0.090 | 0.084 | 0.083 | 0.084 | 0.088 |
| 10 | 0.110 | 0.092 | 0.083 | 0.081 | 0.082 | 0.084 | 0.088 |
| 20 | 0.095 | 0.082 | 0.079 | 0.083 | 0.089 | 0.095 | 0.105 |
| 50 | 0.086 | 0.078 | 0.078 | 0.083 | 0.092 | 0.099 | 0.108 |
| 100 | 0.074 | **0.070** | 0.074 | 0.082 | 0.090 | 0.099 | 0.109 |
| 200 | 0.072 | 0.070 | | | | | |
| 500 | 0.073 | 0.074 | | | | | |
| all (7,725) | 0.079 | 0.079 | 0.081 | | | | |

- Optimum on a diagonal: more genes need smaller C, levelling off at C ≈ 0.003 from ~50 markers. Peak at 100–200 markers; all genes is worse for every patient (+0.010 ± 0.003).
- **Chosen: LR, 100 markers per type (~765 genes), C = 0.003** (pooled 0.0702). Paired one-SE rule: only 200 markers at C = 0.003 is within one SEM (+0.0009 ± 0.0016), and it is not simpler.

| Held-out patient | RF, 10 markers | LR, 100 markers, C = 0.003 |
|---|---|---|
| P1 | 0.080 | 0.055 |
| P2 | 0.147 | 0.133 |
| P3 | 0.085 | 0.071 |
| P4 | 0.086 | 0.075 |
| P7 | 0.077 | 0.050 |
| **Pooled** | **0.090** | **0.070** (−22%) |

### Batch correction and other attempts (vs chosen LR, 0.0702)

| Approach | Pooled B_cell |
|---|---|
| PCA (50) on markers + LR | 0.081 |
| Harmony (batch = sample) + LR, C = 0.1 | 0.0699 (tie) |
| Harmony (batch = patient) + LR | 0.0716 |
| Harmony + RF | 0.088 |
| scANVI (markers), own predictions | 0.193 (overconfident) |
| scANVI (markers) latent + LR, C = 1 | 0.0703 (tie) |
| scANVI (markers) latent + RF | 0.097 |
| scANVI (all genes) latent + LR | 0.077 |
| Temperature scaling of chosen LR (T ≈ 0.93) | 0.0704 (no gain) |

In-fold marker selection already removes most patient effects; batch correction ties at best. RF is worse than LR on every embedding. The remaining error is most likely the myofibroblast/T and NK/T label overlap, not batch effects.

## Bulk deconvolution

Each test bulk sample = mean predicted cell-type probabilities of the test cells from the same sample.

| Estimate (RF, 10 markers) | Bulk Brier |
|---|---|
| Mean predicted probabilities | 0.0174 |
| True single-cell labels | 0.0178 |
| Replicate vs replicate (noise) | 0.0210 |
| Argmax counts | 0.0227 |
| Constant (mean train bulk) | 0.0742 |

The classifier is not the bottleneck: the remaining error is the single-cell vs bulk gap (T cells systematically overestimated). Next lever: per-cell-type correction factors fitted on the 12 training bulk replicates.

## Competition score (local CV)

| Pipeline | Cell term | Bulk term | S |
|---|---|---|---|
| RF, 10 markers | 0.063 | 0.084 | 0.147 |
| **LR, 100 markers, C = 0.003 (current `predict.py`)** | 0.049 | 0.089 | **0.138** |
| LR cells + RF bulk (not implemented) | 0.049 | 0.084 | ~0.133 |

Local estimate only: the task's reference constants are computed on the test rows.
