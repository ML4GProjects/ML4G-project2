# Classifier sweeps (single-cell annotation)

How the cell-type classifier in `src/ml4g_project2/predict.py` was chosen.

## Setup

- **Validation:** leave-one-patient-out CV over the 5 training patients (`StratifiedGroupKFold`, grouped by `patient`). Test patients are unseen, so random cell splits would overestimate performance.
- **Metric:** single-cell Brier score as in the task (sum over the 9 cell types of squared errors), pooled over all cells. Lower is better.
- **Features:** log1p(counts per 10k). Marker genes are the top `n` per cell type from scanpy `rank_genes_groups`, picked inside each fold on training cells only.
- **Comparing settings:** every setting is scored on the same 5 patients, so we use paired differences to the best setting (mean ± SEM over patients). The unpaired SEM (~0.015) mostly measures how much harder P2 is and is too large to separate settings.

## 1. Random forest: number of markers

| Markers per type | 3 | 5 | **10** | 20 | 50 | 100 | all genes |
|---|---|---|---|---|---|---|---|
| Brier (mean over folds) | 0.105 | 0.096 | **0.095** | 0.100 | 0.105 | 0.109 | 0.143 |

Fewer, cell-type-specific genes transfer better to new patients. Best: 10 markers per type (pooled Brier 0.0899).

## 2. Logistic regression: markers × regularization

Multinomial logistic regression with per-gene scaling (`make_lr`). Pooled Brier:

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

The optimum lies on a diagonal: more genes need stronger regularization (smaller C), levelling off at C ≈ 0.003 from about 50 markers on. Performance peaks at 100–200 markers per type. All genes is clearly worse (+0.010 ± 0.003, worse for every patient): the extra genes carry patient-specific signal that regularization shrinks but does not remove.

## Choice

**Logistic regression, 100 markers per cell type (~765 genes), C = 0.003.** By the one-SE rule on paired differences, only 200 markers at C = 0.003 is within one SEM of it (+0.0009 ± 0.0016), and that is not a simpler model.

## Improvement on the single-cell data

| Held-out patient | RF, 10 markers | LR, 100 markers, C = 0.003 |
|---|---|---|
| P1 | 0.080 | 0.055 |
| P2 | 0.147 | 0.133 |
| P3 | 0.085 | 0.071 |
| P4 | 0.086 | 0.075 |
| P7 | 0.077 | 0.050 |
| **Pooled** | **0.090** | **0.070** (−22%) |

Every patient improves. P2 stays hardest: it holds half of all "myofibroblasts", which in most patients express T cell rather than smooth-muscle markers, and are mostly predicted as T cells.

In the competition score (local CV), this lowers the cell half from 0.063 to 0.049 and S from 0.147 to 0.138. The bulk half (mean predicted probabilities per sample) is slightly worse with LR than with RF (0.089 vs 0.084).
