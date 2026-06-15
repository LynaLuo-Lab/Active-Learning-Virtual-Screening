# Linear-SVM separability (ROC-AUC)

Tests whether the ChemProp embedding (`mpn_fp`) and last-layer gradient
(`grad_W`) linearly separate the top-p% from the bottom-p% docking scores. A
soft-margin `LinearSVC` is scored with 5-fold stratified CV ROC-AUC, per
(engine, replicate, feature, threshold ∈ {10%, 25%}).

## Run

```bash
python scripts/chemprop_vina/svm_separability.py
```

## Outputs (`analysis/umaps/`)

- `separability_auc.csv` — one row per (engine, rep, feature, threshold) with
  `auc_mean`, `auc_std`, `n_pos`, `n_neg`. **Also consumed by Figure 2.**
- `separability_auc.png` — ROC-AUC bar chart (engine × feature) at the 25% threshold

## Inputs

- `results/embeddings/<pool>/<engine>_rep<r>/{mpn_fp,grad_W,y_true}.npy`
