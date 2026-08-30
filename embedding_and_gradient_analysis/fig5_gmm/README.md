# Figure 5 — GMM embedding × docking-score overlap

For each (engine, replicate): project the 300-d ChemProp embedding onto its
ridge-regression-to-score direction, stack against the docking score, and fit
an unsupervised `GaussianMixture(n_components=2)`. Reports the Bhattacharyya
distance and |Pearson r| between the embedding projection and the score.

## Run

```bash
python scripts/chemprop_vina/gmm_overlap.py
```

## Outputs (`analysis/umaps/`)

- `gmm_combined.png` — **Figure 5**: (a) ROC-AUC bar, (b) |r| bar, (c) GMM 3×3 scatter
- `gmm_overlap_scatter.png` — 3×3 GMM scatter + 1σ/2σ ellipses (unpublished alternate)
- `gmm_overlap_bar.png` — |Pearson r| by engine (unpublished alternate)

## Inputs

- `results/embeddings/<pool>/<engine>_rep<r>/{mpn_fp,y_true}.npy`
- `analysis/umaps/separability_auc.csv` (from `fig5a_svm_roc_auc/`; included)
