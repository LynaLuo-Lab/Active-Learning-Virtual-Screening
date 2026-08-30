# Embedding and Gradient Analysis

Scripts and data to regenerate the ChemProp embedding / gradient figures and
the ChemProp v2 recovery curves, evaluated across three docking engines
(SILCS, Glide, Vina) and three replicates each. The active-learning protocol
reproduces the MolPAL benchmark with a ChemProp surrogate.

Folder names carry the figure numbers from the paper: `figN_` for the main
text, `figSN_` for the Supporting Information.

## Layout

```
embedding_and_gradient_analysis/
├── requirements.txt
├── fig5_gmm/                       # Figure 5 — GMM(k=2) embedding × docking-score overlap
├── fig5a_svm_roc_auc/              # Figure 5a — linear-SVM separability ROC-AUC
├── figS6-S8_engine_umaps/          # Figures S6/S7/S8 — per-engine embedding UMAPs
└── figS2-S10_chempropv2_curves/    # Figures S2 and S10 — ChemProp v2 recovery@1% curves
```

Each figure folder contains:

```
<figure_dir>/
├── scripts/<category>/<name>.py  # the figure's script, copied verbatim
├── results/  benchmark_data/     # real copies of just the data that script reads
├── analysis/                     # generated PNG(s) / CSV(s) land here
└── README.md                     # figure-specific command + inputs/outputs
```

## Which file is which figure

| Paper figure | File |
|---|---|
| Figure 5 (a, b, c) | `fig5_gmm/analysis/umaps/gmm_combined.png` |
| Figure S2 | `figS2-S10_chempropv2_curves/analysis/recovery/recovery_40k_1pct_5pct.png` |
| Figure S6 (Vina) | `figS6-S8_engine_umaps/analysis/umaps/vina_per_engine_toplabel.png` |
| Figure S7 (Glide) | `figS6-S8_engine_umaps/analysis/umaps/glide_per_engine_toplabel.png` |
| Figure S8 (SILCS) | `figS6-S8_engine_umaps/analysis/umaps/silcs_per_engine_toplabel.png` |
| Figure S10 | `figS2-S10_chempropv2_curves/analysis/recovery/recovery_1p3M.png` |

`fig5a_svm_roc_auc/` has no standalone published figure: it produces
`separability_auc.csv`, from which panel (a) of Figure 5 is drawn, so it must
be run before `fig5_gmm/`.

The remaining outputs are alternates and intermediates that did not go into
the paper: `*_per_engine.png` (side-label variant of S6–S8),
`separability_auc.png`, `gmm_overlap_scatter.png`, `gmm_overlap_bar.png`,
`recovery_40k_batchsweep.png`, `recovery_summary.csv` and
`recovery_summary_table.png`.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # or conda, Python 3.10
pip install -r requirements.txt
```

## Regenerate every figure

Each script resolves its own paths relative to its figure folder, so these can
be run from anywhere. `separability_auc.csv` has to be copied across because
Figure 5's panel (a) is built from it.

```bash
python fig5a_svm_roc_auc/scripts/chemprop_vina/svm_separability.py
cp fig5a_svm_roc_auc/analysis/umaps/separability_auc.csv \
   fig5_gmm/analysis/umaps/separability_auc.csv

python fig5_gmm/scripts/chemprop_vina/gmm_overlap.py
python figS6-S8_engine_umaps/scripts/chemprop_vina/plot_engine_umaps.py \
       --skip-per-rep --skip-cross --skip-combined
python figS2-S10_chempropv2_curves/scripts/al/plot_recovery_curves.py
```
