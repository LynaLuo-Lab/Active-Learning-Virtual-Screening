# Embedding and Gradient Analysis

Scripts and data to regenerate four figures characterising the ChemProp
message-passing embeddings and last-layer gradients learned during an
active-learning docking campaign, evaluated across three docking engines
(SILCS, Glide, Vina) and three replicates each. The active-learning protocol
reproduces a MolPAL active-learning benchmark with a ChemProp v2 surrogate.

## Layout

```
embedding_and_gradient_analysis/
├── requirements.txt
├── fig1_engine_umaps/            # per-engine embedding UMAPs
├── fig2_gmm/                     # GMM(k=2) embedding × docking-score overlap
├── fig3_svm_roc_auc/             # linear-SVM separability ROC-AUC
└── fig4_chempropv2_curves/       # ChemProp-v2 recovery@1% active-learning curves
```

Each `figN_*/` contains:

```
figN_*/
├── scripts/<category>/<name>.py  # the figure's script, copied verbatim
├── results/  benchmark_data/     # real copies of just the data that script reads
├── analysis/                     # generated PNG(s) / CSV(s) land here
└── README.md                     # figure-specific command + inputs/outputs
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # or conda, Python 3.10
pip install -r requirements.txt
```

## Regenerate every figure

```bash
python fig3_svm_roc_auc/scripts/chemprop_vina/svm_separability.py
cp fig3_svm_roc_auc/analysis/umaps/separability_auc.csv \
   fig2_gmm/analysis/umaps/separability_auc.csv

python fig2_gmm/scripts/chemprop_vina/gmm_overlap.py
python fig1_engine_umaps/scripts/chemprop_vina/plot_engine_umaps.py \
       --skip-per-rep --skip-cross --skip-combined
python fig4_chempropv2_curves/scripts/al/plot_recovery_curves.py
```
