# Figures S2 and S10 — ChemProp v2 recovery@1% curves

Recovery@1% active-learning curves for a ChemProp-v2 + MolPAL surrogate
reproducing the reference MolPAL benchmark, across
docking engines (Vina, Glide, SILCS), replicates, and acquisition batch sizes.
Mean over replicates with ±std error bars.

Same-dataset recovery: 40k-trained cells scored on the 40k pool, 1.29M-trained
cells on the 1.29M pool. The incomplete 1.29M 2%-batch cells are reconstructed
from per-iteration `acquired.csv` to a common depth (iteration 4).

## Run

```bash
python scripts/al/plot_recovery_curves.py
```

## Outputs (`analysis/recovery/`)

- `recovery_40k_1pct_5pct.png` — **Figure S2**: 40k, 1% vs 5% batch
- `recovery_1p3M.png` — **Figure S10**: 1.29M recovery curves (0.25 / 0.5 / 1 / 2% batch)
- `recovery_40k_batchsweep.png` — final recovery@1% vs batch size (unpublished)
- `recovery_summary.csv`, `recovery_summary_table.png` (unpublished)

## Inputs

- `results/recovery_batch_sweep_v1/<bs>/<engine>/rep<r>/recovery.json` (40k)
- `results/recovery_batch_sweep_v2/<bs>/<engine>/rep<r>/recovery.json` (1.29M)
- `results/recovery_batch_sweep_v2/bs2/<engine>/rep<r>/per_iter/iter_*/acquired.csv`
- `benchmark_data/vina/1.3M/vina_df.csv`, `benchmark_data/glide/df.csv` (1.29M truth)
