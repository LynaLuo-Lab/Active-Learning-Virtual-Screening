# Per-engine embedding UMAPs

Two-dimensional UMAP projections of the ChemProp molecule embeddings
(`mpn_fp`), one figure per docking engine (SILCS, Glide, Vina), each showing
the three replicates and the cumulative training subset highlighted at every
acquisition round. Colored by ground-truth docking score (top section) and by
ChemProp-predicted score (bottom section).

## Run

```bash
python scripts/chemprop_vina/plot_engine_umaps.py \
       --skip-per-rep --skip-cross --skip-combined
```

UMAP fits are read from `analysis/umaps/_umap_cache/` so the projections
reproduce exactly and the run is fast. Delete that folder to refit from
scratch (deterministic; `random_state=42`).

## Outputs (`analysis/umaps/`)

- `silcs_per_engine.png`, `glide_per_engine.png`, `vina_per_engine.png` — side-label variant
- `*_per_engine_toplabel.png` — horizontal section-label variant

## Inputs

- `results/embeddings/<pool>/<engine>_rep<r>/{mpn_fp,grad_W,y_true,y_hat}.npy`
- `results/embeddings/<pool>/pool_smiles.npz`
- `benchmark_data/benchmark_40k/<engine>/rep<r>/input/train_{1..6}.csv`
