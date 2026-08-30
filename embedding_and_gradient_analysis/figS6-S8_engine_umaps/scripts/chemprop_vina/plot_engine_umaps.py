"""Two-dimensional UMAP projections of MolPAL embeddings and gradients.

For each (scorer, replicate) pair this script:

  1. loads the molecule embeddings (encoder output) and the per molecule last
     layer gradients (residual times the input to the readout) from
     results/embeddings/<pool>/<label>/,
  2. drops rows where the ground truth score was missing,
  3. L2 normalises each row (so distances are angular),
  4. fits two separate UMAP projections, one for the embedding rows and one
     for the gradient rows, with a fixed random seed for reproducibility,
  5. converts the ground truth and ChemProp predicted scores to z scores
     within that (scorer, replicate), so colour ranges are comparable across
     reps and across scorers,
  6. loads per round cumulative training subsets so each acquisition
     round can be highlighted on the same projection.

Two families of figures are produced:

  Per (scorer, replicate) figures (9 total)
    analysis/umaps/<scorer>_<rep>.png
    4 rows by 7 columns. Rows: embedding by ground truth, embedding by
    ChemProp predicted score, gradient by ground truth, gradient by
    ChemProp predicted score. Columns: full pool plus the 6 cumulative
    acquisition rounds.

  Cross scorer figures (4 total)
    analysis/umaps/cross_<representation>_<score>.png
    3 columns by 3 rows. Columns: silcs, glide, vina. Rows: replicates 1
    through 3. Each panel shows the full pool projection colored by the
    chosen score in that scorer-replicate.

The color map is a red to blue diverging palette (RdBu reversed): high
binding affinity (poor binder) is red, low binding affinity (good binder)
is blue. The ChemProp model was trained to predict the absolute affinity
(sign flipped), so the predicted score is negated before plotting to put
it on the same orientation as the ground truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
EMB_ROOT = REPO / "results" / "embeddings"
ANALYSIS_DIR = REPO / "analysis" / "umaps"

BENCHMARK_DIR = REPO / "benchmark_data" / "benchmark_40k"
N_ROUNDS = 6
RANDOM_SEED = 42

# Paper ready text sizes (all the same value so every label, tick and
# title renders consistently).
TEXT_SIZE = 20

SCORER_DISPLAY = {"silcs": "SILCS", "glide": "Glide", "vina": "Vina"}

# Z score range to clip the colour bar to. Outside this range the colour
# saturates. Symmetric around zero so the red and blue ends carry equal
# weight.
Z_CLIP = 2.5


@dataclass(frozen=True)
class RepSpec:
    scorer: str          # "glide" / "silcs" / "vina"
    rep: int             # 1, 2, 3
    pool_dir: Path       # results/embeddings/<pool>
    label: str           # subdir under pool_dir, e.g. "glide_rep1"
    train_subsets_dir: Path  # benchmark_data path with train_*.csv

    @property
    def display_name(self) -> str:
        return f"{SCORER_DISPLAY[self.scorer]} replicate {self.rep}"


SPECS: list[RepSpec] = [
    RepSpec("silcs", 1, EMB_ROOT / "pool_40k_silcs", "silcs_rep1",
            BENCHMARK_DIR / "silcs" / "rep1" / "input"),
    RepSpec("silcs", 2, EMB_ROOT / "pool_40k_silcs", "silcs_rep2",
            BENCHMARK_DIR / "silcs" / "rep2" / "input"),
    RepSpec("silcs", 3, EMB_ROOT / "pool_40k_silcs", "silcs_rep3",
            BENCHMARK_DIR / "silcs" / "rep3" / "input"),
    RepSpec("glide", 1, EMB_ROOT / "pool_40k_A", "glide_rep1",
            BENCHMARK_DIR / "glide" / "rep1" / "input"),
    RepSpec("vina",  1, EMB_ROOT / "pool_40k_A", "vina_rep1",
            BENCHMARK_DIR / "vina"  / "rep1" / "input"),
    RepSpec("vina",  2, EMB_ROOT / "pool_40k_A", "vina_rep2",
            BENCHMARK_DIR / "vina"  / "rep2" / "input"),
    RepSpec("vina",  3, EMB_ROOT / "pool_40k_A", "vina_rep3",
            BENCHMARK_DIR / "vina"  / "rep3" / "input"),
    RepSpec("glide", 2, EMB_ROOT / "pool_40k_B", "glide_rep2",
            BENCHMARK_DIR / "glide" / "rep2" / "input"),
    RepSpec("glide", 3, EMB_ROOT / "pool_40k_B", "glide_rep3",
            BENCHMARK_DIR / "glide" / "rep3" / "input"),
]


@dataclass
class RepData:
    spec: RepSpec
    smiles: np.ndarray
    coords_embed: np.ndarray
    coords_grad: np.ndarray
    gt_z: np.ndarray         # z-scored ground truth, raw orientation
    pred_z: np.ndarray       # z-scored ChemProp prediction, raw orientation
    masks: list[np.ndarray]  # one boolean mask per acquisition round


def set_paper_rc():
    """Bump every text element to one consistent size suitable for printing."""
    mpl.rcParams.update({
        "font.size":           TEXT_SIZE,
        "axes.titlesize":      TEXT_SIZE,
        "axes.labelsize":      TEXT_SIZE,
        "xtick.labelsize":     TEXT_SIZE,
        "ytick.labelsize":     TEXT_SIZE,
        "legend.fontsize":     TEXT_SIZE,
        "figure.titlesize":    TEXT_SIZE,
        "axes.linewidth":      1.2,
        "savefig.dpi":         150,
    })


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def z_score(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    mu = float(finite.mean())
    sigma = float(finite.std())
    if sigma <= 0:
        sigma = 1.0
    return ((values - mu) / sigma).astype(np.float32)


UMAP_CACHE_DIR = ANALYSIS_DIR / "_umap_cache"


def fit_umap(x: np.ndarray, label: str) -> np.ndarray:
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = UMAP_CACHE_DIR / f"{label}_{x.shape[0]}_{x.shape[1]}.npy"
    if cache_path.exists():
        coords = np.load(cache_path)
        print(f"  loaded cached UMAP for {label}: {coords.shape}", flush=True)
        return coords
    print(f"  fitting UMAP on {label}: {x.shape}", flush=True)
    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=RANDOM_SEED,
        n_jobs=1, low_memory=True,
    )
    coords = reducer.fit_transform(x)
    np.save(cache_path, coords)
    print(f"  UMAP fit {label} done in {(time.perf_counter()-t0)/60:.1f} min "
          f"-> shape {coords.shape} (cached)", flush=True)
    return coords


def load_train_masks(spec: RepSpec, smiles: np.ndarray) -> list[np.ndarray]:
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}
    masks: list[np.ndarray] = []
    for k in range(1, N_ROUNDS + 1):
        path = spec.train_subsets_dir / f"train_{k}.csv"
        df = pd.read_csv(path, usecols=["smiles"])
        m = np.zeros(len(smiles), dtype=bool)
        hit = 0
        for s in df["smiles"].astype(str):
            j = smiles_to_idx.get(s)
            if j is not None:
                m[j] = True
                hit += 1
        masks.append(m)
        print(f"  round {k}: train file has {len(df):,} rows, "
              f"{hit:,} mapped to pool indices", flush=True)
    return masks


def compute_rep(spec: RepSpec, relu_gradient: bool = False) -> Optional[RepData]:
    base = spec.pool_dir / spec.label
    needed = [
        spec.pool_dir / "pool_smiles.npz",
        base / "mpn_fp.npy", base / "grad_W.npy",
        base / "y_true.npy", base / "y_hat.npy",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print(f"[skip] {spec.display_name}: missing {missing}", flush=True)
        return None

    print(f"\n=== {spec.display_name} (relu_gradient={relu_gradient}) ===",
          flush=True)
    pool_smiles = np.load(spec.pool_dir / "pool_smiles.npz",
                          allow_pickle=True)["smiles"].astype(str)
    mpn = np.load(base / "mpn_fp.npy", mmap_mode="r")
    grad = np.load(base / "grad_W.npy", mmap_mode="r")
    y_true = np.asarray(np.load(base / "y_true.npy", mmap_mode="r"))
    y_hat = np.asarray(np.load(base / "y_hat.npy", mmap_mode="r"))
    valid = ~np.isnan(y_true)
    smiles = pool_smiles[valid]
    print(f"  {smiles.size:,} molecules with labels", flush=True)

    grad_arr = np.ascontiguousarray(grad[valid]).astype(np.float32)
    if relu_gradient:
        # last_ffn >= 0, so grad_W rows have the sign of the per-molecule
        # residual. ReLU keeps overpredicted rows unchanged and zeroes
        # underpredicted rows entirely.
        before_zero = int((np.abs(grad_arr).sum(axis=1) == 0).sum())
        grad_arr = np.maximum(grad_arr, 0.0)
        after_zero = int((grad_arr.sum(axis=1) == 0).sum())
        print(f"  relu(gradient): {after_zero - before_zero:,} rows "
              f"collapsed to all zeros ({(after_zero - before_zero) / max(grad_arr.shape[0],1):.1%} of pool)",
              flush=True)

    mpn_n = l2_normalize(np.ascontiguousarray(mpn[valid]).astype(np.float32))
    grad_n = l2_normalize(grad_arr)
    coords_embed = fit_umap(mpn_n, "embedding")
    coords_grad = fit_umap(grad_n, "gradient")

    # Raw kcal per mole orientation: y_true was sign flipped during
    # extraction (the --minimize convention), and y_hat is in that same flipped
    # space. Negate both to put red = poor binder, blue = good binder.
    gt_raw = -np.ascontiguousarray(y_true[valid]).astype(np.float32)
    pred_raw = -np.ascontiguousarray(y_hat[valid]).astype(np.float32)
    gt_z = z_score(gt_raw)
    pred_z = z_score(pred_raw)

    masks = load_train_masks(spec, smiles)
    return RepData(
        spec=spec, smiles=smiles,
        coords_embed=coords_embed, coords_grad=coords_grad,
        gt_z=gt_z, pred_z=pred_z, masks=masks,
    )


# --------- panel rendering helpers ---------

def panel_scatter(ax, coords: np.ndarray, color_values: np.ndarray,
                  mask: np.ndarray | None, norm, cmap, point_size=3.5):
    if mask is None:
        ax.scatter(coords[:, 0], coords[:, 1], c=color_values,
                   cmap=cmap, norm=norm, s=point_size, alpha=0.75,
                   linewidths=0, rasterized=True)
    else:
        not_mask = ~mask
        ax.scatter(coords[not_mask, 0], coords[not_mask, 1],
                   c="lightgrey", s=point_size * 0.55, alpha=0.18,
                   linewidths=0, rasterized=True)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color_values[mask], cmap=cmap, norm=norm,
                   s=point_size * 1.8, alpha=0.95, linewidths=0,
                   rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    for spine in ax.spines.values():
        spine.set_color("0.65")


def z_norm() -> Normalize:
    return Normalize(vmin=-Z_CLIP, vmax=Z_CLIP)


def cbar_label_for(score_kind: str) -> str:
    if score_kind == "gt":
        return "Ground truth docking score\n(standard deviations from the mean)"
    return "ChemProp predicted score\n(standard deviations from the mean)"


# --------- per-rep figure ---------

def build_per_rep_figure(rd: RepData, out_path: Path, relu_gradient: bool = False):
    cmap = mpl.colormaps["RdBu_r"]
    norm = z_norm()
    grad_row_label_prefix = ("ReLU of last layer gradient" if relu_gradient
                             else "Last layer gradient")
    rows = [
        ("Molecule embedding,\ncolored by ground truth\ndocking score",
         rd.coords_embed, rd.gt_z, "gt"),
        ("Molecule embedding,\ncolored by ChemProp\npredicted score",
         rd.coords_embed, rd.pred_z, "pred"),
        (f"{grad_row_label_prefix},\ncolored by ground truth\ndocking score",
         rd.coords_grad, rd.gt_z, "gt"),
        (f"{grad_row_label_prefix},\ncolored by ChemProp\npredicted score",
         rd.coords_grad, rd.pred_z, "pred"),
    ]
    train_sizes = [int(m.sum()) for m in rd.masks]
    col_titles = [f"Full pool of {len(rd.smiles):,} molecules"] + [
        f"Trained on {train_sizes[k]:,} molecules\n(iter {k+1})"
        for k in range(N_ROUNDS)
    ]
    n_rows = len(rows)
    n_cols = N_ROUNDS + 1

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(4.0 * n_cols + 4.5, 4.2 * n_rows + 2.0),
        squeeze=False,
        gridspec_kw=dict(left=0.16, right=0.91, top=0.92, bottom=0.03,
                         wspace=0.05, hspace=0.18),
    )

    for r_idx, (row_label, coords, values, kind) in enumerate(rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            mask = None if c_idx == 0 else rd.masks[c_idx - 1]
            panel_scatter(ax, coords, values, mask=mask, norm=norm, cmap=cmap)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=TEXT_SIZE)

        # Row label on left margin, vertically centred for this row
        row_y_top = 0.92
        row_y_bottom = 0.03
        row_h = (row_y_top - row_y_bottom) / n_rows
        row_y = row_y_top - (r_idx + 0.5) * row_h
        fig.text(0.012, row_y, row_label, fontsize=TEXT_SIZE,
                 ha="left", va="center", weight="bold")

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[r_idx, :].tolist(),
                          orientation="vertical",
                          fraction=0.012, pad=0.012, shrink=0.92)
        cb.set_label(cbar_label_for(kind), fontsize=TEXT_SIZE)
        cb.ax.tick_params(labelsize=TEXT_SIZE)

    grad_phrase = ("the ReLU of the last layer gradient"
                   if relu_gradient else "the last layer gradient")
    fig.suptitle(
        f"{rd.spec.display_name}: UMAP projections of the molecule "
        f"embedding and {grad_phrase}.\nColumn one is the full pool. "
        f"Columns two through seven highlight the cumulative training "
        f"subset at each acquisition round.",
        fontsize=TEXT_SIZE + 2, y=0.985,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# --------- cross-scorer figure ---------

def build_cross_scorer_figure(reps: list[RepData], representation: str,
                              score_kind: str, out_path: Path,
                              relu_gradient: bool = False):
    """3 columns (silcs, glide, vina) by 3 rows (replicate 1, 2, 3)."""
    cmap = mpl.colormaps["RdBu_r"]
    norm = z_norm()

    scorers = ["silcs", "glide", "vina"]
    reps_by_key = {(rd.spec.scorer, rd.spec.rep): rd for rd in reps}

    if representation == "embed":
        rep_titles_label = "Molecule embedding"
    elif relu_gradient:
        rep_titles_label = "ReLU of last layer gradient"
    else:
        rep_titles_label = "Last layer gradient"
    score_label = ("ground truth docking score"
                   if score_kind == "gt"
                   else "ChemProp predicted score")
    title = (f"{rep_titles_label} UMAP across docking scorers and "
             f"replicates,\ncolored by {score_label}.")

    n_rows = 3  # replicates
    n_cols = len(scorers)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(5.0 * n_cols + 4.0, 5.0 * n_rows + 2.0),
        squeeze=False,
        gridspec_kw=dict(left=0.135, right=0.90, top=0.92, bottom=0.04,
                         wspace=0.09, hspace=0.13),
    )

    for r_idx in range(n_rows):
        rep_num = r_idx + 1
        for c_idx, scorer in enumerate(scorers):
            ax = axes[r_idx, c_idx]
            key = (scorer, rep_num)
            if key not in reps_by_key:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TEXT_SIZE)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                for s in ax.spines.values():
                    s.set_color("0.65")
                continue
            rd = reps_by_key[key]
            coords = rd.coords_embed if representation == "embed" else rd.coords_grad
            values = rd.gt_z if score_kind == "gt" else rd.pred_z
            panel_scatter(ax, coords, values, mask=None, norm=norm, cmap=cmap,
                          point_size=4.0)
            if r_idx == 0:
                ax.set_title(SCORER_DISPLAY[scorer], fontsize=TEXT_SIZE + 2)

    fig.canvas.draw()

    for r_idx in range(n_rows):
        rep_num = r_idx + 1
        ax = axes[r_idx, 0]
        bbox = ax.get_position()
        row_y = (bbox.y0 + bbox.y1) / 2
        fig.text(bbox.x0 - 0.04, row_y, f"Replicate {rep_num}",
                 fontsize=TEXT_SIZE + 2, ha="center", va="center",
                 weight="bold", rotation=90,
                 transform=fig.transFigure)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    top_bb = axes[0, -1].get_position()
    bot_bb = axes[-1, -1].get_position()
    grid_center = (top_bb.y1 + bot_bb.y0) / 2
    single_h = top_bb.y1 - top_bb.y0
    cb_h = single_h * 2.4
    cb_ax = fig.add_axes([top_bb.x1 + 0.01, grid_center - cb_h / 2, 0.012, cb_h])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label(cbar_label_for(score_kind), fontsize=TEXT_SIZE, rotation=270,
                 labelpad=42)
    cb.ax.tick_params(labelsize=TEXT_SIZE)

    fig.suptitle(title, fontsize=TEXT_SIZE + 2, y=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def build_per_engine_figure(engine_reps: list[RepData], engine: str,
                            out_path: Path, relu_gradient: bool = False):
    """Generate two variants of the per-engine SI figure:
       1. Side-label variant (vertical text on left/right)
       2. Top-label variant (horizontal text above each section)
    """
    for mode in ("side", "top"):
        if mode == "top":
            stem = out_path.stem + "_toplabel"
            p = out_path.with_name(stem + out_path.suffix)
        else:
            p = out_path
        _build_per_engine(engine_reps, engine, p, mode=mode)


def _build_per_engine(engine_reps: list[RepData], engine: str,
                      out_path: Path, mode: str = "side"):
    cmap = mpl.colormaps["RdBu_r"]
    norm = z_norm()

    engine_reps = sorted(engine_reps, key=lambda rd: rd.spec.rep)
    n_reps = len(engine_reps)
    n_cols = N_ROUNDS + 1
    S = 24

    train_sizes = [int(m.sum()) for m in engine_reps[0].masks]
    col_titles = (
        [f"Full pool of\n{len(engine_reps[0].smiles):,} molecules"]
        + [f"Trained on {train_sizes[k]:,}\n(iter {k+1})"
           for k in range(N_ROUNDS)]
    )

    sec_labels = [
        ("gt", "Molecule embedding, colored by ground truth docking score"),
        ("pred", "Molecule embedding, colored by ChemProp predicted score"),
    ]

    total_rows = n_reps * 2 + 1
    if mode == "top":
        height_ratios = [1] * n_reps + [0.14] + [1] * n_reps
        fig = plt.figure(figsize=(4.5 * n_cols, 4.2 * n_reps * 2 + 5.0))
        gs = fig.add_gridspec(
            nrows=total_rows, ncols=n_cols,
            height_ratios=height_ratios,
            left=0.02, right=0.91, top=0.85, bottom=0.01,
            wspace=0.10, hspace=0.09)
    else:
        height_ratios = [1] * n_reps + [0.08] + [1] * n_reps
        fig = plt.figure(figsize=(4.5 * n_cols, 4.2 * n_reps * 2 + 2.0))
        gs = fig.add_gridspec(
            nrows=total_rows, ncols=n_cols,
            height_ratios=height_ratios,
            left=0.025, right=0.91, top=0.88, bottom=0.01,
            wspace=0.10, hspace=0.09)

    axes_top = np.empty((n_reps, n_cols), dtype=object)
    axes_bot = np.empty((n_reps, n_cols), dtype=object)
    for r in range(n_reps):
        for c in range(n_cols):
            axes_top[r, c] = fig.add_subplot(gs[r, c])
            axes_bot[r, c] = fig.add_subplot(gs[n_reps + 1 + r, c])

    for sec_idx, (score_kind, _) in enumerate(sec_labels):
        axes_sec = axes_top if sec_idx == 0 else axes_bot
        for rep_idx, rd in enumerate(engine_reps):
            values = rd.gt_z if score_kind == "gt" else rd.pred_z
            for c_idx in range(n_cols):
                ax = axes_sec[rep_idx, c_idx]
                mask = None if c_idx == 0 else rd.masks[c_idx - 1]
                panel_scatter(ax, rd.coords_embed, values, mask=mask,
                              norm=norm, cmap=cmap, point_size=3.0)
                if sec_idx == 0 and rep_idx == 0:
                    ax.set_title(col_titles[c_idx], fontsize=S)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.canvas.draw()
        sec_top_bb = axes_sec[0, -1].get_position()
        sec_bot_bb = axes_sec[-1, -1].get_position()
        sec_center = (sec_top_bb.y1 + sec_bot_bb.y0) / 2
        sec_single_h = sec_top_bb.y1 - sec_top_bb.y0
        sec_cb_h = sec_single_h * 2.4
        sec_cb_ax = fig.add_axes([sec_top_bb.x1 + 0.01,
                                  sec_center - sec_cb_h / 2,
                                  0.012, sec_cb_h])
        cb = fig.colorbar(sm, cax=sec_cb_ax)
        cb.set_label(cbar_label_for(score_kind), fontsize=S, rotation=270,
                     labelpad=42)
        cb.ax.tick_params(labelsize=S)

    fig.canvas.draw()

    # Rep labels — vertical on the left
    for sec_idx in range(2):
        axes_sec = axes_top if sec_idx == 0 else axes_bot
        for rep_idx, rd in enumerate(engine_reps):
            ax = axes_sec[rep_idx, 0]
            bbox = ax.get_position()
            row_y = (bbox.y0 + bbox.y1) / 2
            fig.text(bbox.x0 - 0.025, row_y, f"Replicate {rd.spec.rep}",
                     fontsize=S, ha="center", va="center",
                     weight="bold", rotation=90,
                     transform=fig.transFigure)

    if mode == "side":
        for sec_idx, (_, label) in enumerate(sec_labels):
            axes_sec = axes_top if sec_idx == 0 else axes_bot
            left_bb = axes_sec[0, 0].get_position()
            top_bb = axes_sec[0, 0].get_position()
            bot_bb = axes_sec[-1, 0].get_position()
            sec_y = (top_bb.y1 + bot_bb.y0) / 2
            fig.text(left_bb.x0 - 0.008, sec_y, label,
                     fontsize=S, ha="center", va="center",
                     weight="bold", rotation=90)
        fig.suptitle(SCORER_DISPLAY[engine], fontsize=S + 10, y=0.95,
                     weight="bold")
    else:
        top_sec_top_bb = axes_top[0, 0].get_position()
        top_sec_bot_bb = axes_top[-1, 0].get_position()
        bot_sec_top_bb = axes_bot[0, 0].get_position()
        gap_center = (top_sec_bot_bb.y0 + bot_sec_top_bb.y1) / 2

        left_x = axes_top[0, 0].get_position().x0
        sec1_label_y = top_sec_top_bb.y1 + 0.035
        title_y = top_sec_top_bb.y1 + 0.065

        fig.text(left_x, sec1_label_y, sec_labels[0][1],
                 fontsize=S, ha="left", va="center", weight="bold")
        fig.text(left_x, gap_center, sec_labels[1][1],
                 fontsize=S, ha="left", va="center", weight="bold")

        fig.text(0.5, title_y, SCORER_DISPLAY[engine],
                 fontsize=S + 10, ha="center", va="center",
                 weight="bold", transform=fig.transFigure)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def build_combined_figure(reps_data: list[RepData], out_path: Path):
    """Combined figure: (a) SVM separability bar chart, (b) cross_embed_gt.

    Panel (a) is regenerated from the saved CSV.
    Panel (b) is a 3x3 grid (scorer × replicate) of embedding UMAPs
    colored by ground truth docking score.
    """
    S = 22

    csv_path = ANALYSIS_DIR / "separability_auc.csv"
    if not csv_path.exists():
        print(f"[skip] combined figure: {csv_path} not found. "
              f"Run svm_separability.py first.", flush=True)
        return

    auc_df = pd.read_csv(csv_path)

    scorers = ["silcs", "glide", "vina"]
    feature_order = ["mpn_fp", "grad_W"]
    feature_labels = {"mpn_fp": "Molecule embedding",
                      "grad_W": "Last layer gradient"}
    feature_colors = {"mpn_fp": "#4C78A8", "grad_W": "#F58518"}

    sub = auc_df[auc_df["percent"] == 25]
    agg = (sub.groupby(["scorer", "feature"])
              .agg(mean=("auc_mean", "mean"),
                   std=("auc_mean", "std"))
              .reset_index())

    cmap = mpl.colormaps["RdBu_r"]
    norm = z_norm()

    fig = plt.figure(figsize=(24, 14))
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.8],
                                left=0.06, right=0.94, top=0.92,
                                bottom=0.05, wspace=0.12)

    # --- Panel (a): bar chart ---
    ax_bar = fig.add_subplot(gs_outer[0, 0])
    bar_width = 0.32
    x_centres = np.arange(len(scorers))
    for i, feat in enumerate(feature_order):
        means = [float(agg[(agg["scorer"] == s) & (agg["feature"] == feat)]["mean"].iloc[0])
                 for s in scorers]
        stds = [float(agg[(agg["scorer"] == s) & (agg["feature"] == feat)]["std"].iloc[0])
                for s in scorers]
        offset = (i - 0.5) * bar_width
        ax_bar.bar(x_centres + offset, means, width=bar_width,
                   yerr=stds, capsize=5, label=feature_labels[feat],
                   color=feature_colors[feat])

    ax_bar.set_xticks(x_centres)
    ax_bar.set_xticklabels([SCORER_DISPLAY[s] for s in scorers], fontsize=S)
    ax_bar.set_ylabel("")
    ax_bar.set_title("Cross validated ROC AUC", fontsize=S)
    ax_bar.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    ax_bar.set_ylim(0.45, 1.05)
    ax_bar.legend(loc="upper right", frameon=False, fontsize=S - 2)
    ax_bar.tick_params(labelsize=S - 1)
    ax_bar.grid(axis="y", alpha=0.3)

    # --- Panel (b): cross_embed_gt 3x3 ---
    gs_right = gs_outer[0, 1].subgridspec(3, 3, wspace=0.12, hspace=0.1)
    reps_by_key = {(rd.spec.scorer, rd.spec.rep): rd for rd in reps_data}

    for r_idx in range(3):
        rep_num = r_idx + 1
        for c_idx, scorer in enumerate(scorers):
            ax = fig.add_subplot(gs_right[r_idx, c_idx])
            key = (scorer, rep_num)
            if key not in reps_by_key:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=S)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                continue
            rd = reps_by_key[key]
            panel_scatter(ax, rd.coords_embed, rd.gt_z, mask=None,
                          norm=norm, cmap=cmap, point_size=4.0)
            if r_idx == 0:
                ax.set_title(SCORER_DISPLAY[scorer], fontsize=S + 2)
            if c_idx == 0:
                ax.set_ylabel(f"Replicate {rep_num}", fontsize=S,
                              weight="bold")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb_ax = fig.add_axes([0.95, 0.15, 0.012, 0.65])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Ground truth docking score\n(standard deviations from the mean)",
                 fontsize=S, rotation=270, labelpad=42)
    cb.ax.tick_params(labelsize=S - 1)

    fig.text(0.03, 0.95, "(a)", fontsize=S + 6, weight="bold")
    fig.text(0.42, 0.95, "(b)", fontsize=S + 6, weight="bold")

    fig.text(0.67, 0.96,
             "Molecule embedding UMAP across docking scorers and replicates,\n"
             "colored by ground truth docking score.",
             fontsize=S, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(ANALYSIS_DIR), type=Path)
    p.add_argument("--scorer", choices=["glide", "silcs", "vina"], default=None,
                   help="Restrict to this scorer for per-rep figures.")
    p.add_argument("--rep", type=int, default=None, choices=[1, 2, 3])
    p.add_argument("--skip-per-rep", action="store_true",
                   help="Only generate cross-scorer and per-engine figures.")
    p.add_argument("--skip-cross", action="store_true",
                   help="Only generate per-rep and per-engine figures.")
    p.add_argument("--skip-per-engine", action="store_true",
                   help="Skip per-engine SI figures.")
    p.add_argument("--skip-combined", action="store_true",
                   help="Skip combined (a) separability + (b) cross embed figure.")
    p.add_argument("--relu-gradient", action="store_true",
                   help="Apply elementwise ReLU to grad_W before "
                        "L2-normalisation and UMAP. Output filenames get "
                        "a '_relugrad' suffix.")
    args = p.parse_args()
    output_dir = Path(args.output_dir)
    set_paper_rc()

    specs = SPECS
    if args.scorer is not None:
        specs = [s for s in specs if s.scorer == args.scorer]
    if args.rep is not None:
        specs = [s for s in specs if s.rep == args.rep]

    suffix = "_relugrad" if args.relu_gradient else ""

    # Compute and cache every requested rep so cross-scorer figures can
    # reuse the same UMAP fits.
    reps_data: list[RepData] = []
    for spec in specs:
        rd = compute_rep(spec, relu_gradient=args.relu_gradient)
        if rd is not None:
            reps_data.append(rd)

    if not args.skip_per_rep:
        for rd in reps_data:
            out_path = output_dir / f"{rd.spec.scorer}_rep{rd.spec.rep}{suffix}.png"
            build_per_rep_figure(rd, out_path, relu_gradient=args.relu_gradient)

    if not args.skip_cross:
        scorers_present = {rd.spec.scorer for rd in reps_data}
        if scorers_present >= {"silcs", "glide", "vina"}:
            for representation in ("embed", "grad"):
                # Only the gradient panels change under --relu-gradient; the
                # embedding panels are identical to the raw run, so we skip
                # them to avoid duplicating identical figures.
                if args.relu_gradient and representation == "embed":
                    continue
                for score_kind in ("gt", "pred"):
                    out_path = (output_dir /
                                f"cross_{representation}_{score_kind}{suffix}.png")
                    build_cross_scorer_figure(
                        reps_data, representation=representation,
                        score_kind=score_kind, out_path=out_path,
                        relu_gradient=args.relu_gradient,
                    )
        else:
            print(f"[note] skipping cross-scorer figures: only "
                  f"{sorted(scorers_present)} present", flush=True)

    if not args.skip_combined:
        scorers_present = {rd.spec.scorer for rd in reps_data}
        if scorers_present >= {"silcs", "glide", "vina"}:
            combined_path = output_dir / "combined_separability_embed.png"
            build_combined_figure(reps_data, combined_path)
        else:
            print(f"[note] skipping combined figure: only "
                  f"{sorted(scorers_present)} present", flush=True)

    if not args.skip_per_engine:
        from collections import defaultdict
        by_engine: dict[str, list[RepData]] = defaultdict(list)
        for rd in reps_data:
            by_engine[rd.spec.scorer].append(rd)
        for engine, ereps in sorted(by_engine.items()):
            if len(ereps) < 2:
                print(f"[note] skipping per-engine figure for {engine}: "
                      f"only {len(ereps)} reps", flush=True)
                continue
            out_path = output_dir / f"{engine}_per_engine{suffix}.png"
            build_per_engine_figure(ereps, engine, out_path,
                                    relu_gradient=args.relu_gradient)

    return 0


if __name__ == "__main__":
    sys.exit(main())
