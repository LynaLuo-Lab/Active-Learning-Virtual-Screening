"""Recovery@1% plots and summary table for the ChemProp v2 MolPAL replica.

Same-dataset recovery only: 40k-trained cells are scored on the 40k dataset,
1.29M-trained cells on the 1.29M dataset. Recovery is read from each cell's
recovery.json (recovery_at_1pct_per_iter), which is already same-dataset.

The 1.29M 2% (bs2) cells were not all completed: the glide reps and vina rep3
stopped at iteration 4. For those, recovery is reconstructed from the
per-iteration acquired molecules (per_iter/iter_k/acquired.csv) up to the common
depth (iteration 4) so glide and vina are compared at equal exploration.

Plots: mean over replicates with +/- std error bars on every point. Colors match
the main paper (Vina green, Glide blue, SILCS black). Tables carry mean +/- std.

Outputs (under analysis/recovery/)
  recovery_40k_1pct_5pct.png   2-panel (1% and 5% batch), 3 workflows
  recovery_40k_batchsweep.png  final recovery@1% vs batch size, line per workflow
  recovery_1p3M.png            1.29M recovery curves (0.25 / 0.5 / 1 / 2% batch)
  recovery_summary.csv         every cell
  recovery_summary_table.png   recovery@1% summary table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis" / "recovery"

SWEEP_40K = PROJECT_ROOT / "results" / "recovery_batch_sweep_v1"
SWEEP_1P3M = PROJECT_ROOT / "results" / "recovery_batch_sweep_v2"
TRUTH_1P3M = {
    "glide": PROJECT_ROOT / "benchmark_data" / "glide" / "df.csv",
    "vina": PROJECT_ROOT / "benchmark_data" / "vina" / "1.3M" / "vina_df.csv",
}

BS_PCT = {"bs0.25": 0.25, "bs0.5": 0.5, "bs1": 1.0, "bs2": 2.0, "bs4": 4.0, "bs5": 5.0}

SCORERS = ["vina", "glide", "silcs"]
SCORER_DISPLAY = {"vina": "Vina-MolPAL", "glide": "Glide-MolPAL", "silcs": "SILCS-MolPAL"}
SCORER_COLOR = {"vina": "#2ca02c", "glide": "#1f77b4", "silcs": "#000000"}
SCORER_MARKER = {"vina": "^", "glide": "s", "silcs": "o"}

REPS = (1, 2, 3)
TEXT = 15


def load_cell(base: Path):
    """Mean/std over reps of recovery_at_1pct_per_iter. Returns (explored, mean, std)."""
    curves, n_init, n_per = [], None, None
    for r in REPS:
        rj = base / f"rep{r}" / "recovery.json"
        if not rj.exists():
            continue
        d = json.loads(rj.read_text())
        curves.append(np.asarray(d["recovery_at_1pct_per_iter"], dtype=float))
        n_init, n_per = d["n_init"], d["n_per_iter"]
    if not curves:
        return None
    K = max(len(c) for c in curves)
    padded = np.array([np.pad(c, (0, K - len(c)), mode="edge") for c in curves])
    explored = n_init + np.arange(K) * n_per
    return explored, padded.mean(0) * 100.0, padded.std(0) * 100.0


def collect(sweep_root: Path, scorers: list[str]) -> dict:
    cells = {}
    for bs_key in BS_PCT:
        for sc in scorers:
            res = load_cell(sweep_root / bs_key / sc)
            if res is not None:
                cells[(bs_key, sc)] = res
    return cells


# --- 1.29M bs2 reconstruction from per_iter (incomplete cells) ---

def _top1pct(sc: str) -> set:
    df = pd.read_csv(TRUTH_1P3M[sc])
    df["smiles"] = df["smiles"].astype(str)
    n1 = int(round(0.01 * len(df)))
    return set(df.nsmallest(n1, "score")["smiles"])


def reconstruct_bs2(sc: str, top: set, max_iter: int = 4):
    """Cumulative greedy recovery from per_iter through max_iter, mean/std over reps.

    per_iter excludes the random init batch, so this is a slight (uniform) lower
    bound on the completed recovery; used only for the not-finished bs2 cells so
    glide and vina are shown to the same depth.
    """
    base = SWEEP_1P3M / "bs2" / sc
    rep_curves = []
    n_per = None
    for r in REPS:
        pdir = base / f"rep{r}" / "per_iter"
        if not pdir.exists():
            continue
        its = sorted(pdir.glob("iter_*"), key=lambda x: int(x.name.split("_")[1]))
        acc, curve = set(), []
        for it in its[:max_iter]:
            f = it / "acquired.csv"
            if not f.exists():
                break
            df = pd.read_csv(f)
            acc |= set(df["smiles"].astype(str))
            curve.append(len(acc & top) / len(top))
            n_per = len(df)
        if curve:
            rep_curves.append(curve)
    if not rep_curves:
        return None
    K = min(len(c) for c in rep_curves)
    arr = np.array([c[:K] for c in rep_curves])
    # x = molecules explored incl. init batch: greedy step k (1-indexed) -> (k+1)*batch
    explored = n_per * (np.arange(1, K + 1) + 1)
    return explored, arr.mean(0) * 100.0, arr.std(0) * 100.0


def set_rc():
    mpl.rcParams.update({
        "font.size": TEXT, "axes.titlesize": TEXT + 1, "axes.labelsize": TEXT,
        "xtick.labelsize": TEXT - 2, "ytick.labelsize": TEXT - 2,
        "legend.fontsize": TEXT - 3, "savefig.dpi": 150, "axes.linewidth": 1.1,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def _plot_curve(ax, x, mean, std, sc, label=None):
    ax.errorbar(x, mean, yerr=std, color=SCORER_COLOR[sc], marker=SCORER_MARKER[sc],
                capsize=3, elinewidth=1.2, markeredgecolor="none",
                label=label or SCORER_DISPLAY[sc])


def fig_40k_1pct_5pct(cells_40k: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, bs_key, title in [(axes[0], "bs1", "1% Batch Size"),
                              (axes[1], "bs5", "5% Batch Size")]:
        for sc in SCORERS:
            if (bs_key, sc) in cells_40k:
                _plot_curve(ax, *cells_40k[(bs_key, sc)], sc)
        ax.set_title(title)
        ax.set_xlabel("Molecules Explored")
        ax.set_ylabel("Top 1% Recovery Rate (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    p = OUT_DIR / "recovery_40k_1pct_5pct.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"[saved] {p}")


def fig_40k_batchsweep(cells_40k: dict):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for sc in SCORERS:
        xs, ys, es = [], [], []
        for bs_key in BS_PCT:
            if (bs_key, sc) in cells_40k:
                _e, mean, std = cells_40k[(bs_key, sc)]
                xs.append(BS_PCT[bs_key]); ys.append(mean[-1]); es.append(std[-1])
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=SCORER_COLOR[sc], marker=SCORER_MARKER[sc],
                        capsize=3, elinewidth=1.2, markeredgecolor="none",
                        label=SCORER_DISPLAY[sc])
    ax.set_xlabel("Batch Size (%)")
    ax.set_ylabel("Top 1% Recovery Rate (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    p = OUT_DIR / "recovery_40k_batchsweep.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"[saved] {p}")


def fig_1p3M(cells_1p3m: dict, bs2_recon: dict):
    panels = [("bs0.25", "0.25% Batch Size"), ("bs0.5", "0.5% Batch Size"),
              ("bs1", "1% Batch Size"), ("bs2", "2% Batch Size")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.0),
                             squeeze=False)
    for ax, (bs_key, title) in zip(axes[0], panels):
        for sc in ("vina", "glide"):
            if bs_key == "bs2":
                if sc in bs2_recon:
                    _plot_curve(ax, *bs2_recon[sc], sc)
            elif (bs_key, sc) in cells_1p3m:
                _plot_curve(ax, *cells_1p3m[(bs_key, sc)], sc)
        ax.set_title(title)
        ax.set_xlabel("Molecules Explored")
        ax.set_ylabel("Top 1% Recovery Rate (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    p = OUT_DIR / "recovery_1p3M.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"[saved] {p}")


def write_csv(cells_40k, cells_1p3m, bs2_recon):
    rows = []
    for dataset, cells in [("40k", cells_40k), ("1.29M", cells_1p3m)]:
        for (bs_key, sc), (_e, mean, std) in sorted(cells.items()):
            rows.append(dict(dataset=dataset, batch_pct=BS_PCT[bs_key], workflow=sc,
                             final_recovery_pct_mean=round(float(mean[-1]), 3),
                             final_recovery_pct_std=round(float(std[-1]), 3),
                             note=""))
    for sc, (_e, mean, std) in bs2_recon.items():
        rows.append(dict(dataset="1.29M", batch_pct=2.0, workflow=sc,
                         final_recovery_pct_mean=round(float(mean[-1]), 3),
                         final_recovery_pct_std=round(float(std[-1]), 3),
                         note="iter4 equal-depth, init excluded"))
    p = OUT_DIR / "recovery_summary.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {p}  ({len(rows)} cells)")


def fig_summary_table(cells_40k, cells_1p3m):
    def s(cells, bs_key, sc):
        if (bs_key, sc) not in cells:
            return "N/A"
        _e, mean, std = cells[(bs_key, sc)]
        return f"{mean[-1]:.2f} ± {std[-1]:.2f}"

    cols = ["Workflow", "Top-1% Recovery\n(40k, 1%)", "Top-1% Recovery\n(40k, 5%)",
            "Top-1% Recovery\n(1.29M, 1%)"]
    body = [[SCORER_DISPLAY[sc], s(cells_40k, "bs1", sc), s(cells_40k, "bs5", sc),
             s(cells_1p3m, "bs1", sc)] for sc in SCORERS]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    t = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2.3)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
    p = OUT_DIR / "recovery_summary_table.png"
    fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig); print(f"[saved] {p}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_rc()
    cells_40k = collect(SWEEP_40K, SCORERS)
    cells_1p3m = collect(SWEEP_1P3M, ["glide", "vina"])
    bs2_recon = {}
    for sc in ("vina", "glide"):
        top = _top1pct(sc)
        rec = reconstruct_bs2(sc, top, max_iter=4)
        if rec is not None:
            bs2_recon[sc] = rec
    print(f"40k cells: {len(cells_40k)}   1.29M cells: {len(cells_1p3m)}   "
          f"bs2 reconstructed: {sorted(bs2_recon)}")
    fig_40k_1pct_5pct(cells_40k)
    fig_40k_batchsweep(cells_40k)
    fig_1p3M(cells_1p3m, bs2_recon)
    write_csv(cells_40k, cells_1p3m, bs2_recon)
    fig_summary_table(cells_40k, cells_1p3m)


if __name__ == "__main__":
    main()
