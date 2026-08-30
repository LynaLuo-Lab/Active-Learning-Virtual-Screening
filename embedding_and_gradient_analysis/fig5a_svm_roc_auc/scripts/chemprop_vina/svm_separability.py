"""Linear separability test on MolPAL embeddings and gradients.

Procedure per (scorer, replicate, feature type, threshold percentile p):
  1. Load the feature matrix X (either mpn_fp or grad_W) and the ground
     truth raw kcal per mole binding affinity y_raw.
  2. Define the binary label by taking the top p percent (the most negative
     raw scores = good binders, class 1) and the bottom p percent (poor
     binders, class 0). Drop the middle.
  3. Standardise X with StandardScaler inside each CV fold (no leakage).
  4. Fit a soft margin LinearSVC (default C=1, hinge loss) on the standard
     scaled X. Use decision_function values for ROC-AUC scoring.
  5. Five fold stratified cross validation; report mean and standard
     deviation across folds.

Outputs:
  analysis/umaps/separability_auc.csv  -- one row per
      (scorer, rep, feature, threshold) with auc_mean, auc_std, n_pos, n_neg.
  analysis/umaps/separability_auc.png  -- bar chart of AUC by scorer,
      grouped by feature type, at the canonical threshold.
      
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
EMB_ROOT = REPO / "results" / "embeddings"
BENCHMARK_DIR = REPO / "benchmark_data" / "benchmark_40k"
ANALYSIS_DIR = REPO / "analysis" / "umaps"

THRESHOLDS = [10, 25]  # top/bottom percentiles to compare
PRIMARY_THRESHOLD = 25  # the one used for the bar chart
FEATURES = ("mpn_fp", "grad_W")
FEATURE_LABEL = {
    "mpn_fp": "Molecule embedding",
    "grad_W": "Last layer gradient",
}
RANDOM_SEED = 42
N_FOLDS = 5

SCORER_DISPLAY = {"silcs": "SILCS", "glide": "Glide", "vina": "Vina"}
SVM_C = 1.0


@dataclass(frozen=True)
class RepSpec:
    scorer: str
    rep: int
    pool_dir: Path
    label: str


SPECS: list[RepSpec] = [
    RepSpec("silcs", 1, EMB_ROOT / "pool_40k_silcs", "silcs_rep1"),
    RepSpec("silcs", 2, EMB_ROOT / "pool_40k_silcs", "silcs_rep2"),
    RepSpec("silcs", 3, EMB_ROOT / "pool_40k_silcs", "silcs_rep3"),
    RepSpec("glide", 1, EMB_ROOT / "pool_40k_A", "glide_rep1"),
    RepSpec("vina",  1, EMB_ROOT / "pool_40k_A", "vina_rep1"),
    RepSpec("vina",  2, EMB_ROOT / "pool_40k_A", "vina_rep2"),
    RepSpec("vina",  3, EMB_ROOT / "pool_40k_A", "vina_rep3"),
    RepSpec("glide", 2, EMB_ROOT / "pool_40k_B", "glide_rep2"),
    RepSpec("glide", 3, EMB_ROOT / "pool_40k_B", "glide_rep3"),
]


def load_features_and_labels(spec: RepSpec, feature: str):
    """Returns X [N,300], y_raw [N] (raw kcal per mole, NaN dropped)."""
    base = spec.pool_dir / spec.label
    X = np.asarray(np.load(base / f"{feature}.npy", mmap_mode="r"), dtype=np.float32)
    y_stored = np.asarray(np.load(base / "y_true.npy", mmap_mode="r"))
    valid = ~np.isnan(y_stored)
    X = np.ascontiguousarray(X[valid])
    y_raw = (-y_stored[valid]).astype(np.float64)
    return X, y_raw


def make_binary_split(y_raw: np.ndarray, percent: float):
    """Top `percent` percent by binding affinity (most negative raw) is class 1.
    Bottom `percent` percent (poor binders, most positive raw) is class 0.
    Returns a boolean keep mask and a binary y array, restricted to kept rows.
    """
    lo_thresh = np.percentile(y_raw, percent)         # below this = top binder
    hi_thresh = np.percentile(y_raw, 100 - percent)   # above this = poor binder
    keep_top = y_raw <= lo_thresh                     # good binders (class 1)
    keep_bot = y_raw >= hi_thresh                     # poor binders (class 0)
    keep = keep_top | keep_bot
    y_bin = np.zeros(int(keep.sum()), dtype=np.int64)
    # Recompute on kept rows so indices line up
    kept_raw = y_raw[keep]
    y_bin[kept_raw <= lo_thresh] = 1
    y_bin[kept_raw >= hi_thresh] = 0
    return keep, y_bin


def cv_auc(X: np.ndarray, y: np.ndarray, n_folds: int, c: float, seed: int):
    """Stratified k fold CV ROC AUC with linear soft SVM."""
    aucs: list[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=c, dual="auto", max_iter=5000)),
        ])
        pipe.fit(X[tr], y[tr])
        scores = pipe.decision_function(X[va])
        aucs.append(float(roc_auc_score(y[va], scores)))
    return float(np.mean(aucs)), float(np.std(aucs))


def run() -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        for feature in FEATURES:
            X, y_raw = load_features_and_labels(spec, feature)
            for pct in THRESHOLDS:
                keep, y_bin = make_binary_split(y_raw, pct)
                X_kept = X[keep]
                t0 = time.perf_counter()
                auc_m, auc_s = cv_auc(X_kept, y_bin, N_FOLDS, SVM_C, RANDOM_SEED)
                elapsed = time.perf_counter() - t0
                print(f"  {spec.label:14s}  feature={feature:8s} "
                      f"pct={pct:>2d}%  n+={int((y_bin==1).sum()):,} "
                      f"n-={int((y_bin==0).sum()):,}  "
                      f"AUC={auc_m:.4f}±{auc_s:.4f}  ({elapsed:.1f}s)",
                      flush=True)
                rows.append(dict(
                    scorer=spec.scorer, rep=spec.rep, feature=feature,
                    percent=pct, n_pos=int((y_bin == 1).sum()),
                    n_neg=int((y_bin == 0).sum()),
                    auc_mean=auc_m, auc_std=auc_s,
                ))
    return pd.DataFrame(rows)


def aggregate_by_scorer(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across replicates for each (scorer, feature, percent)."""
    return (df.groupby(["scorer", "feature", "percent"])
              .agg(auc_mean_across_reps=("auc_mean", "mean"),
                   auc_std_across_reps=("auc_mean", "std"),
                   n_reps=("rep", "count"))
              .reset_index())


def make_bar_figure(df: pd.DataFrame, out_path: Path):
    """Bar chart of AUC per (scorer x feature), errorbars across replicates,
    at the primary threshold. Hard horizontal line at 0.5 for chance.
    """
    sub = df[df["percent"] == PRIMARY_THRESHOLD]
    agg = (sub.groupby(["scorer", "feature"])
              .agg(mean=("auc_mean", "mean"),
                   std=("auc_mean", "std"))
              .reset_index())

    scorers = ["silcs", "glide", "vina"]
    feature_order = ["mpn_fp", "grad_W"]
    bar_width = 0.35
    x_centres = np.arange(len(scorers))

    mpl.rcParams.update({"font.size": 14, "axes.titlesize": 16,
                         "axes.labelsize": 14})
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, feat in enumerate(feature_order):
        means = [
            float(agg[(agg["scorer"] == s) & (agg["feature"] == feat)]["mean"].iloc[0])
            for s in scorers
        ]
        stds = [
            float(agg[(agg["scorer"] == s) & (agg["feature"] == feat)]["std"].iloc[0])
            for s in scorers
        ]
        offset = (i - 0.5) * bar_width
        ax.bar(x_centres + offset, means, width=bar_width,
               yerr=stds, capsize=4,
               label=FEATURE_LABEL[feat],
               color=("#4C78A8" if feat == "mpn_fp" else "#F58518"))

    ax.set_xticks(x_centres)
    ax.set_xticklabels([SCORER_DISPLAY[s] for s in scorers])
    ax.set_ylabel("Cross validated ROC area under curve")
    ax.set_title(
        f"Linear separability of top {PRIMARY_THRESHOLD}% vs\n"
        f"bottom {PRIMARY_THRESHOLD}% docking score"
    )
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    ax.text(len(scorers) - 0.5, 0.51, "chance", color="0.4",
            ha="right", va="bottom", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(ANALYSIS_DIR), type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = run()
    csv_path = args.output_dir / "separability_auc.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}", flush=True)

    agg = aggregate_by_scorer(df)
    print("\n=== aggregated across replicates ===")
    print(agg.to_string(index=False))

    fig_path = args.output_dir / "separability_auc.png"
    make_bar_figure(df, fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
