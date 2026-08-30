"""Bivariate GMM(k=2) overlap analysis on ChemProp embeddings vs docking score.

For each (scorer, replicate):
  1. Load the 300-d mpn_fp embeddings and ground truth docking scores.
  2. Project embeddings to 1-D via PCA (first principal component).
  3. Form the 2-D matrix X = [PC1, gt_score] (standardised).
  4. Fit an unsupervised GaussianMixture(n_components=2) — let it find
     natural clusters without forced label splitting.
  5. Plot scatter colored by GMM assignment + 1σ/2σ contour ellipses.
  6. Compute the Bhattacharyya distance between the two fitted components.

Outputs:
  analysis/umaps/gmm_overlap_scatter.png  -- multi-panel scatter + ellipses
  analysis/umaps/gmm_overlap_bar.png      -- bar chart of Bhattacharyya distance
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
EMB_ROOT = REPO / "results" / "embeddings"
ANALYSIS_DIR = REPO / "analysis" / "umaps"

SCORER_DISPLAY = {"silcs": "SILCS", "glide": "Glide", "vina": "Vina"}
RANDOM_SEED = 42
TEXT_SIZE = 20


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
    RepSpec("glide", 2, EMB_ROOT / "pool_40k_B", "glide_rep2"),
    RepSpec("glide", 3, EMB_ROOT / "pool_40k_B", "glide_rep3"),
    RepSpec("vina", 1, EMB_ROOT / "pool_40k_A", "vina_rep1"),
    RepSpec("vina", 2, EMB_ROOT / "pool_40k_A", "vina_rep2"),
    RepSpec("vina", 3, EMB_ROOT / "pool_40k_A", "vina_rep3"),
]


def bhattacharyya_distance_2d(mu1, cov1, mu2, cov2):
    """Bhattacharyya distance between two 2-D Gaussians."""
    cov_avg = (cov1 + cov2) / 2.0
    diff = mu1 - mu2
    inv_cov_avg = np.linalg.inv(cov_avg)
    term1 = (1.0 / 8.0) * diff @ inv_cov_avg @ diff
    sign1, logdet1 = np.linalg.slogdet(cov_avg)
    sign2, logdet2 = np.linalg.slogdet(cov1)
    sign3, logdet3 = np.linalg.slogdet(cov2)
    term2 = 0.5 * (logdet1 - 0.5 * (logdet2 + logdet3))
    return float(term1 + term2)


def draw_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw an ellipse representing n_std standard deviations of a 2-D Gaussian."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def load_and_project(spec: RepSpec):
    """Load embeddings + scores, project onto max-correlation direction, return 2-D.

    Instead of PCA (max variance, blind to score), we project the 300-d
    embeddings onto the direction that maximally correlates with the docking
    score. This is the OLS coefficient vector (normalised): w = (X^T X)^{-1} X^T y,
    then projection = X @ w_hat. Equivalent to the first CCA component
    between embeddings and score.
    """
    base = spec.pool_dir / spec.label
    mpn = np.load(base / "mpn_fp.npy", mmap_mode="r")
    y_true = np.load(base / "y_true.npy", mmap_mode="r")
    y_true = np.asarray(y_true)
    valid = ~np.isnan(y_true)
    mpn_valid = np.ascontiguousarray(mpn[valid]).astype(np.float32)
    scores = -y_true[valid].astype(np.float32)

    emb_scaler = StandardScaler()
    mpn_std = emb_scaler.fit_transform(mpn_valid)

    # Ridge regression direction (small alpha for numerical stability)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(mpn_std, scores)
    proj = ridge.predict(mpn_std)

    pearson_r = float(np.corrcoef(proj, scores)[0, 1])

    X = np.column_stack([proj, scores])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Display-only de-quantisation: some scorers (Vina) report scores on a
    # coarse grid (0.1 kcal/mol -> 69 levels), which paints horizontal bands.
    # Add uniform jitter of +/- half the detected grid spacing to the score
    # axis FOR PLOTTING ONLY. The GMM fit, |r|, and D_B all use X_std (the
    # unjittered values), so clusters and numbers are unchanged.
    uniq = np.unique(scores)
    spacing = 0.0
    if uniq.size > 1:
        diffs = np.diff(uniq)
        spacing = float(np.median(diffs[diffs > 0]))
    rng = np.random.default_rng(RANDOM_SEED)
    scores_jit = scores + rng.uniform(-spacing / 2.0, spacing / 2.0, size=scores.shape)
    X_disp = scaler.transform(np.column_stack([proj, scores_jit]))
    return X_std, X_disp, pearson_r


def fit_gmm(X: np.ndarray):
    """Fit GMM(k=2) and return labels, means, covariances."""
    gmm = GaussianMixture(
        n_components=2, covariance_type="full",
        random_state=RANDOM_SEED, n_init=5, max_iter=300,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm.means_, gmm.covariances_, gmm


def set_paper_rc():
    mpl.rcParams.update({
        "font.size": TEXT_SIZE,
        "axes.titlesize": TEXT_SIZE,
        "axes.labelsize": TEXT_SIZE,
        "xtick.labelsize": TEXT_SIZE - 2,
        "ytick.labelsize": TEXT_SIZE - 2,
        "legend.fontsize": TEXT_SIZE - 2,
        "figure.titlesize": TEXT_SIZE + 2,
        "axes.linewidth": 1.2,
        "savefig.dpi": 150,
    })


def build_combined_figure(results: list[dict], data_by_spec: dict,
                          auc_csv: Path, output_dir: Path):
    """Combined figure: (a) ROC AUC bar, (b) |r| bar, (c) GMM 3x3 scatter."""
    import pandas as pd
    S = 20

    scorers = ["silcs", "glide", "vina"]
    bar_colors = {"silcs": "#4C78A8", "glide": "#E45756", "vina": "#72B7B2"}

    # --- Load ROC AUC data ---
    auc_df = pd.read_csv(auc_csv)
    feature_order = ["mpn_fp", "grad_W"]
    feature_labels = {"mpn_fp": "Embedding", "grad_W": "Gradient"}
    feature_colors = {"mpn_fp": "#4C78A8", "grad_W": "#F58518"}
    sub = auc_df[auc_df["percent"] == 25]
    agg_auc = (sub.groupby(["scorer", "feature"])
                   .agg(mean=("auc_mean", "mean"),
                        std=("auc_mean", "std"))
                   .reset_index())

    # --- |r| aggregation ---
    df_r = pd.DataFrame(results)
    agg_r = df_r.groupby("scorer")["abs_pearson_r"].agg(["mean", "std"]).reset_index()
    agg_r = agg_r.set_index("scorer").loc[scorers].reset_index()

    # --- Layout ---
    fig = plt.figure(figsize=(22, 13))
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1, 2.0],
                                left=0.06, right=0.96, top=0.92,
                                bottom=0.06, wspace=0.15)

    # Left column: two bar charts stacked
    gs_left = gs_outer[0, 0].subgridspec(2, 1, hspace=0.35)

    # (a) ROC AUC
    ax_auc = fig.add_subplot(gs_left[0])
    bar_width = 0.32
    x = np.arange(len(scorers))
    for i, feat in enumerate(feature_order):
        means = [float(agg_auc[(agg_auc["scorer"] == s) & (agg_auc["feature"] == feat)]["mean"].iloc[0])
                 for s in scorers]
        stds = [float(agg_auc[(agg_auc["scorer"] == s) & (agg_auc["feature"] == feat)]["std"].iloc[0])
                for s in scorers]
        offset = (i - 0.5) * bar_width
        ax_auc.bar(x + offset, means, width=bar_width,
                   yerr=stds, capsize=4, label=feature_labels[feat],
                   color=feature_colors[feat])
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels([SCORER_DISPLAY[s] for s in scorers], fontsize=S)
    ax_auc.set_ylabel("ROC AUC", fontsize=S)
    ax_auc.set_title("Linear separability", fontsize=S)
    ax_auc.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    # Headroom above the bars (which all reach ~0.85-1.0) so the legend sits
    # in clear whitespace instead of overlapping them.
    ax_auc.set_ylim(0.45, 1.28)
    ax_auc.legend(loc="upper center", ncol=2, frameon=False, fontsize=S - 4,
                  columnspacing=1.2, handletextpad=0.5)
    ax_auc.tick_params(labelsize=S - 2)
    ax_auc.grid(axis="y", alpha=0.3)

    # (b) |Pearson r|
    ax_r = fig.add_subplot(gs_left[1])
    ax_r.bar(x, agg_r["mean"], yerr=agg_r["std"], capsize=5,
             color=[bar_colors[s] for s in scorers],
             edgecolor="black", linewidth=0.8, width=0.55)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([SCORER_DISPLAY[s] for s in scorers], fontsize=S)
    ax_r.set_ylabel("|Pearson r|", fontsize=S)
    ax_r.set_title("Embedding–activity correlation", fontsize=S)
    ax_r.grid(axis="y", alpha=0.3)
    ax_r.set_ylim(0, 1.0)
    ax_r.tick_params(labelsize=S - 2)

    # (c) GMM 3x3 scatter
    gs_right = gs_outer[0, 1].subgridspec(3, 3, wspace=0.28, hspace=0.30)
    colors = ["#4C78A8", "#F58518"]

    for r_idx in range(3):
        rep_num = r_idx + 1
        for c_idx, scorer in enumerate(scorers):
            ax = fig.add_subplot(gs_right[r_idx, c_idx])
            key = f"{scorer}_rep{rep_num}"
            if key not in data_by_spec:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=S)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            X_plot, labels, means, covs = data_by_spec[key]

            for k in range(2):
                mask = labels == k
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                           c=colors[k], s=1.5, alpha=0.2, linewidths=0,
                           rasterized=True)
            for k in range(2):
                draw_ellipse(ax, means[k], covs[k], n_std=1.0,
                             facecolor=colors[k], alpha=0.4,
                             edgecolor="0.4", linewidth=1.2)
                draw_ellipse(ax, means[k], covs[k], n_std=2.0,
                             facecolor=colors[k], alpha=0.2,
                             edgecolor="none")
                draw_ellipse(ax, means[k], covs[k], n_std=2.0,
                             facecolor="none", edgecolor="black",
                             linewidth=1.5)

            rec = [r for r in results
                   if r["scorer"] == scorer and r["rep"] == rep_num][0]
            ax.text(0.96, 0.96, f"|r| = {rec['abs_pearson_r']:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=S - 3)

            ax.set_xlabel("")
            ax.set_ylabel("")
            if c_idx == 0:
                ax.set_ylabel(f"Replicate {rep_num}", weight="bold", fontsize=S)
            if r_idx == 0:
                ax.set_title(SCORER_DISPLAY[scorer], fontsize=S + 2,
                             weight="bold")

    # Shared axis labels for panel (c). Panel (c) spans x in ~[0.40, 0.96]
    # (centre ~0.68); the y label sits in the open channel between panel (b)
    # (right edge ~0.34) and the bold "Replicate N" row labels (~0.37), so it
    # does not collide with either.
    fig.text(0.685, 0.025, "Embedding projection (standardised)",
             ha="center", va="center", fontsize=S)
    fig.text(0.355, 0.49, "Docking score (standardised)",
             ha="center", va="center", rotation=90, fontsize=S)

    # Panel labels
    fig.text(0.03, 0.95, "(a)", fontsize=S + 6, weight="bold")
    fig.text(0.03, 0.48, "(b)", fontsize=S + 6, weight="bold")
    fig.text(0.38, 0.95, "(c)", fontsize=S + 6, weight="bold")

    combined_path = output_dir / "gmm_combined.png"
    fig.savefig(combined_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {combined_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(ANALYSIS_DIR), type=Path)
    p.add_argument("--rep", type=int, default=None, choices=[1, 2, 3],
                   help="Restrict to one replicate (default: all).")
    args = p.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_rc()

    specs = SPECS
    if args.rep is not None:
        specs = [s for s in specs if s.rep == args.rep]

    results: list[dict] = []
    data_by_spec: dict[str, tuple] = {}

    for spec in specs:
        base = spec.pool_dir / spec.label
        if not (base / "mpn_fp.npy").exists():
            print(f"[skip] {spec.label}: missing data", flush=True)
            continue
        print(f"[fit] {spec.label}", flush=True)
        X_std, X_disp, pearson_r = load_and_project(spec)
        labels, means, covs, gmm = fit_gmm(X_std)
        db = bhattacharyya_distance_2d(means[0], covs[0], means[1], covs[1])

        # Angle of cluster separation axis relative to embedding (x) axis.
        # 0° = separation is purely along the embedding; 90° = purely along
        # the score axis (embedding contributes nothing).
        delta = means[0] - means[1]
        sep_angle = float(np.degrees(np.arctan2(abs(delta[1]), abs(delta[0]))))

        print(f"  D_B={db:.4f}  |r|={abs(pearson_r):.4f}  "
              f"sep_angle={sep_angle:.1f}°", flush=True)
        results.append({
            "scorer": spec.scorer, "rep": spec.rep, "label": spec.label,
            "bhattacharyya": db,
            "abs_pearson_r": abs(pearson_r),
            "sep_angle": sep_angle,
        })
        data_by_spec[spec.label] = (X_disp, labels, means, covs)

    # --- Scatter figure ---
    scorers_present = sorted({r["scorer"] for r in results},
                             key=lambda s: ["silcs", "glide", "vina"].index(s))
    reps_present = sorted({r["rep"] for r in results})
    n_cols = len(scorers_present)
    n_rows = len(reps_present)

    colors = ["#4C78A8", "#F58518"]

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(5.0 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.06,
                        wspace=0.32, hspace=0.35)

    for r_idx, rep_num in enumerate(reps_present):
        for c_idx, scorer in enumerate(scorers_present):
            ax = axes[r_idx, c_idx]
            key = f"{scorer}_rep{rep_num}"
            if key not in data_by_spec:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TEXT_SIZE)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            X_plot, labels, means, covs = data_by_spec[key]

            for k in range(2):
                mask = labels == k
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                           c=colors[k], s=2, alpha=0.25, linewidths=0,
                           rasterized=True)

            for k in range(2):
                draw_ellipse(ax, means[k], covs[k], n_std=1.0,
                             facecolor=colors[k], alpha=0.4,
                             edgecolor="0.4", linewidth=1.2)
                draw_ellipse(ax, means[k], covs[k], n_std=2.0,
                             facecolor=colors[k], alpha=0.2,
                             edgecolor="none")
                draw_ellipse(ax, means[k], covs[k], n_std=2.0,
                             facecolor="none", edgecolor="black",
                             linewidth=1.8)

            rec = [r for r in results
                   if r["scorer"] == scorer and r["rep"] == rep_num][0]
            ax.text(0.96, 0.96,
                    f"|r| = {rec['abs_pearson_r']:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=TEXT_SIZE - 2)

            ax.set_xlabel("")
            ax.set_ylabel("")
            if c_idx == 0:
                ax.set_ylabel(f"Replicate {rep_num}", weight="bold")
            if r_idx == 0:
                ax.set_title(SCORER_DISPLAY[scorer], fontsize=TEXT_SIZE + 2,
                             weight="bold")

    # Shared axis labels via fig.text
    fig.text(0.5, 0.01, "Embedding projection (standardised)",
             ha="center", fontsize=TEXT_SIZE)
    fig.text(0.01, 0.5, "Docking score (standardised)",
             ha="center", va="center", rotation=90, fontsize=TEXT_SIZE)

    scatter_path = output_dir / "gmm_overlap_scatter.png"
    fig.savefig(scatter_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n[saved] {scatter_path}", flush=True)

    # --- Bar chart: |Pearson r| between embedding PC1 and docking score ---
    import pandas as pd
    df = pd.DataFrame(results)
    agg_r = df.groupby("scorer")["abs_pearson_r"].agg(["mean", "std"]).reset_index()
    scorer_order = [s for s in ["silcs", "glide", "vina"] if s in agg_r["scorer"].values]
    agg_r = agg_r.set_index("scorer").loc[scorer_order].reset_index()

    fig_bar, ax_bar = plt.subplots(figsize=(5, 4.5))
    fig_bar.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.12)
    x = np.arange(len(scorer_order))
    ax_bar.bar(x, agg_r["mean"], yerr=agg_r["std"], capsize=6,
               color=["#4C78A8", "#E45756", "#72B7B2"],
               edgecolor="black", linewidth=0.8, width=0.55)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([SCORER_DISPLAY[s] for s in scorer_order])
    ax_bar.set_ylabel("|Pearson r|")
    ax_bar.set_title("Embedding–activity correlation")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(0, 1.0)

    bar_path = output_dir / "gmm_overlap_bar.png"
    fig_bar.savefig(bar_path, bbox_inches="tight", dpi=150)
    plt.close(fig_bar)
    print(f"[saved] {bar_path}", flush=True)

    # --- Combined figure: (a) ROC AUC, (b) |r|, (c) GMM scatter 3x3 ---
    csv_path = output_dir / "separability_auc.csv"
    if csv_path.exists():
        build_combined_figure(results, data_by_spec, csv_path, output_dir)
    else:
        print(f"[skip] combined figure: {csv_path} not found", flush=True)

    # Print summary table
    print("\n=== Summary ===")
    print(f"  {'label':15s}  {'D_B':>8s}  {'|r|':>8s}  {'angle':>7s}")
    for r in sorted(results, key=lambda x: (x["scorer"], x["rep"])):
        print(f"  {r['label']:15s}  {r['bhattacharyya']:8.4f}  "
              f"{r['abs_pearson_r']:8.4f}  {r['sep_angle']:6.1f}°")
    print(f"\n  Mean |r| by scorer:")
    for _, row in agg_r.iterrows():
        print(f"    {SCORER_DISPLAY[row['scorer']]:6s}: {row['mean']:.4f} +/- {row['std']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
