"""
Maximal Information Coefficient (MIC) analysis of features vs accuracy.

Methodology (paper-ready):
  - Maximal Information Coefficient (Reshef et al., 2011) as effect size for general dependence
  - Permutation-based p-value for MIC (N=200 shuffles; Phipson & Smyth, 2010)
  - Benjamini-Hochberg FDR correction on MIC permutation p-values (primary significance)
  - Spearman rank correlation retained as a reference baseline

Reads per-dataset CSVs produced by divide_by_dataset.py.
Runs per dataset, per epoch (1, 5, 50).

Usage:
    python -m ab.stat.mic_analysis
"""

from __future__ import annotations

import os
import time
import warnings
from typing import List

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minepy import MINE
from scipy.stats import spearmanr
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    "celeba-gender",
    "cifar-10",
    "cifar-100",
    "imagenette",
    "mnist",
    "places365",
    "svhn",
]

EPOCHS = [1, 5, 50]

MIN_SAMPLES = 30
FDR_ALPHA = 0.05

# Permutation test parameters (MIC significance)
PERM_N = 200
PERM_SEED = 42
PERM_MIC_MIN = 0.05

# MINE hyperparameters (defaults from Reshef et al., 2011)
MINE_ALPHA = 0.6
MINE_C = 15
MINE_EST = "mic_approx"

# Feature allowlist
FEATURE_ALLOWLIST = [
    "nn_total_params",
    "nn_flops",
    "nn_total_layers",
    "nn_max_depth",
    "nn_has_attention",
    "nn_has_residual",
    "nn_dropout_count",
    "prm__lr",
    "prm__batch",
    "prm__dropout",
    "prm__weight_decay",
    "prm__momentum",
]

_base = os.path.join(os.path.dirname(__file__), "..", "..")
INPUT_DIR = os.path.join(_base, "dataset_splits", "epoch_1_5_50")
OUTDIR = os.path.join(_base, "mic_out")


# ============================================================================
# Statistical utilities
# ============================================================================

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = sorted_pvals * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(pvals)
    result[sorted_idx] = adjusted
    return result


def compute_mic_pval(
    x: np.ndarray,
    y: np.ndarray,
    observed_mic: float,
    rng: np.random.Generator,
    n_perm: int = PERM_N,
) -> float:
    if observed_mic <= PERM_MIC_MIN:
        return 1.0

    mine = MINE(alpha=MINE_ALPHA, c=MINE_C, est=MINE_EST)
    y_perm = y.copy()
    null_mics = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(y_perm)
        mine.compute_score(x, y_perm)
        null_mics[i] = mine.mic()

    count = int(np.sum(null_mics >= observed_mic))
    return (count + 1) / (n_perm + 1)


def compute_mic_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mine = MINE(alpha=MINE_ALPHA, c=MINE_C, est=MINE_EST)
    mine.compute_score(x, y)
    mic = float(mine.mic())

    sp_corr, sp_pval = spearmanr(x, y)

    return {
        "mic": round(mic, 6),
        "spearman": round(float(sp_corr), 6),
        "spearman_pval": round(float(sp_pval), 6),
        "abs_spearman": round(abs(float(sp_corr)), 6),
    }


# ============================================================================
# I/O helpers
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in FEATURE_ALLOWLIST:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() < MIN_SAMPLES:
            continue
        cols.append(c)
    return cols


# ============================================================================
# Analysis engine
# ============================================================================

def analyze_features_vs_target(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(PERM_SEED)
    results = []
    total = len(features)
    feat_pbar = tqdm(features, desc="    features", unit="feat", leave=False, dynamic_ncols=True)
    for i, feat in enumerate(feat_pbar, 1):
        feat_pbar.set_postfix({"current": feat[:25]})

        pair = df[[feat, target_col]].dropna()
        n = len(pair)
        if n < MIN_SAMPLES:
            continue

        x = pair[feat].values.astype(float)
        y = pair[target_col].values.astype(float)

        if x.std() == 0 or y.std() == 0:
            continue

        stats = compute_mic_stats(x, y)
        mic_pval = compute_mic_pval(x, y, stats["mic"], rng)
        stats["mic_pval"] = round(mic_pval, 6)

        stats["feature"] = feat
        stats["n_samples"] = n
        results.append(stats)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    col_order = [
        "feature", "n_samples",
        "mic", "mic_pval",
        "spearman", "spearman_pval", "abs_spearman",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    out = out.sort_values("mic", ascending=False).reset_index(drop=True)
    return out


def apply_fdr_correction(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()
    df["mic_pval_fdr"] = benjamini_hochberg(df["mic_pval"].values)
    df["mic_significant"] = df["mic_pval_fdr"] < FDR_ALPHA
    df["spearman_pval_fdr"] = benjamini_hochberg(df["spearman_pval"].values)
    df["spearman_significant"] = df["spearman_pval_fdr"] < FDR_ALPHA
    return df


# ============================================================================
# Visualization functions
# ============================================================================

def _short_name(feat: str) -> str:
    return feat.replace("prm__", "p:").replace("nn_", "").replace("is_", "").replace("_like", "")


def plot_fig1_mean_mic(avg: pd.DataFrame, out_path: str) -> None:
    df = avg.reset_index().sort_values("mean_mic", ascending=True)
    if df.empty:
        return
    labels = [_short_name(f) for f in df["feature"]]
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
    bars = ax.barh(y_pos, df["mean_mic"].values,
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["mean_mic"].values):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean MIC (averaged across all datasets & epochs)", fontsize=9)
    ax.set_title("Fig 1 — Feature Importance: Mean MIC across All Datasets & Epochs",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, min(1.0, df["mean_mic"].max() * 1.45 + 0.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_heatmap(combined: pd.DataFrame, target_tag: str,
                      title: str, out_path: str) -> None:
    sub = combined[combined["target"].str.contains(target_tag, regex=False)].copy()
    sub = sub[sub["dataset"].isin(DATASETS)]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="feature", columns="dataset",
                            values="mic", aggfunc="mean")
    ordered_cols = [c for c in DATASETS if c in pivot.columns]
    pivot = pivot[ordered_cols]
    sig_pivot = sub.pivot_table(index="feature", columns="dataset",
                                values="mic_significant", aggfunc="max")
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    sig_pivot = sig_pivot.reindex(row_order)
    row_labels = [_short_name(f) for f in pivot.index]
    col_labels = list(pivot.columns)
    annot = np.full(pivot.shape, "", dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                annot[i, j] = ""
            else:
                try:
                    sig = bool(sig_pivot.iloc[i, j])
                except (ValueError, TypeError):
                    sig = False
                annot[i, j] = f"{val:.3f}" + ("*" if sig else " ")
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.3),
                                    max(4, len(row_labels) * 0.55)))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="YlOrRd", vmin=0, vmax=0.5,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "MIC (Maximal Information Coefficient)", "shrink": 0.7},
                yticklabels=row_labels, xticklabels=col_labels)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(row_labels, fontsize=8, rotation=0)
    fig.text(0.01, -0.01,
             "* MIC permutation p-value FDR-significant (Benjamini-Hochberg, α=0.05)",
             fontsize=7, style="italic")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_epoch_comparison(combined: pd.DataFrame, ds_name: str,
                               out_path: str) -> None:
    epoch_map = {f"Epoch {e}": f"accuracy_epoch_{e}" for e in EPOCHS}
    colors = ["#3A7FD5", "#E8653A", "#2EAB6F"]
    edge_colors = ["#1D5BA8", "#C0441C", "#1A8050"]

    records = []
    for label, target in epoch_map.items():
        sub = combined[(combined["dataset"] == ds_name) &
                       (combined["target"] == target)]
        for _, row in sub.iterrows():
            records.append({"feature": _short_name(row["feature"]),
                            "epoch": label, "mic": row["mic"]})
    if not records:
        return

    plot_df = pd.DataFrame(records)
    feat_order = (plot_df.groupby("feature")["mic"]
                  .mean().sort_values(ascending=False).index.tolist())

    n_feat = len(feat_order)
    n_ep = len(epoch_map)
    width = 0.22
    gap = 0.07
    group_w = n_ep * width + gap
    x = np.arange(n_feat) * group_w

    fig_w = max(12, n_feat * group_w * 1.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    for i, (label, color, ec) in enumerate(zip(epoch_map.keys(), colors, edge_colors)):
        vals = []
        for feat in feat_order:
            row = plot_df[(plot_df["feature"] == feat) & (plot_df["epoch"] == label)]
            vals.append(float(row["mic"].values[0]) if len(row) else 0.0)
        offset = (i - (n_ep - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=label, color=color, edgecolor=ec,
                      linewidth=0.8, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            if val >= 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color="#333333", fontweight="bold")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_order, rotation=38, ha="right", fontsize=9,
                       fontweight="semibold")
    ax.set_ylabel("MIC (Maximal Information Coefficient)", fontsize=10, labelpad=8)
    ax.set_ylim(bottom=0, top=min(1.0, plot_df["mic"].max() * 1.35 + 0.04))

    epoch_labels = " vs ".join(f"Epoch {e}" for e in EPOCHS)
    ax.set_title(f"Fig 5 — Epoch Comparison: {epoch_labels}\nDataset: {ds_name}",
                 fontsize=11, fontweight="bold", pad=14)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#BBBBBB",
              loc="upper right", bbox_to_anchor=(1.0, 1.0))
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=8.5)

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    ensure_dir(OUTDIR)

    print("=" * 70)
    print("MAXIMAL INFORMATION COEFFICIENT (MIC) ANALYSIS (paper-ready)")
    print("=" * 70)
    print(f"  Input dir      : {os.path.abspath(INPUT_DIR)}")
    print(f"  Output dir     : {os.path.abspath(OUTDIR)}")
    print(f"  Datasets       : {DATASETS}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Min samples    : {MIN_SAMPLES}")
    print(f"  FDR alpha      : {FDR_ALPHA}")
    print(f"  MINE alpha     : {MINE_ALPHA}  (exploration param)")
    print(f"  MINE c         : {MINE_C}      (clumping param)")
    print(f"  MINE estimator : {MINE_EST!r}")
    print(f"  Perm N         : {PERM_N}    (MIC null-distribution samples, seed={PERM_SEED})")
    print(f"  Perm pre-filter: MIC > {PERM_MIC_MIN} (trivial pairs skipped)")
    print("=" * 70)
    print()

    all_results = []

    ds_pbar = tqdm(DATASETS, desc="Datasets", unit="ds", position=0, leave=True, dynamic_ncols=True)
    for ds_name in ds_pbar:
        ds_pbar.set_postfix({"current": ds_name})
        csv_path = os.path.join(INPUT_DIR, f"{ds_name}.csv")
        if not os.path.exists(csv_path):
            tqdm.write(f"[SKIP] {ds_name}: CSV not found at {csv_path}")
            continue

        tqdm.write(f"\n{'─' * 70}")
        tqdm.write(f"DATASET: {ds_name}")
        tqdm.write(f"{'─' * 70}")

        df = pd.read_csv(csv_path)
        tqdm.write(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        if "accuracy" not in df.columns:
            tqdm.write(f"  [SKIP] No 'accuracy' column")
            continue

        if "epoch" in df.columns:
            df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

        features = get_numeric_features(df)
        tqdm.write(f"  Numeric features with >= {MIN_SAMPLES} non-null: {len(features)}")
        if not features:
            tqdm.write(f"  [SKIP] No usable features")
            continue

        ds_outdir = os.path.join(OUTDIR, ds_name)
        ensure_dir(ds_outdir)

        for epoch_val in EPOCHS:
            df_epoch = df[df["epoch"] == epoch_val] if "epoch" in df.columns else df
            n = len(df_epoch)
            if n < MIN_SAMPLES:
                tqdm.write(f"\n  Epoch {epoch_val}: only {n} rows (< {MIN_SAMPLES}), skipping")
                continue

            tqdm.write(f"\n  Epoch {epoch_val}: {n:,} rows")
            t0 = time.time()
            result = analyze_features_vs_target(df_epoch, features, "accuracy")
            elapsed = time.time() - t0

            if result.empty:
                tqdm.write(f"    No features passed filtering")
                continue

            result.insert(0, "dataset", ds_name)
            result.insert(1, "target", f"accuracy_epoch_{epoch_val}")

            out_path = os.path.join(ds_outdir, f"epoch_{epoch_val}.csv")
            result.to_csv(out_path, index=False)
            all_results.append(result)

            tqdm.write(f"    Computed in {elapsed:.1f}s | {len(result)} features")

    # Combined summary with FDR correction
    print(f"\n\n{'=' * 70}")
    print("COMBINED SUMMARY (with FDR correction)")
    print(f"{'=' * 70}")

    if not all_results:
        print("  No results to summarize.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[combined["dataset"].isin(DATASETS)].reset_index(drop=True)
    print(f"\n  Total feature-target pairs: {len(combined):,}  "
          f"(datasets: {sorted(combined['dataset'].unique().tolist())})")

    combined = apply_fdr_correction(combined)

    n_mic_sig = combined["mic_significant"].sum()
    n_sp_sig = combined["spearman_significant"].sum()
    print(f"  Significant after FDR (alpha={FDR_ALPHA}):")
    print(f"    MIC permutation (primary)  : {n_mic_sig} / {len(combined)}")
    print(f"    Spearman analytical (ref)  : {n_sp_sig} / {len(combined)}")

    # Output folders
    summary_dir = os.path.join(OUTDIR, "summary")
    figures_dir = os.path.join(OUTDIR, "figures")
    per_ds_figs = os.path.join(figures_dir, "per_dataset")
    ensure_dir(summary_dir)
    ensure_dir(figures_dir)
    ensure_dir(per_ds_figs)

    # Save summary CSVs
    combined_path = os.path.join(summary_dir, "summary_all_datasets.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  Full results: {combined_path}")

    significant = combined[combined["mic_significant"]].sort_values("mic", ascending=False)
    sig_path = os.path.join(summary_dir, "significant_results.csv")
    significant.to_csv(sig_path, index=False)
    print(f"  MIC-significant results: {sig_path} ({len(significant)} rows)")

    avg = combined.groupby("feature").agg(
        mean_mic=("mic", "mean"),
        mean_abs_spearman=("abs_spearman", "mean"),
        n_tests=("mic", "count"),
        n_mic_significant=("mic_significant", "sum"),
    ).sort_values("mean_mic", ascending=False)

    avg_path = os.path.join(summary_dir, "avg_mic_per_feature.csv")
    avg.to_csv(avg_path)
    print(f"\n  Average MIC per feature saved to {avg_path}")

    # Generate figures
    print(f"\n  Generating figures ...")
    try:
        plot_fig1_mean_mic(avg, out_path=os.path.join(figures_dir, "fig1_mean_mic.png"))
        print("    fig1_mean_mic.png  ✓")
    except Exception as e:
        print(f"    fig1_mean_mic.png  FAILED: {e}")

    for epoch, tag in [(1, "epoch1"), (5, "epoch5"), (50, "epoch50")]:
        try:
            plot_fig2_heatmap(
                combined, target_tag=f"epoch_{epoch}",
                title=f"Fig — MIC Heatmap: Feature vs Dataset  (Epoch {epoch})",
                out_path=os.path.join(figures_dir, f"fig_heatmap_{tag}.png"),
            )
            print(f"    fig_heatmap_{tag}.png  ✓")
        except Exception as e:
            print(f"    fig_heatmap_{tag}.png  FAILED: {e}")

    print("    Per-dataset epoch comparison charts:")
    for ds_name in DATASETS:
        if combined[combined["dataset"] == ds_name].empty:
            continue
        try:
            out_path = os.path.join(per_ds_figs, f"{ds_name}_epoch_comparison.png")
            plot_fig4_epoch_comparison(combined, ds_name, out_path=out_path)
            print(f"      {ds_name}_epoch_comparison.png  ✓")
        except Exception as e:
            print(f"      {ds_name}_epoch_comparison.png  FAILED: {e}")

    # Methodology file
    method_path = os.path.join(OUTDIR, "METHODOLOGY.txt")
    with open(method_path, "w") as f:
        f.write("MIC (Maximal Information Coefficient) Analysis — Methodology\n")
        f.write("=" * 60 + "\n\n")
        f.write("Statistical measures:\n")
        f.write("  - MIC (Maximal Information Coefficient, Reshef et al., 2011):\n")
        f.write("    Measures general (including non-linear) statistical dependence.\n")
        f.write("    Range: [0, 1]. MIC = 0 implies independence.\n")
        f.write("  - Spearman rank correlation: reference baseline for monotonic dependence.\n\n")
        f.write("Significance testing (PRIMARY — MIC permutation test):\n")
        f.write(f"  - Permutation p-value (Phipson & Smyth, 2010): p = (|null >= obs| + 1) / ({PERM_N} + 1).\n")
        f.write(f"  - Pre‑filter: if observed MIC ≤ {PERM_MIC_MIN}, p = 1.0 (skip permutations).\n")
        f.write(f"  - Benjamini-Hochberg FDR correction applied jointly across all tests, alpha = {FDR_ALPHA}.\n")
        f.write(f"  - mic_significant = True iff mic_pval_fdr < {FDR_ALPHA}.\n\n")
        f.write("Filters:\n")
        f.write(f"  - Minimum {MIN_SAMPLES} non‑null sample pairs per test.\n")
        f.write(f"  - Constant features (std = 0) excluded.\n\n")
        f.write("Reproducibility:\n")
        f.write(f"  - Permutation RNG: numpy.random.default_rng(seed={PERM_SEED}).\n")
        f.write(f"  - All results are deterministic.\n")
    print(f"\n  Methodology notes: {method_path}")

    print(f"\n{'=' * 70}")
    print("DONE. Output directory:", os.path.abspath(OUTDIR))
    print("=" * 70)


if __name__ == "__main__":
    main()