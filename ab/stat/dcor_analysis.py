"""
Distance Correlation (dcor) analysis of features vs accuracy.

Methodology:
  - Distance correlation (Székely et al., 2007) for general dependence.
  - Analytical t-test p-values for dcor (Székely & Rizzo, 2013).
  - Spearman rank correlation as reference baseline.
  - Benjamini-Hochberg FDR correction on both dcor and Spearman p-values.

Reads per-dataset CSVs produced by divide_by_dataset.py.
Runs per dataset, per epoch (1, 5, 50).

Usage:
    python -m ab.stat.dcor_analysis
"""

from __future__ import annotations

import os
import time
import warnings
from typing import List

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import dcor
import dcor.independence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
FAST_METHOD = "avl"                     # O(n log n) algorithm, identical to O(n²)

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
OUTDIR = os.path.join(_base, "dcor_out")


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


def compute_dcor_stats(x: np.ndarray, y: np.ndarray) -> dict:
    dc_val = float(dcor.distance_correlation(x, y, method=FAST_METHOD))
    t_result = dcor.independence.distance_correlation_t_test(x, y)
    dc_pval = float(t_result.pvalue)

    sp_corr, sp_pval = spearmanr(x, y)

    return {
        "dcor": round(dc_val, 6),
        "dcor_pval": round(dc_pval, 6),
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

        stats = compute_dcor_stats(x, y)
        stats["feature"] = feat
        stats["n_samples"] = n
        results.append(stats)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    col_order = ["feature", "n_samples", "dcor", "dcor_pval",
                 "spearman", "spearman_pval", "abs_spearman"]
    out = out[[c for c in col_order if c in out.columns]]
    out = out.sort_values("dcor", ascending=False).reset_index(drop=True)
    return out


def apply_fdr_correction(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()
    df["dcor_pval_fdr"] = benjamini_hochberg(df["dcor_pval"].values)
    df["dcor_significant"] = df["dcor_pval_fdr"] < FDR_ALPHA
    df["spearman_pval_fdr"] = benjamini_hochberg(df["spearman_pval"].values)
    df["spearman_significant"] = df["spearman_pval_fdr"] < FDR_ALPHA
    return df


# ============================================================================
# Visualization functions
# ============================================================================

def _short_name(feat: str) -> str:
    return feat.replace("prm__", "p:").replace("nn_", "").replace("is_", "").replace("_like", "")


def plot_fig1_mean_dcor(avg: pd.DataFrame, out_path: str) -> None:
    df = avg.reset_index().sort_values("mean_dcor", ascending=True)
    if df.empty:
        return
    labels = [_short_name(f) for f in df["feature"]]
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
    bars = ax.barh(y_pos, df["mean_dcor"].values,
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["mean_dcor"].values):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Distance Correlation (averaged across all datasets & epochs)", fontsize=9)
    ax.set_title("Fig 1 — Feature Importance: Mean dcor across All Datasets & Epochs",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, min(1.0, df["mean_dcor"].max() * 1.45 + 0.02))
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
                            values="dcor", aggfunc="mean")
    ordered_cols = [c for c in DATASETS if c in pivot.columns]
    pivot = pivot[ordered_cols]
    sig_pivot = sub.pivot_table(index="feature", columns="dataset",
                                values="dcor_significant", aggfunc="max")
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
                cbar_kws={"label": "Distance Correlation", "shrink": 0.7},
                yticklabels=row_labels, xticklabels=col_labels)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(row_labels, fontsize=8, rotation=0)
    fig.text(0.01, -0.01, "* FDR-significant (Benjamini-Hochberg, α=0.05)",
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
                            "epoch": label, "dcor": row["dcor"]})
    if not records:
        return

    plot_df = pd.DataFrame(records)
    feat_order = (plot_df.groupby("feature")["dcor"]
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
            vals.append(float(row["dcor"].values[0]) if len(row) else 0.0)
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
    ax.set_ylabel("Distance Correlation", fontsize=10, labelpad=8)
    ax.set_ylim(bottom=0, top=min(1.0, plot_df["dcor"].max() * 1.35 + 0.04))

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
    print("DISTANCE CORRELATION ANALYSIS (per-epoch only)")
    print("=" * 70)
    print(f"  Input dir      : {os.path.abspath(INPUT_DIR)}")
    print(f"  Output dir     : {os.path.abspath(OUTDIR)}")
    print(f"  Datasets       : {DATASETS}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Min samples    : {MIN_SAMPLES}")
    print(f"  FDR alpha      : {FDR_ALPHA}")
    print(f"  dcor method    : {FAST_METHOD!r} (AVL O(n log n) algorithm)")
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
    print(f"\n  Total feature-target pairs: {len(combined):,}  (datasets: {sorted(combined['dataset'].unique().tolist())})")

    combined = apply_fdr_correction(combined)

    n_dc_sig = combined["dcor_significant"].sum()
    n_sp_sig = combined["spearman_significant"].sum()
    print(f"  Significant after FDR (alpha={FDR_ALPHA}):")
    print(f"    dcor (t-test) : {n_dc_sig} / {len(combined)}")
    print(f"    spearman      : {n_sp_sig} / {len(combined)}")

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

    significant = combined[combined["dcor_significant"]].sort_values("dcor", ascending=False)
    sig_path = os.path.join(summary_dir, "significant_results.csv")
    significant.to_csv(sig_path, index=False)
    print(f"  dcor-significant results: {sig_path} ({len(significant)} rows)")

    avg = combined.groupby("feature").agg(
        mean_dcor=("dcor", "mean"),
        mean_abs_spearman=("abs_spearman", "mean"),
        n_tests=("dcor", "count"),
    ).sort_values("mean_dcor", ascending=False)

    avg_path = os.path.join(summary_dir, "avg_dcor_per_feature.csv")
    avg.to_csv(avg_path)
    print(f"\n  Average dcor per feature saved to {avg_path}")

    # Generate figures
    print(f"\n  Generating figures ...")
    try:
        plot_fig1_mean_dcor(avg, out_path=os.path.join(figures_dir, "fig1_mean_dcor.png"))
        print("    fig1_mean_dcor.png  ✓")
    except Exception as e:
        print(f"    fig1_mean_dcor.png  FAILED: {e}")

    for epoch, tag in [(1, "epoch1"), (5, "epoch5"), (50, "epoch50")]:
        try:
            plot_fig2_heatmap(
                combined, target_tag=f"epoch_{epoch}",
                title=f"Fig — dcor Heatmap: Feature vs Dataset  (Epoch {epoch})",
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
        f.write("Distance Correlation Analysis — Methodology (per-epoch)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Statistical measures:\n")
        f.write("  - Distance correlation (Székely et al., 2007): measures general\n")
        f.write("    (including non-linear) statistical dependence.\n")
        f.write("    Range: [0, 1]. dcor = 0 iff X and Y are independent.\n\n")
        f.write("  - Spearman rank correlation: measures monotonic dependence.\n")
        f.write("    Range: [-1, +1]. Included as a reference baseline.\n\n")
        f.write("Significance testing:\n")
        f.write("  - dcor p-value: analytical t-test (Székely & Rizzo, 2013).\n")
        f.write("    t-distribution approximation — no permutation resampling.\n")
        f.write("  - Spearman p-value: asymptotic approximation (scipy.stats.spearmanr).\n")
        f.write(f"  - Multiple testing correction: Benjamini-Hochberg FDR, alpha = {FDR_ALPHA}.\n")
        f.write("    Applied jointly across ALL tests (all datasets, epochs, features).\n\n")
        f.write("Filters:\n")
        f.write(f"  - Minimum {MIN_SAMPLES} non-null sample pairs per test.\n")
        f.write("  - Constant features (std = 0) excluded.\n\n")
        f.write(f"Computational details:\n")
        f.write(f"  - dcor algorithm: {FAST_METHOD!r} (AVL fast method, O(n log n)).\n")
        f.write("    This produces IDENTICAL results to the naive O(n²) algorithm.\n")
        f.write("    Reference: Huo, X., Székely, G.J. (2016). Fast computing for\n")
        f.write("    distance covariance. Technometrics, 58(4), 435-447.\n")
        f.write("  - Python package: dcor\n\n")
        f.write("Reproducibility:\n")
        f.write("  - No random sampling used. All results are fully deterministic.\n")
    print(f"\n  Methodology notes: {method_path}")

    print(f"\n{'=' * 70}")
    print("DONE. Output directory:", os.path.abspath(OUTDIR))
    print("=" * 70)


if __name__ == "__main__":
    main()