"""
Phase 1 — Universal Correlation Analysis.

Two complementary approaches:

Approach A — Pooled dcor / MIC (primary)
  dcor : full pooled dataset (246,908 rows), analytical t-test p-value, BH-FDR
  MIC  : stratified subsample (3,000/group, ~43,500 rows), 200 permutations,
         BH-FDR (secondary significance; dcor is primary)
  Target  : accuracy_znorm_ds
  Features: 12 (epoch_log + 11 architecture/hyper features)

Approach B — Random-effects meta-analysis (secondary, paper-critical)
  Reads per-dataset dcor outputs from dcor_out/{dataset}/epoch_{epoch}.csv
  DerSimonian-Laird random-effects model per feature across all
  (dataset, epoch) studies.
  Reports: pooled_dcor, 95% CI, Cochran's Q, I², p_heterogeneity.

All parameters (MIN_SAMPLES, FDR_ALPHA, MINE_*, dcor method) are identical
to the existing per-dataset scripts (dcor_analysis.py, mic_analysis.py).

Outputs
-------
  universal_out/correlation/pooled_dcor_mic.csv
  universal_out/correlation/meta_analysis.csv
  universal_out/correlation/mic_meta_analysis.csv
  universal_out/correlation/fig_pooled_bar.png
  universal_out/correlation/fig_forest_plot.png
  universal_out/correlation/fig_I2_bars.png
  universal_out/correlation/fig_dcor_heatmap.png
  universal_out/correlation/fig_mic_I2_bars.png
  universal_out/correlation/fig_mic_forest.png
  universal_out/correlation/fig_mic_heatmap.png
  universal_out/correlation/METHODOLOGY_universal_correlation.txt

Usage
-----
    python -m ab.stat.universal_correlation
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import dcor
import dcor.independence
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minepy import MINE
from scipy.stats import chi2, spearmanr
from tqdm import tqdm


# ============================================================================
# Configuration — identical to dcor_analysis.py / mic_analysis.py
# ============================================================================

DATASETS: List[str] = [
    "celeba-gender",
    "cifar-10",
    "cifar-100",
    "imagenette",
    "mnist",
    "places365",
    "svhn",
]

EPOCHS: List[int] = [1, 5, 50]

FEATURE_ALLOWLIST: List[str] = [
    "epoch_log",
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
    "prm__momentum",
]

# --- dcor (same as dcor_analysis.py) ---
MIN_SAMPLES = 30
FDR_ALPHA   = 0.05
FAST_METHOD = "avl"             # O(n log n), identical result to O(n²)

# --- MIC (same as mic_analysis.py) ---
MIC_SAMPLE_PER_GROUP = 1000     # rows per (dataset × epoch) group → ~10,500 total
MIC_PERM_N           = 200
MIC_PERM_SEED        = 42
MIC_PERM_MIN         = 0.05    # skip permutations if MIC ≤ this
MINE_ALPHA           = 0.6
MINE_C               = 15
MINE_EST             = "mic_approx"

# --- meta-analysis ---
I2_THRESHOLD = 50.0            # I² < 50% → feature is "universal"

_base        = os.path.join(os.path.dirname(__file__), "..", "..")
POOLED_CSV   = os.path.join(_base, "universal_out", "pooled_all_datasets.csv")
DCOR_OUT_DIR = os.path.join(_base, "dcor_out")
MIC_OUT_DIR  = os.path.join(_base, "mic_out")
OUTDIR       = os.path.join(_base, "universal_out", "correlation")


# ============================================================================
# Utilities
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    idx = np.argsort(pvals)
    adj = pvals[idx] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty_like(pvals)
    out[idx] = adj
    return out


def short_name(feat: str) -> str:
    return (feat.replace("prm__", "p:")
                .replace("nn_", "")
                .replace("is_", "")
                .replace("_like", ""))


def get_features(df: pd.DataFrame) -> List[str]:
    return [
        f for f in FEATURE_ALLOWLIST
        if f in df.columns
        and pd.api.types.is_numeric_dtype(df[f])
        and df[f].notna().sum() >= MIN_SAMPLES
    ]


# ============================================================================
# Approach A1 — Pooled dcor (full data)
# ============================================================================

def run_pooled_dcor(
    df: pd.DataFrame,
    features: List[str],
    target: str = "accuracy_znorm_ds",
) -> pd.DataFrame:
    """
    dcor on FULL pooled dataset for each feature vs target.
    Analytical t-test p-value — no permutations needed.
    """
    print(f"\n  n={len(df):,}  features={len(features)}")
    results = []

    pbar = tqdm(features, desc="  dcor", unit="feat", dynamic_ncols=True)
    for feat in pbar:
        pbar.set_postfix({"feat": feat[:25]})
        pair = df[[feat, target]].dropna()
        n = len(pair)
        if n < MIN_SAMPLES:
            continue
        x = pair[feat].values.astype(float)
        y = pair[target].values.astype(float)
        if x.std() == 0 or y.std() == 0:
            continue

        t0      = time.time()
        dc_val  = float(dcor.distance_correlation(x, y, method=FAST_METHOD))
        t_res   = dcor.independence.distance_correlation_t_test(x, y)
        dc_pval = float(t_res.pvalue)
        sp_corr, sp_pval = spearmanr(x, y)

        results.append({
            "feature":       feat,
            "n_samples":     n,
            "dcor":          round(dc_val, 6),
            "dcor_pval":     round(dc_pval, 6),
            "spearman":      round(float(sp_corr), 6),
            "spearman_pval": round(float(sp_pval), 6),
            "abs_spearman":  round(abs(float(sp_corr)), 6),
            "elapsed_s":     round(time.time() - t0, 3),
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("dcor", ascending=False).reset_index(drop=True)
    # FDR correction is applied jointly with MIC in main() — not here
    return out


# ============================================================================
# Approach A2 — Pooled MIC (stratified subsample)
# ============================================================================

def make_stratified_subsample(
    df: pd.DataFrame,
    n_per_group: int = MIC_SAMPLE_PER_GROUP,
    seed: int = MIC_PERM_SEED,
) -> pd.DataFrame:
    """
    Sample min(n_per_group, group_size) rows from each (dataset × epoch) group.
    Groups smaller than n_per_group contribute all their rows.
    Preserves dataset and epoch columns in the output.
    """
    frames = [
        g.sample(min(len(g), n_per_group), random_state=seed)
        for _, g in df.groupby(["dataset", "epoch"])
    ]
    return pd.concat(frames, ignore_index=True)


def compute_mic_pval(
    x: np.ndarray,
    y: np.ndarray,
    observed_mic: float,
    rng: np.random.Generator,
    n_perm: int = MIC_PERM_N,
) -> float:
    """Permutation p-value for MIC (Phipson & Smyth, 2010)."""
    if observed_mic <= MIC_PERM_MIN:
        return 1.0
    mine   = MINE(alpha=MINE_ALPHA, c=MINE_C, est=MINE_EST)
    y_perm = y.copy()
    nulls  = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(y_perm)
        mine.compute_score(x, y_perm)
        nulls[i] = mine.mic()
    return (int(np.sum(nulls >= observed_mic)) + 1) / (n_perm + 1)


def run_pooled_mic(
    sub: pd.DataFrame,
    features: List[str],
    target: str = "accuracy_znorm_ds",
) -> pd.DataFrame:
    """
    MIC on a pre-built stratified subsample (3,000/group) with 200 permutations.
    Subsample is built once in main() and shared with run_pooled_dcor so both
    methods run on identical data.
    """
    n_sub  = len(sub)
    gcounts = sub.groupby(["dataset", "epoch"]).size()

    print(f"\n  subsample n={n_sub:,}  groups={len(gcounts)}"
          f"  perms={MIC_PERM_N}  features={len(features)}")
    print(f"  group sizes: min={gcounts.min()}  max={gcounts.max()}  total={n_sub:,}")

    rng     = np.random.default_rng(MIC_PERM_SEED)
    mine    = MINE(alpha=MINE_ALPHA, c=MINE_C, est=MINE_EST)
    results = []

    pbar = tqdm(features, desc="  MIC ", unit="feat", dynamic_ncols=True)
    for feat in pbar:
        pbar.set_postfix({"feat": feat[:25]})
        pair = sub[[feat, target]].dropna()
        n = len(pair)
        if n < MIN_SAMPLES:
            continue
        x = pair[feat].values.astype(float)
        y = pair[target].values.astype(float)
        if x.std() == 0 or y.std() == 0:
            continue

        t0 = time.time()
        mine.compute_score(x, y)
        mic_val  = float(mine.mic())
        mic_pval = compute_mic_pval(x, y, mic_val, rng)
        elapsed  = time.time() - t0

        results.append({
            "feature":       feat,
            "n_samples_mic": n,
            "mic":           round(mic_val, 6),
            "mic_pval":      round(mic_pval, 6),
            "elapsed_s":     round(elapsed, 3),
        })
        tqdm.write(f"    {feat:<30s}  mic={mic_val:.4f}"
                   f"  pval={mic_pval:.4f}  t={elapsed:.0f}s")

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("mic", ascending=False).reset_index(drop=True)
    # FDR correction is applied jointly with dcor in main() — not here
    return out


# ============================================================================
# Approach B — Random-effects meta-analysis (DerSimonian-Laird)
# ============================================================================

def load_per_dataset_dcor() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        for ep in EPOCHS:
            path = os.path.join(DCOR_OUT_DIR, ds, f"epoch_{ep}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df["dataset"] = ds
            df["epoch"]   = ep
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_per_dataset_mic() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        for ep in EPOCHS:
            path = os.path.join(MIC_OUT_DIR, ds, f"epoch_{ep}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df["dataset"] = ds
            df["epoch"]   = ep
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dersimonian_laird(
    effects: np.ndarray,
    n_samples: np.ndarray,
    method: str = "dcor",
) -> dict:
    """
    DerSimonian-Laird random-effects meta-analysis.

    Within-study variance:
      dcor : v_i ≈ (1 - dcor²)² / (n_i - 3)   [Székely & Rizzo, 2013]
      mic  : v_i ≈ mic_i * (1 - mic_i) / (n_i - 1)  [proportion-based;
             MIC ∈ [0,1] so the dcor formula has no theoretical basis for MIC]
    Heterogeneity: Cochran's Q, I² statistic
    Pooled estimate: random-effects weighted mean with 95% CI
    """
    k = len(effects)
    if k < 2:
        return {
            "pooled_dcor": float(effects[0]) if k == 1 else np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "Q": np.nan, "I2": np.nan, "tau2": np.nan,
            "p_heterogeneity": np.nan, "n_studies": k,
            "weighted_mean": float(effects[0]) if k == 1 else np.nan,
        }

    # Within-study variance — formula depends on effect type
    if method == "mic":
        # MIC is bounded [0,1]; use proportion-based variance
        v = (effects * (1 - effects)) / np.maximum(n_samples - 1, 1)
    else:
        # dcor asymptotic variance (Székely & Rizzo, 2013)
        v = (1 - effects ** 2) ** 2 / np.maximum(n_samples - 3, 1)
    v  = np.where(v <= 0, 1e-8, v)

    # Fixed-effects
    w        = 1.0 / v
    sw       = w.sum()
    theta_fe = np.sum(w * effects) / sw

    # Cochran's Q and df
    Q  = float(np.sum(w * (effects - theta_fe) ** 2))
    df = k - 1

    # DerSimonian-Laird τ²
    C    = sw - np.sum(w ** 2) / sw
    tau2 = max(0.0, (Q - df) / C)

    # Random-effects
    w_re     = 1.0 / (v + tau2)
    sw_re    = w_re.sum()
    theta_re = np.sum(w_re * effects) / sw_re
    se_re    = np.sqrt(1.0 / sw_re)

    I2    = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0
    p_het = float(1 - chi2.cdf(Q, df)) if df > 0 else np.nan

    return {
        "pooled_dcor":     round(float(theta_re), 6),
        "ci_lower":        round(float(theta_re - 1.96 * se_re), 6),
        "ci_upper":        round(float(theta_re + 1.96 * se_re), 6),
        "weighted_mean":   round(float(theta_fe), 6),
        "Q":               round(Q, 4),
        "I2":              round(I2, 2),
        "tau2":            round(float(tau2), 6),
        "p_heterogeneity": round(p_het, 6) if not np.isnan(p_het) else np.nan,
        "n_studies":       int(k),
    }


def run_meta_analysis(dcor_all: pd.DataFrame) -> pd.DataFrame:
    features_in_data = [f for f in FEATURE_ALLOWLIST
                        if f in dcor_all["feature"].values]
    rows = []
    for feat in features_in_data:
        sub = dcor_all[dcor_all["feature"] == feat].dropna(
            subset=["dcor", "n_samples"])
        if len(sub) < 2:
            continue
        result = dersimonian_laird(
            sub["dcor"].values.astype(float),
            sub["n_samples"].values.astype(float),
        )
        result["feature"]          = feat
        result["n_datasets"]       = sub["dataset"].nunique()
        result["n_epochs"]         = sub["epoch"].nunique()
        result["dcor_mean_simple"] = round(float(sub["dcor"].mean()), 6)
        result["dcor_std"]         = round(float(sub["dcor"].std()), 6)
        result["dcor_min"]         = round(float(sub["dcor"].min()), 6)
        result["dcor_max"]         = round(float(sub["dcor"].max()), 6)
        result["universal"]        = (
            result["pooled_dcor"] > 0.0
            and not np.isnan(result["I2"])
            and result["I2"] < I2_THRESHOLD
        )
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    col_order = [
        "feature", "n_studies", "n_datasets", "n_epochs",
        "pooled_dcor", "ci_lower", "ci_upper", "weighted_mean",
        "Q", "I2", "tau2", "p_heterogeneity",
        "dcor_mean_simple", "dcor_std", "dcor_min", "dcor_max",
        "universal",
    ]
    out = pd.DataFrame(rows)
    out = out[[c for c in col_order if c in out.columns]]
    return out.sort_values("pooled_dcor", ascending=False).reset_index(drop=True)


def run_mic_meta_analysis(mic_all: pd.DataFrame) -> pd.DataFrame:
    """
    DerSimonian-Laird random-effects meta-analysis on per-dataset MIC outputs.
    Within-study variance: v_i ≈ (1 - mic²)² / (n_i - 3)  [same form as dcor].
    """
    features_in_data = [f for f in FEATURE_ALLOWLIST
                        if f in mic_all["feature"].values]
    rows = []
    for feat in features_in_data:
        sub = mic_all[mic_all["feature"] == feat].dropna(subset=["mic", "n_samples"])
        if len(sub) < 2:
            continue
        result = dersimonian_laird(
            sub["mic"].values.astype(float),
            sub["n_samples"].values.astype(float),
            method="mic",
        )
        # rename pooled key to pooled_mic for clarity
        result["pooled_mic"]      = result.pop("pooled_dcor")
        result["feature"]         = feat
        result["n_datasets"]      = sub["dataset"].nunique()
        result["n_epochs"]        = sub["epoch"].nunique()
        result["mic_mean_simple"] = round(float(sub["mic"].mean()), 6)
        result["mic_std"]         = round(float(sub["mic"].std()), 6)
        result["mic_min"]         = round(float(sub["mic"].min()), 6)
        result["mic_max"]         = round(float(sub["mic"].max()), 6)
        result["universal"]       = (
            result["pooled_mic"] > 0.0
            and not np.isnan(result["I2"])
            and result["I2"] < I2_THRESHOLD
        )
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    col_order = [
        "feature", "n_studies", "n_datasets", "n_epochs",
        "pooled_mic", "ci_lower", "ci_upper", "weighted_mean",
        "Q", "I2", "tau2", "p_heterogeneity",
        "mic_mean_simple", "mic_std", "mic_min", "mic_max",
        "universal",
    ]
    out = pd.DataFrame(rows)
    out = out[[c for c in col_order if c in out.columns]]
    return out.sort_values("pooled_mic", ascending=False).reset_index(drop=True)


# ============================================================================
# Visualizations
# ============================================================================

def plot_pooled_bar(dcor_df: pd.DataFrame, mic_df: pd.DataFrame,
                    out_path: str) -> None:
    merged = dcor_df[["feature", "dcor", "dcor_significant"]].copy()
    if not mic_df.empty:
        merged = merged.merge(
            mic_df[["feature", "mic", "mic_significant"]],
            on="feature", how="left")
    else:
        merged["mic"] = np.nan
        merged["mic_significant"] = False

    merged = merged.sort_values("dcor", ascending=False).reset_index(drop=True)
    labels = [short_name(f) for f in merged["feature"]]
    x      = np.arange(len(labels))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9 + 2), 5))
    b1 = ax.bar(x - width / 2, merged["dcor"].fillna(0), width,
                label="dcor (full data)", color="#2196F3",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, merged["mic"].fillna(0), width,
                label=f"MIC (subsample {MIC_SAMPLE_PER_GROUP}/group)",
                color="#FF9800", edgecolor="white", linewidth=0.5)

    for bar, sig in zip(b1, merged["dcor_significant"].fillna(False)):
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, "*",
                    ha="center", va="bottom", fontsize=10, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Effect size", fontsize=10)
    ax.set_ylim(0, min(1.0, max(
        merged["dcor"].max(),
        merged["mic"].fillna(0).max()) * 1.35 + 0.05))
    ax.set_title(
        "Universal Correlation — Pooled dcor (full data) vs MIC (subsample)\n"
        "* dcor FDR-significant (Benjamini-Hochberg, α=0.05)",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forest(meta_df: pd.DataFrame, dcor_all: pd.DataFrame,
                out_path: str) -> None:
    if meta_df.empty or dcor_all.empty:
        return

    features = meta_df["feature"].tolist()
    n_feat   = len(features)
    ds_colors = {ds: c for ds, c in zip(DATASETS, [
        "#E53935", "#8E24AA", "#1E88E5", "#43A047",
        "#FB8C00", "#00ACC1", "#6D4C41"])}

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 1.2 + 2)))
    y_pos   = np.arange(n_feat) * 2.0

    for i, feat in enumerate(features):
        y    = y_pos[i]
        row  = meta_df[meta_df["feature"] == feat].iloc[0]
        stud = dcor_all[dcor_all["feature"] == feat]

        jitter = np.linspace(-0.5, 0.5, len(stud))
        for j, (_, s) in enumerate(stud.iterrows()):
            ax.scatter(s["dcor"], y + jitter[j],
                       color=ds_colors.get(s["dataset"], "#888"),
                       s=20, alpha=0.6, zorder=3)

        pe, lo, hi = row["pooled_dcor"], row["ci_lower"], row["ci_upper"]
        ax.plot([lo, hi], [y - 0.7, y - 0.7], color="black", lw=1.5, zorder=4)
        ax.scatter(pe, y - 0.7, marker="D", color="black", s=50, zorder=5)
        ax.text(max(hi + 0.01, 0.55), y - 0.7,
                f"I²={row['I2']:.0f}%", va="center", fontsize=7, color="#555")

    ax.set_yticks(y_pos - 0.35)
    ax.set_yticklabels([short_name(f) for f in features], fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Distance Correlation (dcor)", fontsize=10)
    ax.set_title(
        "Forest Plot — Per-dataset dcor (dots) + DerSimonian-Laird pooled (◆ ±95% CI)\n"
        "Colours = datasets;  I² = heterogeneity",
        fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=ds)
               for ds, c in ds_colors.items()]
    patches.append(plt.Line2D([0], [0], marker="D", color="black",
                               linestyle="None", markersize=6,
                               label="Pooled estimate"))
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_i2_bars(meta_df: pd.DataFrame, out_path: str) -> None:
    if meta_df.empty:
        return
    df     = meta_df.sort_values("I2", ascending=True).reset_index(drop=True)
    labels = [short_name(f) for f in df["feature"]]
    colors = ["#43A047" if v < I2_THRESHOLD else "#E53935" for v in df["I2"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    bars = ax.barh(labels, df["I2"], color=colors,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["I2"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", ha="left", fontsize=8)
    ax.axvline(I2_THRESHOLD, color="black", lw=1.2, linestyle="--",
               label=f"Threshold {I2_THRESHOLD:.0f}%")
    ax.set_xlabel("I² — Heterogeneity (%)", fontsize=10)
    ax.set_title(
        f"Meta-analysis Heterogeneity (I²) per Feature\n"
        f"Green = universal (I²<{I2_THRESHOLD:.0f}%)  "
        f"Red = heterogeneous",
        fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dcor_heatmap(dcor_all: pd.DataFrame, out_path: str) -> None:
    if dcor_all.empty:
        return
    pivot = (
        dcor_all[dcor_all["feature"].isin(FEATURE_ALLOWLIST)]
        .groupby(["feature", "dataset"])["dcor"]
        .mean()
        .unstack("dataset")
    )
    ordered_ds = [d for d in DATASETS if d in pivot.columns]
    pivot = pivot[ordered_ds]
    pivot.index = [short_name(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_ds) * 1.3),
                                    max(4, len(pivot) * 0.55)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=0.5, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Mean dcor (avg across epochs)",
                           "shrink": 0.7})
    ax.set_title(
        "dcor Heatmap — Feature × Dataset (mean across epochs)\n"
        "From per-dataset dcor_analysis outputs",
        fontsize=10, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                        ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mic_i2_bars(mic_meta_df: pd.DataFrame, out_path: str) -> None:
    if mic_meta_df.empty:
        return
    df     = mic_meta_df.sort_values("I2", ascending=True).reset_index(drop=True)
    labels = [short_name(f) for f in df["feature"]]
    colors = ["#43A047" if v < I2_THRESHOLD else "#E53935" for v in df["I2"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    bars = ax.barh(labels, df["I2"], color=colors,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["I2"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", ha="left", fontsize=8)
    ax.axvline(I2_THRESHOLD, color="black", lw=1.2, linestyle="--",
               label=f"Threshold {I2_THRESHOLD:.0f}%")
    ax.set_xlabel("I² — Heterogeneity (%)", fontsize=10)
    ax.set_title(
        f"MIC Meta-analysis Heterogeneity (I²) per Feature\n"
        f"Green = universal (I²<{I2_THRESHOLD:.0f}%)  "
        f"Red = heterogeneous",
        fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mic_forest(mic_meta_df: pd.DataFrame, mic_all: pd.DataFrame,
                    out_path: str) -> None:
    if mic_meta_df.empty or mic_all.empty:
        return

    features = mic_meta_df["feature"].tolist()
    n_feat   = len(features)
    ds_colors = {ds: c for ds, c in zip(DATASETS, [
        "#E53935", "#8E24AA", "#1E88E5", "#43A047",
        "#FB8C00", "#00ACC1", "#6D4C41"])}

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 1.2 + 2)))
    y_pos   = np.arange(n_feat) * 2.0

    for i, feat in enumerate(features):
        y    = y_pos[i]
        row  = mic_meta_df[mic_meta_df["feature"] == feat].iloc[0]
        stud = mic_all[mic_all["feature"] == feat]

        jitter = np.linspace(-0.5, 0.5, len(stud))
        for j, (_, s) in enumerate(stud.iterrows()):
            ax.scatter(s["mic"], y + jitter[j],
                       color=ds_colors.get(s["dataset"], "#888"),
                       s=20, alpha=0.6, zorder=3)

        pe, lo, hi = row["pooled_mic"], row["ci_lower"], row["ci_upper"]
        ax.plot([lo, hi], [y - 0.7, y - 0.7], color="black", lw=1.5, zorder=4)
        ax.scatter(pe, y - 0.7, marker="D", color="black", s=50, zorder=5)
        ax.text(max(hi + 0.01, 0.55), y - 0.7,
                f"I²={row['I2']:.0f}%", va="center", fontsize=7, color="#555")

    ax.set_yticks(y_pos - 0.35)
    ax.set_yticklabels([short_name(f) for f in features], fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Maximal Information Coefficient (MIC)", fontsize=10)
    ax.set_title(
        "Forest Plot — Per-dataset MIC (dots) + DerSimonian-Laird pooled (◆ ±95% CI)\n"
        "Colours = datasets;  I² = heterogeneity",
        fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=ds)
               for ds, c in ds_colors.items()]
    patches.append(plt.Line2D([0], [0], marker="D", color="black",
                               linestyle="None", markersize=6,
                               label="Pooled estimate"))
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mic_heatmap(mic_all: pd.DataFrame, out_path: str) -> None:
    if mic_all.empty:
        return
    pivot = (
        mic_all[mic_all["feature"].isin(FEATURE_ALLOWLIST)]
        .groupby(["feature", "dataset"])["mic"]
        .mean()
        .unstack("dataset")
    )
    ordered_ds = [d for d in DATASETS if d in pivot.columns]
    pivot = pivot[ordered_ds]
    pivot.index = [short_name(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_ds) * 1.3),
                                    max(4, len(pivot) * 0.55)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=0.5, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Mean MIC (avg across epochs)",
                           "shrink": 0.7})
    ax.set_title(
        "MIC Heatmap — Feature × Dataset (mean across epochs)\n"
        "From per-dataset mic_analysis outputs",
        fontsize=10, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                        ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Methodology file
# ============================================================================

def write_methodology(path: str) -> None:
    with open(path, "w") as f:
        f.write("Universal Correlation Analysis — Methodology\n")
        f.write("=" * 60 + "\n\n")
        f.write("TARGET VARIABLE\n")
        f.write("  accuracy_znorm_ds — z-scored within each dataset (NOT within\n")
        f.write("  epoch). Removes dataset offset; preserves epoch trend.\n\n")
        f.write("APPROACH A — POOLED ANALYSIS\n\n")
        f.write("  A1. Distance Correlation (dcor) — PRIMARY\n")
        f.write("    Data        : full pooled dataset (246,908 rows)\n")
        f.write(f"    Algorithm   : AVL O(n log n) — identical to O(n²)\n")
        f.write("    p-value     : analytical t-test (Székely & Rizzo, 2013)\n")
        f.write(f"    Correction  : Benjamini-Hochberg FDR, α={FDR_ALPHA}\n\n")
        f.write("  A2. Maximal Information Coefficient (MIC) — SECONDARY\n")
        f.write(f"    Data        : stratified subsample, {MIC_SAMPLE_PER_GROUP}"
                f" rows per (dataset × epoch) group\n")
        f.write(f"    Estimator   : {MINE_EST}  (α={MINE_ALPHA}, c={MINE_C})\n")
        f.write(f"    p-value     : permutation test ({MIC_PERM_N} shuffles, "
                f"seed={MIC_PERM_SEED})\n")
        f.write(f"                  p = (|null ≥ obs| + 1) / ({MIC_PERM_N} + 1)\n")
        f.write(f"    Correction  : Benjamini-Hochberg FDR, α={FDR_ALPHA}\n")
        f.write("    Role        : confirmatory effect size; dcor is primary\n\n")
        f.write("APPROACH B — RANDOM-EFFECTS META-ANALYSIS\n\n")
        f.write("  B1. dcor meta-analysis\n")
        f.write("    Input   : per-dataset dcor outputs "
                "(dcor_out/{dataset}/epoch_{e}.csv)\n")
        f.write("    Unit    : each (dataset × epoch) pair = one study\n")
        f.write("    Model   : DerSimonian-Laird random-effects\n")
        f.write("              (DerSimonian & Laird, 1986)\n")
        f.write("    v_i     : (1 - dcor_i²)² / (n_i - 3)  [within-study variance]\n\n")
        f.write("  B2. MIC meta-analysis\n")
        f.write("    Input   : per-dataset MIC outputs "
                "(mic_out/{dataset}/epoch_{e}.csv)\n")
        f.write("    Unit    : each (dataset × epoch) pair = one study\n")
        f.write("    Model   : DerSimonian-Laird random-effects\n")
        f.write("    v_i     : (1 - mic_i²)² / (n_i - 3)  [same form as dcor]\n\n")
        f.write("  Shared:\n")
        f.write("  Q       : Cochran's heterogeneity statistic\n")
        f.write("  I²      : (Q - df) / Q × 100%\n")
        f.write(f"  Universal criterion: I² < {I2_THRESHOLD}%\n\n")
        f.write("REPRODUCIBILITY\n")
        f.write("  dcor : fully deterministic (AVL, analytical p-value)\n")
        f.write(f"  MIC  : numpy.random.default_rng(seed={MIC_PERM_SEED})\n")
        f.write(f"  Subsample seed = {MIC_PERM_SEED}\n")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ensure_dir(OUTDIR)

    print("=" * 70)
    print("PHASE 1 — UNIVERSAL CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"  Pooled CSV  : {os.path.abspath(POOLED_CSV)}")
    print(f"  Output dir  : {os.path.abspath(OUTDIR)}")
    print(f"  Target      : accuracy_znorm_ds")
    print(f"  Features    : {len(FEATURE_ALLOWLIST)}")
    print(f"  dcor        : full data, AVL, analytical t-test + BH-FDR α={FDR_ALPHA}")
    print(f"  MIC         : {MIC_SAMPLE_PER_GROUP}/group subsample, "
          f"{MIC_PERM_N} permutations + BH-FDR")
    print(f"  Meta-analysis: DerSimonian-Laird, I² threshold={I2_THRESHOLD}%")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load pooled data
    # ------------------------------------------------------------------
    print("\nLoading pooled dataset ...")
    if not os.path.exists(POOLED_CSV):
        print(f"  ERROR: {POOLED_CSV} not found. Run universal_pooled.py first.")
        return

    pooled   = pd.read_csv(POOLED_CSV, low_memory=False)
    features = get_features(pooled)
    print(f"  Rows: {len(pooled):,}  Usable features: {len(features)}  {features}")

    if not features:
        print("  ERROR: No usable features.")
        return

    # ------------------------------------------------------------------
    # Build shared stratified subsample (used by BOTH dcor and MIC so
    # both methods run on identical data — equal footing for the paper)
    # ------------------------------------------------------------------
    print("\nBuilding stratified subsample ...")
    sub      = make_stratified_subsample(pooled)
    gcounts  = sub.groupby(["dataset", "epoch"]).size()
    print(f"  Subsample: {len(sub):,} rows  "
          f"groups={len(gcounts)}  "
          f"min={gcounts.min()}  max={gcounts.max()}")

    # ------------------------------------------------------------------
    # Approach A1 — dcor (stratified subsample, same data as MIC)
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("APPROACH A1 — POOLED dcor (stratified subsample, equal to MIC)")
    print("─" * 70)
    t0      = time.time()
    dcor_df = run_pooled_dcor(sub, features)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Approach A2 — MIC (same stratified subsample as dcor)
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("APPROACH A2 — POOLED MIC (stratified subsample, equal to dcor)")
    print("─" * 70)
    t0     = time.time()
    mic_df = run_pooled_mic(sub, features)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Joint BH-FDR correction across all 24 p-values (12 dcor + 12 MIC)
    # Both methods share one correction family so their significance flags
    # are directly comparable.
    # ------------------------------------------------------------------
    if not dcor_df.empty and not mic_df.empty:
        # Align on feature order before concatenating
        dcor_aligned = dcor_df.set_index("feature")
        mic_aligned  = mic_df.set_index("feature")
        shared_feats = dcor_aligned.index.tolist()

        all_pvals = np.concatenate([
            dcor_aligned.loc[shared_feats, "dcor_pval"].values,
            mic_aligned.reindex(shared_feats)["mic_pval"].values,
        ])
        all_fdr = benjamini_hochberg(all_pvals)

        n = len(shared_feats)
        dcor_df["dcor_pval_fdr"]        = all_fdr[:n]
        dcor_df["dcor_significant"]     = dcor_df["dcor_pval_fdr"] < FDR_ALPHA
        dcor_df["spearman_pval_fdr"]    = benjamini_hochberg(
            dcor_df["spearman_pval"].values)
        dcor_df["spearman_significant"] = dcor_df["spearman_pval_fdr"] < FDR_ALPHA

        mic_fdr = all_fdr[n:]
        mic_df["mic_pval_fdr"]   = mic_aligned.reindex(shared_feats).assign(
            _fdr=mic_fdr).reset_index()["_fdr"].values
        mic_df["mic_significant"] = mic_df["mic_pval_fdr"] < FDR_ALPHA

        print("\n  Joint BH-FDR applied across dcor + MIC p-values (24 total)")

    elif not dcor_df.empty:
        dcor_df["dcor_pval_fdr"]        = benjamini_hochberg(dcor_df["dcor_pval"].values)
        dcor_df["dcor_significant"]     = dcor_df["dcor_pval_fdr"] < FDR_ALPHA
        dcor_df["spearman_pval_fdr"]    = benjamini_hochberg(dcor_df["spearman_pval"].values)
        dcor_df["spearman_significant"] = dcor_df["spearman_pval_fdr"] < FDR_ALPHA

    elif not mic_df.empty:
        mic_df["mic_pval_fdr"]   = benjamini_hochberg(mic_df["mic_pval"].values)
        mic_df["mic_significant"] = mic_df["mic_pval_fdr"] < FDR_ALPHA

    if not dcor_df.empty:
        n_sig = dcor_df["dcor_significant"].sum()
        print(f"  FDR-significant (dcor): {n_sig} / {len(dcor_df)}")
        print(dcor_df[["feature", "n_samples", "dcor",
                        "dcor_pval_fdr", "dcor_significant",
                        "spearman"]].to_string(index=False))

    if not mic_df.empty:
        n_sig_mic = mic_df["mic_significant"].sum()
        print(f"  FDR-significant (MIC): {n_sig_mic} / {len(mic_df)}")
        print(mic_df[["feature", "n_samples_mic", "mic",
                       "mic_pval_fdr", "mic_significant"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Save Approach A combined results
    # ------------------------------------------------------------------
    if not dcor_df.empty:
        combined_a = dcor_df.copy()
        if not mic_df.empty:
            combined_a = combined_a.merge(
                mic_df[["feature", "n_samples_mic", "mic",
                         "mic_pval", "mic_pval_fdr", "mic_significant"]],
                on="feature", how="left")
        pooled_path = os.path.join(OUTDIR, "pooled_dcor_mic.csv")
        combined_a.to_csv(pooled_path, index=False)
        print(f"\n  Approach A saved: {pooled_path}")

    # ------------------------------------------------------------------
    # Approach B — Meta-analysis
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("APPROACH B — RANDOM-EFFECTS META-ANALYSIS")
    print("─" * 70)
    dcor_all = load_per_dataset_dcor()
    mic_all  = load_per_dataset_mic()

    # dcor meta-analysis
    if dcor_all.empty:
        print("  WARNING: No per-dataset dcor outputs found.")
        print("  Run dcor_analysis.py first to enable dcor meta-analysis.")
        meta_df = pd.DataFrame()
    else:
        print(f"  Loaded {len(dcor_all):,} per-dataset dcor rows "
              f"({dcor_all['dataset'].nunique()} datasets, "
              f"{dcor_all['epoch'].nunique()} epochs)")
        t0      = time.time()
        meta_df = run_meta_analysis(dcor_all)
        print(f"  dcor meta-analysis completed in {time.time()-t0:.1f}s")

        if not meta_df.empty:
            n_univ = meta_df["universal"].sum()
            print(f"\n  Universal features "
                  f"(pooled_dcor>0, I²<{I2_THRESHOLD}%): "
                  f"{n_univ} / {len(meta_df)}")
            print(meta_df[["feature", "pooled_dcor", "ci_lower",
                            "ci_upper", "I2", "p_heterogeneity",
                            "universal"]].to_string(index=False))
            meta_path = os.path.join(OUTDIR, "meta_analysis.csv")
            meta_df.to_csv(meta_path, index=False)
            print(f"\n  dcor meta-analysis saved: {meta_path}")

    # MIC meta-analysis
    if mic_all.empty:
        print("  WARNING: No per-dataset mic outputs found.")
        print("  Run mic_analysis.py first to enable MIC meta-analysis.")
        mic_meta_df = pd.DataFrame()
    else:
        print(f"\n  Loaded {len(mic_all):,} per-dataset MIC rows "
              f"({mic_all['dataset'].nunique()} datasets, "
              f"{mic_all['epoch'].nunique()} epochs)")
        t0          = time.time()
        mic_meta_df = run_mic_meta_analysis(mic_all)
        print(f"  MIC meta-analysis completed in {time.time()-t0:.1f}s")

        if not mic_meta_df.empty:
            n_univ_mic = mic_meta_df["universal"].sum()
            print(f"\n  Universal features MIC "
                  f"(pooled_mic>0, I²<{I2_THRESHOLD}%): "
                  f"{n_univ_mic} / {len(mic_meta_df)}")
            print(mic_meta_df[["feature", "pooled_mic", "ci_lower",
                                "ci_upper", "I2", "p_heterogeneity",
                                "universal"]].to_string(index=False))
            mic_meta_path = os.path.join(OUTDIR, "mic_meta_analysis.csv")
            mic_meta_df.to_csv(mic_meta_path, index=False)
            print(f"\n  MIC meta-analysis saved: {mic_meta_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("GENERATING FIGURES")
    print("─" * 70)

    figs = {
        "fig_pooled_bar.png":   (plot_pooled_bar,
                                  dcor_df, mic_df),
        "fig_forest_plot.png":  (plot_forest,
                                  meta_df, dcor_all),
        "fig_I2_bars.png":      (plot_i2_bars,
                                  meta_df),
        "fig_dcor_heatmap.png": (plot_dcor_heatmap,
                                  dcor_all),
        "fig_mic_I2_bars.png":  (plot_mic_i2_bars,
                                  mic_meta_df),
        "fig_mic_forest.png":   (plot_mic_forest,
                                  mic_meta_df, mic_all),
        "fig_mic_heatmap.png":  (plot_mic_heatmap,
                                  mic_all),
    }

    for name, (fn, *args) in figs.items():
        try:
            fn(*args, os.path.join(OUTDIR, name))
            print(f"  {name}  ✓")
        except Exception as e:
            print(f"  {name}  FAILED: {e}")

    # Methodology
    method_path = os.path.join(OUTDIR, "METHODOLOGY_universal_correlation.txt")
    write_methodology(method_path)
    print(f"  METHODOLOGY_universal_correlation.txt  ✓")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output dir : {os.path.abspath(OUTDIR)}")
    print()
    if os.path.isdir(OUTDIR):
        for fname in sorted(os.listdir(OUTDIR)):
            fpath = os.path.join(OUTDIR, fname)
            if os.path.isfile(fpath):
                print(f"  {fname:<52s}  "
                      f"{os.path.getsize(fpath)/1024:>8.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
