

from __future__ import annotations

import os
import json
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ab.nn.api as api
from ab.nn.util.db.Query import JoinConf

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

# ---- Global Plot Style ----
sns.set_style("whitegrid")

plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})


# ----------------------------
# USER SETTINGS (edit these)
# ----------------------------
OUTDIR = "corr_out"
PLOT_DIR = os.path.join(OUTDIR, "plots")

METHOD = "spearman"     
TOP_K = 6            
MIN_NONNULL = 200      


METRIC_FILTER: Optional[str] = "acc"   


EPOCHS_FOR_CURVES: Optional[List[int]] = None


CURVE_PARAMS: List[str] = []


# ----------------------------
# Helpers: filesystem
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))


# ----------------------------
# Helpers: prm expansion
# ----------------------------
def is_dict_or_json_dict(x: Any) -> bool:
    if isinstance(x, dict):
        return True
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                return isinstance(d, dict)
            except Exception:
                return False
    return False


def parse_maybe_json_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


def expand_prm(df: pd.DataFrame) -> pd.DataFrame:
    if "prm" not in df.columns:
        return df.copy()
    s = df["prm"].dropna()
    if len(s) == 0:
        return df.copy()

    frac = s.head(500).map(is_dict_or_json_dict).mean()
    if frac < 0.2:
        return df.copy()

    parsed = df["prm"].map(parse_maybe_json_dict)
    prm_flat = pd.json_normalize(parsed).add_prefix("prm__")
    out = pd.concat([df.drop(columns=["prm"]), prm_flat], axis=1)
    return out


# ----------------------------
# JoinConf builder (robust)
# ----------------------------
def make_joinconf_for_nn() -> Optional[JoinConf]:
    try:
        sig = inspect.signature(JoinConf)
        kwargs = {}
        for k in ("join_nn", "with_nn", "include_nn", "nn"):
            if k in sig.parameters:
                kwargs[k] = True
        return JoinConf(**kwargs) if kwargs else JoinConf()
    except Exception:
        return None


# ----------------------------
# Load data
# ----------------------------
def load_df() -> pd.DataFrame:
    sql = make_joinconf_for_nn()

    sig = inspect.signature(api.data)
    kwargs = {}
    if "sql" in sig.parameters:
        kwargs["sql"] = sql
    if "only_best_accuracy" in sig.parameters:
        kwargs["only_best_accuracy"] = False

    df = api.data(**kwargs)
    df = expand_prm(df)

    if METRIC_FILTER is not None and "metric" in df.columns:
        df = df[df["metric"] == METRIC_FILTER].copy()

    return df



# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in str(s))

def get_numeric_param_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "task","dataset","metric","metric_code","metric_id",
        "nn","nn_code","nn_id","epoch","accuracy","duration",
        "transform_code","transform_id","prm_id"
    }
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

# ----------------------------
# MI ranking (non-linear dependence)
# ----------------------------
def mi_rank_params(
    df: pd.DataFrame,
    target: str = "accuracy",
    min_n: int = 2000,
    random_state: int = 0
) -> pd.DataFrame:
    num_cols = get_numeric_param_cols(df)
    y = pd.to_numeric(df[target], errors="coerce")

    rows = []
    for col in num_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < min_n:
            continue

        X = x[m].values.reshape(-1, 1)
        Y = y[m].values
        mi = mutual_info_regression(X, Y, random_state=random_state)
        rows.append((col, float(mi[0]), n))

    out = pd.DataFrame(rows, columns=["param", "mutual_info", "n"]).sort_values("mutual_info", ascending=False)
    return out

# ----------------------------
# Spearman ranking (monotonic dependence)
# ----------------------------
def spearman_rank_params(
    df: pd.DataFrame,
    target: str = "accuracy",
    min_n: int = 2000
) -> pd.DataFrame:
    num_cols = get_numeric_param_cols(df)
    y = pd.to_numeric(df[target], errors="coerce")

    rows = []
    for col in num_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < min_n:
            continue

        rho, p = spearmanr(x[m].values, y[m].values)
        rows.append((col, float(rho), float(p), n))

    out = pd.DataFrame(rows, columns=["param", "spearman_rho", "p_value", "n"])
    out["abs_spearman"] = out["spearman_rho"].abs()
    out = out.sort_values("abs_spearman", ascending=False)
    return out

# ----------------------------
# MI vs Spearman per (task, dataset)
# ----------------------------
def mi_vs_spearman_per_task_dataset(
    df: pd.DataFrame,
    metric_filter: str = "acc",
    min_group_rows: int = 5000,
    min_n_per_param: int = 2000,
    top_k: int = 30,
    outdir: str = "corr_out/mi_task_dataset"
) -> pd.DataFrame:
    ensure_dir(outdir)

    d = df.copy()
    if metric_filter is not None and "metric" in d.columns:
        d = d[d["metric"] == metric_filter].copy()

    d["accuracy"] = pd.to_numeric(d["accuracy"], errors="coerce")
    d = d.dropna(subset=["accuracy"])

    all_rows = []

    for (task, dataset), g in d.groupby(["task", "dataset"], dropna=False):
        if len(g) < min_group_rows:
            continue

        mi_tbl = mi_rank_params(g, min_n=min_n_per_param).head(top_k)
        sp_tbl = spearman_rank_params(g, min_n=min_n_per_param)

        merged = mi_tbl.merge(
            sp_tbl[["param", "spearman_rho", "p_value", "n"]],
            on="param",
            how="left",
            suffixes=("_mi", "_sp")
        )
        merged["task"] = task
        merged["dataset"] = dataset
        merged["group_rows"] = len(g)
        merged["abs_spearman"] = merged["spearman_rho"].abs()

        # Save per-group report
        fname = f"mi_vs_spearman__task_{task}__dataset_{dataset}__metric_{metric_filter}.csv"
        path = os.path.join(outdir, safe_filename(fname))
        merged.to_csv(path, index=False)

        all_rows.append(merged)
        print(f"Saved MI vs Spearman: {path}  (group_rows={len(g)})")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined_path = os.path.join(outdir, f"mi_vs_spearman__ALL__metric_{metric_filter}.csv")
    combined.to_csv(combined_path, index=False)
    print("Saved combined MI vs Spearman:", combined_path)

    return combined

# ----------------------------
# Cross-dataset summary: most consistently important params
# ----------------------------
def cross_dataset_importance_summary(
    mi_vs_sp_all: pd.DataFrame,
    top_k_per_group: int = 10,
    out_path: str = "corr_out/mi_task_dataset/summary__param_consistency.csv"
) -> pd.DataFrame:
    """
    Input: combined MI vs Spearman table (ALL groups).
    Output: parameters ranked by frequency in top-k and mean MI across appearances.

    Robust to different sample-size column names (n, n_mi, n_sp, n_x, n_y).
    """
    if mi_vs_sp_all.empty:
        print("No rows to summarize.")
        return mi_vs_sp_all

    df = mi_vs_sp_all.copy()

    # Determine which n-column to use
    n_candidates = ["n", "n_mi", "n_sp", "n_x", "n_y"]
    n_col = next((c for c in n_candidates if c in df.columns), None)

    # If none found, create a safe fallback
    if n_col is None:
        df["n_used"] = np.nan
        n_col = "n_used"

    # Keep top-k per (task,dataset) by MI
    df["rank_in_group"] = df.groupby(["task", "dataset"])["mutual_info"].rank(
        ascending=False, method="first"
    )
    top = df[df["rank_in_group"] <= top_k_per_group].copy()

    # Ensure abs_spearman exists
    if "abs_spearman" not in top.columns and "spearman_rho" in top.columns:
        top["abs_spearman"] = top["spearman_rho"].abs()

    summary = top.groupby("param").agg(
        appearances=("param", "size"),
        mean_mi=("mutual_info", "mean"),
        median_mi=("mutual_info", "median"),
        mean_abs_spearman=("abs_spearman", "mean"),
        median_abs_spearman=("abs_spearman", "median"),
        min_n=(n_col, "min"),
        max_n=(n_col, "max"),
    ).reset_index()

    summary = summary.sort_values(["appearances", "mean_mi"], ascending=[False, False])

    ensure_dir(os.path.dirname(out_path))
    summary.to_csv(out_path, index=False)
    print("Saved cross-dataset summary:", out_path)

    return summary

# ----------------------------
# Nonlinear plots: binned mean curve
# ----------------------------
def binned_mean_plot(
    df: pd.DataFrame,
    param: str,
    metric_filter: str = "acc",
    bins: int = 25,
    min_points: int = 2000,
    save_path: str | None = None,
    title: str | None = None
) -> None:
    d = df.copy()
    if metric_filter is not None and "metric" in d.columns:
        d = d[d["metric"] == metric_filter].copy()

    x = pd.to_numeric(d.get(param), errors="coerce")
    y = pd.to_numeric(d.get("accuracy"), errors="coerce")
    if x is None:
        print("Param not found:", param)
        return
    m = x.notna() & y.notna()
    if int(m.sum()) < min_points:
        print(f"Too few points for plot: {param}  n={int(m.sum())}")
        return

    x = x[m]
    y = y[m]

    # Quantile bins (works well for skewed hyperparams)
    q = pd.qcut(x, q=bins, duplicates="drop")
    g = pd.DataFrame({"bin": q, "x": x, "y": y}).groupby("bin").agg(
        x_mean=("x","mean"),
        y_mean=("y","mean"),
        y_std=("y","std"),
        n=("y","size")
    ).reset_index(drop=True)

    dataset_colors = {
    "cifar-10": "#1f77b4",
    "celeba-gender": "#2E8B57",
    "wikitext": "#6A0DAD",
    }
    dataset_name = d["dataset"].iloc[0] if "dataset" in d.columns else None
    color = dataset_colors.get(dataset_name, "#3B5F8A")

    plt.figure(figsize=(7, 4))

    # Main line
    plt.plot(
        g["x_mean"],
        g["y_mean"],
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4
    )
    plt.fill_between(
    g["x_mean"],
    g["y_mean"] - g["y_std"] / np.sqrt(g["n"]),
    g["y_mean"] + g["y_std"] / np.sqrt(g["n"]),
    alpha=0.2,
    color=color
    )

    # Optional: light fill for emphasis
    plt.fill_between(
        g["x_mean"],
        g["y_mean"],
        alpha=0.15,
        color=color,
    )

    plt.xlabel(param)
    plt.ylabel("Mean Accuracy")
    plt.tight_layout()
    plt.title(title or f"Binned mean: {param} vs accuracy (metric={metric_filter})")

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

# ----------------------------
# Auto-generate plots for top params per group
# ----------------------------
def generate_top_param_plots_per_group(
    df: pd.DataFrame,
    mi_vs_sp_all: pd.DataFrame,
    metric_filter: str = "acc",
    top_k_per_group: int = 3,
    bins: int = 25,
    outdir: str = "corr_out/mi_task_dataset/plots"
) -> None:
    ensure_dir(outdir)
    if mi_vs_sp_all.empty:
        print("No MI vs Spearman results available for plotting.")
        return

    # choose top-k params per (task,dataset) by MI
    mi_vs_sp_all = mi_vs_sp_all.copy()
    mi_vs_sp_all["rank_in_group"] = mi_vs_sp_all.groupby(["task","dataset"])["mutual_info"].rank(ascending=False, method="first")
    pick = mi_vs_sp_all[mi_vs_sp_all["rank_in_group"] <= top_k_per_group].copy()

    for (task, dataset), g in pick.groupby(["task","dataset"]):
        params = g.sort_values("mutual_info", ascending=False)["param"].tolist()
        subset = df[(df["task"] == task) & (df["dataset"] == dataset)].copy()

        for p in params:
            fname = f"binned__task_{task}__dataset_{dataset}__param_{p}__metric_{metric_filter}.png"
            path = os.path.join(outdir, safe_filename(fname))
            binned_mean_plot(
                subset,
                p,
                metric_filter=metric_filter,
                bins=bins,
                min_points=1000,
                save_path=path,
                title=f"{task} / {dataset} | {p} (MI={g[g['param']==p]['mutual_info'].iloc[0]:.3f})"
            )
        print(f"Saved plots for group: {task} / {dataset}")


def find_mi_beats_spearman(mi_vs_sp_all: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = mi_vs_sp_all.copy()
    df["mi_rank"] = df.groupby(["task","dataset"])["mutual_info"].rank(ascending=False, method="first")
    df["sp_rank"] = df.groupby(["task","dataset"])["abs_spearman"].rank(ascending=False, method="first")
    df["rank_gain"] = df["sp_rank"] - df["mi_rank"]  # positive => MI ranks it higher (better)

    # keep strong MI candidates per group, then sort by biggest gain
    cand = df[df["mi_rank"] <= top_n].sort_values("rank_gain", ascending=False)
    return cand


def compute_epoch_snapshot(df, epoch_mode="epoch1"):
    """
    epoch_mode:
        - 'epoch1'
        - 'epoch5'
        - 'final'
    """
    if epoch_mode == "epoch1":
        d = df[df["epoch"] == 1].copy()

    elif epoch_mode == "epoch5":
        d = df[df["epoch"] == 5].copy()

    elif epoch_mode == "final":
        # final epoch per (task, dataset, nn)
        idx = (
            df.groupby(["task", "dataset", "nn"])["epoch"]
            .transform("max") == df["epoch"]
        )
        d = df[idx].copy()

    else:
        raise ValueError("Invalid epoch_mode")

    print(f"\nSnapshot: {epoch_mode}")
    print("Rows:", len(d))

    return d
# ----------------------------
# Main workflow
# ----------------------------
def main() -> None:
    df = load_df()
    print("Loaded rows:", len(df))
    print("Columns:", len(df.columns))
    print("Metric filter:", METRIC_FILTER)
    return df

    
    

if __name__ == "__main__":
    df_loaded = main()  
   # print("\nEpoch summary:")
   # print(df_loaded["epoch"].describe())
   # print("\nUnique epochs:")
   # print(sorted(df_loaded["epoch"].unique()))
   # print("\nEpoch counts:")
   # print(df_loaded["epoch"].value_counts().sort_index())

    # 1) MI vs Spearman for each (task, dataset)
    mi_vs_sp_all = mi_vs_spearman_per_task_dataset(
        df_loaded,
        metric_filter="acc",
        min_group_rows=5000,
        min_n_per_param=2000,
        top_k=30,
        outdir="corr_out/mi_task_dataset"
    )
    print("mi_vs_sp_all columns:", mi_vs_sp_all.columns.tolist())

    # 2) Cross-dataset summary: which params are consistently important
    summary = cross_dataset_importance_summary(
        mi_vs_sp_all,
        top_k_per_group=10,
        out_path="corr_out/mi_task_dataset/summary__param_consistency.csv"
    )

    print("\nTop 20 most consistent params across task/dataset:")
    print(summary.head(20))

    # 3) Nonlinear plots (binned mean curves) for top params per group
    generate_top_param_plots_per_group(
        df_loaded,
        mi_vs_sp_all,
        metric_filter="acc",
        top_k_per_group=3,
        bins=25,
        outdir="corr_out/mi_task_dataset/plots"
    )

    cand = find_mi_beats_spearman(mi_vs_sp_all, top_n=10)
    print("\nTop cases where MI ranks a param much higher than Spearman (potential non-linear effects):")
    print(cand.head(30)[["task","dataset","param","mutual_info","abs_spearman","mi_rank","sp_rank","rank_gain","n_mi","group_rows"]])
    import pandas as pd

    all_tbl = pd.read_csv("corr_out/mi_task_dataset/mi_vs_spearman__ALL__metric_acc.csv")

    nonlinear = mi_vs_sp_all[
    (mi_vs_sp_all["mutual_info"] >= mi_vs_sp_all["mutual_info"].quantile(0.90)) &
    (mi_vs_sp_all["abs_spearman"] < 0.20)
    ].sort_values("mutual_info", ascending=False)

    print("\nCandidates: high MI (top 10%) + low Spearman (<0.20):")
    print(nonlinear.head(30))

    print("\nTop nonlinear-only effects (high MI, low Spearman):")
    print(nonlinear.head(20))

        # ---- Manual nonlinear inspection example ----
    subset = df_loaded[
        (df_loaded["task"] == "img-classification") &
        (df_loaded["dataset"] == "cifar-10")
    ]

    binned_mean_plot(
        subset,
        "prm__weight_decay",
        metric_filter="acc",
        bins=25,
        save_path="corr_out/mi_task_dataset/plots/manual__cifar10_weight_decay.png"
    )

    print("Saved manual nonlinear plot for CIFAR-10 weight_decay.")
    print("\nDONE. Check outputs in: corr_out/mi_task_dataset/")
    subset = df_loaded[
    (df_loaded["task"] == "txt-generation") &
    (df_loaded["dataset"] == "wikitext")
    ]

    binned_mean_plot(
        subset,
        "prm__momentum",
        metric_filter="acc",
        bins=20,
        save_path="corr_out/mi_task_dataset/plots/manual__wikitext_momentum.png"
    )

    df_epoch1 = compute_epoch_snapshot(df_loaded, "epoch1")
    df_epoch5 = compute_epoch_snapshot(df_loaded, "epoch5")
    df_final  = compute_epoch_snapshot(df_loaded, "final")

    print("\nTop MI at Epoch 1")
    print(mi_rank_params(df_epoch1).head(10))

    print("\nTop MI at Epoch 5")
    print(mi_rank_params(df_epoch5).head(10))

    print("\nTop MI at Final Epoch")
    print(mi_rank_params(df_final).head(10))


