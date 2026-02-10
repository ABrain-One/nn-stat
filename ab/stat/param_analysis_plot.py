
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
# Param selection (no hardcoding)
# ----------------------------
def select_param_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {
        "task", "dataset", "metric", "metric_id", "metric_code",
        "nn", "nn_id", "nn_code",
        "epoch", "accuracy", "duration",
        "transform_code", "transform_id", "prm_id",
    }
    cols = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def rank_params_by_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if "accuracy" not in df.columns:
        raise ValueError("accuracy column missing")

    num_cols, _ = select_param_columns(df)

    tmp = df[["accuracy"] + num_cols].copy()
    tmp = tmp.dropna(subset=["accuracy"])

    for c in num_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    corrs = tmp.corr(method=METHOD)["accuracy"].dropna().drop(labels=["accuracy"], errors="ignore")
    n = tmp[num_cols].notna().sum().reindex(corrs.index).fillna(0).astype(int)

    out = pd.DataFrame({"corr": corrs, "n": n})
    out["abs_corr"] = out["corr"].abs()
    out = out[out["n"] >= MIN_NONNULL].sort_values("abs_corr", ascending=False)
    return out


# ----------------------------
# Plotting
# ----------------------------
def plot_corr_heatmap(df: pd.DataFrame, save_path: str, max_cols: int = 40) -> None:
    """
    Parameter-parameter correlation heatmap for numeric parameters.
    To keep it readable, we plot only the top `max_cols` parameters by abs correlation with accuracy.
    """
    rank = rank_params_by_accuracy(df)
    top_params = rank.head(max_cols).index.tolist()

    X = df[top_params].copy()
    for c in top_params:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    corr = X.corr(method=METHOD)

    plt.figure(figsize=(14, 12))
    plt.imshow(corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(top_params)), top_params, rotation=90, fontsize=7)
    plt.yticks(range(len(top_params)), top_params, fontsize=7)
    title = f"Param–Param Correlation Heatmap ({METHOD})"
    if METRIC_FILTER:
        title += f" | metric={METRIC_FILTER}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_top_scatter(df: pd.DataFrame, params: List[str], base_dir: str) -> None:
    """
    Scatter plots: each param vs accuracy
    """
    for p in params:
        if p not in df.columns:
            continue
        x = pd.to_numeric(df[p], errors="coerce")
        y = pd.to_numeric(df["accuracy"], errors="coerce")
        m = x.notna() & y.notna()

        if m.sum() < 50:
            continue

        plt.figure(figsize=(7, 5))
        plt.scatter(x[m], y[m], alpha=0.25, s=8)
        plt.xlabel(p)
        plt.ylabel("accuracy")
        title = f"{p} vs accuracy"
        if METRIC_FILTER:
            title += f" | metric={METRIC_FILTER}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"scatter__{safe_filename(p)}.png"), dpi=200)
        plt.close()


def plot_histograms(df: pd.DataFrame, params: List[str], base_dir: str) -> None:
    """
    Histograms for parameter distributions (sanity checks).
    """
    for p in params:
        if p not in df.columns:
            continue
        x = pd.to_numeric(df[p], errors="coerce").dropna()
        if len(x) < 50:
            continue

        plt.figure(figsize=(7, 4))
        plt.hist(x.values, bins=50)
        plt.xlabel(p)
        plt.ylabel("count")
        title = f"Distribution of {p}"
        if METRIC_FILTER:
            title += f" | metric={METRIC_FILTER}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"hist__{safe_filename(p)}.png"), dpi=200)
        plt.close()


def corr_param_with_accuracy_over_epochs(df: pd.DataFrame, param: str, epochs: Optional[List[int]] = None) -> pd.DataFrame:
    if "epoch" not in df.columns:
        raise ValueError("No epoch column in df")
    if param not in df.columns:
        raise ValueError(f"Param '{param}' not found")

    e_all = pd.to_numeric(df["epoch"], errors="coerce")
    if epochs is None:
        epochs = sorted(e_all.dropna().unique().astype(int).tolist())

    rows = []
    for e in epochs:
        d = df.loc[e_all == e, ["accuracy", param]].copy()
        d["accuracy"] = pd.to_numeric(d["accuracy"], errors="coerce")
        d[param] = pd.to_numeric(d[param], errors="coerce")
        d = d.dropna()

        n = len(d)
        if n < 30:
            continue

        corr = d.corr(method=METHOD).iloc[0, 1]
        rows.append({"epoch": int(e), "corr": float(corr), "n": int(n)})

    return pd.DataFrame(rows)


def plot_epoch_curves(df: pd.DataFrame, params: List[str], base_dir: str) -> None:
    """
    For each param, plot corr(param, accuracy) over epochs.
    """
    if "epoch" not in df.columns:
        return

    for p in params:
        if p not in df.columns:
            continue

        curve = corr_param_with_accuracy_over_epochs(df, p, epochs=EPOCHS_FOR_CURVES)
        if curve.empty:
            continue

        plt.figure(figsize=(7, 4))
        plt.plot(curve["epoch"], curve["corr"])
        plt.xlabel("epoch")
        plt.ylabel(f"corr({p}, accuracy)")
        title = f"Epoch-wise correlation ({METHOD})"
        if METRIC_FILTER:
            title += f" | metric={METRIC_FILTER}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"epoch_curve__{safe_filename(p)}.png"), dpi=200)
        plt.close()

        curve.to_csv(os.path.join(base_dir, f"epoch_curve__{safe_filename(p)}.csv"), index=False)


def plot_architecture_boxplots(df: pd.DataFrame, base_dir: str) -> None:
    """
    Boxplots for common architecture flags if present (is_resnet_like, has_attention, etc.)
    """
    candidates = [c for c in df.columns if any(k in c.lower() for k in [
        "has_attention", "has_residual", "is_resnet_like", "is_vgg_like", "is_transformer_like",
        "is_unet_like", "is_mobilenet_like", "is_efficientnet_like"
    ])]

    good = []
    for c in candidates:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if len(vals) <= 3:
            good.append(c)

    if not good:
        return

    y = pd.to_numeric(df["accuracy"], errors="coerce")

    for c in good[:10]:
        x = pd.to_numeric(df[c], errors="coerce")
        m0 = (x == 0) & y.notna()
        m1 = (x == 1) & y.notna()

        if m0.sum() < 30 or m1.sum() < 30:
            continue

        data = [y[m0].values, y[m1].values]

        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=["0", "1"])
        plt.xlabel(c)
        plt.ylabel("accuracy")
        title = f"{c} vs accuracy"
        if METRIC_FILTER:
            title += f" | metric={METRIC_FILTER}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"box__{safe_filename(c)}.png"), dpi=200)
        plt.close()


# ----------------------------
# Main workflow
# ----------------------------
def main() -> None:
    ensure_dir(OUTDIR)
    ensure_dir(PLOT_DIR)

    df = load_df()
    print("Loaded rows:", len(df))
    print("Columns:", len(df.columns))
    print("Metric filter:", METRIC_FILTER)

    ranking = rank_params_by_accuracy(df)
    ranking_path = os.path.join(OUTDIR, f"ranking_param_acc__{METHOD}" + (f"__metric_{METRIC_FILTER}" if METRIC_FILTER else "") + ".csv")
    ranking.to_csv(ranking_path, index=True)
    print("Saved ranking:", ranking_path)

    if CURVE_PARAMS:
        top_params = [p for p in CURVE_PARAMS if p in df.columns]
    else:
        top_params = ranking.head(TOP_K).index.tolist()

    print("Top params:", top_params)

    # 1) Heatmap of param-param correlations (top N by acc-corr)
    heatmap_path = os.path.join(PLOT_DIR, f"heatmap_param_param__{METHOD}" + (f"__metric_{METRIC_FILTER}" if METRIC_FILTER else "") + ".png")
    plot_corr_heatmap(df, heatmap_path, max_cols=40)
    print("Saved heatmap:", heatmap_path)

    # 2) Scatter plots (top params vs accuracy)
    plot_top_scatter(df, top_params, PLOT_DIR)
    print("Saved scatter plots.")

    # 3) Histograms (top param distributions)
    plot_histograms(df, top_params, PLOT_DIR)
    print("Saved histograms.")

    # 4) Epoch correlation curves for top params
    if "epoch" in df.columns:
        plot_epoch_curves(df, top_params, PLOT_DIR)
        print("Saved epoch-correlation curves.")

    # 5) Architecture flag boxplots (if metadata is joined and present)
    plot_architecture_boxplots(df, PLOT_DIR)
    print("Saved architecture boxplots (if columns exist).")

    # Small console summary
    print("\nTop correlations:")
    print(ranking.head(20))

    print("\nDone. See:", PLOT_DIR)


if __name__ == "__main__":
    main()
