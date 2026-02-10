from __future__ import annotations

import os
import json
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import ab.nn.api as api

import numpy as np
import pandas as pd

from ab.nn.util.db.Query import JoinConf


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class CorrConfig:
    outdir: str = "corr_out"
    method: str = "spearman"          
    top_k: int = 30
    min_nonnull_param_acc: int = 50   
    min_nonnull_param_param: int = 200  
    min_n_epoch_query: int = 30       
    epochs_to_analyze: List[int] = None  

    per_metric: bool = True
    per_task: bool = True


# ----------------------------
# Filesystem
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


# ----------------------------
# Dict/JSON parsing for prm
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
    """
    Expand df['prm'] dict into flat columns prm__<key> (no hardcoding).
    Keeps other columns intact.
    """
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
# JoinConf (robust construction)
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
# Load data (only uses your api.data)
# ----------------------------
def load_runs(
    only_best_accuracy: bool = False,
    task: Optional[str] = None,
    dataset: Optional[str] = None,
    metric: Optional[str] = None,
    nn: Optional[str] = None,
    epoch: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    sql = make_joinconf_for_nn()

    sig = inspect.signature(api.data)
    kwargs = {}
    for k, v in [
        ("only_best_accuracy", only_best_accuracy),
        ("task", task),
        ("dataset", dataset),
        ("metric", metric),
        ("nn", nn),
        ("epoch", epoch),
        ("max_rows", max_rows),
        ("sql", sql),
    ]:
        if k in sig.parameters:
            kwargs[k] = v

    df = api.data(**kwargs)
    return df


# ----------------------------
# Column selection (no hardcoding hyperparameter names)
# ----------------------------
def select_param_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {
        # identifiers
        "task", "dataset", "metric", "metric_id", "metric_code",
        "nn", "nn_id", "nn_code",
        # training index / target / runtime
        "epoch", "accuracy", "duration",
        # code blobs / IDs
        "transform_code", "transform_id", "prm_id",
    }

    cols = [c for c in df.columns if c not in exclude]

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    return num_cols, cat_cols


# ----------------------------
# Correlation: param ↔ accuracy
# ----------------------------
def corr_params_with_accuracy(
    df: pd.DataFrame,
    method: str,
    min_nonnull: int,
) -> pd.DataFrame:
    
    if "accuracy" not in df.columns:
        raise ValueError("accuracy column not found")

    num_cols, _ = select_param_columns(df)
    if not num_cols:
        return pd.DataFrame(columns=["corr", "n", "abs_corr"])

    tmp = df[["accuracy"] + num_cols].copy()
    tmp = tmp.dropna(subset=["accuracy"])

    for c in num_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    corrs = tmp.corr(method=method)["accuracy"].dropna().drop(labels=["accuracy"], errors="ignore")
    n = tmp[num_cols].notna().sum()

    out = pd.DataFrame({"corr": corrs, "n": n.reindex(corrs.index).fillna(0).astype(int)})
    out["abs_corr"] = out["corr"].abs()
    out = out[out["n"] >= min_nonnull].sort_values("abs_corr", ascending=False)
    return out


# ----------------------------
# Correlation: param ↔ param (dependencies)
# ----------------------------
def top_param_param_correlations(
    df: pd.DataFrame,
    method: str,
    top_k: int,
    min_nonnull: int,
) -> pd.DataFrame:
   
    num_cols, _ = select_param_columns(df)
    if len(num_cols) < 2:
        return pd.DataFrame(columns=["param_a", "param_b", "corr", "n", "abs_corr"])

    X = df[num_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    keep = [c for c in num_cols if X[c].notna().sum() >= min_nonnull]
    X = X[keep]
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["param_a", "param_b", "corr", "n", "abs_corr"])

    C = X.corr(method=method)

    pairs = []
    cols = C.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            val = C.loc[a, b]
            if pd.isna(val):
                continue
            n = int((X[[a, b]].notna().all(axis=1)).sum())
            if n < min_nonnull:
                continue
            pairs.append((a, b, float(val), n, abs(float(val))))

    out = pd.DataFrame(pairs, columns=["param_a", "param_b", "corr", "n", "abs_corr"])
    out = out.sort_values("abs_corr", ascending=False).head(top_k)
    return out


# ----------------------------
# Epoch helpers
# ----------------------------
def get_first_epoch_value(df: pd.DataFrame) -> Optional[int]:
    if "epoch" not in df.columns:
        return None
    e = pd.to_numeric(df["epoch"], errors="coerce").dropna()
    if len(e) == 0:
        return None
    return int(e.min())


def filter_epoch(df: pd.DataFrame, epoch_value: int) -> pd.DataFrame:
    if "epoch" not in df.columns:
        return df.iloc[0:0].copy()
    e = pd.to_numeric(df["epoch"], errors="coerce")
    return df.loc[e == epoch_value].copy()


def final_epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    
    needed = ["task", "dataset", "metric", "nn", "epoch"]
    if not all(c in df.columns for c in needed):
        if "epoch" in df.columns:
            mx = pd.to_numeric(df["epoch"], errors="coerce").max()
            return filter_epoch(df, int(mx)) if pd.notna(mx) else df.iloc[0:0].copy()
        return df.iloc[0:0].copy()

    tmp = df.copy()
    tmp["epoch_num"] = pd.to_numeric(tmp["epoch"], errors="coerce")
    tmp = tmp.dropna(subset=["epoch_num"])
    idx = tmp.groupby(["task", "dataset", "metric", "nn"])["epoch_num"].idxmax()
    return tmp.loc[idx].drop(columns=["epoch_num"])


# ----------------------------
# Query functions YOU asked for
# ----------------------------
def corr_param_with_accuracy_at_epoch(
    df: pd.DataFrame,
    param: str,
    epoch_value: Optional[int] = None,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    Correlate ONE parameter with accuracy.
    If epoch_value is None: uses all epochs.
    """
    if param not in df.columns:
        return {"param": param, "epoch": epoch_value, "error": "param not found"}

    d = df.copy()
    if epoch_value is not None:
        d = filter_epoch(d, epoch_value)

    d = d.dropna(subset=["accuracy", param]).copy()
    d[param] = pd.to_numeric(d[param], errors="coerce")
    d = d.dropna(subset=[param])

    if len(d) < 10:
        return {"param": param, "epoch": epoch_value, "n": int(len(d)), "corr": np.nan}

    corr = d[[param, "accuracy"]].corr(method=method).iloc[0, 1]
    return {"param": param, "epoch": epoch_value, "n": int(len(d)), "corr": float(corr)}


def corr_param_with_accuracy_over_epochs(
    df: pd.DataFrame,
    param: str,
    epochs: Optional[List[int]] = None,
    method: str = "spearman",
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Correlation curve corr(param, accuracy) per epoch.
    If epochs is None: uses all distinct epochs present.
    """
    if "epoch" not in df.columns:
        raise ValueError("No 'epoch' column in df")
    if param not in df.columns:
        raise ValueError(f"Parameter '{param}' not found")

    if epochs is None:
        epochs = sorted(pd.to_numeric(df["epoch"], errors="coerce").dropna().unique().astype(int).tolist())

    rows = []
    for e in epochs:
        res = corr_param_with_accuracy_at_epoch(df, param, epoch_value=e, method=method)
        if res.get("n", 0) >= min_n:
            rows.append(res)

    return pd.DataFrame(rows)


# ----------------------------
# Bundles (global/per-metric/per-task)
# ----------------------------
def bundle_param_accuracy(
    df: pd.DataFrame,
    cfg: CorrConfig,
    label: str,
) -> Dict[str, pd.DataFrame]:
    out = {}
    out[f"{label}__global"] = corr_params_with_accuracy(df, cfg.method, cfg.min_nonnull_param_acc)

    if cfg.per_metric and "metric" in df.columns:
        for m, g in df.groupby("metric"):
            out[f"{label}__metric__{m}"] = corr_params_with_accuracy(g, cfg.method, cfg.min_nonnull_param_acc)

    if cfg.per_task and "task" in df.columns:
        for t, g in df.groupby("task"):
            out[f"{label}__task__{t}"] = corr_params_with_accuracy(g, cfg.method, cfg.min_nonnull_param_acc)

    return out


def export_tables(tables: Dict[str, pd.DataFrame], outdir: str) -> None:
    ensure_dir(outdir)
    for name, df in tables.items():
        path = os.path.join(outdir, f"{safe_filename(name)}.csv")
        df.to_csv(path, index=True)


def print_top(title: str, df_corr: pd.DataFrame, top_k: int) -> None:
    print(f"\n{title}")
    if df_corr.empty:
        print("  (empty after filtering)")
        return
    show = df_corr.head(top_k)
    for idx, row in show.iterrows():
        print(f"  {idx:30s} corr={row['corr']:+.4f}  n={int(row['n'])}")


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = CorrConfig(
        outdir="corr_out",
        method="spearman",
        top_k=30,
        min_nonnull_param_acc=50,
        min_nonnull_param_param=200,
        min_n_epoch_query=30,
        epochs_to_analyze=[1, 5, 10, 20],  # you can put ANY epoch numbers here
        per_metric=True,
        per_task=True,
    )

    ensure_dir(cfg.outdir)

    # 1) Load from DB
    df = load_runs(only_best_accuracy=False, max_rows=None)
    print("Loaded rows:", len(df))
    print("Loaded columns:", len(df.columns))

    # 2) Expand prm hyperparameters
    df = expand_prm(df)

    # 3) Confirm architecture stats presence 
    arch_hits = [c for c in df.columns if any(k in c.lower() for k in [
        "total_layers", "leaf_layers", "max_depth", "total_params", "trainable_params", "frozen_params",
        "flops", "model_size", "buffer_size", "total_memory",
        "dropout_count", "has_attention", "has_residual", "is_resnet", "is_transformer", "is_vgg", "is_unet",
    ])]
    print("Architecture-like columns found:", arch_hits[:40], ("..." if len(arch_hits) > 40 else ""))

    # ---------------------------
    # A) Parameter-parameter correlations (dependencies)
    # ---------------------------
    dep_all = top_param_param_correlations(
        df=df,
        method=cfg.method,
        top_k=200,   
        min_nonnull=cfg.min_nonnull_param_param,
    )
    dep_all_path = os.path.join(cfg.outdir, "TOP_param_param__all_epochs.csv")
    dep_all.to_csv(dep_all_path, index=False)
    print("\n[PARAM ↔ PARAM] Top dependencies (all epochs):")
    print(dep_all.head(cfg.top_k).to_string(index=False))

    # ---------------------------
    # B) Parameter-accuracy correlations
    # ---------------------------
    all_epochs = bundle_param_accuracy(df, cfg, label="param_acc__all_epochs")
    print_top("[PARAM ↔ ACC] All epochs (GLOBAL) top:", all_epochs["param_acc__all_epochs__global"], cfg.top_k)
    export_tables(all_epochs, cfg.outdir)

    # ---------------------------
    # C) First epoch
    # ---------------------------
    first_epoch = get_first_epoch_value(df)
    if first_epoch is not None:
        df_first = filter_epoch(df, first_epoch)
        first_bundle = bundle_param_accuracy(df_first, cfg, label=f"param_acc__epoch_{first_epoch}")
        print_top(f"[PARAM ↔ ACC] Epoch {first_epoch} (GLOBAL) top:", first_bundle[f"param_acc__epoch_{first_epoch}__global"], cfg.top_k)
        export_tables(first_bundle, cfg.outdir)
    else:
        print("\nNo epoch column or no epoch values -> cannot compute first-epoch correlations.")

    # ---------------------------
    # D) Any epochs you specify
    # ---------------------------
    if cfg.epochs_to_analyze:
        for e in cfg.epochs_to_analyze:
            df_e = filter_epoch(df, e)
            if len(df_e) == 0:
                continue
            bundle = bundle_param_accuracy(df_e, cfg, label=f"param_acc__epoch_{e}")
            print_top(f"[PARAM ↔ ACC] Epoch {e} (GLOBAL) top:", bundle[f"param_acc__epoch_{e}__global"], cfg.top_k)
            export_tables(bundle, cfg.outdir)

    # ---------------------------
    # E) Final epoch per run
    # ---------------------------
    df_final = final_epoch_rows(df)
    final_bundle = bundle_param_accuracy(df_final, cfg, label="param_acc__final_epoch_per_run")
    print_top("[PARAM ↔ ACC] Final epoch per run (GLOBAL) top:", final_bundle["param_acc__final_epoch_per_run__global"], cfg.top_k)
    export_tables(final_bundle, cfg.outdir)

    # ---------------------------
    # F) Convenience: function demos (any param, any epoch)
    # ---------------------------
   
    best_global = all_epochs["param_acc__all_epochs__global"]
    if not best_global.empty and "epoch" in df.columns:
        best_param = best_global.index[0]
        curve = corr_param_with_accuracy_over_epochs(
            df=df,
            param=best_param,
            epochs=None,
            method=cfg.method,
            min_n=cfg.min_n_epoch_query,
        )
        curve_path = os.path.join(cfg.outdir, f"CURVE_param_acc__{safe_filename(best_param)}.csv")
        curve.to_csv(curve_path, index=False)
        print(f"\n[CURVE] Saved correlation curve for best param '{best_param}' -> {curve_path}")

    summary_txt = os.path.join(cfg.outdir, "SUMMARY.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Top param↔param dependencies (all epochs):\n")
        f.write(dep_all.head(cfg.top_k).to_string(index=False))
        f.write("\n\nTop param↔accuracy (all epochs, global):\n")
        f.write(all_epochs["param_acc__all_epochs__global"].head(cfg.top_k).to_string())
        f.write("\n")
    print("\nWrote:", summary_txt)
    print("\nDone. All CSVs in:", cfg.outdir)

    corr_param_with_accuracy_at_epoch(df, "prm__lr", epoch_value=5)
    corr_param_with_accuracy_over_epochs(df, "prm__lr", epochs=[1,5,10,20])
    top_param_param_correlations(df, method="spearman", top_k=50, min_nonnull=200)

    return df


if __name__ == "__main__":
    df_loaded = main()
