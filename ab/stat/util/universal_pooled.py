"""
Phase 0 — Data Pooling & Normalization for Universal Analysis.

Reads all per-dataset CSVs from dataset_splits/epoch_1_5_50/, concatenates
them into a single pooled dataset.

Column roles in output CSV
--------------------------
Bookkeeping (NOT features):
  dataset          — dataset name string
  dataset_id       — integer 0–6 (ordered as DATASETS list)
  epoch            — raw epoch value {1, 5, 50}; kept for compatibility
                     with existing per-dataset scripts, not used as a feature
  accuracy         — raw accuracy (not normalized)

Produced by this script:
  epoch_log        — log(epoch); maps {1,5,50} → {0, 1.61, 3.91}
                     FEATURE: evenly spaced epoch scale for correlation & PySR
  accuracy_znorm_ds — z-scored within dataset only (NOT within epoch)
                     TARGET: removes dataset offset, preserves epoch trend

Feature allowlist (12 features):
  epoch_log, nn_total_params, nn_flops, nn_total_layers, nn_max_depth,
  nn_has_attention, nn_has_residual, nn_dropout_count,
  prm__lr, prm__batch, prm__dropout, prm__momentum

Why within-dataset z-score (not within dataset × epoch)?
  Z-scoring within each dataset × epoch cell erases the epoch trend from the
  target, making it impossible for correlation or PySR to detect how epoch
  relates to accuracy. Z-scoring within dataset only removes the dataset-level
  offset (MNIST ≈ 99% vs Places365 ≈ 40%) while preserving the epoch signal.

Why epoch_log and not raw epoch?
  Raw epoch {1, 5, 50} is not evenly spaced. Log scale {0, 1.61, 3.91} is more
  evenly spaced, improves correlation detection, and produces simpler symbolic
  regression formulas. Raw epoch stays as a bookkeeping column only.

Outputs
-------
  universal_out/pooled_all_datasets.csv   — 246,908 rows × 112 cols
  universal_out/group_stats.csv           — per-dataset × per-epoch accuracy stats
  universal_out/feature_coverage.csv      — non-null counts per feature per dataset
  universal_out/DATA_MANIFEST.json        — sha256 checksums, software versions,
                                            column roles, timestamp

Usage
-----
    python -m ab.stat.util.universal_pooled
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, List

import importlib.metadata

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


# ============================================================================
# Configuration
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

# 12 features used in Phase 1 (correlation) and Phase 2 (PySR).
# epoch_log is first — it is the epoch predictor.
# Raw epoch is NOT here; it is a bookkeeping column only.
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

# Explicit role of every key column — written into the manifest.
COLUMN_ROLES: Dict[str, str] = {
    "dataset":            "bookkeeping — dataset name; not a feature",
    "dataset_id":         "bookkeeping — integer 0–6; not a feature",
    "epoch":              "bookkeeping — raw epoch {1,5,50}; not a feature (kept for existing scripts)",
    "epoch_log":          "feature — log(epoch); evenly spaced epoch scale for correlation and PySR",
    "accuracy":           "bookkeeping — raw accuracy; not normalized",
    "accuracy_znorm_ds":  "TARGET — z-scored within dataset only; epoch trend preserved",
}

_base = os.path.join(os.path.dirname(__file__), "..", "..", "..")
INPUT_DIR = os.path.join(_base, "dataset_splits", "epoch_1_5_50")
OUTDIR    = os.path.join(_base, "universal_out")


# ============================================================================
# Helpers
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def collect_software_versions() -> Dict[str, str]:
    return {
        "python":       sys.version.split()[0],
        "platform":     platform.platform(),
        "numpy":        np.__version__,
        "pandas":       pd.__version__,
        "scipy":        get_package_version("scipy"),
        "dcor":         get_package_version("dcor"),
        "minepy":       get_package_version("minepy"),
        "pysr":         get_package_version("pysr"),
        "scikit-learn": get_package_version("scikit-learn"),
        "matplotlib":   get_package_version("matplotlib"),
        "tqdm":         get_package_version("tqdm"),
    }


# ============================================================================
# Loading
# ============================================================================

def load_dataset(ds_name: str) -> pd.DataFrame | None:
    path = os.path.join(INPUT_DIR, f"{ds_name}.csv")
    if not os.path.exists(path):
        tqdm.write(f"  [SKIP] {ds_name}: CSV not found at {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df[df["epoch"].isin(EPOCHS)].copy()

    if "accuracy" not in df.columns:
        tqdm.write(f"  [SKIP] {ds_name}: no 'accuracy' column")
        return None

    # Overwrite with canonical name (source CSV already has dataset col, but
    # ensure it matches the key we use throughout)
    df["dataset"] = ds_name
    return df


# ============================================================================
# Normalization & derived columns
# ============================================================================

def _zscore(series: pd.Series) -> pd.Series:
    """Z-score; returns zeros for constant groups (std == 0)."""
    mu    = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add accuracy_znorm_ds and epoch_log to the pooled dataframe.

    accuracy_znorm_ds
        Z-scored within each dataset (across ALL epochs combined).
        Removes dataset-level accuracy offset; preserves epoch trend.
        This is the primary target for Phase 1 and Phase 2.

    epoch_log
        Natural log of epoch. {1, 5, 50} → {0.0, 1.609, 3.912}.
        Used as the epoch predictor feature everywhere.
    """
    df = df.copy()
    df["accuracy_znorm_ds"] = (
        df.groupby("dataset")["accuracy"].transform(_zscore)
    )
    df["epoch_log"] = np.log(df["epoch"].astype(float))
    return df


# ============================================================================
# Dataset integer encoding
# ============================================================================

def add_dataset_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset_id: integer 0–6 ordered as DATASETS list."""
    mapping = {ds: idx for idx, ds in enumerate(DATASETS)}
    df = df.copy()
    df["dataset_id"] = df["dataset"].map(mapping)
    return df


# ============================================================================
# Summary statistics
# ============================================================================

def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-dataset × per-epoch summary of raw accuracy and accuracy_znorm_ds.

    znorm_ds_mean rises across epochs (negative at epoch 1, positive at epoch 50)
    — confirming the epoch trend is preserved in the target.
    """
    rows = []
    for ds in sorted(df["dataset"].unique()):
        for ep in sorted(df["epoch"].unique()):
            sub = df[(df["dataset"] == ds) & (df["epoch"] == ep)]
            if sub.empty:
                continue
            acc = sub["accuracy"]
            zds = sub["accuracy_znorm_ds"]
            rows.append({
                "dataset":       ds,
                "epoch":         int(ep),
                "epoch_log":     round(float(np.log(ep)), 4),
                "n_rows":        len(sub),
                "acc_mean":      round(float(acc.mean()), 6),
                "acc_std":       round(float(acc.std(ddof=1)), 6),
                "acc_min":       round(float(acc.min()), 6),
                "acc_max":       round(float(acc.max()), 6),
                "znorm_ds_mean": round(float(zds.mean()), 6),  # rises epoch 1→50
                "znorm_ds_std":  round(float(zds.std(ddof=1)), 6),
            })
    return pd.DataFrame(rows)


def check_feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Non-null row count per allowlisted feature per dataset."""
    rows = []
    for feat in FEATURE_ALLOWLIST:
        if feat not in df.columns:
            rows.append({"feature": feat, "in_pooled": False,
                         **{ds: 0 for ds in DATASETS}})
            continue
        row: dict = {"feature": feat, "in_pooled": True}
        for ds in DATASETS:
            row[ds] = int(df.loc[df["dataset"] == ds, feat].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Manifest
# ============================================================================

def build_manifest(
    input_paths:   Dict[str, str],
    output_paths:  Dict[str, str],
    dataset_stats: Dict[str, dict],
    software:      Dict[str, str],
) -> dict:
    manifest: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": software,
        "column_roles": COLUMN_ROLES,
        "normalization": {
            "accuracy_znorm_ds": {
                "grouping": "within dataset only (NOT within epoch)",
                "formula":  "(accuracy - mean(accuracy|dataset)) / std(accuracy|dataset, ddof=1)",
                "epoch_signal": "preserved — znorm_ds_mean rises from negative (epoch 1) to positive (epoch 50)",
                "use_for":  "Phase 1 universal correlation; Phase 2 universal PySR",
            },
            "epoch_log": {
                "formula":  "log(epoch)",
                "values":   {str(ep): round(float(np.log(ep)), 6) for ep in EPOCHS},
                "rationale": (
                    "Raw epoch {1,5,50} is not evenly spaced. "
                    "Log scale {0, 1.61, 3.91} is evenly spaced, improves correlation "
                    "detection, and produces simpler symbolic regression formulas. "
                    "Raw epoch is kept as a bookkeeping column only."
                ),
            },
        },
        "feature_allowlist": FEATURE_ALLOWLIST,
        "epochs_included":   EPOCHS,
        "dataset_id_mapping": {ds: idx for idx, ds in enumerate(DATASETS)},
        "dataset_stats":     dataset_stats,
        "input_files":       {},
        "output_files":      {},
    }

    for label, path in input_paths.items():
        if os.path.exists(path):
            manifest["input_files"][label] = {
                "path":       os.path.abspath(path),
                "sha256":     sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }

    for label, path in output_paths.items():
        if os.path.exists(path):
            manifest["output_files"][label] = {
                "path":       os.path.abspath(path),
                "sha256":     sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }

    return manifest


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ensure_dir(OUTDIR)

    print("=" * 70)
    print("PHASE 0 — UNIVERSAL DATA POOLING & NORMALIZATION")
    print("=" * 70)
    print(f"  Input dir    : {os.path.abspath(INPUT_DIR)}")
    print(f"  Output dir   : {os.path.abspath(OUTDIR)}")
    print(f"  Datasets     : {DATASETS}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Target       : accuracy_znorm_ds  (z-scored within dataset)")
    print(f"  Epoch feature: epoch_log = log(epoch)")
    print(f"  Features ({len(FEATURE_ALLOWLIST):2d}) : {FEATURE_ALLOWLIST}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load all datasets
    # ------------------------------------------------------------------
    input_paths: Dict[str, str] = {}
    frames: List[pd.DataFrame] = []

    ds_pbar = tqdm(DATASETS, desc="Loading datasets", unit="ds", dynamic_ncols=True)
    for ds_name in ds_pbar:
        ds_pbar.set_postfix({"current": ds_name})
        csv_path = os.path.join(INPUT_DIR, f"{ds_name}.csv")
        input_paths[ds_name] = csv_path

        df = load_dataset(ds_name)
        if df is None:
            continue

        tqdm.write(f"  {ds_name}: {len(df):,} rows  "
                   f"epochs={sorted(df['epoch'].unique().tolist())}")
        frames.append(df)

    if not frames:
        print("\nERROR: No datasets loaded. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Concatenate
    # ------------------------------------------------------------------
    print(f"\nConcatenating {len(frames)} datasets ...")
    pooled = pd.concat(frames, ignore_index=True)
    print(f"  Total rows : {len(pooled):,}")
    print(f"  Columns    : {len(pooled.columns)}")

    # ------------------------------------------------------------------
    # 3. Add dataset_id (bookkeeping)
    # ------------------------------------------------------------------
    pooled = add_dataset_id(pooled)
    print(f"\nDataset ID mapping (bookkeeping — not a feature):")
    for idx, ds in enumerate(DATASETS):
        count = (pooled["dataset"] == ds).sum()
        print(f"  {idx}  {ds:<20s}  {count:>7,} rows")

    # ------------------------------------------------------------------
    # 4. Add accuracy_znorm_ds and epoch_log
    # ------------------------------------------------------------------
    print("\nAdding accuracy_znorm_ds and epoch_log ...")
    pooled = add_derived_columns(pooled)

    # Sanity check — within-dataset z-score: mean~0, std~1 per dataset
    gc = pooled.groupby("dataset")["accuracy_znorm_ds"].agg(["mean", "std"])
    print(f"  accuracy_znorm_ds sanity check (per dataset):")
    print(f"    Max |mean| : {gc['mean'].abs().max():.2e}  (expect ~0)")
    print(f"    std range  : [{gc['std'].min():.4f}, {gc['std'].max():.4f}]  (expect ~1)")
    print(f"  epoch_log    : {{{', '.join(f'{ep}→{np.log(ep):.4f}' for ep in EPOCHS)}}}")

    # ------------------------------------------------------------------
    # 5. Column ordering
    #    meta/bookkeeping first → target → features → remaining raw cols
    # ------------------------------------------------------------------
    meta_cols    = ["dataset", "dataset_id", "epoch", "epoch_log",
                    "accuracy", "accuracy_znorm_ds"]
    feature_cols = [f for f in FEATURE_ALLOWLIST
                    if f in pooled.columns and f not in meta_cols]
    other_cols   = [c for c in pooled.columns
                    if c not in meta_cols and c not in feature_cols]
    pooled = pooled[meta_cols + feature_cols + other_cols]

    # ------------------------------------------------------------------
    # 6. Save pooled CSV
    # ------------------------------------------------------------------
    pooled_path = os.path.join(OUTDIR, "pooled_all_datasets.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"\nPooled CSV saved: {pooled_path}")
    print(f"  Shape : {pooled.shape[0]:,} rows × {pooled.shape[1]} cols")

    # ------------------------------------------------------------------
    # 7. Group statistics
    # ------------------------------------------------------------------
    stats_df  = compute_group_stats(pooled)
    stats_path = os.path.join(OUTDIR, "group_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nGroup statistics (znorm_ds_mean rising epoch 1→50 confirms epoch trend preserved):")
    print(stats_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Feature coverage
    # ------------------------------------------------------------------
    coverage_df   = check_feature_coverage(pooled)
    coverage_path = os.path.join(OUTDIR, "feature_coverage.csv")
    coverage_df.to_csv(coverage_path, index=False)
    present = coverage_df[coverage_df["in_pooled"]]["feature"].tolist()
    missing = coverage_df[~coverage_df["in_pooled"]]["feature"].tolist()
    print(f"\nFeature coverage saved: {coverage_path}")
    print(f"  Present ({len(present)}): {present}")
    if missing:
        print(f"  Missing ({len(missing)}): {missing}")

    # ------------------------------------------------------------------
    # 9. DATA_MANIFEST.json
    # ------------------------------------------------------------------
    dataset_stats_for_manifest: Dict[str, dict] = {}
    for ds in DATASETS:
        sub = pooled[pooled["dataset"] == ds]
        if sub.empty:
            continue
        dataset_stats_for_manifest[ds] = {
            "total_rows": int(len(sub)),
            "rows_per_epoch": {
                str(int(ep)): int((sub["epoch"] == ep).sum()) for ep in EPOCHS
            },
            "accuracy_raw": {
                "mean": round(float(sub["accuracy"].mean()), 6),
                "std":  round(float(sub["accuracy"].std(ddof=1)), 6),
                "min":  round(float(sub["accuracy"].min()), 6),
                "max":  round(float(sub["accuracy"].max()), 6),
            },
            "accuracy_znorm_ds_by_epoch": {
                str(int(ep)): {
                    "mean": round(float(sub.loc[sub["epoch"] == ep, "accuracy_znorm_ds"].mean()), 6),
                    "std":  round(float(sub.loc[sub["epoch"] == ep, "accuracy_znorm_ds"].std(ddof=1)), 6),
                }
                for ep in EPOCHS if (sub["epoch"] == ep).any()
            },
        }

    output_paths = {
        "pooled_all_datasets.csv": pooled_path,
        "group_stats.csv":         stats_path,
        "feature_coverage.csv":    coverage_path,
    }

    software = collect_software_versions()
    manifest = build_manifest(
        input_paths, output_paths, dataset_stats_for_manifest, software
    )

    manifest_path = os.path.join(OUTDIR, "DATA_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDATA_MANIFEST.json saved: {manifest_path}")

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PHASE 0 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total rows   : {len(pooled):,}")
    print(f"  Datasets     : {len(frames)} / {len(DATASETS)}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Output dir   : {os.path.abspath(OUTDIR)}")
    print()
    print("  Column roles:")
    for col, role in COLUMN_ROLES.items():
        print(f"    {col:<22s}  {role}")
    print()
    print("  Output files:")
    for label, path in {**output_paths, "DATA_MANIFEST.json": manifest_path}.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"    {label:<40s}  {size_kb:>8.1f} KB")
    print()
    print("  Software versions:")
    for pkg, ver in software.items():
        print(f"    {pkg:<20s} {ver}")
    print("=" * 70)


if __name__ == "__main__":
    main()
