#!/usr/bin/env python3
"""
Divide all statistics from the database by the 7 image-classification datasets.

Outputs:
  - dataset_splits/          directory with per-dataset CSVs
  - dataset_splits/summary/  summary stats per dataset (row counts, columns, etc.)

Usage:
    python -m ab.stat.util.divide_by_dataset
"""

# =======================
# Standard library imports
# =======================
import os
import json
import inspect

# =======================
# Third-party imports
# =======================
import pandas as pd

# =======================
# Project-specific imports
# =======================
import ab.nn.api as api


# =======================
# Configuration constants
# =======================
DATASETS = [
    "celeba-gender",
    "cifar-10",
    "cifar-100",
    "imagenette",
    "mnist",
    "places365",
    "svhn",
]

# Epoch filter: set to a list like [1, 5, 50] to keep only those epochs.
# Set to None to keep ALL epochs.
EPOCHS = [1, 5, 50]  # or None

_base_outdir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset_splits")
if EPOCHS is not None:
    _epoch_tag = "epoch_" + "_".join(str(e) for e in sorted(EPOCHS))
    OUTDIR = os.path.join(_base_outdir, _epoch_tag)
else:
    OUTDIR = os.path.join(_base_outdir, "all_epochs")
SUMMARY_DIR = os.path.join(OUTDIR, "summary")

# Cache file for raw data (saves hitting the API every time)
RAW_CACHE = os.path.join(OUTDIR, "raw_data_cache.csv")

# Columns to drop – these are the ones you do NOT want in the final dataset.
DROP_COLS = [
    'nn_uses_sequential',
    'nn_uses_modulelist',
    'nn_uses_moduledict',
    'nn_stats_meta',
    'nn_stats_error',
    'transform_code',
    'transform_id',
    'metric',
    'metric_code',
    'metric_id',
    'nn_code',
    'prm_id'
]


# =======================
# Helper functions
# =======================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_dict(x) -> dict:
    """
    Convert a value to a dict.
    - If it's already a dict, return it.
    - If it's a JSON string, parse it and return the dict.
    - Otherwise return an empty dict.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            d = json.loads(x.strip())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def expand_prm(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten df['prm'] JSON/dict into prm__<key> columns."""
    if "prm" not in df.columns:
        return df.copy()
    # Quick heuristic: if less than 20% of rows look like dicts, skip expansion
    sample = df["prm"].dropna().head(500)
    if len(sample) == 0:
        return df.copy()
    ratio = sum(1 for v in sample if isinstance(v, dict) or (isinstance(v, str) and v.strip().startswith("{"))) / len(sample)
    if ratio < 0.2:
        return df.copy()

    parsed = df["prm"].map(parse_dict)
    prm_flat = pd.json_normalize(parsed).add_prefix("prm__")
    return pd.concat([df.drop(columns=["prm"]), prm_flat], axis=1)


def load_raw_with_cache() -> pd.DataFrame:
    """
    Fetch raw data from the database, caching to a CSV file.
    If the cache file exists, load from it; otherwise query the API and save.
    """
    ensure_dir(OUTDIR)  # ensure cache directory exists

    if os.path.exists(RAW_CACHE):
        print(f"  Loading cached data from {RAW_CACHE}")
        return pd.read_csv(RAW_CACHE)

    print("  Fetching fresh data from API...")
    sig = inspect.signature(api.data)
    kwargs = {}
    if "only_best_accuracy" in sig.parameters:
        kwargs["only_best_accuracy"] = False
    if "include_nn_stats" in sig.parameters:
        kwargs["include_nn_stats"] = True
    df = api.data(**kwargs)

    # Save to cache
    df.to_csv(RAW_CACHE, index=False)
    print(f"  Cached to {RAW_CACHE}")
    return df


def prepare_data() -> pd.DataFrame:
    """
    Load data, expand hyperparameters, and drop unwanted columns.
    Returns a cleaned DataFrame ready for splitting.
    """
    # Step 1: load raw data (with caching)
    df = load_raw_with_cache()

    # Step 2: expand prm column
    df = expand_prm(df)

    # Step 3: drop unwanted columns (only those that exist)
    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    if existing_drop_cols:
        print(f"  Dropping columns: {existing_drop_cols}")
        df = df.drop(columns=existing_drop_cols)
    else:
        print("  No columns to drop (none of the DROP_COLS were found).")

    return df


# =======================
# Main pipeline
# =======================
def main():
    ensure_dir(OUTDIR)
    ensure_dir(SUMMARY_DIR)

    # --------------------------------------------------------------------
    # Step 1: Load and prepare data (using prepare_data)
    # --------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading and preparing data ...")
    print("=" * 70)
    df = prepare_data()
    print(f"  Total rows loaded : {len(df):,}")
    print(f"  Total columns     : {len(df.columns)}")
    print(f"  Columns           : {list(df.columns)}")

    # Print all column names in a numbered list for easy reference
    print("\n" + "-" * 50)
    print("ALL COLUMN NAMES AFTER PREPARATION:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:3d}. {col}")
    print("-" * 50 + "\n")

    # --------------------------------------------------------------------
    # Step 2: Filter by epochs (if configured)
    # --------------------------------------------------------------------
    if EPOCHS is not None:
        print(f"STEP 2: Filtering epochs to {EPOCHS} ...")
        if "epoch" not in df.columns:
            print("  ERROR: No 'epoch' column found. Cannot filter epochs.")
            return
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        before = len(df)
        df = df[df["epoch"].isin(EPOCHS)].copy()
        after = len(df)
        print(f"  Rows before filter : {before:,}")
        print(f"  Rows after filter  : {after:,}")
        print(f"  Rows removed       : {before - after:,}")
        available_epochs = sorted(df["epoch"].dropna().unique().tolist())
        requested_missing = [e for e in EPOCHS if e not in available_epochs]
        if requested_missing:
            print(f"  WARNING: These requested epochs have NO data: {requested_missing}")
        print()
    else:
        print("STEP 2: No epoch filter applied (keeping all epochs).")
        print()

    # --------------------------------------------------------------------
    # Step 3: Fill nn_stat columns per nn, then drop only on core columns
    # --------------------------------------------------------------------
    print("STEP 3: Filling nn_stat columns and removing rows with missing core values ...")

    # nn_stat columns are architecture-level (same for all runs of a nn).
    # The API join matches on (nn_name, prm_id), so only one run per nn gets
    # stats filled. Propagate those values to ALL runs of the same nn.
    nn_stat_cols = [
        "nn_total_params", "nn_trainable_params", "nn_frozen_params",
        "nn_total_layers", "nn_leaf_layers", "nn_max_depth",
        "nn_flops", "nn_model_size_mb", "nn_buffer_size_mb", "nn_total_memory_mb",
        "nn_dropout_count", "nn_has_attention", "nn_has_residual",
        "nn_is_resnet_like", "nn_is_vgg_like", "nn_is_inception_like",
        "nn_is_densenet_like", "nn_is_unet_like", "nn_is_transformer_like",
        "nn_is_mobilenet_like", "nn_is_efficientnet_like",
        "nn_code_length", "nn_num_classes", "nn_num_functions",
    ]
    existing_stat_cols = [c for c in nn_stat_cols if c in df.columns]
    if existing_stat_cols and "nn" in df.columns:
        before_fill = df["nn_total_params"].isna().sum() if "nn_total_params" in df.columns else 0
        df[existing_stat_cols] = df.groupby("nn")[existing_stat_cols].transform("first")
        after_fill = df["nn_total_params"].isna().sum() if "nn_total_params" in df.columns else 0
        print(f"  nn_total_params NaN before fill : {before_fill:,}")
        print(f"  nn_total_params NaN after fill  : {after_fill:,}")

    # Drop only rows missing essential columns — NOT optional hyperparameters or nn_stats
    core_cols = ["accuracy", "epoch", "nn", "dataset", "task", "duration",
                 "prm__lr", "prm__batch", "prm__momentum"]
    existing_core = [c for c in core_cols if c in df.columns]
    before_nan = len(df)
    df = df.dropna(subset=existing_core).copy()
    after_nan = len(df)
    print(f"  Rows before NaN removal : {before_nan:,}")
    print(f"  Rows after NaN removal  : {after_nan:,}")
    print(f"  Rows removed            : {before_nan - after_nan:,}")
    if after_nan == 0:
        print("  WARNING: No rows left after dropping NaNs. Exiting.")
        return
    print()

    # --------------------------------------------------------------------
    # Step 4: Check available datasets
    # --------------------------------------------------------------------
    print("STEP 4: Checking available datasets in the data ...")
    if "dataset" not in df.columns:
        print("  ERROR: No 'dataset' column found. Cannot split. Exiting.")
        return
    all_datasets = sorted(df["dataset"].dropna().unique())
    print(f"  All datasets in DB ({len(all_datasets)}): {all_datasets}")

    missing = [d for d in DATASETS if d not in all_datasets]
    if missing:
        print(f"  WARNING: These requested datasets are NOT in the data: {missing}")
    print()

    # --------------------------------------------------------------------
    # Step 5: Identify feature columns for correlation analysis
    # --------------------------------------------------------------------
    print("STEP 5: Identifying feature columns for correlation analysis ...")
    # We are not excluding any columns – all columns are potential features.
    feature_cols = [c for c in df.columns]   # all columns
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"  Numeric feature columns  ({len(numeric_features)}): {numeric_features[:30]}{'...' if len(numeric_features) > 30 else ''}")
    print(f"  Non-numeric columns      ({len(non_numeric_features)}): {non_numeric_features[:20]}{'...' if len(non_numeric_features) > 20 else ''}")
    print()

    # --------------------------------------------------------------------
    # Step 6: Split and save per dataset
    # --------------------------------------------------------------------
    print("STEP 6: Splitting data by dataset and saving CSVs ...")
    print("-" * 70)

    dataset_frames = {}
    for ds_name in DATASETS:
        df_ds = df[df["dataset"] == ds_name].copy()
        dataset_frames[ds_name] = df_ds

        csv_path = os.path.join(OUTDIR, f"{ds_name}.csv")
        df_ds.to_csv(csv_path, index=False)

        n_rows = len(df_ds)
        n_models = df_ds["nn"].nunique() if "nn" in df_ds.columns else "N/A"
        n_epochs = sorted(df_ds["epoch"].dropna().unique().tolist()) if "epoch" in df_ds.columns else []
        metrics = sorted(df_ds["metric"].dropna().unique().tolist()) if "metric" in df_ds.columns else []
        tasks = sorted(df_ds["task"].dropna().unique().tolist()) if "task" in df_ds.columns else []

        num_feat_nonnull = {c: int(df_ds[c].notna().sum()) for c in numeric_features}
        usable_features = {k: v for k, v in num_feat_nonnull.items() if v > 0}

        print(f"\n  Dataset: {ds_name}")
        print(f"    Rows           : {n_rows:,}")
        print(f"    Unique models  : {n_models}")
        print(f"    Epochs         : {n_epochs[:10]}{'...' if len(n_epochs) > 10 else ''}")
        print(f"    Metrics        : {metrics}")
        print(f"    Tasks          : {tasks}")
        print(f"    Usable numeric features (non-null > 0): {len(usable_features)}")
        print(f"    Saved to       : {csv_path}")

        if "accuracy" in df_ds.columns and n_rows > 0:
            acc = df_ds["accuracy"].dropna()
            print(f"    Accuracy stats : count={len(acc)}, mean={acc.mean():.4f}, std={acc.std():.4f}, min={acc.min():.4f}, max={acc.max():.4f}")

        if "accuracy" in df_ds.columns and "nn" in df_ds.columns and n_rows > 0:
            best_per_model = df_ds.groupby("nn")["accuracy"].max().sort_values(ascending=False)
            print(f"    Top 5 models by best accuracy:")
            for model_name, best_acc in best_per_model.head(5).items():
                print(f"      {model_name:30s} -> {best_acc:.4f}")

    print("\n" + "-" * 70)

    # --------------------------------------------------------------------
    # Step 7: Save combined summary CSV
    # --------------------------------------------------------------------
    print("\nSTEP 7: Saving summary ...")
    summary_rows = []
    for ds_name in DATASETS:
        df_ds = dataset_frames[ds_name]
        row = {
            "dataset": ds_name,
            "total_rows": len(df_ds),
            "unique_models": df_ds["nn"].nunique() if "nn" in df_ds.columns else 0,
            "unique_epochs": df_ds["epoch"].nunique() if "epoch" in df_ds.columns else 0,
            "metrics": ", ".join(sorted(df_ds["metric"].dropna().unique())) if "metric" in df_ds.columns else "",
        }
        if "accuracy" in df_ds.columns and len(df_ds) > 0:
            acc = df_ds["accuracy"].dropna()
            row["acc_mean"] = round(acc.mean(), 4)
            row["acc_std"] = round(acc.std(), 4)
            row["acc_min"] = round(acc.min(), 4)
            row["acc_max"] = round(acc.max(), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SUMMARY_DIR, "dataset_overview.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved to: {summary_path}")
    print()
    print("  Dataset Overview:")
    print(summary_df.to_string(index=False))

    # --------------------------------------------------------------------
    # Step 8: Save per-dataset feature availability matrix
    # --------------------------------------------------------------------
    print("\n\nSTEP 8: Feature availability matrix (non-null counts per dataset) ...")
    avail_rows = []
    for ds_name in DATASETS:
        df_ds = dataset_frames[ds_name]
        row = {"dataset": ds_name}
        for c in numeric_features:
            row[c] = int(df_ds[c].notna().sum())
        avail_rows.append(row)

    avail_df = pd.DataFrame(avail_rows).set_index("dataset")
    avail_path = os.path.join(SUMMARY_DIR, "feature_availability.csv")
    avail_df.to_csv(avail_path)
    print(f"  Saved to: {avail_path}")

    has_data = avail_df.columns[avail_df.sum() > 0].tolist()
    print(f"  Features with data in at least one dataset ({len(has_data)}):")
    if has_data:
        print(avail_df[has_data].to_string())
    else:
        print("  (none)")

    # --------------------------------------------------------------------
    # Done
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE. Output directory:", os.path.abspath(OUTDIR))
    print("=" * 70)
    print("\nNext steps for your correlation analysis:")
    print("  1. Load any dataset CSV:  pd.read_csv('dataset_splits/cifar-10.csv')")
    print("  2. Check 'feature_availability.csv' to see which features are usable per dataset")
    print("  3. Use dcor / minepy / PySR on the numeric feature columns vs 'accuracy'")

    return dataset_frames


# =======================
# Entry point
# =======================
if __name__ == "__main__":
    dataset_frames = main()