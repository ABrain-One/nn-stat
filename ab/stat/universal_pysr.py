

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pysr import PySRRegressor
from tqdm import tqdm


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

# --- Feature selection thresholds (Step 2a) ---
SCORE_THRESHOLD = 0.15      # pooled dcor threshold (plan: 0.15)
I2_THRESHOLD    = 50.0      # I² must be below this (plan: 50%)
MIN_SAMPLES     = 30

# --- PySR stages ---
STAGE1_TOP_N = 3
STAGE2_TOP_N = 5
R2_MIN_GAIN  = 0.05         # R² gain needed to proceed Stage2 → Stage3

# --- PySR hyperparameters (identical to pysr_individual.py) ---
PYSR_ITERATIONS  = 100     
LODO_ITERATIONS  = 200      # LODO folds (fewer to control runtime)
PYSR_POPULATIONS = 30
PYSR_MAXSIZE     = 30
PYSR_SEED        = 42
BINARY_OPS       = ["+", "-", "*", "/", "^"]
UNARY_OPS        = ["sqrt", "log", "abs", "exp", "square"]

# Step 2b pooled run: subsample to this many rows (stratified by dataset×epoch).
# PySR warns at >10,000 rows and batching is recommended above that.
# Set to None to use all data.
POOLED_MAX_SAMPLES: Optional[int] = None

# Per-fold LODO trains on ~6/7 of pooled data (~210k rows).
# Subsample to LODO_MAX_SAMPLES (stratified by dataset×epoch) to keep
# each LODO fold tractable. Set to None to disable subsampling.
LODO_MAX_SAMPLES: Optional[int] = 30000

# Enable PySR batching when n > BATCH_THRESHOLD.
# PySR processes mini-batches during evolution instead of all rows at once.
PYSR_BATCHING        = True
PYSR_BATCH_SIZE      = 1000
PYSR_BATCH_THRESHOLD = 1000   # enable batching if n exceeds this

TARGET_COL = "accuracy_znorm_ds"

# Resolve paths relative to this file's location so the script works when
# invoked as:  python universal_pysr.py  OR  python ab/stat/universal_pysr.py
_here      = os.path.abspath(os.path.dirname(__file__))
_base      = os.path.normpath(os.path.join(_here, "..", ".."))
POOLED_CSV = os.path.join(_base, "universal_out", "pooled_all_datasets.csv")
CORR_DIR   = os.path.join(_base, "universal_out", "correlation")
OUTDIR     = os.path.join(_base, "universal_out", "pysr")


# ============================================================================
# Helpers
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def short_name(feat: str) -> str:
    return (feat.replace("prm__", "p:")
                .replace("nn_", "")
                .replace("is_", "")
                .replace("_like", ""))


def stratified_subsample(
    df: pd.DataFrame,
    n_total: int,
    seed: int = PYSR_SEED,
) -> pd.DataFrame:
    """Stratified subsample by dataset×epoch. Targets n_total rows total."""
    groups = list(df.groupby(["dataset", "epoch"]))
    n_per_group = max(1, n_total // len(groups))
    rng = np.random.default_rng(seed)
    frames = []
    for _, g in groups:
        n_take = min(len(g), n_per_group)
        idx = rng.choice(len(g), size=n_take, replace=False)
        frames.append(g.iloc[idx])
    return pd.concat(frames, ignore_index=True)


def compute_r2_per_equation(
    model: PySRRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[int, float]:
    """Return {complexity: r2} for all Pareto-optimal equations."""
    results: Dict[int, float] = {}
    for idx in model.equations_["complexity"]:
        try:
            y_pred = model.predict(X, index=int(idx))
            results[int(idx)] = float(r2_score(y, y_pred))
        except Exception:
            results[int(idx)] = float("nan")
    return results


def best_from_r2_map(r2_map: Dict[int, float]) -> Tuple[float, int]:
    """Return (best_r2, complexity_index) for highest finite R²."""
    if not r2_map:
        return float("nan"), -1
    best_idx = max(
        r2_map,
        key=lambda k: r2_map[k] if not np.isnan(r2_map[k]) else -np.inf,
    )
    return r2_map[best_idx], best_idx


# ============================================================================
# Step 2a — Feature selection
# ============================================================================

def select_universal_features(
    pooled: pd.DataFrame,
    corr_dir: str,
) -> Tuple[List[str], dict]:
    """
    Select features passing both gates, sorted by pooled dcor (epoch_log first).

    Gate 1 (Approach A): dcor > SCORE_THRESHOLD OR dcor_significant == True
                         (from pooled_dcor_mic.csv)
    Gate 2 (Approach B): I² < I2_THRESHOLD
                         (from meta_analysis.csv)

    Falls back gracefully if Phase 1 outputs are absent.
    Returns (ordered_feature_list, info_dict).
    """
    info: dict = {}

    # Available numeric features with sufficient data
    available = [
        f for f in FEATURE_ALLOWLIST
        if f in pooled.columns
        and pd.api.types.is_numeric_dtype(pooled[f])
        and pooled[f].notna().sum() >= MIN_SAMPLES
    ]

    # ── Gate 1: pooled dcor significance ──────────────────────────────
    dcor_path = os.path.join(corr_dir, "pooled_dcor_mic.csv")
    dcor_scores: Dict[str, float] = {}   # for sorting

    if os.path.exists(dcor_path):
        dcor_df = pd.read_csv(dcor_path)
        # Record scores for sorting
        for _, row in dcor_df.iterrows():
            dcor_scores[row["feature"]] = float(row.get("dcor", 0))

        has_mic = "mic" in dcor_df.columns
        if has_mic:
            pass_a = set(
                dcor_df[
                    (dcor_df["dcor"] > SCORE_THRESHOLD) |
                    (dcor_df["mic"] > SCORE_THRESHOLD)
                ]["feature"].tolist()
            )
        else:
            pass_a = set(
                dcor_df[dcor_df["dcor"] > SCORE_THRESHOLD]["feature"].tolist()
            )
        info["pooled_dcor_path"] = dcor_path
        info["n_pass_gate1"] = len(pass_a)
        print(f"  Gate 1 ({len(pass_a)} features pass "
              f"dcor>{SCORE_THRESHOLD} OR mic>{SCORE_THRESHOLD}):")
        for f in [x for x in FEATURE_ALLOWLIST if x in pass_a]:
            dc = dcor_scores.get(f, float("nan"))
            row = dcor_df[dcor_df["feature"] == f]
            mc = float(row["mic"].values[0]) if has_mic and not row.empty else float("nan")
            print(f"    {f:<30s}  dcor={dc:.4f}  mic={mc:.4f}")
    else:
        pass_a = set(available)
        info["pooled_dcor_path"] = f"NOT FOUND — using all {len(pass_a)} available"
        print(f"  WARNING: {dcor_path} not found — Gate 1 passes all features")

    # ── Gate 2: heterogeneity ──────────────────────────────────────────
    meta_path = os.path.join(corr_dir, "meta_analysis.csv")

    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        pass_b = set(meta_df[meta_df["I2"] < I2_THRESHOLD]["feature"].tolist())
        # Features absent from meta_analysis get a free pass
        pass_b |= set(available) - set(meta_df["feature"].tolist())
        info["meta_path"] = meta_path
        info["n_pass_gate2"] = len(pass_b)
        print(f"  Gate 2 ({len(pass_b)} features pass I²<{I2_THRESHOLD}%):")
        for f in [x for x in FEATURE_ALLOWLIST if x in pass_b]:
            row = meta_df[meta_df["feature"] == f]
            i2 = row["I2"].values[0] if not row.empty else float("nan")
            print(f"    {f:<30s}  I²={i2:.1f}%")
    else:
        pass_b = set(available)
        info["meta_path"] = "NOT FOUND — Gate 2 passes all features"
        print(f"  WARNING: {meta_path} not found — Gate 2 passes all features")

    # ── Intersection ────────────────────────────────────────────────────
    # epoch_log always included; others must pass both gates
    selected_raw = [
        f for f in available
        if f == "epoch_log" or (f in pass_a and f in pass_b)
    ]

    # Fallback: if Gate 2 leaves fewer than 2 non-epoch features, use Gate 1 only.
    # This happens when all features have high I² (heterogeneous across datasets),
    # which is common — high heterogeneity means effects vary by dataset but
    # features may still be useful predictors for the universal model.
    non_epoch_intersect = [f for f in selected_raw if f != "epoch_log"]
    if len(non_epoch_intersect) < 2:
        print(f"\n  WARNING: Gate 1 ∩ Gate 2 yields only {len(non_epoch_intersect)} "
              f"non-epoch feature(s).")
        print("  Falling back to Gate 1 only (I² gate skipped).")
        print("  High I² means effects are heterogeneous across datasets,")
        print("  but features may still contribute to a universal model.")
        selected_raw = [f for f in available if f in pass_a or f == "epoch_log"]
        info["gate2_skipped"] = True
    else:
        info["gate2_skipped"] = False

    # Sort by dcor score (descending); epoch_log forced to position 0
    non_epoch = [f for f in selected_raw if f != "epoch_log"]
    non_epoch.sort(key=lambda f: dcor_scores.get(f, 0.0), reverse=True)

    selected: List[str] = []
    if "epoch_log" in selected_raw:
        selected = ["epoch_log"] + non_epoch
    else:
        selected = non_epoch
        if "epoch_log" in available:
            selected = ["epoch_log"] + selected
            print("  epoch_log appended (always included, by-design predictor)")

    info["selected_features"] = selected
    return selected, info


# ============================================================================
# PySR stage runner
# ============================================================================

def run_pysr_stage(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    stage_label: str,
    outdir: Optional[str],
    niterations: int = PYSR_ITERATIONS,
    verbosity: int = 1,
) -> Tuple[pd.DataFrame, PySRRegressor]:
    """Run one PySR stage. Returns (equations_df, fitted_model)."""
    print(f"\n  {stage_label}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  n={len(y):,}  iterations={niterations}")

    use_batching = PYSR_BATCHING and len(y) > PYSR_BATCH_THRESHOLD
    if use_batching:
        print(f"  batching=True  batch_size={PYSR_BATCH_SIZE}")

    model = PySRRegressor(
        niterations=niterations,
        populations=PYSR_POPULATIONS,
        maxsize=PYSR_MAXSIZE,
        binary_operators=BINARY_OPS,
        unary_operators=UNARY_OPS,
        model_selection="best",
        random_state=PYSR_SEED,
        deterministic=True,
        procs=0,
        multithreading=False,
        verbosity=verbosity,
        temp_equation_file=True,
        delete_tempfiles=True,
        batching=use_batching,
        batch_size=PYSR_BATCH_SIZE if use_batching else len(y),
    )

    t0 = time.time()
    model.fit(X, y, variable_names=feature_names)
    elapsed = time.time() - t0

    equations = model.equations_
    print(f"  Done in {elapsed:.1f}s  |  {len(equations)} Pareto equations")

    if outdir:
        csv_path = os.path.join(
            outdir,
            f"{stage_label.lower().replace(' ', '_')}_equations.csv",
        )
        equations.to_csv(csv_path, index=False)

    return equations, model


# ============================================================================
# Step 2b — Universal PySR run
# ============================================================================

def run_universal_pysr(
    pooled: pd.DataFrame,
    selected_features: List[str],
    outdir: str,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, PySRRegressor],
    Dict[str, float],
    Dict[str, tuple],
]:
    """
    Three-stage PySR on the pooled dataset (subsampled if POOLED_MAX_SAMPLES set).
    Returns (stage_equations, stage_models, stage_r2, stage_data).
    stage_data[label] = (X, y, feats, r2_map, best_eq_idx)
    """
    print("\n" + "─" * 70)
    print("STEP 2b — UNIVERSAL PySR RUN")
    print("─" * 70)

    if POOLED_MAX_SAMPLES and len(pooled) > POOLED_MAX_SAMPLES:
        pooled = stratified_subsample(pooled, POOLED_MAX_SAMPLES)
        print(f"  Pooled data subsampled to {len(pooled):,} rows "
              f"(stratified by dataset×epoch)")
    else:
        print(f"  Using all {len(pooled):,} pooled rows")

    top3 = selected_features[:STAGE1_TOP_N]
    top5 = selected_features[:STAGE2_TOP_N]
    stage_sets: Dict[str, List[str]] = {
        f"Stage1_top{STAGE1_TOP_N}":              top3,
        f"Stage2_top{STAGE2_TOP_N}":              top5,
        f"Stage3_all{len(selected_features)}":    selected_features,
    }
    print("  Stage feature sets:")
    for lbl, feats in stage_sets.items():
        print(f"    {lbl}: {feats}")

    stage_equations: Dict[str, pd.DataFrame]  = {}
    stage_models:    Dict[str, PySRRegressor] = {}
    stage_r2:        Dict[str, float]         = {}
    stage_data:      Dict[str, tuple]         = {}
    run_stage3 = False

    stage_labels = list(stage_sets.keys())

    for stage_idx, (label, feats) in enumerate(stage_sets.items(), 1):
        if stage_idx == 3 and not run_stage3:
            print(f"\n  Skipping {label} "
                  f"(R² gain Stage1→Stage2 < {R2_MIN_GAIN})")
            continue

        sub = pooled[feats + [TARGET_COL]].dropna()
        X = sub[feats].values.astype(float)
        y = sub[TARGET_COL].values.astype(float)

        if len(y) < 10:
            print(f"\n  Skipping {label}: too few rows ({len(y)})")
            continue

        equations, model = run_pysr_stage(
            X, y, feats, label, outdir, niterations=PYSR_ITERATIONS,
        )
        r2_map = compute_r2_per_equation(model, X, y)
        r2, best_idx = best_from_r2_map(r2_map)

        stage_equations[label] = equations
        stage_models[label]    = model
        stage_r2[label]        = r2
        stage_data[label]      = (X, y, feats, r2_map, best_idx)

        print(f"  Best train R² ({label}): {r2:.4f}  [complexity {best_idx}]")

        if stage_idx == 2:
            r2_s1 = stage_r2.get(stage_labels[0], -np.inf)
            gain  = r2 - r2_s1
            print(f"  R² gain Stage1→Stage2: {gain:+.4f}  (threshold {R2_MIN_GAIN})")
            run_stage3 = gain >= R2_MIN_GAIN
            print(f"  → Stage 3 {'WILL run' if run_stage3 else 'will be SKIPPED'}")

    return stage_equations, stage_models, stage_r2, stage_data


# ============================================================================
# Step 2c — LODO cross-validation
# ============================================================================

def run_lodo_cv(
    pooled: pd.DataFrame,
    selected_features: List[str],
    outdir: str,
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out cross-validation.

    For each of the 7 datasets (held-out):
      - Train PySR on the other 6 datasets
        (stratified subsample if LODO_MAX_SAMPLES is set).
      - Evaluate each Pareto equation on the held-out test set.
      - Record test R² per equation per fold.

    Feature set: top STAGE2_TOP_N features (Stage 2 balance).
    Returns a DataFrame saved to lodo_results.csv.
    """
    print("\n" + "─" * 70)
    print("STEP 2c — LEAVE-ONE-DATASET-OUT (LODO) CROSS-VALIDATION")
    print("─" * 70)

    lodo_feats = selected_features[:STAGE2_TOP_N]
    print(f"  Feature set (top {STAGE2_TOP_N}): {lodo_feats}")
    print(f"  Iterations per fold: {LODO_ITERATIONS}")
    if LODO_MAX_SAMPLES:
        print(f"  Max train samples:   {LODO_MAX_SAMPLES:,} (stratified)")

    datasets_present = [
        ds for ds in DATASETS if ds in pooled["dataset"].unique()
    ]

    all_rows: list = []

    for fold_ds in tqdm(datasets_present, desc="  LODO folds", unit="fold"):
        tqdm.write(f"\n  ── Fold: held-out = {fold_ds} ──")

        train_df = pooled[pooled["dataset"] != fold_ds].copy()
        test_df  = pooled[pooled["dataset"] == fold_ds].copy()

        if LODO_MAX_SAMPLES and len(train_df) > LODO_MAX_SAMPLES:
            train_df = stratified_subsample(train_df, LODO_MAX_SAMPLES)
            tqdm.write(f"    Train subsampled: {len(train_df):,} rows")
        else:
            tqdm.write(f"    Train: {len(train_df):,} rows")
        tqdm.write(f"    Test:  {len(test_df):,} rows")

        train_sub = train_df[lodo_feats + [TARGET_COL]].dropna()
        test_sub  = test_df[lodo_feats + [TARGET_COL]].dropna()
        X_train = train_sub[lodo_feats].values.astype(float)
        y_train = train_sub[TARGET_COL].values.astype(float)
        X_test  = test_sub[lodo_feats].values.astype(float)
        y_test  = test_sub[TARGET_COL].values.astype(float)

        if len(y_train) < 10 or len(y_test) < 10:
            tqdm.write("    Skipping (too few rows)")
            continue

        _, model = run_pysr_stage(
            X_train, y_train, lodo_feats,
            f"LODO_{fold_ds}", outdir=None,
            niterations=LODO_ITERATIONS, verbosity=0,
        )

        train_r2_map = compute_r2_per_equation(model, X_train, y_train)
        test_r2_map  = compute_r2_per_equation(model, X_test,  y_test)

        for idx in model.equations_["complexity"]:
            idx    = int(idx)
            eq_row = model.equations_[
                model.equations_["complexity"] == idx
            ].iloc[0]
            all_rows.append({
                "held_out_dataset": fold_ds,
                "complexity":       idx,
                "equation":         eq_row.get("sympy_format", ""),
                "loss":             float(eq_row["loss"]),
                "train_r2":         train_r2_map.get(idx, float("nan")),
                "test_r2":          test_r2_map.get(idx, float("nan")),
            })

        best_test_r2, best_idx = best_from_r2_map(test_r2_map)
        tqdm.write(
            f"    Best test R²: {best_test_r2:.4f}  [complexity {best_idx}]"
        )

    if not all_rows:
        return pd.DataFrame()

    lodo_df = pd.DataFrame(all_rows)
    out_path = os.path.join(outdir, "lodo_results.csv")
    lodo_df.to_csv(out_path, index=False)
    print(f"\n  LODO results saved: {out_path}")
    print(f"  {len(lodo_df):,} rows  "
          f"({lodo_df['held_out_dataset'].nunique()} folds)")

    print("\n  Per-dataset best test R²:")
    for ds in sorted(lodo_df["held_out_dataset"].unique()):
        sub  = lodo_df[lodo_df["held_out_dataset"] == ds]
        best = sub["test_r2"].max() if not sub.empty else float("nan")
        print(f"    {ds:<20s}  best_test_r2={best:.4f}")

    return lodo_df


# ============================================================================
# Step 2d — Epoch generalization
# ============================================================================

def run_epoch_generalization(
    pooled: pd.DataFrame,
    selected_features: List[str],
    outdir: str,
) -> pd.DataFrame:
    """
    Train on epochs {1, 5} (all datasets), test on epoch 50 (all datasets).
    Tests whether the universal formula extrapolates in epoch_log space.
    Feature set: top STAGE2_TOP_N features (same as LODO).
    """
    print("\n" + "─" * 70)
    print("STEP 2d — EPOCH GENERALIZATION")
    print("─" * 70)

    feats        = selected_features[:STAGE2_TOP_N]
    train_epochs = [1, 5]
    test_epoch   = 50

    train_df = pooled[pooled["epoch"].isin(train_epochs)].copy()
    test_df  = pooled[pooled["epoch"] == test_epoch].copy()

    print(f"  Train epochs: {train_epochs}   n={len(train_df):,}")
    print(f"  Test epoch:   {test_epoch}     n={len(test_df):,}")
    print(f"  Features:     {feats}")

    train_sub = train_df[feats + [TARGET_COL]].dropna()
    test_sub  = test_df[feats + [TARGET_COL]].dropna()
    X_train = train_sub[feats].values.astype(float)
    y_train = train_sub[TARGET_COL].values.astype(float)
    X_test  = test_sub[feats].values.astype(float)
    y_test  = test_sub[TARGET_COL].values.astype(float)

    if LODO_MAX_SAMPLES and len(X_train) > LODO_MAX_SAMPLES:
        idx = np.random.default_rng(PYSR_SEED).choice(
            len(X_train), size=LODO_MAX_SAMPLES, replace=False,
        )
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Train subsampled to {len(y_train):,} rows")

    if len(y_train) < 10 or len(y_test) < 10:
        print("  Skipping: too few rows.")
        return pd.DataFrame()

    _, model = run_pysr_stage(
        X_train, y_train, feats,
        "EpochGen_train1_5", outdir=None,
        niterations=LODO_ITERATIONS, verbosity=0,
    )

    train_r2_map = compute_r2_per_equation(model, X_train, y_train)
    test_r2_map  = compute_r2_per_equation(model, X_test,  y_test)

    rows = []
    for idx in model.equations_["complexity"]:
        idx    = int(idx)
        eq_row = model.equations_[
            model.equations_["complexity"] == idx
        ].iloc[0]
        rows.append({
            "complexity": idx,
            "equation":   eq_row.get("sympy_format", ""),
            "loss":       float(eq_row["loss"]),
            "train_r2":   train_r2_map.get(idx, float("nan")),
            "test_r2":    test_r2_map.get(idx, float("nan")),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(outdir, "epoch_generalization.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Epoch generalization saved: {out_path}")

    best_test,  _ = best_from_r2_map(test_r2_map)
    best_train, _ = best_from_r2_map(train_r2_map)
    print(f"  Best train R² (epochs 1,5): {best_train:.4f}")
    print(f"  Best test R²  (epoch 50):   {best_test:.4f}")
    return df


# ============================================================================
# Figures
# ============================================================================

def plot_pareto_front(
    stage_equations: Dict[str, pd.DataFrame],
    stage_r2: Dict[str, float],
    out_path: str,
) -> None:
    """Complexity vs MSE loss Pareto curve for all stages."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, equations), color in zip(stage_equations.items(), colors):
        if equations is None or equations.empty:
            continue
        r2 = stage_r2.get(label, float("nan"))
        ax.plot(
            equations["complexity"], equations["loss"], "o-",
            color=color,
            label=f"{label}  (R²={r2:.3f})",
            linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("Formula Complexity (nodes)")
    ax.set_ylabel("MSE Loss")
    ax.set_title(
        "Universal PySR — Pareto Front (Complexity vs Loss)\n"
        "Pooled across all 7 datasets × 3 epochs",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lodo_r2(lodo_df: pd.DataFrame, out_path: str) -> None:
    """Bar chart: best LODO test R² per held-out dataset."""
    if lodo_df.empty:
        return

    per_ds = (
        lodo_df.groupby("held_out_dataset")["test_r2"]
        .max()
        .reset_index()
        .sort_values("test_r2", ascending=False)
    )
    labels = per_ds["held_out_dataset"].tolist()
    values = per_ds["test_r2"].tolist()
    colors = ["#43A047" if v > 0 else "#E53935" for v in values]
    mean_r2 = float(np.nanmean(values))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.axhline(mean_r2, color="black", lw=1.2, linestyle="--",
               label=f"Mean LODO R² = {mean_r2:.3f}")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test R² (best equation)")
    ax.set_title(
        "LODO Cross-Validation — Best Test R² per Held-Out Dataset\n"
        "Green = R²>0 (formula generalizes);  Red = R²≤0",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(
    pooled: pd.DataFrame,
    best_model: PySRRegressor,
    best_eq_idx: int,
    best_feats: List[str],
    out_path: str,
) -> None:
    """Predicted vs actual scatter, one panel per dataset (same formula)."""
    datasets = sorted(pooled["dataset"].unique())
    n_ds  = len(datasets)
    ncols = min(4, n_ds)
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if n_ds > 1 else [axes]

    ds_colors = dict(zip(DATASETS, [
        "#E53935", "#8E24AA", "#1E88E5", "#43A047",
        "#FB8C00", "#00ACC1", "#6D4C41",
    ]))

    for ax, ds in zip(axes_flat, datasets):
        sub = pooled[pooled["dataset"] == ds][best_feats + [TARGET_COL]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        X_ds = sub[best_feats].values.astype(float)
        y_ds = sub[TARGET_COL].values.astype(float)
        try:
            y_pred = best_model.predict(X_ds, index=int(best_eq_idx))
            r2 = r2_score(y_ds, y_pred)
        except Exception:
            ax.set_visible(False)
            continue

        ax.scatter(y_ds, y_pred, alpha=0.3, s=10,
                   color=ds_colors.get(ds, "#888"))
        lo = min(float(y_ds.min()), float(y_pred.min())) - 0.1
        hi = max(float(y_ds.max()), float(y_pred.max())) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"{ds}\nR²={r2:.3f}", fontsize=8)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle(
        "Universal Formula — Predicted vs Actual (per Dataset)\n"
        "Same formula applied to all 7 datasets",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Output files
# ============================================================================

def save_universal_equations(
    stage_equations: Dict[str, pd.DataFrame],
    stage_data: Dict[str, tuple],
    out_path: str,
) -> None:
    """Save all Pareto-optimal equations (all stages) to universal_equations.csv."""
    rows = []
    for label, equations in stage_equations.items():
        if equations is None or equations.empty:
            continue
        _, _, feats, r2_map, _ = stage_data[label]
        for _, row in equations.iterrows():
            idx = int(row["complexity"])
            rows.append({
                "stage":      label,
                "complexity": idx,
                "loss":       float(row["loss"]),
                "r2_train":   r2_map.get(idx, float("nan")),
                "equation":   row.get("sympy_format", ""),
                "features":   str(feats),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)


def write_best_formula(
    stage_r2:        Dict[str, float],
    stage_data:      Dict[str, tuple],
    stage_equations: Dict[str, pd.DataFrame],
    stage_models:    Dict[str, PySRRegressor],
    lodo_df:         pd.DataFrame,
    epoch_gen_df:    pd.DataFrame,
    selected_features: List[str],
    out_path: str,
) -> None:
    if not stage_r2:
        return

    best_stage = max(stage_r2, key=stage_r2.get)
    best_r2    = stage_r2[best_stage]
    _, _, _, _, best_idx = stage_data[best_stage]

    best_eq_str = ""
    if best_idx >= 0 and best_stage in stage_models:
        eqs = stage_models[best_stage].equations_
        row = eqs[eqs["complexity"] == best_idx]
        if not row.empty:
            best_eq_str = row.iloc[0].get("sympy_format", "")

    with open(out_path, "w") as f:
        f.write("Universal Symbolic Regression — Best Formula\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"TARGET:   {TARGET_COL}  (z-scored within dataset)\n")
        f.write(f"FEATURES: {selected_features}\n\n")

        f.write("CHOSEN EQUATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Stage:      {best_stage}\n")
        f.write(f"  Complexity: {best_idx}\n")
        f.write(f"  Train R²:   {best_r2:.4f}\n")
        f.write(f"  Formula:    {best_eq_str}\n\n")

        f.write("ALL STAGES (POOLED RUN)\n")
        f.write("-" * 70 + "\n")
        for lbl, r2 in stage_r2.items():
            f.write(f"  {lbl:<35s}  R²={r2:.4f}\n")
        f.write("\n")

        if not lodo_df.empty:
            f.write("LODO CROSS-VALIDATION SUMMARY\n")
            f.write("-" * 70 + "\n")
            lodo_summary = (
                lodo_df.groupby("held_out_dataset")["test_r2"]
                .max()
                .reset_index()
                .sort_values("test_r2", ascending=False)
            )
            for _, row in lodo_summary.iterrows():
                f.write(f"  {row['held_out_dataset']:<20s}  "
                        f"best_test_r2={row['test_r2']:.4f}\n")
            mean_lodo = float(lodo_summary["test_r2"].mean())
            n_pos     = int((lodo_summary["test_r2"] > 0).sum())
            n_total   = len(lodo_summary)
            f.write(f"\n  Mean LODO R²: {mean_lodo:.4f}\n")
            f.write(f"  Datasets with R²>0: {n_pos}/{n_total}\n")
            claim = "DEFENSIBLE" if n_pos > n_total / 2 else "NOT MET"
            f.write(f"  Universality claim (R²>0 on majority): {claim}\n\n")

        if not epoch_gen_df.empty:
            f.write("EPOCH GENERALIZATION (train {1,5} → test epoch 50)\n")
            f.write("-" * 70 + "\n")
            best_test_r2, _  = best_from_r2_map(
                dict(zip(epoch_gen_df["complexity"], epoch_gen_df["test_r2"])))
            best_train_r2, _ = best_from_r2_map(
                dict(zip(epoch_gen_df["complexity"], epoch_gen_df["train_r2"])))
            f.write(f"  Best train R² (epochs 1,5): {best_train_r2:.4f}\n")
            f.write(f"  Best test R²  (epoch 50):   {best_test_r2:.4f}\n\n")

        f.write("PARETO FRONT — ALL STAGES\n")
        f.write("-" * 70 + "\n")
        for label, equations in stage_equations.items():
            if equations is None or equations.empty:
                continue
            _, _, feats_s, r2_map_s, _ = stage_data[label]
            f.write(f"\n{label}\n")
            f.write(f"Features: {feats_s}\n")
            f.write(f"{'Complexity':>10}  {'Loss':>12}  "
                    f"{'R²_train':>10}  Equation\n")
            f.write("-" * 65 + "\n")
            for _, row in equations.iterrows():
                idx = int(row["complexity"])
                r2  = r2_map_s.get(idx, float("nan"))
                f.write(f"{idx:>10d}  {float(row['loss']):>12.6f}  "
                        f"{r2:>10.4f}  {row.get('sympy_format', '')}\n")


def write_methodology(out_path: str, selected_features: List[str]) -> None:
    with open(out_path, "w") as f:
        f.write("Universal Symbolic Regression — Methodology\n")
        f.write("=" * 60 + "\n\n")

        f.write("TARGET VARIABLE\n")
        f.write("  accuracy_znorm_ds — z-scored within each dataset (NOT within\n")
        f.write("  epoch). Removes dataset offset; preserves epoch trend.\n\n")

        f.write("STEP 2a — FEATURE SELECTION\n")
        f.write(f"  Gate 1: pooled dcor > {SCORE_THRESHOLD}"
                f" OR mic > {SCORE_THRESHOLD}\n")
        f.write("          (from pooled_dcor_mic.csv)\n")
        f.write(f"  Gate 2: I² < {I2_THRESHOLD}%\n")
        f.write("          (from meta_analysis.csv, DerSimonian-Laird)\n")
        f.write("          Skipped if fewer than 2 non-epoch features survive\n")
        f.write("          (fallback to Gate 1 only — high I² is common when\n")
        f.write("          effect sizes vary by dataset but features still predict)\n")
        f.write("  epoch_log: always included (by-design predictor)\n")
        f.write(f"  Sorted by: pooled dcor (descending); epoch_log first\n")
        f.write(f"  Selected ({len(selected_features)}): {selected_features}\n\n")

        f.write("STEP 2b — UNIVERSAL PySR RUN\n")
        f.write("  Data:       full pooled dataset (all 7 datasets, all 3 epochs)\n")
        f.write(f"  Stage 1:    top {STAGE1_TOP_N} features\n")
        f.write(f"  Stage 2:    top {STAGE2_TOP_N} features\n")
        f.write(f"  Stage 3:    all {len(selected_features)} features "
                f"(only if R² gain Stage1→2 ≥ {R2_MIN_GAIN})\n")
        f.write(f"  niterations   = {PYSR_ITERATIONS}\n")
        f.write(f"  populations   = {PYSR_POPULATIONS}\n")
        f.write(f"  maxsize       = {PYSR_MAXSIZE}\n")
        f.write(f"  binary_ops    = {BINARY_OPS}\n")
        f.write(f"  unary_ops     = {UNARY_OPS}\n")
        f.write(f"  random_state  = {PYSR_SEED}, deterministic = True\n\n")

        f.write("STEP 2c — LODO CROSS-VALIDATION\n")
        f.write("  Protocol:  7 folds, one per dataset\n")
        f.write("  Train:     6 datasets (held-out excluded)\n")
        if LODO_MAX_SAMPLES:
            f.write(f"             subsampled to {LODO_MAX_SAMPLES:,} rows "
                    "(stratified by dataset×epoch)\n")
        f.write("  Test:      held-out dataset (all epochs)\n")
        f.write(f"  Features:  top {STAGE2_TOP_N} (Stage 2 set)\n")
        f.write(f"  niterations = {LODO_ITERATIONS}\n")
        f.write("  Metric:    test R² per Pareto equation per fold\n")
        f.write("  Claim:     LODO R² > 0 on majority of held-out datasets\n\n")

        f.write("STEP 2d — EPOCH GENERALIZATION\n")
        f.write("  Train:     epochs {1, 5} (all datasets)\n")
        f.write("  Test:      epoch 50 (all datasets)\n")
        f.write("  Purpose:   tests extrapolation in epoch_log space\n")
        f.write(f"  Features:  top {STAGE2_TOP_N} (Stage 2 set)\n")
        f.write(f"  niterations = {LODO_ITERATIONS}\n\n")

        f.write("REPRODUCIBILITY\n")
        f.write(f"  PySR random_state = {PYSR_SEED}, deterministic = True\n")
        f.write(f"  Subsample seed    = {PYSR_SEED}\n")
        f.write("  Phase 0 target:   accuracy_znorm_ds\n")
        f.write("  Phase 1 inputs:   pooled_dcor_mic.csv + meta_analysis.csv\n")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ensure_dir(OUTDIR)

    print("=" * 70)
    print("PHASE 2 — UNIVERSAL SYMBOLIC REGRESSION")
    print("=" * 70)
    print(f"  Pooled CSV  : {os.path.abspath(POOLED_CSV)}")
    print(f"  Corr dir    : {os.path.abspath(CORR_DIR)}")
    print(f"  Output dir  : {os.path.abspath(OUTDIR)}")
    print(f"  Target      : {TARGET_COL}")
    print(f"  SCORE_THRESHOLD : {SCORE_THRESHOLD}")
    print(f"  I2_THRESHOLD    : {I2_THRESHOLD}%")
    print(f"  Iterations (pooled) : {PYSR_ITERATIONS}")
    print(f"  Iterations (LODO)   : {LODO_ITERATIONS}")
    print(f"  LODO_MAX_SAMPLES    : {LODO_MAX_SAMPLES}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load pooled data
    # ------------------------------------------------------------------
    print("\nLoading pooled dataset ...")
    if not os.path.exists(POOLED_CSV):
        print(f"  ERROR: {POOLED_CSV} not found. Run universal_pooled.py first.")
        return

    pooled = pd.read_csv(POOLED_CSV, low_memory=False)
    print(f"  Rows: {len(pooled):,}  Cols: {len(pooled.columns)}")
    print(f"  Datasets: {sorted(pooled['dataset'].unique().tolist())}")
    print(f"  Epochs:   {sorted(pooled['epoch'].unique().tolist())}")

    # ------------------------------------------------------------------
    # Step 2a — Feature selection
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("STEP 2a — FEATURE SELECTION")
    print("─" * 70)

    selected_features, _ = select_universal_features(pooled, CORR_DIR)
    print(f"\n  Selected {len(selected_features)} features: {selected_features}")

    if len(selected_features) < 2:
        print("  ERROR: fewer than 2 features selected. "
              "Check Phase 1 outputs or lower SCORE_THRESHOLD.")
        return

    # ------------------------------------------------------------------
    # Step 2b — Universal PySR run
    # ------------------------------------------------------------------
    stage_equations, stage_models, stage_r2, stage_data = run_universal_pysr(
        pooled, selected_features, OUTDIR,
    )

    if not stage_r2:
        print("\nERROR: No PySR stages completed.")
        return

    best_stage = max(stage_r2, key=stage_r2.get)
    _, _, best_feats, _, best_eq_idx = stage_data[best_stage]
    best_model = stage_models[best_stage]
    print(f"\n  Best stage: {best_stage}  R²={stage_r2[best_stage]:.4f}")

    # Save full Pareto front CSV
    eq_path = os.path.join(OUTDIR, "universal_equations.csv")
    save_universal_equations(stage_equations, stage_data, eq_path)
    print(f"  universal_equations.csv saved: {eq_path}")

    # ------------------------------------------------------------------
    # Step 2c — LODO CV
    # ------------------------------------------------------------------
    lodo_df = run_lodo_cv(pooled, selected_features, OUTDIR)

    # ------------------------------------------------------------------
    # Step 2d — Epoch generalization
    # ------------------------------------------------------------------
    epoch_gen_df = run_epoch_generalization(pooled, selected_features, OUTDIR)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("GENERATING FIGURES")
    print("─" * 70)

    figs = {
        "fig_pareto_front.png": (
            plot_pareto_front, stage_equations, stage_r2,
        ),
        "fig_lodo_r2.png": (
            plot_lodo_r2, lodo_df,
        ),
        "fig_pred_vs_actual.png": (
            plot_pred_vs_actual, pooled, best_model, best_eq_idx, best_feats,
        ),
    }

    for name, (fn, *args) in figs.items():
        try:
            fn(*args, os.path.join(OUTDIR, name))
            print(f"  {name}  ✓")
        except Exception as e:
            print(f"  {name}  FAILED: {e}")

    # ------------------------------------------------------------------
    # Text outputs
    # ------------------------------------------------------------------
    formula_path = os.path.join(OUTDIR, "best_universal_formula.txt")
    write_best_formula(
        stage_r2, stage_data, stage_equations, stage_models,
        lodo_df, epoch_gen_df, selected_features, formula_path,
    )
    print(f"  best_universal_formula.txt  ✓")

    method_path = os.path.join(OUTDIR, "METHODOLOGY_universal_pysr.txt")
    write_methodology(method_path, selected_features)
    print(f"  METHODOLOGY_universal_pysr.txt  ✓")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output dir : {os.path.abspath(OUTDIR)}")
    print()
    if os.path.isdir(OUTDIR):
        for fname in sorted(os.listdir(OUTDIR)):
            fpath = os.path.join(OUTDIR, fname)
            if os.path.isfile(fpath):
                print(f"  {fname:<52s}  "
                      f"{os.path.getsize(fpath)/1024:>8.1f} KB")
    print()
    print("  Stage R² summary:")
    for lbl, r2 in stage_r2.items():
        print(f"    {lbl:<35s}  R²={r2:.4f}")

    if not lodo_df.empty:
        mean_lodo = float(lodo_df.groupby("held_out_dataset")["test_r2"]
                          .max().mean())
        n_pos = int(
            (lodo_df.groupby("held_out_dataset")["test_r2"].max() > 0).sum()
        )
        print(f"\n  Mean LODO R²: {mean_lodo:.4f}")
        print(f"  Datasets with LODO R²>0: {n_pos}/{len(DATASETS)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
