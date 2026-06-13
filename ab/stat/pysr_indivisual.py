"""
PySR symbolic regression with optional convergence analysis and 5‑fold CV.

Usage examples:
    # Normal run (three-stage regression, no CV)
    python pysr_individual.py --dataset mnist --epoch 1

    # Normal run with 5‑fold CV
    python pysr_individual.py --dataset mnist --epoch 1 --cv

    # Convergence analysis (runs full three-stage pipeline for each iteration count)
    python pysr_individual.py --dataset mnist --epoch 1 --convergence

    # Note: --convergence and --cv cannot be used together. Run them separately.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from pysr import PySRRegressor

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STAGE1_TOP_N = 3          # top 3 features
STAGE2_TOP_N = 5          # top 5 features
R2_MIN_GAIN = 0.05        # gain needed to proceed to stage 3
SCORE_THRESHOLD = 0.2     # dcor or mic must exceed this to be "significant"

PYSR_ITERATIONS = 300        # default for normal runs

DATASET_BEST_ITERATIONS = {
    "celeba-gender": 500,
    "cifar-10":      500,
    "cifar-100":     500,
    "imagenette":    200,
    "mnist":         200,
    "places365":      50,
    "svhn":          200,
}
PYSR_POPULATIONS = 30
PYSR_MAXSIZE = 30
PYSR_SEED = 42

BINARY_OPS = ["+", "-", "*", "/", "^"]
UNARY_OPS = ["sqrt", "log", "abs", "exp", "square"]

TARGET_COL = "accuracy"

# Paths (relative to script location)
_base = os.path.join(os.path.dirname(__file__), "..", "..")
DCOR_OUT_DIR = os.path.join(_base, "dcor_out")
MIC_OUT_DIR = os.path.join(_base, "mic_out")
DATASET_DIR = os.path.join(_base, "dataset_splits", "epoch_1_5_50")
PYSR_OUT_DIR = os.path.join(_base, "pysr_out")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_best_r2_and_index(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Return (best_r2, best_index) of equation with max R²."""
    best_r2 = -np.inf
    best_idx = None
    for idx in model.equations_["complexity"]:
        try:
            y_pred = model.predict(X, index=int(idx))
            r2 = r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_idx = idx
        except Exception:
            continue
    return best_r2, best_idx


def run_pysr_stage(X: np.ndarray, y: np.ndarray,
                   feature_names: List[str], stage_label: str,
                   outdir: str, niterations: int = PYSR_ITERATIONS) -> Tuple[pd.DataFrame, PySRRegressor]:
    """Run PySR and return equations DataFrame and fitted model."""
    print(f"\n  {stage_label}")
    print(f"  Features: {feature_names}")
    print(f"  n = {len(y):,}  |  iterations = {niterations}")

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
        verbosity=1,
        temp_equation_file=True,
        delete_tempfiles=True,
    )

    t0 = time.time()
    model.fit(X, y, variable_names=feature_names)
    elapsed = time.time() - t0

    equations = model.equations_
    print(f"  Done in {elapsed:.1f}s | {len(equations)} Pareto‑optimal equations")

    # Save equations CSV (if outdir is given)
    if outdir:
        csv_path = os.path.join(outdir, f"{stage_label.replace(' ', '_').lower()}_equations.csv")
        equations.to_csv(csv_path, index=False)

    return equations, model


def plot_pareto(stage_results: List[Tuple[str, pd.DataFrame]], outpath: str) -> None:
    """Plot Pareto front (complexity vs loss) for all stages."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (label, equations), color in zip(stage_results, colors):
        if equations is None or equations.empty:
            continue
        ax.plot(equations["complexity"], equations["loss"],
                "o-", color=color, label=label, linewidth=1.5, markersize=5)
    ax.set_xlabel("Formula Complexity (nodes)")
    ax.set_ylabel("MSE Loss")
    ax.set_title("PySR Pareto Front — Complexity vs Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_r2_bars(stage_r2: dict, outpath: str) -> None:
    """Bar chart of best R² per stage."""
    labels = list(stage_r2.keys())
    values = [stage_r2[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values,
                  color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(labels)],
                  edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best R²")
    ax.set_title("PySR — Best R² per Stage")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def run_convergence_analysis(dataset: str, epoch: int, outdir: str) -> None:
    """Run convergence analysis: 5-fold CV across iteration counts to find best without overfitting."""
    print("\n" + "=" * 70)
    print("Convergence Analysis — 5-Fold CV per Iteration Count")
    print("=" * 70)

    # Paths
    dcor_csv = os.path.join(DCOR_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    mic_csv = os.path.join(MIC_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    data_csv = os.path.join(DATASET_DIR, f"{dataset}.csv")

    for fpath in [dcor_csv, mic_csv, data_csv]:
        if not os.path.exists(fpath):
            print(f"ERROR: missing {fpath}")
            return

    # Load dcor/mic and select significant features
    dcor_df = pd.read_csv(dcor_csv)[["feature", "dcor", "spearman"]]
    mic_df = pd.read_csv(mic_csv)[["feature", "mic"]]
    scores = dcor_df.merge(mic_df, on="feature", how="outer").fillna(0)
    significant = scores[(scores["dcor"] > SCORE_THRESHOLD) |
                         (scores["mic"] > SCORE_THRESHOLD)].copy()
    significant.sort_values("dcor", ascending=False, inplace=True)
    sig_features = significant["feature"].tolist()

    if not sig_features:
        print("No significant features. Exiting.")
        return

    print(f"\nSignificant features ({len(sig_features)}):")
    for i, row in significant.iterrows():
        print(f"  {i+1:2d}. {row['feature']:32s}  dcor={row['dcor']:.4f}  "
              f"mic={row['mic']:.4f}  spearman={row['spearman']:+.4f}")

    # Load data, filter epoch, keep target and significant features
    df = pd.read_csv(data_csv, low_memory=False)
    df = df[df["epoch"] == epoch].copy()
    if df.empty:
        print(f"No data for epoch {epoch}")
        return

    # Remove constant 'epoch' column and remove from sig_features
    if "epoch" in df.columns:
        df = df.drop(columns=["epoch"])
    if "epoch" in sig_features:
        sig_features.remove("epoch")
        print("Removed 'epoch' from features (data filtered to a single epoch).")

    df[TARGET_COL] = df["accuracy"]
    keep_cols = [TARGET_COL] + [f for f in sig_features if f in df.columns]
    df = df[keep_cols].dropna(subset=[TARGET_COL])

    # Define stage sets (same as main)
    top3 = sig_features[:STAGE1_TOP_N]
    top5 = sig_features[:STAGE2_TOP_N]
    stage_sets = {
        f"Stage1_top{STAGE1_TOP_N}": top3,
        f"Stage2_top{STAGE2_TOP_N}": top5,
        f"Stage3_all{len(sig_features)}": sig_features,
    }

    # Build full X, y from all significant features
    X_all = df[sig_features].values.astype(float)
    y_all = df[TARGET_COL].values.astype(float)
    nan_mask = ~np.isnan(X_all).any(axis=1)
    X_all, y_all = X_all[nan_mask], y_all[nan_mask]

    print(f"\nTotal samples after NaN removal: {len(y_all)}")

    # Iteration values to test
    iter_values = [50, 100, 150, 200, 250, 300, 400, 500]
    kf = KFold(n_splits=5, shuffle=True, random_state=PYSR_SEED)
    results = []

    for iters in iter_values:
        print(f"\n{'='*50}")
        print(f"Iterations: {iters}  (5-fold CV)")
        print('='*50)

        fold_train_r2 = []
        fold_val_r2 = []
        fold_best_stages = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
            print(f"\n  --- Fold {fold}/5 ---")
            X_train_full, X_val_full = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            stage_r2_train = {}
            stage_r2_val = {}
            run_stage3 = False

            for stage_idx, (label, feats) in enumerate(stage_sets.items(), 1):
                if stage_idx == 3 and not run_stage3:
                    print(f"    Skipping {label} (val R² gain stage1→2 < {R2_MIN_GAIN})")
                    continue

                # Slice columns for this stage
                feat_indices = [sig_features.index(f) for f in feats]
                X_tr = X_train_full[:, feat_indices]
                X_vl = X_val_full[:, feat_indices]

                if len(y_train) < 10:
                    print(f"    Not enough train rows ({len(y_train)}), skipping {label}")
                    continue

                _, model = run_pysr_stage(
                    X_tr, y_train, feats,
                    f"{label}_fold{fold}", outdir=None, niterations=iters
                )

                # Train R²
                r2_train, best_idx = compute_best_r2_and_index(model, X_tr, y_train)

                # Val R²
                try:
                    y_pred_val = model.predict(X_vl, index=int(best_idx))
                    r2_val = r2_score(y_val, y_pred_val)
                except Exception:
                    r2_val = np.nan

                stage_r2_train[label] = r2_train
                stage_r2_val[label] = r2_val

                print(f"    {label}: train_r²={r2_train:.4f}  val_r²={r2_val:.4f}")

                if stage_idx == 2:
                    s1_label = list(stage_sets.keys())[0]
                    gain = r2_val - stage_r2_val.get(s1_label, -np.inf)
                    print(f"    val R² gain stage1→stage2: {gain:+.4f} (threshold {R2_MIN_GAIN})")
                    run_stage3 = gain >= R2_MIN_GAIN
                    print(f"    → Stage 3 {'will run' if run_stage3 else 'will be skipped'}")

            if not stage_r2_val:
                continue

            # Pick best stage by val R² for this fold
            best_stage = max(
                stage_r2_val,
                key=lambda k: stage_r2_val[k] if not np.isnan(stage_r2_val[k]) else -np.inf
            )
            fold_train_r2.append(stage_r2_train[best_stage])
            fold_val_r2.append(stage_r2_val[best_stage])
            fold_best_stages.append(best_stage)

            print(f"  Fold {fold} best: {best_stage}  "
                  f"train_r²={stage_r2_train[best_stage]:.4f}  "
                  f"val_r²={stage_r2_val[best_stage]:.4f}")

        if not fold_val_r2:
            print("No folds completed. Skipping.")
            continue

        mean_train = float(np.mean(fold_train_r2))
        std_train  = float(np.std(fold_train_r2))
        mean_val   = float(np.mean(fold_val_r2))
        std_val    = float(np.std(fold_val_r2))
        gap        = mean_train - mean_val
        most_common_stage = max(set(fold_best_stages), key=fold_best_stages.count)

        print(f"\n  Summary — {iters} iterations:")
        print(f"  train_r²: {mean_train:.4f} ± {std_train:.4f}")
        print(f"  val_r²:   {mean_val:.4f} ± {std_val:.4f}")
        print(f"  gap:      {gap:.4f}")

        results.append({
            "iterations":     iters,
            "train_r2_mean":  mean_train,
            "train_r2_std":   std_train,
            "val_r2_mean":    mean_val,
            "val_r2_std":     std_val,
            "gap":            gap,
            "best_stage":     most_common_stage,
        })

    # Save results
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results collected.")
        return

    csv_path = os.path.join(outdir, "convergence_analysis.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 78)
    print("CONVERGENCE SUMMARY")
    print("=" * 78)
    print(f"  {'Iterations':>10}  {'train_r2':>10}  {'val_r2':>10}  {'gap':>8}  best_stage")
    print("  " + "-" * 74)
    for _, row in df_res.iterrows():
        print(f"  {int(row['iterations']):>10}  "
              f"{row['train_r2_mean']:>10.4f}  "
              f"{row['val_r2_mean']:>10.4f}  "
              f"{row['gap']:>8.4f}  "
              f"{row['best_stage']}")

    best_row = df_res.loc[df_res["val_r2_mean"].idxmax()]
    print(f"\n>>> Best iteration count by val_r²: {int(best_row['iterations'])}")
    print(f"    train_r² = {best_row['train_r2_mean']:.4f} ± {best_row['train_r2_std']:.4f}")
    print(f"    val_r²   = {best_row['val_r2_mean']:.4f} ± {best_row['val_r2_std']:.4f}")
    print(f"    gap      = {best_row['gap']:.4f}")

    # Plot
    iters_arr  = df_res["iterations"].values
    train_mean = df_res["train_r2_mean"].values
    train_std  = df_res["train_r2_std"].values
    val_mean   = df_res["val_r2_mean"].values
    val_std    = df_res["val_r2_std"].values
    gap_arr    = df_res["gap"].values
    best_iters = int(best_row["iterations"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: train vs val R²
    ax1.plot(iters_arr, train_mean, "o-", color="green", label="train_r²", linewidth=1.8, markersize=6)
    ax1.fill_between(iters_arr, train_mean - train_std, train_mean + train_std, alpha=0.15, color="green")
    ax1.plot(iters_arr, val_mean, "o-", color="blue", label="val_r²", linewidth=1.8, markersize=6)
    ax1.fill_between(iters_arr, val_mean - val_std, val_mean + val_std, alpha=0.15, color="blue")
    ax1.axvline(best_iters, color="red", linestyle="--", linewidth=1.2, label=f"best={best_iters}")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("R²")
    ax1.set_title("Train vs Val R² — 5-Fold CV")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: overfitting gap
    ax2.plot(iters_arr, gap_arr, "o-", color="red", linewidth=1.8, markersize=6, label="gap")
    ax2.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="overfit threshold (0.05)")
    ax2.axvline(best_iters, color="red", linestyle="--", linewidth=1.2, label=f"best={best_iters}")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Gap (train_r² − val_r²)")
    ax2.set_title("Overfitting Gap — 5-Fold CV")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(outdir, "convergence_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved to {plot_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--epoch", required=True, type=int, help="Epoch number")
    parser.add_argument("--cv", action="store_true", help="Perform 5‑fold CV on the best equation")
    parser.add_argument("--convergence", action="store_true", help="Run convergence analysis (full pipeline for each iteration count) and exit")
    args = parser.parse_args()

    dataset = args.dataset
    epoch = args.epoch
    niterations = DATASET_BEST_ITERATIONS.get(dataset, PYSR_ITERATIONS) if args.cv else PYSR_ITERATIONS

    # Define paths
    dcor_csv = os.path.join(DCOR_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    mic_csv = os.path.join(MIC_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    data_csv = os.path.join(DATASET_DIR, f"{dataset}.csv")
    outdir = os.path.join(PYSR_OUT_DIR, dataset, f"epoch_{epoch}")
    ensure_dir(outdir)

    # If convergence analysis requested, run it and exit
    if args.convergence:
        run_convergence_analysis(dataset, epoch, outdir)
        return

    # ------------------------------------------------------------------
    # 1. Load dcor and mic results, combine, select significant features
    # ------------------------------------------------------------------
    for fpath in [dcor_csv, mic_csv, data_csv]:
        if not os.path.exists(fpath):
            print(f"ERROR: missing {fpath}")
            return

    dcor_df = pd.read_csv(dcor_csv)[["feature", "dcor", "spearman"]]
    mic_df = pd.read_csv(mic_csv)[["feature", "mic"]]
    scores = dcor_df.merge(mic_df, on="feature", how="outer").fillna(0)

    significant = scores[(scores["dcor"] > SCORE_THRESHOLD) |
                         (scores["mic"] > SCORE_THRESHOLD)].copy()
    significant.sort_values("dcor", ascending=False, inplace=True)
    sig_features = significant["feature"].tolist()

    print(f"\nSignificant features ({len(sig_features)}):")
    for i, row in significant.iterrows():
        print(f"  {i+1:2d}. {row['feature']:32s}  dcor={row['dcor']:.4f}  "
              f"mic={row['mic']:.4f}  spearman={row['spearman']:+.4f}")

    if not sig_features:
        print("No significant features. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Load data, filter epoch, and remove constant 'epoch' column
    # ------------------------------------------------------------------
    df = pd.read_csv(data_csv, low_memory=False)
    df = df[df["epoch"] == epoch].copy()
    if df.empty:
        print(f"No data for epoch {epoch}")
        return

    # Remove the constant 'epoch' column and remove it from sig_features
    if "epoch" in df.columns:
        df = df.drop(columns=["epoch"])
    if "epoch" in sig_features:
        sig_features.remove("epoch")
        print("Removed 'epoch' from features (data filtered to a single epoch).")

    df[TARGET_COL] = df["accuracy"]
    keep_cols = [TARGET_COL] + [f for f in sig_features if f in df.columns]
    df = df[keep_cols].dropna(subset=[TARGET_COL])

    # ------------------------------------------------------------------
    # 3. Define feature sets for stages
    # ------------------------------------------------------------------
    top3 = sig_features[:STAGE1_TOP_N]
    top5 = sig_features[:STAGE2_TOP_N]
    stage_sets = {
        f"Stage1_top{STAGE1_TOP_N}": top3,
        f"Stage2_top{STAGE2_TOP_N}": top5,
        f"Stage3_all{len(sig_features)}": sig_features,
    }

    # ------------------------------------------------------------------
    # 4. Run stages
    # ------------------------------------------------------------------
    stage_results = []          # (label, equations_df)
    stage_r2 = {}
    stage_models = {}
    stage_data = {}             # label -> (X, y, features)
    stage_best_idx = {}         # label -> best equation index
    run_stage3 = False

    for stage_idx, (label, feats) in enumerate(stage_sets.items(), 1):
        if stage_idx == 3 and not run_stage3:
            print(f"\nSkipping {label} (R² gain from stage 1→2 < {R2_MIN_GAIN})")
            stage_results.append((label, None))
            continue

        X = df[feats].values.astype(float)
        y = df[TARGET_COL].values.astype(float)
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        if len(y) < 10:
            print(f"Not enough rows after dropping NaNs ({len(y)}), skipping {label}")
            stage_results.append((label, None))
            continue

        equations, model = run_pysr_stage(X, y, feats, label, outdir, niterations=niterations)
        r2, best_idx = compute_best_r2_and_index(model, X, y)

        stage_results.append((label, equations))
        stage_r2[label] = r2
        stage_models[label] = model
        stage_data[label] = (X, y, feats)
        stage_best_idx[label] = best_idx

        if stage_idx == 2:
            r2_s1 = stage_r2.get(list(stage_sets.keys())[0], -np.inf)
            gain = r2 - r2_s1
            print(f"  R² gain Stage1→Stage2: {gain:+.4f} (threshold {R2_MIN_GAIN})")
            run_stage3 = gain >= R2_MIN_GAIN
            if not run_stage3:
                print("  → Stage 3 will be skipped")
            else:
                print("  → Stage 3 will run")

    # ------------------------------------------------------------------
    # 5. Determine best stage
    # ------------------------------------------------------------------
    if not stage_r2:
        print("No stages completed. Exiting.")
        return

    best_stage_label = max(stage_r2, key=stage_r2.get)
    best_stage_r2 = stage_r2[best_stage_label]
    best_X, best_y, best_feats = stage_data[best_stage_label]
    best_model = stage_models[best_stage_label]
    best_eq_idx = stage_best_idx[best_stage_label]
    best_eq_str = best_model.equations_.loc[best_model.equations_["complexity"] == best_eq_idx, "sympy_format"].values[0]

    print(f"\nBest stage: {best_stage_label} (R² = {best_stage_r2:.4f})")
    print(f"Best equation: {best_eq_str}")

    # ------------------------------------------------------------------
    # 6. Cross‑validation (simple 5‑fold)
    # ------------------------------------------------------------------
    if args.cv:
        print(f"\nPerforming 5‑fold cross‑validation on best stage: {best_stage_label}")
        print(f"Features: {best_feats}")
        print(f"n = {len(best_y)}")

        kf = KFold(n_splits=5, shuffle=True, random_state=PYSR_SEED)
        fold_algo = []   # R² of algorithm's best equation per fold on validation
        fold_ref = []    # R² of reference equation per fold on validation

        cv_dir = os.path.join(outdir, "cv_5fold")
        ensure_dir(cv_dir)

        for fold, (train_idx, val_idx) in enumerate(kf.split(best_X), 1):
            X_train, X_val = best_X[train_idx], best_X[val_idx]
            y_train, y_val = best_y[train_idx], best_y[val_idx]

            # Train model on fold
            model_fold = PySRRegressor(
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
                verbosity=0,
                temp_equation_file=True,
                delete_tempfiles=True,
            )
            model_fold.fit(X_train, y_train, variable_names=best_feats)

            # Best equation on training set
            _, best_idx_fold = compute_best_r2_and_index(model_fold, X_train, y_train)
            y_pred_algo = model_fold.predict(X_val, index=int(best_idx_fold))
            r2_val_algo = r2_score(y_val, y_pred_algo)

            # Reference equation on validation set
            y_pred_ref = best_model.predict(X_val, index=int(best_eq_idx))
            r2_val_ref = r2_score(y_val, y_pred_ref)

            fold_algo.append(r2_val_algo)
            fold_ref.append(r2_val_ref)

            print(f"  Fold {fold}: algo R²_val = {r2_val_algo:.4f}, ref R²_val = {r2_val_ref:.4f}")

        # Summary statistics
        mean_algo = np.mean(fold_algo)
        std_algo = np.std(fold_algo)
        mean_ref = np.mean(fold_ref)
        std_ref = np.std(fold_ref)

        # Save results
        with open(os.path.join(cv_dir, "cv_results.txt"), "w") as f:
            f.write(f"Cross‑validation results for best stage \"{best_stage_label}\":\n")
            f.write(f"  5‑fold CV\n")
            f.write(f"  Algorithm (best per fold) : R² = {mean_algo:.4f} ± {std_algo:.4f}\n")
            f.write(f"  Reference equation        : R² = {mean_ref:.4f} ± {std_ref:.4f}\n\n")
            f.write("Per‑fold values (algo):\n")
            f.write(", ".join(f"{x:.4f}" for x in fold_algo) + "\n\n")
            f.write("Per‑fold values (ref):\n")
            f.write(", ".join(f"{x:.4f}" for x in fold_ref) + "\n")

        print(f"\nCV results saved to {os.path.join(cv_dir, 'cv_results.txt')}")

        # Boxplot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot([fold_algo, fold_ref], labels=["Algorithm (best per fold)", "Reference equation"])
            ax.set_ylabel("R² on validation set")
            ax.set_title(f"5‑fold CV – {dataset} epoch {epoch}\nBest stage: {best_stage_label}")
            plt.tight_layout()
            fig.savefig(os.path.join(cv_dir, "cv_boxplot.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot boxplot: {e}")

    # ------------------------------------------------------------------
    # 7. Generate plots and summary
    # ------------------------------------------------------------------
    valid = [(lbl, eq) for lbl, eq in stage_results if eq is not None]
    if valid:
        plot_pareto(valid, os.path.join(outdir, "pareto_front.png"))
    if stage_r2:
        plot_r2_bars(stage_r2, os.path.join(outdir, "r2_comparison.png"))

    # Save text summary
    with open(os.path.join(outdir, "best_equations.txt"), "w") as f:
        f.write(f"PySR Symbolic Regression — {dataset} epoch {epoch}\n")
        f.write("=" * 70 + "\n\n")
        for stage_idx, (label, equations) in enumerate(stage_results, 1):
            if equations is None:
                continue
            f.write(f"{label}\n")
            f.write(f"Features: {list(stage_sets.values())[stage_idx-1]}\n")
            f.write(f"Best R²: {stage_r2.get(label, float('nan')):.4f}\n\n")
            f.write("Complexity   Loss        R²      Equation\n")
            f.write("-" * 60 + "\n")
            for _, row in equations.iterrows():
                try:
                    X_cur, y_cur, _ = stage_data[label]
                    y_pred = stage_models[label].predict(X_cur, index=int(row["complexity"]))
                    r2 = r2_score(y_cur, y_pred)
                except:
                    r2 = np.nan
                f.write(f"{int(row['complexity']):>10d}  {row['loss']:>12.6f}  {r2:>8.4f}  {row['sympy_format']}\n")
            f.write("\n")

        if stage_r2:
            best_stage = max(stage_r2, key=stage_r2.get)
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"RECOMMENDED: {best_stage}  (R²={stage_r2[best_stage]:.4f})\n")

    print(f"\nOutput saved to {outdir}")


if __name__ == "__main__":
    main()