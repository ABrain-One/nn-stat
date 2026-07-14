"""
EXTERNAL replication — nn-Meter HELD-OUT-SPLIT RE-LOCK (airtight, blind).

Removes the last non-blindness: communality is fit on a FIT half and predictions are later
revealed on a never-seen TEST half.

Steps 1-3 (this script):
  1. per space (11; nasbench201s + shufflenetv2s excluded) 50/50 FIT/TEST split, seed 42;
     report per-half n (>=100 required). Saves the split assignment.
  2. fit one-factor communality on FIT half only, all 4 channels -> R2_factor.
  3. KK predictions for 3 VPU pairs from FIT communalities -> lock + sha256.
COMPUTES NO observed Spearman on TEST. Reveal awaits "go".
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

SEED = 42
MIN_HALF = 100
DATA = Path(__file__).resolve().parents[1] / "data" / "nnmeter_raw"
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOCK = OUT / "predictions_locked_nnmeter_heldout.csv"
SPLIT = OUT / "nnmeter_heldout_split.csv"
COMM = OUT / "nnmeter_heldout_fit_communality.csv"
EXCLUDE = {"nasbench201s", "shufflenetv2s"}

CH = {
    "Pixel4 / CPU": "cortexA76cpu_tflite21",
    "Mi9 / GPU": "adreno640gpu_tflite21",
    "Pixel3XL / GPU": "adreno630gpu_tflite21",
    "Myriad / VPU": "myriadvpu_openvino2019r2",
}
LABELS = list(CH)
VPU = "Myriad / VPU"
PARTNERS = ["Pixel4 / CPU", "Pixel3XL / GPU", "Mi9 / GPU"]


def kk(r1, r2):
    return float((6.0 / np.pi) * np.arcsin(np.sqrt(r1 * r2) / 2.0))


def space_matrix(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            rows.append({lab: r.get(k) for lab, k in CH.items()})
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    m = (df > 0).all(axis=1) & df.notna().all(axis=1)
    return np.log(df[m].reset_index(drop=True))   # deterministic file order


def communality_fit(Mfit):
    np.random.seed(SEED)
    v = Mfit[LABELS].values
    Z = (v - v.mean(0)) / v.std(0, ddof=0)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z)
    load = fa.loadings_.ravel(); uniq = fa.get_uniquenesses()
    return {lab: float(load[j] ** 2 / (load[j] ** 2 + uniq[j])) for j, lab in enumerate(LABELS)}


def main():
    rng = np.random.default_rng(SEED)
    split_rows, comm_rows, pred_rows, size_rows = [], [], [], []
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        space = Path(fp).stem
        if space in EXCLUDE:
            continue
        M = space_matrix(fp)
        n = len(M)
        perm = rng.permutation(n)
        half = np.array(["FIT"] * n, dtype=object)
        half[perm[n // 2:]] = "TEST"
        n_fit = int((half == "FIT").sum()); n_test = int((half == "TEST").sum())
        size_rows.append({"space": space, "n_total": n, "n_fit": n_fit, "n_test": n_test,
                          "fit_ok": n_fit >= MIN_HALF, "test_ok": n_test >= MIN_HALF})
        for pos, hlab in enumerate(half):
            split_rows.append({"space": space, "pos": pos, "half": hlab})
        if n_fit < MIN_HALF or n_test < MIN_HALF:
            continue  # excluded from lock if either half too small
        Mfit = M[half == "FIT"]
        r2 = communality_fit(Mfit)
        comm_rows.append({"space": space, "n_fit": n_fit,
                          **{f"R2fit_{lab}": round(r2[lab], 4) for lab in LABELS}})
        for p in PARTNERS:
            pred_rows.append({
                "space": space, "n_fit": n_fit, "n_test": n_test,
                "vpu_pair": f"{p} x {VPU}", "channel_partner": p,
                "R2_partner_fit": round(r2[p], 4), "R2_vpu_fit": round(r2[VPU], 4),
                "rho_S_pred_heldout_FACTOR": round(kk(r2[p], r2[VPU]), 4),
                "source": "nn-Meter held-out (FIT->TEST, real measured)",
            })

    pd.DataFrame(split_rows).to_csv(SPLIT, index=False)
    sizes = pd.DataFrame(size_rows)
    comm = pd.DataFrame(comm_rows)
    pred = pd.DataFrame(pred_rows).sort_values(["space", "vpu_pair"]).reset_index(drop=True)
    comm.to_csv(COMM, index=False)
    pred.to_csv(LOCK, index=False)
    h = hashlib.sha256(open(LOCK, "rb").read()).hexdigest()

    print("="*86)
    print("nn-Meter HELD-OUT RE-LOCK — predictions locked on FIT half (NO TEST observed)")
    print("="*86)
    print("\n--- Step 1: 50/50 split sizes (seed 42) ---")
    print(sizes.to_string(index=False))
    below = sizes[~(sizes.fit_ok & sizes.test_ok)]
    print(f"  all halves >= {MIN_HALF}? {'YES' if below.empty else 'NO -> ' + ','.join(below.space)}")
    print(f"  spaces entering lock: {pred.space.nunique()}  -> {len(pred)} VPU-pair predictions")
    print("\n--- Step 2: FIT-half communalities (R2_factor, all 4 channels) ---")
    print(comm.to_string(index=False))
    print("\n--- Step 3: LOCKED held-out KK predictions per VPU pair ---")
    piv = pred.pivot(index="space", columns="channel_partner",
                     values="rho_S_pred_heldout_FACTOR")[PARTNERS]
    print(piv.to_string())
    print(f"\nLOCK FILE: {LOCK}")
    print(f"sha256:    {h}")
    print(f"split saved: {SPLIT.name}  | FIT communality: {COMM.name}")
    print("\nSTOPPED before any TEST-half observed Spearman. Reveal on \"go\".")


if __name__ == "__main__":
    main()
