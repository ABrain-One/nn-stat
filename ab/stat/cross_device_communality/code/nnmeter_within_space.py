"""
ROBUSTNESS — within-space (un-pooled) Spearman for the 3 VPU pairs.

Question: is the VPU under-transfer (pooled observed ~0.38-0.46 vs predicted ~0.74) real
within homogeneous search spaces, or a pooling/mixture artifact? Compute Spearman per search
space (each ~1.8-2k archs) for each VPU pair; compare to the pooled value and the locked pred.

Read-only. Writes a CSV; touches nothing in the paper, no lock files.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parents[1] / "data" / "nnmeter_raw"
OUT = Path(__file__).resolve().parents[1] / "outputs"
EXCLUDE = {"nasbench201s"}

KEY = {
    "Pixel4 / CPU": "cortexA76cpu_tflite21",
    "Mi9 / GPU": "adreno640gpu_tflite21",
    "Pixel3XL / GPU": "adreno630gpu_tflite21",
    "Myriad / VPU": "myriadvpu_openvino2019r2",
}
VPU = "Myriad / VPU"
VPU_PAIRS = [("Pixel4 / CPU", VPU), ("Pixel3XL / GPU", VPU), ("Mi9 / GPU", VPU)]
POOLED = {"Pixel4 / CPU": 0.3767, "Pixel3XL / GPU": 0.4177, "Mi9 / GPU": 0.4618}
PRED = {"Pixel4 / CPU": 0.7353, "Pixel3XL / GPU": 0.7404, "Mi9 / GPU": 0.7404}


def space_frame(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            rows.append({lab: r.get(k) for lab, k in KEY.items()})
    return pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")


def main():
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        space = Path(fp).stem
        if space in EXCLUDE:
            continue
        df = space_frame(fp)
        for other, _ in VPU_PAIRS:
            sub = df[[other, VPU]]
            m = (sub > 0).all(axis=1) & sub.notna().all(axis=1)
            sub = sub[m]
            if len(sub) < 30:
                rho = np.nan
            else:
                rho = float(spearmanr(np.log(sub[other]), np.log(sub[VPU])).statistic)
            rows.append({"space": space, "vpu_partner": other,
                         "n": int(len(sub)), "rho_S_within": round(rho, 4) if not np.isnan(rho) else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "nnmeter_within_space_vpu.csv", index=False)

    print("="*78)
    print("WITHIN-SPACE Spearman — 3 VPU pairs (un-pooled robustness)")
    print("="*78)
    piv = tab.pivot(index="space", columns="vpu_partner", values="rho_S_within")
    npiv = tab.pivot(index="space", columns="vpu_partner", values="n")
    order = ["Pixel4 / CPU", "Pixel3XL / GPU", "Mi9 / GPU"]
    piv = piv[order]
    print("\nper-space rho_S (VPU vs partner):")
    print(piv.to_string())
    print("\nper-space n:")
    print(npiv[order].astype(int).to_string())

    print("\n--- SUMMARY per VPU pair ---")
    print(f"{'partner':<16} {'pooled':>7} {'pred':>7} {'within:median':>14} "
          f"{'min':>7} {'max':>7} {'#spaces<pooled':>15}")
    for p in order:
        col = piv[p].dropna()
        nlt = int((col < POOLED[p]).sum())
        print(f"{p:<16} {POOLED[p]:>7.3f} {PRED[p]:>7.3f} {col.median():>14.3f} "
              f"{col.min():>7.3f} {col.max():>7.3f} {nlt:>11}/{len(col)}")

    allvals = piv.values[~np.isnan(piv.values)]
    print(f"\nVERDICT: across all {len(allvals)} within-space VPU estimates, "
          f"range [{allvals.min():.3f}, {allvals.max():.3f}], median {np.median(allvals):.3f}.")
    print(f"Every within-space value is FAR below the locked prediction ~0.74"
          if allvals.max() < 0.74 else "Some within-space values approach the prediction.")
    print(f"\nSaved: {OUT/'nnmeter_within_space_vpu.csv'}")


if __name__ == "__main__":
    main()
