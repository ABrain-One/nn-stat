"""
EXTERNAL replication — nn-Meter, WITHIN-SPACE RE-LOCK (clean, pre-registered).

Pre-registered question: does the WITHIN-SPACE communality->KK prediction match WITHIN-SPACE
observed rho_S for the VPU pairs?

This script (steps 1-3):
  (1) re-estimate one-factor communality WITHIN each search space for all 4 channels;
  (2) compute within-space KK predictions per VPU pair;
  (3) write + sha256-hash predictions_locked_nnmeter_withinspace.csv.
It computes NO within-space observed Spearman. Reveal happens later, on "go".

Excluded: nasbench201s (leakage reserve) and shufflenetv2s (both GPU channels null -> cannot
fit a 4-channel within-space factor model).
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
DATA = Path(__file__).resolve().parents[1] / "data" / "nnmeter_raw"
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOCK = OUT / "predictions_locked_nnmeter_withinspace.csv"
COMM_CSV = OUT / "nnmeter_withinspace_communality.csv"
EXCLUDE = {"nasbench201s", "shufflenetv2s"}

CH = {  # label -> jsonl key ; first is the latent indicator order
    "Pixel4 / CPU": "cortexA76cpu_tflite21",
    "Mi9 / GPU": "adreno640gpu_tflite21",
    "Pixel3XL / GPU": "adreno630gpu_tflite21",
    "Myriad / VPU": "myriadvpu_openvino2019r2",
}
LABELS = list(CH)
VPU = "Myriad / VPU"
VPU_PARTNERS = ["Pixel4 / CPU", "Pixel3XL / GPU", "Mi9 / GPU"]


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
    return df[m].reset_index(drop=True)


def communalities(M):
    """One-factor ML communality per channel on within-space standardized log-latency."""
    np.random.seed(SEED)
    logT = np.log(M[LABELS].values)
    Z = (logT - logT.mean(0)) / logT.std(0, ddof=0)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z)
    load = fa.loadings_.ravel()
    uniq = fa.get_uniquenesses()
    return {lab: float(load[j] ** 2 / (load[j] ** 2 + uniq[j])) for j, lab in enumerate(LABELS)}


def main():
    comm_rows, pred_rows = [], []
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        space = Path(fp).stem
        if space in EXCLUDE:
            continue
        M = space_matrix(fp)
        n = len(M)
        r2 = communalities(M)
        comm_rows.append({"space": space, "n_space": n,
                          **{f"R2_{lab}": round(r2[lab], 4) for lab in LABELS}})
        for partner in VPU_PARTNERS:
            pred_rows.append({
                "space": space, "n_space": n,
                "vpu_pair": f"{partner} x {VPU}",
                "channel_partner": partner, "channel_vpu": VPU,
                "R2_partner_withinspace": round(r2[partner], 4),
                "R2_vpu_withinspace": round(r2[VPU], 4),
                "rho_S_pred_withinspace_FACTOR": round(kk(r2[partner], r2[VPU]), 4),
                "source": "nn-Meter within-space (real, measured)",
            })

    comm = pd.DataFrame(comm_rows).sort_values("space").reset_index(drop=True)
    comm.to_csv(COMM_CSV, index=False)               # pre-lock communality table (transparency)

    pred = pd.DataFrame(pred_rows).sort_values(["space", "vpu_pair"]).reset_index(drop=True)
    pred.to_csv(LOCK, index=False)
    h = hashlib.sha256(open(LOCK, "rb").read()).hexdigest()

    print("="*84)
    print("nn-Meter WITHIN-SPACE RE-LOCK — predictions locked (NO observed computed)")
    print("="*84)
    print(f"Spaces: {comm.space.nunique()} (nasbench201s + shufflenetv2s excluded). "
          f"VPU pairs/space: 3 -> {len(pred)} locked predictions.\n")
    print(f"LOCK FILE: {LOCK}")
    print(f"sha256:    {h}\n")
    print("--- within-space communalities (R2_factor, all 4 channels) ---")
    print(comm.to_string(index=False))
    print("\n--- LOCKED within-space KK predictions per VPU pair (rho_S_pred) ---")
    piv = pred.pivot(index="space", columns="channel_partner",
                     values="rho_S_pred_withinspace_FACTOR")[VPU_PARTNERS]
    print(piv.to_string())
    print(f"\n  predicted rho_S range: [{pred.rho_S_pred_withinspace_FACTOR.min():.4f}, "
          f"{pred.rho_S_pred_withinspace_FACTOR.max():.4f}]")
    print("\nSTOPPED before any within-space observed Spearman. Reveal on \"go\".")
    print(f"Saved: {LOCK.name}, {COMM_CSV.name}")


if __name__ == "__main__":
    main()
