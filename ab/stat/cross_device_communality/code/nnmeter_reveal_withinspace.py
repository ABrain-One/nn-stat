"""
EXTERNAL replication — nn-Meter WITHIN-SPACE REVEAL (runs on "go").

READS predictions_locked_nnmeter_withinspace.csv (verifies hash), computes WITHIN-SPACE
observed Spearman per VPU pair on the SAME 4-channel complete-case rows used for the
communality fits, and answers the pre-registered question:
    does within-space communality->KK match within-space observed rho_S for the VPU pairs?

All 33 points are the same estimand (within-space VPU-pair rho_S), so a single MAE across
them is legitimate here (unlike the earlier pooled-vs-saturated mix).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parents[1] / "data" / "nnmeter_raw"
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOCK = Path(__file__).resolve().parents[1] / "locks" / "predictions_locked_nnmeter_withinspace.csv"
LOCK_SHA = "7d65820f7e04bf4df2605a9468978d36addf3ea4dd9f6e2ef8f57dc7f49faa09"
EXCLUDE = {"nasbench201s", "shufflenetv2s"}
N_BOOT = 2000
SEED = 42

CH = {
    "Pixel4 / CPU": "cortexA76cpu_tflite21",
    "Mi9 / GPU": "adreno640gpu_tflite21",
    "Pixel3XL / GPU": "adreno630gpu_tflite21",
    "Myriad / VPU": "myriadvpu_openvino2019r2",
}
LABELS = list(CH)
VPU = "Myriad / VPU"


def space_matrix(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            rows.append({lab: r.get(k) for lab, k in CH.items()})
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    m = (df > 0).all(axis=1) & df.notna().all(axis=1)
    return np.log(df[m].reset_index(drop=True))   # 4-channel complete-case, log-latency


def main():
    h = hashlib.sha256(open(LOCK, "rb").read()).hexdigest()
    assert h == LOCK_SHA, f"LOCK CHANGED! {h}"
    pred = pd.read_csv(LOCK)

    mats = {Path(fp).stem: space_matrix(fp)
            for fp in sorted(glob.glob(str(DATA / "*.jsonl")))
            if Path(fp).stem not in EXCLUDE}
    rng = np.random.default_rng(SEED)

    rows = []
    for _, r in pred.iterrows():
        M = mats[r.space]
        x = M[r.channel_partner].to_numpy(); y = M[VPU].to_numpy()
        rho = float(spearmanr(x, y).statistic)
        s = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(x), len(x))
            s[b] = spearmanr(x[idx], y[idx]).statistic
        lo, hi = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
        pf = float(r.rho_S_pred_withinspace_FACTOR)
        rows.append({
            "space": r.space, "vpu_pair": r.vpu_pair, "channel_partner": r.channel_partner,
            "n_space": int(r.n_space), "rho_S_pred": pf, "rho_S_obs": round(rho, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "abs_error": round(abs(pf - rho), 4),
            "signed_err_obs_minus_pred": round(rho - pf, 4),
            "pred_in_CI": bool(lo <= pf <= hi),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "nnmeter_withinspace_reveal.csv", index=False)

    print("="*88)
    print(f"nn-Meter WITHIN-SPACE REVEAL — lock hash verified {h[:12]}…  ({len(tab)} VPU-pair points)")
    print("="*88)
    obs_p = tab.pivot(index="space", columns="channel_partner", values="rho_S_obs")
    prd_p = tab.pivot(index="space", columns="channel_partner", values="rho_S_pred")
    order = ["Pixel4 / CPU", "Pixel3XL / GPU", "Mi9 / GPU"]
    print("\nPREDICTED (locked):")
    print(prd_p[order].to_string())
    print("\nOBSERVED:")
    print(obs_p[order].to_string())
    print("\nABS ERROR:")
    print((obs_p[order] - prd_p[order]).abs().round(4).to_string())

    mae = tab.abs_error.mean(); mx = tab.abs_error.max()
    mxrow = tab.loc[tab.abs_error.idxmax()]
    cov = int(tab.pred_in_CI.sum())
    rank = spearmanr(tab.rho_S_pred, tab.rho_S_obs).statistic
    bias = tab.signed_err_obs_minus_pred.mean()
    print("\n--- DIAGNOSTICS (all 33 within-space VPU-pair points; same estimand) ---")
    print(f"  MAE = {mae:.4f}   max|err| = {mx:.4f} ({mxrow.space}, {mxrow.channel_partner})")
    print(f"  mean signed error (obs-pred) = {bias:+.4f}   pred-in-95%CI: {cov}/{len(tab)}")
    print(f"  ACROSS-SPACE rank-order Spearman(pred, obs) over 33 pts = {rank:.4f}")
    for p in order:
        sub = tab[tab.channel_partner == p]
        rp = spearmanr(sub.rho_S_pred, sub.rho_S_obs).statistic
        print(f"    {p:<16} MAE={sub.abs_error.mean():.4f}  "
              f"rank-order ρ(pred,obs) over 11 spaces = {rp:.4f}")
    print(f"\nSaved: {OUT/'nnmeter_withinspace_reveal.csv'}")


if __name__ == "__main__":
    main()
