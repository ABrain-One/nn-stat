"""
nn-Meter HELD-OUT REVEAL (runs on "go"). Fully blind: predictions were fit on the FIT half,
observed Spearman is computed on the never-seen TEST half.

Verifies the lock hash, reads the saved split, computes observed rho_S per VPU pair on TEST
rows only, vs the FIT-locked predictions. Same estimand across all 33 -> single MAE legit.
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
LOCK = Path(__file__).resolve().parents[1] / "locks" / "predictions_locked_nnmeter_heldout.csv"
SPLIT = Path(__file__).resolve().parents[1] / "data" / "nnmeter" / "nnmeter_heldout_split.csv"
LOCK_SHA = "fabac5a2a34d03fd6b0b9435c772fc56831606780abda74fe1499b52c118f7ec"
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
PARTNERS = ["Pixel4 / CPU", "Pixel3XL / GPU", "Mi9 / GPU"]


def space_matrix(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            rows.append({lab: r.get(k) for lab, k in CH.items()})
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    m = (df > 0).all(axis=1) & df.notna().all(axis=1)
    return np.log(df[m].reset_index(drop=True))


def main():
    h = hashlib.sha256(open(LOCK, "rb").read()).hexdigest()
    assert h == LOCK_SHA, f"LOCK CHANGED! {h}"
    pred = pd.read_csv(LOCK)
    split = pd.read_csv(SPLIT)

    # TEST-row matrices per space
    test_idx = {sp: split[(split.space == sp) & (split.half == "TEST")].pos.to_numpy()
                for sp in split.space.unique()}
    mats = {}
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        sp = Path(fp).stem
        if sp in EXCLUDE:
            continue
        M = space_matrix(fp)
        mats[sp] = M.iloc[test_idx[sp]].reset_index(drop=True)   # TEST half only

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
        pf = float(r.rho_S_pred_heldout_FACTOR)
        rows.append({
            "space": r.space, "vpu_pair": r.vpu_pair, "channel_partner": r.channel_partner,
            "n_test": int(len(x)), "R2_vpu_fit": float(r.R2_vpu_fit),
            "rho_S_pred": pf, "rho_S_obs_TEST": round(rho, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "abs_error": round(abs(pf - rho), 4),
            "signed_err_obs_minus_pred": round(rho - pf, 4),
            "in_95CI": bool(lo <= pf <= hi),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "nnmeter_heldout_reveal.csv", index=False)

    order = PARTNERS
    obs_p = tab.pivot(index="space", columns="channel_partner", values="rho_S_obs_TEST")[order]
    err_p = tab.pivot(index="space", columns="channel_partner", values="abs_error")[order]
    print("="*88)
    print(f"nn-Meter HELD-OUT REVEAL (TEST half, blind) — lock {h[:12]}…  ({len(tab)} points)")
    print("="*88)
    print("\nOBSERVED rho_S on TEST half:")
    print(obs_p.to_string())
    print("\nABS ERROR vs FIT-locked prediction:")
    print(err_p.to_string())

    mae = tab.abs_error.mean(); mx = tab.abs_error.max()
    mxr = tab.loc[tab.abs_error.idxmax()]
    cov = int(tab.in_95CI.sum())
    rank = spearmanr(tab.rho_S_pred, tab.rho_S_obs_TEST).statistic
    bias = tab.signed_err_obs_minus_pred.mean()
    print("\n--- DIAGNOSTICS (33 held-out points; same estimand) ---")
    print(f"  MAE = {mae:.4f}   max|err| = {mx:.4f} ({mxr.space}, {mxr.channel_partner})")
    print(f"  mean signed error (obs-pred) = {bias:+.4f}   pred-in-95%CI: {cov}/{len(tab)}")
    print(f"  ACROSS-SPACE rank-order Spearman(pred, obs) = {rank:.4f}")
    for p in order:
        sub = tab[tab.channel_partner == p]
        rp = spearmanr(sub.rho_S_pred, sub.rho_S_obs_TEST).statistic
        print(f"    {p:<16} MAE={sub.abs_error.mean():.4f}  in-CI {int(sub.in_95CI.sum())}/{len(sub)}  "
              f"rank-order rho={rp:.4f}")
    print(f"\n  COMPARISON: non-split within-space MAE was 0.0285 (14/33 in CI). "
          f"Held-out MAE = {mae:.4f} ({cov}/33 in CI).")
    print(f"\nSaved: {OUT/'nnmeter_heldout_reveal.csv'}")


if __name__ == "__main__":
    main()
