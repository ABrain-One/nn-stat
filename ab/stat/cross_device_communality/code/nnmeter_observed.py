"""
EXTERNAL replication — nn-Meter, OBSERVED STEP (Exp 2 analog). Runs on user "go".

READS predictions_locked_nnmeter.csv (verifies hash), computes observed Spearman rho_S per
device-pair on the shared n=21,643 set, per-pair 95% bootstrap CIs (seed 42).

REPORTING RULE (user-mandated): the 3 DISCRIMINATING (VPU) pairs are the result and are
reported as their OWN group (MAE/max within-group). The 3 SATURATED pairs are reported
separately and are NEVER pooled with the VPU pairs.
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
LOCK = Path(__file__).resolve().parents[1] / "locks" / "predictions_locked_nnmeter.csv"
LOCK_SHA = "78a2efeebd05f71e7ffd44e635be825ce7ff3fa14ef546ef9d11842971ce776c"
EXCLUDE_SPACES = {"nasbench201s"}
N_BOOT = 2000
SEED = 42

KEY = {  # channel label -> jsonl key
    "Pixel4 / CPU": "cortexA76cpu_tflite21",
    "Mi9 / GPU": "adreno640gpu_tflite21",
    "Pixel3XL / GPU": "adreno630gpu_tflite21",
    "Myriad / VPU": "myriadvpu_openvino2019r2",
}


def shared_matrix() -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        if Path(fp).stem in EXCLUDE_SPACES:
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rows.append({lab: r.get(k) for lab, k in KEY.items()})
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    mask = (df > 0).all(axis=1) & df.notna().all(axis=1)
    return np.log(df[mask].reset_index(drop=True))   # natural-log latency; Spearman is invariant


def boot_ci(x, y, rng):
    n = len(x)
    s = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        s[b] = spearmanr(x[idx], y[idx]).statistic
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    h = hashlib.sha256(open(LOCK, "rb").read()).hexdigest()
    assert h == LOCK_SHA, f"LOCK FILE CHANGED! {h}"
    pred = pd.read_csv(LOCK)
    M = shared_matrix()
    n = len(M)
    rng = np.random.default_rng(SEED)

    rows = []
    for _, r in pred.iterrows():
        c1, c2 = r.channel_1, r.channel_2
        x, y = M[c1].to_numpy(), M[c2].to_numpy()
        rho = float(spearmanr(x, y).statistic)
        lo, hi = boot_ci(x, y, rng)
        pf = float(r.rho_S_pred_FACTOR)
        rows.append({
            "pair": r.pair, "pair_class": r.pair_class, "structure": r.structure,
            "rho_S_pred_FACTOR": pf, "rho_S_obs": round(rho, 4),
            "rho_S_obs_CI_lo": round(lo, 4), "rho_S_obs_CI_hi": round(hi, 4),
            "abs_error": round(abs(pf - rho), 4),
            "signed_error_obs_minus_pred": round(rho - pf, 4),
            "pred_in_CI": bool(lo <= pf <= hi), "n": n,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "nnmeter_observed_comparison.csv", index=False)

    disc = tab[tab.pair_class == "discriminating"]
    sat = tab[tab.pair_class == "saturated"]

    print("="*82)
    print(f"nn-Meter OBSERVED reveal — shared set n={n}.  Lock hash verified: {h[:12]}…")
    print("="*82)
    show = ["pair", "structure", "rho_S_pred_FACTOR", "rho_S_obs",
            "rho_S_obs_CI_lo", "rho_S_obs_CI_hi", "abs_error", "pred_in_CI"]

    print("\n######## DISCRIMINATING (VPU) PAIRS — THE RESULT ########")
    print(disc[show].to_string(index=False))
    print(f"  >> VPU-group MAE = {disc.abs_error.mean():.4f}   "
          f"max|err| = {disc.abs_error.max():.4f}   "
          f"pred-in-CI {int(disc.pred_in_CI.sum())}/{len(disc)}")

    print("\n-------- SATURATED (CPU/GPU-only) PAIRS — reported separately, NOT pooled --------")
    print(sat[show].to_string(index=False))
    print(f"  (saturated-group MAE = {sat.abs_error.mean():.4f}; shown only for completeness — "
          f"these test nothing)")
    print("\nNOTE: no MAE pooled across all 6 pairs, per the pre-registered reporting rule.")
    print(f"\nSaved: {OUT/'nnmeter_observed_comparison.csv'}")


if __name__ == "__main__":
    main()
