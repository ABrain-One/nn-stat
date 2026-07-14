"""
EXTERNAL replication — nn-Meter, STEP 1 ONLY (Exp 1 analog).

Ingest the nn-Meter bench dataset, build the architecture x device-channel matrix for the
FOUR REAL devices, verify the complete-case shared set, fit one-factor latent-speed
communalities (R2_factor = route of record). REPORT communalities + cleaning.

DOES NOT lock predictions and DOES NOT compute any cross-device observed Spearman.
NAS-Bench-201 EXCLUDED from this headline pool (published-rho leakage); reserved for LOFO.

Devices (all real on-device, single measured latency in ms, no per-run std):
  cortexA76cpu_tflite21        Pixel4   CPU (Cortex-A76)
  adreno640gpu_tflite21        Mi9      GPU (Adreno 640)
  adreno630gpu_tflite21        Pixel3XL GPU (Adreno 630)
  myriadvpu_openvino2019r2     Intel Myriad VPU
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "nnmeter_raw"
OUT = Path(__file__).resolve().parents[1] / "outputs"
EXCLUDE_SPACES = {"nasbench201s"}            # leakage reserve -> LOFO secondary only

CH = {
    "cortexA76cpu_tflite21": ("Pixel4 / CPU", "Cortex-A76 CPU"),
    "adreno640gpu_tflite21": ("Mi9 / GPU", "Adreno 640 GPU"),
    "adreno630gpu_tflite21": ("Pixel3XL / GPU", "Adreno 630 GPU"),
    "myriadvpu_openvino2019r2": ("Myriad / VPU", "Intel Myriad VPU"),
}
KEYS = list(CH)


def load_long() -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*.jsonl"))):
        space = Path(fp).stem
        if space in EXCLUDE_SPACES:
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rows.append({"model_id": r["id"], "space": space,
                             **{k: r.get(k) for k in KEYS}})
    return pd.DataFrame(rows)


def main():
    df = load_long()
    n_raw = len(df)

    # ---- cleaning report per channel ----
    clean_rows = []
    for k in KEYS:
        v = pd.to_numeric(df[k], errors="coerce")
        n_missing = int(v.isna().sum())
        n_nonpos = int((v <= 0).sum())
        n_valid = int(((v > 0) & v.notna()).sum())
        clean_rows.append({"channel": CH[k][0], "device": CH[k][1], "key": k,
                           "n_rows": n_raw, "missing_or_null": n_missing,
                           "nonpositive": n_nonpos, "valid": n_valid})
    clean = pd.DataFrame(clean_rows)
    clean.to_csv(OUT / "nnmeter_cleaning_report.csv", index=False)

    # ---- complete-case shared matrix (all 4 channels positive) ----
    num = df[KEYS].apply(pd.to_numeric, errors="coerce")
    mask = (num > 0).all(axis=1) & num.notna().all(axis=1)
    M = num[mask].copy()
    M.columns = [CH[k][0] for k in KEYS]
    n = len(M)
    logT = np.log(M.values)                      # natural log of ms

    # per-space contribution to the shared set
    space_counts = df.loc[mask, "space"].value_counts().sort_index()

    # ---- one-factor latent-speed fit (route of record), mirrors internal Exp 1 ----
    np.random.seed(SEED)
    Z = (logT - logT.mean(0)) / logT.std(0, ddof=0)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z)
    load = fa.loadings_.ravel()
    uniq = fa.get_uniquenesses()

    rows = []
    for j, k in enumerate(KEYS):
        beta = float(load[j]); sig = float(uniq[j])
        r2_factor = beta**2 / (beta**2 + sig)
        col = logT[:, j]
        rows.append({
            "channel": CH[k][0], "device": CH[k][1], "source": "nn-Meter (real, measured)",
            "n": n, "beta_loading": round(beta, 4), "uniqueness": round(sig, 4),
            "R2_factor": round(r2_factor, 4),
            "R2_variance": "N/A (no per-run std in nn-Meter)",
            "loglat_var": round(float(col.var(ddof=0)), 4),
            "lat_ms_min": round(float(M.iloc[:, j].min()), 3),
            "lat_ms_median": round(float(M.iloc[:, j].median()), 3),
            "lat_ms_max": round(float(M.iloc[:, j].max()), 1),
        })
    comm = pd.DataFrame(rows)
    comm.to_csv(OUT / "nnmeter_communality.csv", index=False)

    # ---- report ----
    print("="*78)
    print("nn-Meter EXTERNAL replication — STEP 1 (communalities only; NOT locked)")
    print("="*78)
    print(f"\nRaw records pooled (12 diverse spaces, nasbench201 excluded): {n_raw}")
    print(f"COMPLETE-CASE SHARED SET across all 4 real devices: n = {n}\n")
    print("--- Cleaning report (per channel) ---")
    print(clean[["channel", "device", "n_rows", "missing_or_null",
                 "nonpositive", "valid"]].to_string(index=False))
    print("\n  (shufflenetv2s contributes 0: both Adreno GPU channels are null — "
          "channel-shuffle ops unsupported by the GPU delegate; drops out via complete-case.)")
    print("\n--- Per-space contribution to the shared set ---")
    print(space_counts.to_string())
    print("\n--- COMMUNALITY TABLE (R2_factor = route of record) ---")
    print(comm[["channel", "device", "n", "beta_loading", "uniqueness",
                "R2_factor", "loglat_var", "lat_ms_median"]].to_string(index=False))
    print("\nNOTES:")
    print("- Single measured latency per (model,device); NO per-run std_dev in nn-Meter ->")
    print("  variance/reliability route NOT computable here; factor route is the only route")
    print("  (and is our route of record anyway).")
    print("- All 4 channels are DISTINCT physical devices (Pixel4 CPU, Mi9 GPU, Pixel3XL GPU,")
    print("  Myriad VPU) -> NO within-device CPU<->GPU pair like our internal set (affects Exp 5")
    print("  design later, not Step 1). Adreno 630 vs 640 = same-vendor GPU near-family.")
    print("- STOPPED before locking. No predictions written, no observed Spearman computed.")
    print(f"\nSaved: {OUT/'nnmeter_communality.csv'}, {OUT/'nnmeter_cleaning_report.csv'}")


if __name__ == "__main__":
    main()
