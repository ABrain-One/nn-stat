"""
EXTERNAL replication — nn-Meter, LOCK STEP (Exp 2 step 1-2 analog).

Pre-register KK predictions for all 6 device-pairs from the FOUR USER-CONFIRMED
communalities. Writes predictions_locked_nnmeter.csv ONLY. Computes NO observed Spearman.

KK identity:  rho_S_pred = (6/pi) * arcsin( sqrt(R2_c1 * R2_c2) / 2 )

Pair classes:
  discriminating = involves Myriad/VPU (R2=0.574) -> the REAL test.
  saturated      = CPU/GPU-only (all R2 >= 0.983) -> expected near-ceiling, tests nothing.

REVEAL-TIME REPORTING RULE (recorded here so it is honoured later): report the 3 VPU
pairs SEPARATELY from the 3 saturated pairs. DO NOT pool them into one MAE — the saturated
pairs flatter the average. The VPU pairs are the result.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from pathlib import Path

OUT = str(Path(__file__).resolve().parents[1] / "outputs" / "predictions_locked_nnmeter.csv")

# user-confirmed communalities (verified, locked)
COMM = {
    "Pixel4 / CPU":   {"r2": 0.983, "device": "Cortex-A76 CPU",  "kind": "CPU"},
    "Mi9 / GPU":      {"r2": 0.996, "device": "Adreno 640 GPU",  "kind": "GPU"},
    "Pixel3XL / GPU": {"r2": 0.996, "device": "Adreno 630 GPU",  "kind": "GPU"},
    "Myriad / VPU":   {"r2": 0.574, "device": "Intel Myriad VPU", "kind": "VPU"},
}


def kk(r1: float, r2: float) -> float:
    return float((6.0 / np.pi) * np.arcsin(np.sqrt(r1 * r2) / 2.0))


def main() -> None:
    rows = []
    for a, b in combinations(COMM, 2):
        ca, cb = COMM[a], COMM[b]
        involves_vpu = "VPU" in (ca["kind"], cb["kind"])
        pair_class = "discriminating" if involves_vpu else "saturated"
        # same-vendor GPU near-family flag (Adreno 630 vs 640)
        structure = ("same-vendor-GPU-near-family"
                     if ca["kind"] == "GPU" and cb["kind"] == "GPU" else "cross-device")
        rows.append({
            "pair": f"{a} x {b}",
            "channel_1": a, "channel_2": b,
            "device_1": ca["device"], "device_2": cb["device"],
            "R2_factor_c1": ca["r2"], "R2_factor_c2": cb["r2"],
            "rho_S_pred_FACTOR": round(kk(ca["r2"], cb["r2"]), 4),
            "pair_class": pair_class,
            "structure": structure,
            "source": "nn-Meter (real, measured)",
            "report_group": "VPU/discriminating (THE RESULT)" if involves_vpu
                            else "saturated (do NOT pool into MAE)",
        })
    pred = pd.DataFrame(rows)
    # order: discriminating first so the result is foregrounded
    pred = pred.sort_values(["pair_class", "pair"], ascending=[True, True]).reset_index(drop=True)
    pred.to_csv(OUT, index=False)

    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print("LOCKED nn-Meter predictions written (NO observed values computed).")
    print(f"file:   {OUT}")
    print(f"sha256: {h}\n")
    show = ["pair", "pair_class", "structure", "R2_factor_c1", "R2_factor_c2", "rho_S_pred_FACTOR"]
    print("--- DISCRIMINATING (VPU) pairs — the real test ---")
    print(pred[pred.pair_class == "discriminating"][show].to_string(index=False))
    print("\n--- SATURATED (CPU/GPU-only) pairs — expected near-ceiling, test nothing ---")
    print(pred[pred.pair_class == "saturated"][show].to_string(index=False))
    print("\nReveal-time rule (recorded): report the 3 VPU pairs separately; do NOT pool a "
          "single MAE across all 6.")


if __name__ == "__main__":
    main()
