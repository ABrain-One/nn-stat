"""
EXPERIMENT 2 (LOCK STEP) — pre-register rho_S predictions for GPU device-pairs.

R²_{s,d} of record = FACTOR/communality route (locked decision, option C).
Variance route reported alongside as a saturated UPPER BOUND.

Kruskal-Kendall identity:  rho_S_pred = (6/pi) * arcsin( sqrt(R2_c1 * R2_c2) / 2 )

WRITES predictions_locked.csv ONLY. Computes NO observed Spearman.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from exp_common import OUT, SOC, PROVENANCE


def ks(r1: float, r2: float) -> float:
    return float((6.0 / np.pi) * np.arcsin(np.sqrt(r1 * r2) / 2.0))


def main() -> None:
    t = pd.read_csv(OUT / "exp1_latent_speed_R2.csv")
    rows = []
    for prec in ["fp32", "int8"]:
        g = t[(t.precision == prec) & (t.backend == "GPU")].set_index("channel")
        for a, b in combinations(g.index, 2):
            ra, rb = g.loc[a], g.loc[b]
            soc1, soc2 = ra["soc"], rb["soc"]
            # independence level (within-phone CPU<->GPU handled in Exp 5, not here)
            if soc1 == soc2:
                indep = "same-family-SD720G"
            else:
                indep = "cross-family"
            prov = ("involves-PRIOR" if "PRIOR-context" in (ra["provenance"], rb["provenance"])
                    else "NEW-only")
            rows.append({
                "precision": prec,
                "pair": f"{ra['phone']} x {rb['phone']}",
                "channel_1": a, "channel_2": b,
                "soc_1": soc1, "soc_2": soc2,
                "provenance_pair": prov,
                "independence_level": indep,
                "R2_factor_c1": round(float(ra["R2_factor"]), 4),
                "R2_factor_c2": round(float(rb["R2_factor"]), 4),
                "R2_variance_c1": round(float(ra["R2_variance"]), 4),
                "R2_variance_c2": round(float(rb["R2_variance"]), 4),
                "rho_S_pred_FACTOR": round(ks(ra["R2_factor"], rb["R2_factor"]), 4),
                "rho_S_pred_variance_UPPERBOUND": round(ks(ra["R2_variance"], rb["R2_variance"]), 4),
            })
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions_locked.csv", index=False)
    print("LOCKED predictions written to predictions_locked.csv (NO observed values computed).")
    print(f"R2 of record = FACTOR route. Seed=42. n=491 (fp32) / 503 (int8), GPU, 10 pairs each.\n")
    for prec in ["fp32", "int8"]:
        print(f"--- {prec} (GPU pairs) ---")
        cols = ["pair", "soc_1", "soc_2", "independence_level", "provenance_pair",
                "R2_factor_c1", "R2_factor_c2", "rho_S_pred_FACTOR",
                "rho_S_pred_variance_UPPERBOUND"]
        print(pred[pred.precision == prec][cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
