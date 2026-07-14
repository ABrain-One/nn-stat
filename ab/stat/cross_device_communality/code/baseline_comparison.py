"""
BASELINE COMPARISON (paper Section 5 / validation table).

Compares the communality/arcsin (Kruskal-Kendall) predictor of cross-device
Spearman rank-transfer against trivial baselines, on the 10 GPU device-pairs,
for fp32 and int8. Reads only already-revealed reference CSVs — recomputes no
latencies and touches no lock file.

Predictors evaluated against observed Spearman rho_S per pair:
  (a) communality / arcsin (ours):  rho_S_pred_FACTOR
  (b) constant = mean observed rho_S    <-- THIS is the paper's "constant predictor"
                                    (fp32 MAE 0.0091, int8 MAE 0.0199)
  (c) constant rho_S == 1           naive "ranks transfer perfectly"
                                    (fp32 MAE 0.0213, int8 MAE 0.0333)
  (d) reliability route             KK on the variance/reliability R^2 (saturates ~1)
  (e) raw Pearson-as-Spearman       observed Pearson used as a Spearman proxy
                                    (within-sample oracle ordering, NOT a prediction)

NOTE: the paper's baseline table reports the CONSTANT predictor as the constant
equal to the mean observed rho_S (row (b), 0.0091 fp32). Row (c), the rho_S==1
naive baseline, is a distinct, additional reference and is a LARGER MAE (0.0213).
Do not confuse the two. (The MAE-optimal constant is actually the median of the
observed rho_S, 0.0082 fp32; the mean-observed constant is the reported choice.)

Metrics: MAE overall, MAE on the 4 Kirin-710 pairs (the discriminating floor),
and rank-order Spearman between predictor and observed across the 10 pairs.
Inputs:
  data/exp2_observed_comparison.csv   (rho_S_obs, rho_S_pred_FACTOR, rho_P_obs)
  locks/predictions_locked.csv        (rho_S_pred_variance_UPPERBOUND = reliability route)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
OBS = REPO / "data" / "exp2_observed_comparison.csv"
LOCK = REPO / "locks" / "predictions_locked.csv"
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def rank_order(pred: np.ndarray, obs: np.ndarray) -> float:
    if np.allclose(pred, pred[0]):
        return np.nan  # constant predictor has no ordering
    return float(spearmanr(pred, obs).statistic)


def run(precision: str) -> pd.DataFrame:
    o = pd.read_csv(OBS)
    l = pd.read_csv(LOCK)
    o = o[o.precision == precision].reset_index(drop=True)
    l = l[l.precision == precision].reset_index(drop=True)
    assert (o.pair.values == l.pair.values).all(), "pair order mismatch"

    y = o.rho_S_obs.to_numpy()
    is_kirin = (o.soc_1 == "Kirin710") | (o.soc_2 == "Kirin710")

    preds = {
        "communality/arcsin (ours)": o.rho_S_pred_FACTOR.to_numpy(),
        "constant=mean obs [PAPER]": np.full_like(y, y.mean()),
        "constant rho_S=1 (naive)": np.ones_like(y),
        "reliability route": l.rho_S_pred_variance_UPPERBOUND.to_numpy(),
        "raw Pearson-as-Spearman": o.rho_P_obs.to_numpy(),
    }
    rows = []
    for name, p in preds.items():
        ae = np.abs(p - y)
        rows.append({
            "precision": precision, "predictor": name,
            "MAE": round(ae.mean(), 4),
            "MAE_kirin": round(ae[is_kirin].mean(), 4),
            "MAE_nonkirin": round(ae[~is_kirin].mean(), 4),
            "rank_order_rho": (round(rank_order(p, y), 4)
                               if not np.isnan(rank_order(p, y)) else np.nan),
        })
    return pd.DataFrame(rows)


def main() -> None:
    tab = pd.concat([run("fp32"), run("int8")], ignore_index=True)
    tab.to_csv(OUT / "baseline_comparison.csv", index=False)
    for prec in ["fp32", "int8"]:
        print(f"\n### Baseline comparison — {prec} (10 GPU pairs) ###")
        print(tab[tab.precision == prec].to_string(index=False))
    print(f"\nSaved: {OUT / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
