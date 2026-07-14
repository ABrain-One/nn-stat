"""
EXPERIMENT 1 — Latent-speed fit and R²_{s,c} per channel.

For each of the 10 device-channels (5 phones x {CPU,GPU}):
- one-factor latent-speed model logT[a,c] = alpha_c + beta_c*s_a + eps[a,c]
- R²_{s,c} two ways: (a) factor communality, (b) variance/reliability route
- kappa sensitivity on the measurement noise.

Does NOT compute any cross-device Spearman (that is Exp 2, pre-registered).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from exp_common import (BACKENDS, M_RUNS, OUT, SEED, SOC, PROVENANCE,
                        build_channel_long, cleaning_report, shared_matrix)

np.random.seed(SEED)


def run(precision: str) -> pd.DataFrame:
    long = build_channel_long(precision)
    channels = [f"{ph} / {bk.upper()}" for bk in BACKENDS for ph in SOC]
    M_logT, M_noise, models = shared_matrix(long, channels)
    n = len(models)
    print(f"\n[{precision}] shared models clean on all {len(channels)} channels: n={n}")

    # ---- one-factor model on standardized log-latency ----
    Z = (M_logT - M_logT.mean()) / M_logT.std(ddof=0)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z.values)
    loadings = fa.loadings_.ravel()              # corr(channel, latent factor) since standardized
    uniq = fa.get_uniquenesses()                 # channel-specific (noise) variance on std scale
    # latent scores -> var(s); standardized factor has var ~1
    scores = fa.transform(Z.values).ravel()
    var_s = float(np.var(scores, ddof=0))

    rows = []
    for j, ch in enumerate(channels):
        ph = ch.split(" / ")[0]
        bk = ch.split(" / ")[1]
        beta_c = float(loadings[j])              # on standardized scale
        sigma2_c = float(uniq[j])                # standardized residual variance
        # (a) factor communality reliability
        r2_factor = beta_c ** 2 / (beta_c ** 2 + sigma2_c)

        # (b) variance / reliability route (on natural-log scale, not standardized)
        v_total = float(M_logT[ch].var(ddof=0))                 # across-architecture var of mean logT
        sigma2_run = float(M_noise[ch].mean())                  # mean within-arch per-run log-var
        sigma2_mean = sigma2_run / M_RUNS                       # noise of the per-arch mean (m=20)
        def r2_var(kappa):
            return max(0.0, (v_total - kappa * sigma2_mean) / v_total)
        r2_variance = r2_var(1.0)

        rows.append({
            "precision": precision, "channel": ch, "phone": ph, "soc": SOC[ph],
            "backend": bk, "provenance": PROVENANCE[ph], "n": n,
            "beta_c": round(beta_c, 4), "sigma2_c": round(sigma2_c, 4),
            "v_total_logvar": round(v_total, 4),
            "sigma2_run_log": round(sigma2_run, 6),
            "sigma2_mean_log": round(sigma2_mean, 8),
            "R2_factor": round(r2_factor, 4),
            "R2_variance": round(r2_variance, 4),
            "R2_at_k1.5": round(r2_var(1.5), 4),
            "R2_at_k2": round(r2_var(2.0), 4),
            "R2_at_k3": round(r2_var(3.0), 4),
        })
    out = pd.DataFrame(rows)
    out["var_s_factor"] = round(var_s, 4)
    return out


def main():
    clean = pd.concat([cleaning_report("fp32"), cleaning_report("int8")], ignore_index=True)
    clean.to_csv(OUT / "exp1_cleaning_report.csv", index=False)
    print("### Cleaning report (per channel) ###")
    print(clean.to_string(index=False))

    res = []
    for prec in ["fp32", "int8"]:
        res.append(run(prec))
    table = pd.concat(res, ignore_index=True)
    table.to_csv(OUT / "exp1_latent_speed_R2.csv", index=False)

    show = ["precision", "channel", "soc", "backend", "provenance", "n",
            "beta_c", "sigma2_c", "R2_factor", "R2_variance",
            "R2_at_k1.5", "R2_at_k2", "R2_at_k3"]
    print("\n### EXP 1 — R²_{s,c} per channel ###")
    for prec in ["fp32", "int8"]:
        print(f"\n--- {prec} ---")
        print(table[table.precision == prec][show].to_string(index=False))
    print(f"\nSaved: {OUT/'exp1_latent_speed_R2.csv'} , {OUT/'exp1_cleaning_report.csv'}")


if __name__ == "__main__":
    main()
