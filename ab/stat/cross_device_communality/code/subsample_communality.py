"""
SUBSAMPLING STABILITY CHECK — how much do the one-factor communalities (R2_factor)
wobble when refit on random subsamples of the existing architectures?

No new profiling. Reuses the exact Exp-1 fit:
  FactorAnalyzer(n_factors=1, rotation=None, method="ml") on standardized log-latency,
  R2_{s,c} = loading_c^2 / (loading_c^2 + uniqueness_c).

For each precision and each subsample size n in {50,100,200}, draw REPS random
subsamples (without replacement) of the architectures from the full complete-case
matrix, refit, and record per-channel R2_factor. Report the spread vs the full-data
point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from exp_common import BACKENDS, OUT, SEED, SOC, build_channel_long, shared_matrix

SIZES = [50, 100, 200]
REPS = 300


def fit_communalities(M_logT: pd.DataFrame) -> np.ndarray:
    """Return R2_factor per channel for a (models x channels) log-latency matrix."""
    sd = M_logT.std(ddof=0)
    Z = (M_logT - M_logT.mean()) / sd
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z.values)
    load = fa.loadings_.ravel()
    uniq = fa.get_uniquenesses()
    return load ** 2 / (load ** 2 + uniq)


def run(precision: str):
    long = build_channel_long(precision)
    channels = [f"{ph} / {bk.upper()}" for bk in BACKENDS for ph in SOC]
    M_logT, _, models = shared_matrix(long, channels)
    N = len(models)
    full = fit_communalities(M_logT)
    full_s = pd.Series(full, index=channels)

    rng = np.random.default_rng(SEED)
    rep_rows = []
    for n in SIZES:
        if n > N:
            continue
        draws = []  # REPS x n_channels
        n_fail = 0
        for _ in range(REPS):
            idx = rng.choice(N, size=n, replace=False)
            sub = M_logT.iloc[idx]
            try:
                draws.append(fit_communalities(sub))
            except Exception:
                n_fail += 1
        draws = np.array(draws)
        for j, ch in enumerate(channels):
            col = draws[:, j]
            rep_rows.append({
                "precision": precision, "n_subsample": n, "reps": len(draws),
                "channel": ch, "backend": ch.split(" / ")[1], "soc": SOC[ch.split(" / ")[0]],
                "R2_full": round(full_s[ch], 4),
                "mean": round(col.mean(), 4),
                "bias": round(col.mean() - full_s[ch], 4),
                "sd": round(col.std(ddof=1), 4),
                "p05": round(np.percentile(col, 5), 4),
                "p95": round(np.percentile(col, 95), 4),
                "range90": round(np.percentile(col, 95) - np.percentile(col, 5), 4),
                "min": round(col.min(), 4),
                "max": round(col.max(), 4),
                "n_fail": n_fail,
            })
    return full_s, N, pd.DataFrame(rep_rows)


def main():
    all_rows = []
    summary = []
    for prec in ["fp32", "int8"]:
        full_s, N, df = run(prec)
        all_rows.append(df)
        gpu = df[df.backend == "GPU"]
        print(f"\n================  {prec}  (full N={N}, REPS={REPS})  ================")
        print("Full-data communalities:")
        for ch in full_s.index:
            print(f"   {ch:28s} {full_s[ch]:.4f}")
        for n in SIZES:
            d = df[df.n_subsample == n]
            if d.empty:
                continue
            dg = d[d.backend == "GPU"]
            print(f"\n--- n={n} ---  (mean SD across 10 ch = {d['sd'].mean():.4f}; "
                  f"GPU-only mean SD = {dg['sd'].mean():.4f})")
            print(d[["channel", "R2_full", "mean", "bias", "sd",
                     "p05", "p95", "range90"]].to_string(index=False))
            summary.append({
                "precision": prec, "n_subsample": n,
                "sd_all10_mean": round(d["sd"].mean(), 4),
                "sd_all10_max": round(d["sd"].max(), 4),
                "sd_gpu_mean": round(dg["sd"].mean(), 4),
                "range90_all10_mean": round(d["range90"].mean(), 4),
                "range90_gpu_mean": round(dg["range90"].mean(), 4),
                "bias_all10_mean": round(d["bias"].mean(), 4),
                "worst_ch": d.loc[d["sd"].idxmax(), "channel"],
                "worst_sd": round(d["sd"].max(), 4),
            })

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT / "subsample_communality.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "subsample_communality_summary.csv", index=False)
    print("\n\n################  SUMMARY (wobble of R2_factor)  ################")
    print(sm.to_string(index=False))
    print(f"\nSaved: {OUT/'subsample_communality.csv'} , {OUT/'subsample_communality_summary.csv'}")


if __name__ == "__main__":
    main()
