"""
EXPERIMENT 6 — leave-one-SoC-family-out (LOFO) generalisation.

Question: can we predict a previously-unseen SoC family's cross-device rank-transfer
(mean rho_S to the rest of the fleet) from its latent-speed R2_factor alone, via KK, using
only a SMALL reference set of already-characterised devices?

Procedure (GPU channels, R2_factor route of record):
  For each held-out family F:
    target pairs = {(f, r): f in F's GPU channels, r in reference GPU channels (not in F)}
    predicted mean rho_S(F) = mean_{f,r} KK(R2f[f], R2f[r])
    observed  mean rho_S(F) = mean_{f,r} Spearman(logT_f, logT_r)
  Then sweep reference-set size k = 1..K: report the WORST-CASE |pred_mean - obs_mean|
  over all size-k reference subsets, and the smallest k whose worst case <= 0.05.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp_common import OUT, SOC, PROVENANCE, build_channel_long, shared_matrix

ALL_CHANNELS = [f"{ph} / {bk.upper()}" for bk in ("cpu", "gpu") for ph in SOC]
TOL = 0.05
FAMILIES = ["SD888", "HelioMT6768", "SD720G", "Kirin710"]


def kk(r1, r2):
    return float((6.0 / np.pi) * np.arcsin(np.sqrt(r1 * r2) / 2.0))


def run(precision: str):
    long = build_channel_long(precision)
    M_logT, _, models = shared_matrix(long, ALL_CHANNELS)
    n = len(models)

    # R2_factor per GPU channel from the locked Exp1 table (route of record)
    t = pd.read_csv(OUT / "exp1_latent_speed_R2.csv")
    g = t[(t.precision == precision) & (t.backend == "GPU")].set_index("channel")
    r2f = {ch: float(g.loc[ch, "R2_factor"]) for ch in g.index}
    gpu = list(g.index)
    fam_of = {ch: SOC[ch.split(" / ")[0]] for ch in gpu}

    # precompute pairwise pred / obs
    pred_pair, obs_pair = {}, {}
    for a in gpu:
        for b in gpu:
            if a == b:
                continue
            pred_pair[(a, b)] = kk(r2f[a], r2f[b])
            obs_pair[(a, b)] = float(spearmanr(M_logT[a], M_logT[b]).statistic)

    rows = []
    for fam in FAMILIES:
        target = [c for c in gpu if fam_of[c] == fam]
        refs = [c for c in gpu if fam_of[c] != fam]
        # full-reference family means
        pred_full = np.mean([pred_pair[(f, r)] for f in target for r in refs])
        obs_full = np.mean([obs_pair[(f, r)] for f in target for r in refs])
        # sweep reference-set size: worst-case error over size-k subsets
        min_k = None
        per_k = {}
        for k in range(1, len(refs) + 1):
            worst = 0.0
            for sub in combinations(refs, k):
                pm = np.mean([pred_pair[(f, r)] for f in target for r in sub])
                om = np.mean([obs_pair[(f, r)] for f in target for r in sub])
                worst = max(worst, abs(pm - om))
            per_k[k] = round(worst, 4)
            if worst <= TOL and min_k is None:
                min_k = k
        # best single reference device
        single = {r: abs(np.mean([pred_pair[(f, r)] for f in target])
                         - np.mean([obs_pair[(f, r)] for f in target])) for r in refs}
        best_r = min(single, key=single.get)
        prov = ("involves-PRIOR" if any(PROVENANCE[c.split(' / ')[0]] == "PRIOR-context"
                for c in target) else "held-out-NEW")
        rows.append({
            "precision": precision, "held_out_family": fam,
            "held_out_provenance": prov, "n_target_channels": len(target),
            "n_reference_channels": len(refs),
            "pred_mean_rhoS_fullref": round(float(pred_full), 4),
            "obs_mean_rhoS_fullref": round(float(obs_full), 4),
            "abs_error_fullref": round(abs(pred_full - obs_full), 4),
            "min_reference_set_size_within_0.05": min_k,
            "worstcase_err_k1": per_k.get(1), "worstcase_err_k2": per_k.get(2),
            "best_single_reference": best_r,
            "best_single_reference_err": round(float(single[best_r]), 4),
        })
    return pd.DataFrame(rows), n


def main():
    tabs = {}
    for prec in ("fp32", "int8"):
        tab, n = run(prec)
        tabs[prec] = tab
        tabs[prec]["n"] = n
    full = pd.concat(tabs.values(), ignore_index=True)
    full.to_csv(OUT / "exp6_lofo.csv", index=False)
    show = ["held_out_family", "held_out_provenance", "pred_mean_rhoS_fullref",
            "obs_mean_rhoS_fullref", "abs_error_fullref",
            "min_reference_set_size_within_0.05", "worstcase_err_k1",
            "best_single_reference", "best_single_reference_err"]
    for prec in ("fp32", "int8"):
        print(f"\n=== EXP 6 LOFO — {prec} (n={tabs[prec].n.iloc[0]}) ===")
        print(tabs[prec][show].to_string(index=False))
    print("\n--- VERDICT ---")
    print(f"min_reference_set_size_within_0.05 reports the smallest #reference devices whose")
    print(f"WORST-CASE family mean-rho_S error stays <= {TOL}. A value of 1 means a single")
    print("already-characterised device suffices to predict a brand-new SoC family's fleet rank-")
    print("transfer to within 0.05 from its R2_factor alone. Kirin710 is the held-out stress case")
    print("(lowest R2 -> lowest transfer); provenance: SD888 = involves-PRIOR, others held-out-NEW.")
    print("\nSaved: exp6_lofo.csv")


if __name__ == "__main__":
    main()
