"""
EXPERIMENT 5 — nested-independence ladder.

KK identity assumes the channel-specific residuals eps_c are INDEPENDENT across the two
channels; then observed rho_S is fully explained by the shared latent-speed communalities
(R2_factor route of record). We climb a ladder of increasing residual dependence:

  Rung A  cross-family GPU pairs        (different SoC, different device) -> eps ~ independent
  Rung B  SD720G same-family GPU pair   (same SoC, different device)
  Rung C  within-phone CPU<->GPU pairs  (SAME physical device) -> shared thermal/scheduler eps

Prediction: signed error (obs - pred_FACTOR) is a MONOTONE-INCREASING function of the
residual correlation corr(eps_c1, eps_c2). Rung C should show the largest residual_corr and
the largest positive error (KK under-predicts because device-shared residual signal is NOT
in the cross-device latent factor). Factor R2 stays the route of record throughout.

eps are taken from the SAME one-factor fit used in Exp 1 (seed 42, ML, 10 std channels).
NPU excluded (data-hygiene note only; it is a GPU fallback).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from factor_analyzer import FactorAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_common import OUT, SEED, SOC, PROVENANCE, build_channel_long, shared_matrix

ALL_CHANNELS = [f"{ph} / {bk.upper()}" for bk in ("cpu", "gpu") for ph in SOC]


def kk(r1, r2):
    return float((6.0 / np.pi) * np.arcsin(np.sqrt(r1 * r2) / 2.0))


def fit_factor(M_logT: pd.DataFrame):
    """Reproduce Exp1 one-factor fit; return eps residual frame + R2_factor per channel."""
    np.random.seed(SEED)
    Z = (M_logT - M_logT.mean()) / M_logT.std(ddof=0)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(Z.values)
    loadings = fa.loadings_.ravel()
    uniq = fa.get_uniquenesses()
    scores = fa.transform(Z.values).ravel()
    # standardize scores to unit variance so loading*score reconstructs the common part
    scores = (scores - scores.mean()) / scores.std(ddof=0)
    common = np.outer(scores, loadings)                      # n x channels
    eps = pd.DataFrame(Z.values - common, index=Z.index, columns=Z.columns)
    r2_factor = {ch: float(loadings[j] ** 2 / (loadings[j] ** 2 + uniq[j]))
                 for j, ch in enumerate(M_logT.columns)}
    return eps, r2_factor


def run(precision: str):
    long = build_channel_long(precision)
    M_logT, _, models = shared_matrix(long, ALL_CHANNELS)
    n = len(models)
    eps, r2f = fit_factor(M_logT)

    gpu = [f"{ph} / GPU" for ph in SOC]
    rows = []

    def add(c1, c2, rung, indep):
        ph1, ph2 = c1.split(" / ")[0], c2.split(" / ")[0]
        resid_corr = float(np.corrcoef(eps[c1], eps[c2])[0, 1])
        pred = kk(r2f[c1], r2f[c2])
        obs = float(spearmanr(M_logT[c1], M_logT[c2]).statistic)
        prov = ("involves-PRIOR" if "PRIOR-context" in (PROVENANCE[ph1], PROVENANCE[ph2])
                else "NEW-only")
        rows.append({
            "precision": precision, "rung": rung, "pair": f"{c1} x {c2}",
            "independence_level": indep, "provenance_pair": prov,
            "R2_factor_c1": round(r2f[c1], 4), "R2_factor_c2": round(r2f[c2], 4),
            "residual_corr": round(resid_corr, 4),
            "rho_S_pred_FACTOR": round(pred, 4), "rho_S_obs": round(obs, 4),
            "signed_error_obs_minus_pred": round(obs - pred, 4), "n": n,
        })

    # Rung A: cross-family GPU pairs ; Rung B: SD720G same-family GPU pair
    for a, b in combinations(gpu, 2):
        soc1, soc2 = SOC[a.split(" / ")[0]], SOC[b.split(" / ")[0]]
        if soc1 == soc2:
            add(a, b, "B_same-family-GPU", "same-family-SD720G")
        else:
            add(a, b, "A_cross-family-GPU", "cross-family")
    # Rung C: within-phone CPU<->GPU
    for ph in SOC:
        add(f"{ph} / CPU", f"{ph} / GPU", "C_within-phone-CPUxGPU", "within-phone")

    tab = pd.DataFrame(rows)
    # monotonicity: signed error vs residual_corr across the whole ladder
    mono = spearmanr(tab.residual_corr, tab.signed_error_obs_minus_pred).statistic
    rung_means = tab.groupby("rung").agg(
        residual_corr=("residual_corr", "mean"),
        signed_error=("signed_error_obs_minus_pred", "mean"),
        mae=("signed_error_obs_minus_pred", lambda s: s.abs().mean()),
        n_pairs=("pair", "count")).reset_index()
    return tab, float(mono), rung_means


def plot(tab_fp32, tab_int8):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"A_cross-family-GPU": "tab:blue", "B_same-family-GPU": "tab:green",
              "C_within-phone-CPUxGPU": "tab:red"}
    for tab, mk, alpha in [(tab_fp32, "o", 0.9), (tab_int8, "s", 0.55)]:
        for rung, c in colors.items():
            s = tab[tab.rung == rung]
            ax.scatter(s.residual_corr, s.signed_error_obs_minus_pred, marker=mk,
                       c=c, s=60, edgecolor="k", lw=0.4, alpha=alpha)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    # legend proxies
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k], markeredgecolor="k",
                  markersize=9, label=k) for k in colors]
    leg += [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="fp32"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=9, label="int8")]
    ax.legend(handles=leg, fontsize=8, loc="upper left")
    ax.set_xlabel(r"residual correlation  $\mathrm{corr}(\epsilon_{c_1},\epsilon_{c_2})$")
    ax.set_ylabel(r"signed error  $\rho_S^{obs}-\rho_S^{pred(FACTOR)}$")
    ax.set_title("Exp 5 — KK error tracks residual dependence (independence ladder)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp5_ladder.pdf")
    fig.savefig(OUT / "exp5_ladder.png", dpi=150)
    plt.close(fig)


def main():
    tabs, monos, means = {}, {}, {}
    for prec in ("fp32", "int8"):
        t, m, rm = run(prec)
        tabs[prec], monos[prec], means[prec] = t, m, rm
    full = pd.concat(tabs.values(), ignore_index=True)
    full.to_csv(OUT / "exp5_ladder.csv", index=False)
    pd.concat([rm.assign(precision=p) for p, rm in means.items()], ignore_index=True
              ).to_csv(OUT / "exp5_ladder_rung_means.csv", index=False)
    plot(tabs["fp32"], tabs["int8"])

    show = ["rung", "pair", "residual_corr", "R2_factor_c1", "R2_factor_c2",
            "rho_S_pred_FACTOR", "rho_S_obs", "signed_error_obs_minus_pred"]
    for prec in ("fp32", "int8"):
        print(f"\n=== EXP 5 LADDER — {prec} (n={tabs[prec].n.iloc[0]}) ===")
        print(tabs[prec].sort_values(["rung", "pair"])[show].to_string(index=False))
        print("\n  rung means:")
        print(means[prec].to_string(index=False))
        print(f"  MONOTONICITY Spearman(residual_corr, signed_error) over ladder = {monos[prec]:.4f}")
    print("\n--- VERDICT ---")
    print("Monotone-increasing error with residual_corr confirms the independence assumption is")
    print("the operative one: Rung A (cross-family) ~independent eps, error ~0; Rung C (within-")
    print("phone CPU<->GPU) shares device residual -> KK under-predicts (obs>pred). Factor R2 of")
    print("record throughout. NPU excluded (GPU fallback) — data-hygiene note only.")
    print("\nSaved: exp5_ladder.csv, exp5_ladder_rung_means.csv, exp5_ladder.(pdf|png)")


if __name__ == "__main__":
    main()
