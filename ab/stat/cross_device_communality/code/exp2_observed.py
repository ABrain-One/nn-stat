"""
EXPERIMENT 2 (OBSERVED STEP) — compute observed Spearman rho_S on the GPU shared set
and compare against the PRE-REGISTERED predictions in predictions_locked.csv.

Runs ONLY after the lock file exists and was reported (it does; hash verified). This
script READS predictions_locked.csv and never rewrites it.

Outputs:
  exp2_observed_comparison.csv  (per-pair: R2s, pred FACTOR, obs, abs_error, boot CI)
  exp2_rank_order_check.csv     (pred-vs-obs Spearman across 10 pairs + Kirin/SD720G checks)
  exp2_kk_curve.{pdf,png}       (Kruskal-Kendall curve + observed pairs + pred-vs-obs panel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_common import OUT, SEED, build_channel_long, shared_matrix, SOC

ALL_CHANNELS = [f"{ph} / {bk.upper()}" for bk in ("cpu", "gpu") for ph in SOC]
N_BOOT = 5000
KIRIN_PHONE = "STK-L21"
SD720G_SAME = ("Redmi Note 9 Pro", "SM-P613")  # the same-family SD720G GPU pair


def kk_curve(rho_p: np.ndarray) -> np.ndarray:
    return (6.0 / np.pi) * np.arcsin(rho_p / 2.0)


def boot_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    n = len(x)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stats[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def run(precision: str, pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    long = build_channel_long(precision)
    M_logT, _, models = shared_matrix(long, ALL_CHANNELS)
    n = len(models)
    rng = np.random.default_rng(SEED)

    p = pred[pred.precision == precision].copy()
    rows = []
    for _, r in p.iterrows():
        c1, c2 = r.channel_1, r.channel_2
        x = M_logT[c1].to_numpy()
        y = M_logT[c2].to_numpy()
        rho_s = spearmanr(x, y).statistic
        rho_p = pearsonr(x, y).statistic
        lo, hi = boot_spearman(x, y, rng)
        pred_f = float(r.rho_S_pred_FACTOR)
        rows.append({
            "precision": precision,
            "pair": r.pair,
            "soc_1": r.soc_1, "soc_2": r.soc_2,
            "independence_level": r.independence_level,
            "provenance_pair": r.provenance_pair,
            "R2_factor_c1": r.R2_factor_c1, "R2_factor_c2": r.R2_factor_c2,
            "rho_S_pred_FACTOR": pred_f,
            "rho_S_obs": round(float(rho_s), 4),
            "rho_S_obs_CI_lo": round(lo, 4), "rho_S_obs_CI_hi": round(hi, 4),
            "rho_P_obs": round(float(rho_p), 4),
            "abs_error": round(abs(pred_f - float(rho_s)), 4),
            "signed_error_obs_minus_pred": round(float(rho_s) - pred_f, 4),
            "pred_in_CI": bool(lo <= pred_f <= hi),
            "n": n,
        })
    tab = pd.DataFrame(rows)

    # ---- aggregate error metrics ----
    mae = float(tab.abs_error.mean())
    maxerr = float(tab.abs_error.max())
    maxerr_pair = tab.loc[tab.abs_error.idxmax(), "pair"]
    cov = int(tab.pred_in_CI.sum())

    # ---- rank-order check across the 10 pairs ----
    rho_rank = spearmanr(tab.rho_S_pred_FACTOR, tab.rho_S_obs).statistic

    # Kirin pairs (any pair involving STK-L21) observed lowest?
    tab["is_kirin"] = tab.pair.str.contains(KIRIN_PHONE)
    obs_sorted = tab.sort_values("rho_S_obs")
    n_kirin = int(tab.is_kirin.sum())
    bottom_k = obs_sorted.head(n_kirin)
    kirin_lowest = bool(bottom_k.is_kirin.all())

    # SD720G same-family pair highest among non-Kirin?
    same_mask = tab.pair.str.contains(SD720G_SAME[0]) & tab.pair.str.contains(SD720G_SAME[1])
    non_kirin = tab[~tab.is_kirin]
    sd720_obs = float(tab[same_mask].rho_S_obs.iloc[0])
    sd720_highest_nonkirin = bool(sd720_obs >= non_kirin.rho_S_obs.max() - 1e-9)

    summary = {
        "precision": precision, "n": n, "mae": mae, "max_error": maxerr,
        "max_error_pair": maxerr_pair, "pred_in_CI_count": cov,
        "rank_order_spearman_pred_vs_obs": float(rho_rank),
        "kirin_pairs_observed_lowest": kirin_lowest,
        "sd720g_samefamily_highest_among_nonkirin": sd720_highest_nonkirin,
        "sd720g_samefamily_obs": sd720_obs,
        "nonkirin_max_obs": float(non_kirin.rho_S_obs.max()),
    }
    return tab, summary


def plot(fp32: pd.DataFrame, int8: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel 1: KK curve rho_S = (6/pi) arcsin(rho_P/2) + diagonal, observed pairs overlaid
    rp = np.linspace(0.80, 1.0, 400)
    ax[0].plot(rp, kk_curve(rp), "k-", lw=2, label=r"KK identity $\rho_S=\frac{6}{\pi}\arcsin(\rho_P/2)$")
    ax[0].plot(rp, rp, "k--", lw=1, alpha=0.5, label="diagonal $\\rho_S=\\rho_P$")
    for tab, mk, c, lab in [(fp32, "o", "tab:blue", "fp32"), (int8, "s", "tab:orange", "int8")]:
        ax[0].scatter(tab.rho_P_obs, tab.rho_S_obs, marker=mk, c=c, s=45,
                      edgecolor="k", lw=0.4, alpha=0.85, label=f"observed ({lab})")
    ax[0].set_xlabel(r"observed Pearson $\rho_P$ (mean log-latency)")
    ax[0].set_ylabel(r"observed Spearman $\rho_S$")
    ax[0].set_title("Observed pairs vs Kruskal–Kendall curve (GPU)")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3)

    # Panel 2: predicted (FACTOR) vs observed Spearman, identity line
    lim = [0.90, 1.001]
    ax[1].plot(lim, lim, "k--", lw=1, alpha=0.6, label="identity")
    for tab, mk, c, lab in [(fp32, "o", "tab:blue", "fp32"), (int8, "s", "tab:orange", "int8")]:
        yerr = np.vstack([tab.rho_S_obs - tab.rho_S_obs_CI_lo,
                          tab.rho_S_obs_CI_hi - tab.rho_S_obs])
        ax[1].errorbar(tab.rho_S_pred_FACTOR, tab.rho_S_obs, yerr=yerr, fmt=mk, c=c,
                       ms=6, capsize=2, elinewidth=0.8, mec="k", mew=0.4, label=lab)
    ax[1].set_xlabel(r"predicted $\rho_S$ (FACTOR route, pre-registered)")
    ax[1].set_ylabel(r"observed $\rho_S$ (95% bootstrap CI)")
    ax[1].set_title("Pre-registered prediction vs observation")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "exp2_kk_curve.pdf")
    fig.savefig(OUT / "exp2_kk_curve.png", dpi=150)
    plt.close(fig)


def main():
    pred = pd.read_csv(OUT / "predictions_locked.csv")
    out_tabs, summaries = [], []
    for prec in ["fp32", "int8"]:
        tab, summ = run(prec, pred)
        out_tabs.append(tab)
        summaries.append(summ)
    full = pd.concat(out_tabs, ignore_index=True)
    full.drop(columns=["is_kirin"]).to_csv(OUT / "exp2_observed_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "exp2_rank_order_check.csv", index=False)
    plot(out_tabs[0], out_tabs[1])

    show = ["pair", "soc_1", "soc_2", "independence_level", "provenance_pair",
            "rho_S_pred_FACTOR", "rho_S_obs", "rho_S_obs_CI_lo", "rho_S_obs_CI_hi",
            "abs_error", "pred_in_CI"]
    for tab, summ in zip(out_tabs, summaries):
        prec = summ["precision"]
        print(f"\n=== EXP 2 OBSERVED — {prec} (GPU, n={summ['n']}) ===")
        print(tab[show].to_string(index=False))
        print(f"  MAE={summ['mae']:.4f}  max|err|={summ['max_error']:.4f} ({summ['max_error_pair']})"
              f"  pred-in-CI {summ['pred_in_CI_count']}/10")
        print(f"  rank-order Spearman(pred,obs) over 10 pairs = {summ['rank_order_spearman_pred_vs_obs']:.4f}")
        print(f"  Kirin pairs observed lowest? {summ['kirin_pairs_observed_lowest']}")
        print(f"  SD720G same-family highest among non-Kirin? {summ['sd720g_samefamily_highest_among_nonkirin']}"
              f"  ({summ['sd720g_samefamily_obs']:.4f} vs non-Kirin max {summ['nonkirin_max_obs']:.4f})")
    print(f"\nSaved: exp2_observed_comparison.csv, exp2_rank_order_check.csv, exp2_kk_curve.(pdf|png)")


if __name__ == "__main__":
    main()
