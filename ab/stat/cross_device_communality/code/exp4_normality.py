"""
EXPERIMENT 4 — bivariate-normality diagnostics for the Kruskal-Kendall identity.

The identity rho_S = (6/pi) arcsin(rho_P/2) assumes the (logT_c1, logT_c2) pair is
bivariate normal. We test that assumption on representative GPU pairs:
  - cross-family, high transfer : SM-F926B x 23106RN0DA  (SD888 x HelioMT6768)
  - cross-family, low  transfer : SM-F926B x STK-L21     (SD888 x Kirin710)
  - same-family SD720G          : Redmi Note 9 Pro x SM-P613

Tests: Henze-Zirkler (pingouin) + Mardia multivariate skewness/kurtosis.
Plots: chi-square Q-Q of Mahalanobis distances + BVN density-contour scatter.
Ceiling/floor truncation check: fraction of points within 1% of marginal min/max.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_common import OUT, SOC, build_channel_long, shared_matrix

ALL_CHANNELS = [f"{ph} / {bk.upper()}" for bk in ("cpu", "gpu") for ph in SOC]

PAIRS = [
    ("SM-F926B / GPU", "23106RN0DA / GPU", "cross-family-high (SD888 x HelioMT6768)"),
    ("SM-F926B / GPU", "STK-L21 / GPU", "cross-family-low (SD888 x Kirin710)"),
    ("Redmi Note 9 Pro / GPU", "SM-P613 / GPU", "same-family-SD720G"),
]


def mardia(X: np.ndarray) -> dict:
    """Mardia multivariate skewness & kurtosis with test statistics. X: n x p."""
    n, p = X.shape
    xbar = X.mean(0)
    S = np.cov(X, rowvar=False, bias=True)          # MLE covariance (divide by n)
    Sinv = np.linalg.inv(S)
    Xc = X - xbar
    D = Xc @ Sinv @ Xc.T                              # n x n matrix of mahalanobis-type products
    b1p = (D ** 3).sum() / (n ** 2)                   # multivariate skewness
    b2p = (np.diag(D) ** 2).sum() / n                 # multivariate kurtosis
    # skewness test: (n/6) b1p ~ chi2 with df = p(p+1)(p+2)/6
    df = p * (p + 1) * (p + 2) / 6
    skew_stat = (n / 6.0) * b1p
    skew_p = float(chi2.sf(skew_stat, df))
    # kurtosis test: (b2p - p(p+2)) / sqrt(8 p(p+2)/n) ~ N(0,1)
    kurt_z = (b2p - p * (p + 2)) / np.sqrt(8.0 * p * (p + 2) / n)
    kurt_p = float(2 * norm.sf(abs(kurt_z)))
    return {"mardia_skew": float(b1p), "mardia_skew_stat": float(skew_stat),
            "mardia_skew_p": skew_p, "mardia_kurt": float(b2p),
            "mardia_kurt_z": float(kurt_z), "mardia_kurt_p": kurt_p}


def truncation(col: np.ndarray, tol_frac=0.01) -> dict:
    rng = col.max() - col.min()
    near_lo = float(np.mean(col <= col.min() + tol_frac * rng))
    near_hi = float(np.mean(col >= col.max() - tol_frac * rng))
    return {"frac_near_min": near_lo, "frac_near_max": near_hi}


def run(precision: str):
    long = build_channel_long(precision)
    M_logT, _, models = shared_matrix(long, ALL_CHANNELS)
    n = len(models)
    rows = []
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(5 * len(PAIRS), 9))
    for k, (c1, c2, label) in enumerate(PAIRS):
        X = M_logT[[c1, c2]].to_numpy()
        # Henze-Zirkler
        hz = pg.multivariate_normality(X, alpha=0.05)
        md = mardia(X)
        t1 = truncation(X[:, 0]); t2 = truncation(X[:, 1])
        rows.append({
            "precision": precision, "pair": f"{c1} x {c2}", "type": label, "n": n,
            "HZ_stat": round(float(hz.hz), 4), "HZ_p": round(float(hz.pval), 4),
            "HZ_normal": bool(hz.normal),
            "mardia_skew": round(md["mardia_skew"], 4),
            "mardia_skew_p": round(md["mardia_skew_p"], 4),
            "mardia_kurt": round(md["mardia_kurt"], 4),
            "mardia_kurt_z": round(md["mardia_kurt_z"], 3),
            "mardia_kurt_p": round(md["mardia_kurt_p"], 4),
            "frac_near_min_c1": round(t1["frac_near_min"], 4),
            "frac_near_max_c1": round(t1["frac_near_max"], 4),
            "frac_near_min_c2": round(t2["frac_near_min"], 4),
            "frac_near_max_c2": round(t2["frac_near_max"], 4),
        })
        # --- chi2 Q-Q of Mahalanobis distances ---
        xbar = X.mean(0); S = np.cov(X, rowvar=False, bias=True)
        d2 = np.einsum("ij,jk,ik->i", X - xbar, np.linalg.inv(S), X - xbar)
        d2s = np.sort(d2)
        q = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=2)
        ax = axes[0, k]
        ax.scatter(q, d2s, s=8, alpha=0.5)
        m = max(q.max(), d2s.max())
        ax.plot([0, m], [0, m], "r--", lw=1)
        ax.set_title(f"{label}\nchi2(2) Q-Q [{precision}]", fontsize=9)
        ax.set_xlabel("theoretical chi2(2) quantile"); ax.set_ylabel("ordered Mahalanobis $d^2$")
        # --- BVN contour scatter ---
        ax2 = axes[1, k]
        ax2.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(X.T)
            xi = np.linspace(X[:, 0].min(), X[:, 0].max(), 60)
            yi = np.linspace(X[:, 1].min(), X[:, 1].max(), 60)
            XX, YY = np.meshgrid(xi, yi)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax2.contour(XX, YY, ZZ, levels=6, colors="tab:red", linewidths=0.7, alpha=0.8)
        except Exception:
            pass
        ax2.set_xlabel(c1, fontsize=8); ax2.set_ylabel(c2, fontsize=8)
        ax2.set_title("log-latency joint + KDE contours", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"exp4_normality_{precision}.pdf")
    fig.savefig(OUT / f"exp4_normality_{precision}.png", dpi=140)
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    tabs = [run(p) for p in ("fp32", "int8")]
    full = pd.concat(tabs, ignore_index=True)
    full.to_csv(OUT / "exp4_normality.csv", index=False)
    show = ["pair", "type", "HZ_stat", "HZ_p", "HZ_normal",
            "mardia_skew_p", "mardia_kurt_z", "mardia_kurt_p",
            "frac_near_min_c1", "frac_near_max_c1"]
    for prec, tab in zip(("fp32", "int8"), tabs):
        print(f"\n=== EXP 4 NORMALITY — {prec} ===")
        print(tab[show].to_string(index=False))
    # verdict
    print("\n--- VERDICT ---")
    strict = full.HZ_normal.any()
    print(f"Strict HZ normality holds on any pair? {strict}")
    print("HZ/Mardia typically REJECT at n~500 (high power vs tiny departures). Read the")
    print("Q-Q plots: near-linear up the body with mild tail curvature => approx-BVN, KK valid")
    print("as an approximation. Truncation fractions near 0 => no ceiling/floor clipping.")
    print(f"\nSaved: exp4_normality.csv, exp4_normality_(fp32|int8).(pdf|png)")


if __name__ == "__main__":
    main()
