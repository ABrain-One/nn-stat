"""
FIGURE — external replication (nn-Meter), held-out blind test.

Standalone regenerator for fig_external_replication.{pdf,png}: predicted vs.
observed Spearman rho_S for the 3 VPU device-pairs across the 11 within-space
held-out TEST halves (n=33). Reads the shipped reveal CSV; recomputes nothing.

Input:  data/nnmeter/nnmeter_heldout_reveal.csv
Output: outputs/fig_external_replication.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REVEAL = REPO / "data" / "nnmeter" / "nnmeter_heldout_reveal.csv"
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rev = pd.read_csv(REVEAL).rename(columns={"rho_S_obs_TEST": "rho_S_obs"})
    colors = {"Pixel4 / CPU": "tab:red", "Pixel3XL / GPU": "tab:blue", "Mi9 / GPU": "tab:green"}
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lim = [0.74, 1.0]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="identity")
    for p, c in colors.items():
        s = rev[rev.channel_partner == p]
        ax.scatter(s.rho_S_pred, s.rho_S_obs, c=c, s=55, edgecolor="k", lw=0.4, alpha=0.85,
                   label=p.replace(" / ", "/") + r" $\times$ VPU")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"FIT-locked predicted $\rho_S$ (communality$\to$KK)")
    ax.set_ylabel(r"observed $\rho_S$ (held-out TEST, blind)")
    ax.set_title("External replication (nn-Meter): VPU pairs, 11 spaces")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    mae = rev.abs_error.mean()
    ax.text(0.97, 0.04, f"MAE = {mae:.3f}  (n={len(rev)}, held-out TEST)", transform=ax.transAxes,
            ha="right", fontsize=8.5, bbox=dict(boxstyle="round", fc="w", ec="0.7"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_external_replication.pdf")
    fig.savefig(OUT / "fig_external_replication.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_external_replication.pdf'} (MAE={mae:.3f}, n={len(rev)})")


if __name__ == "__main__":
    main()
