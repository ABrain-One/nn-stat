"""
Communality-split figure: GPU vs CPU latent-speed communality (R2_factor) per phone,
fp32 and int8. Styled to match exp2_kk_curve.pdf / exp5_ladder.pdf.

Message: GPU channels are ~fully shared latent speed (communality ~0.95-1.0) -> rank
transfers; CPU channels (esp. fp32) carry channel-specific true signal (lower communality)
that does NOT transfer. Dumbbells show the CPU->GPU gap per device.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from exp_common import OUT, SOC

PHONES = list(SOC)  # locked order: SM-F926B, 23106RN0DA, Redmi Note 9 Pro, SM-P613, STK-L21
CPU_C, GPU_C = "tab:red", "tab:blue"
DODGE = 0.11  # horizontal offset of CPU/GPU markers around each phone's tick


def main():
    t = pd.read_csv(OUT / "exp1_latent_speed_R2.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    for ax, prec in zip(axes, ("fp32", "int8")):
        g = t[t.precision == prec].set_index("channel")
        x = np.arange(len(PHONES))
        cpu = np.array([g.loc[f"{ph} / CPU", "R2_factor"] for ph in PHONES])
        gpu = np.array([g.loc[f"{ph} / GPU", "R2_factor"] for ph in PHONES])

        # Dodge CPU/GPU slightly off-centre: at int8 the two communalities nearly coincide, and
        # on a shared x the markers would sit on top of each other and hide the CPU point.
        xc, xg = x - DODGE, x + DODGE
        for xci, xgi, c, gp in zip(xc, xg, cpu, gpu):
            ax.plot([xci, xgi], [c, gp], color="0.6", lw=2, zorder=1)
        ax.scatter(xc, cpu, marker="o", s=80, c=CPU_C, edgecolor="k", lw=0.5, zorder=3, label="CPU")
        ax.scatter(xg, gpu, marker="s", s=80, c=GPU_C, edgecolor="k", lw=0.5, zorder=3, label="GPU")

        # Channel-specific fraction (1 - communality) on the CPU points. Labels are the bare
        # percentage, centred UNDER the marker, so they never run into the neighbouring column;
        # the right-hand axis below tells the reader what the percentage means.
        for xi, c in zip(xc, cpu):
            ax.annotate(f"{(1 - c) * 100:.0f}%", (xi, c), textcoords="offset points",
                        xytext=(0, -11), fontsize=7.5, color=CPU_C, weight="bold",
                        ha="center", va="top", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

        ax.set_xticks(x)
        ax.set_xticklabels([f"{ph}\n({SOC[ph]})" for ph in PHONES], fontsize=7.5, rotation=20, ha="right")
        ax.set_title(f"Latent-speed communality $R^2_{{\\mathrm{{factor}}}}$ — {prec}")
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.6)

    axes[0].set_ylabel(r"communality $R^2_{\mathrm{factor}}$ (shared latent-speed fraction)")
    axes[0].set_ylim(0.6, 1.01)

    # Right-hand twin axis: the complement, 1 - R^2 = the device-specific fraction the CPU
    # labels report. Gives the "% device-specific" annotations their meaning without printing
    # the words next to every marker.
    tw = axes[1].twinx()
    tw.set_ylim(axes[1].get_ylim())
    ticks = [0.6, 0.7, 0.8, 0.9, 1.0]
    tw.set_yticks(ticks)
    tw.set_yticklabels([f"{(1 - t) * 100:.0f}%" for t in ticks], fontsize=8, color=CPU_C)
    tw.set_ylabel("device-specific fraction  $1-R^2$", color=CPU_C, fontsize=9)
    tw.tick_params(axis="y", colors=CPU_C, length=3)

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=GPU_C, markeredgecolor="k",
                      markersize=10, label="GPU channel"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=CPU_C, markeredgecolor="k",
                      markersize=10, label="CPU channel"),
               Line2D([0], [0], color="0.6", lw=2, label="CPU$\\to$GPU communality gap"),
               Line2D([0], [0], color="w", label="% under CPU marker = device-specific")]
    axes[1].legend(handles=handles, fontsize=8.5, loc="lower left", framealpha=0.95)

    fig.suptitle("Communality split: GPU latent speed is fleet-shared; CPU carries device-specific signal",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_communality_split.pdf")
    fig.savefig(OUT / "fig_communality_split.png", dpi=150)
    plt.close(fig)
    print("Saved:", OUT / "fig_communality_split.pdf", "and .png")
    # quick console echo of the split
    for prec in ("fp32", "int8"):
        g = t[t.precision == prec].set_index("channel")
        cpu = np.array([g.loc[f"{ph} / CPU", "R2_factor"] for ph in PHONES])
        gpu = np.array([g.loc[f"{ph} / GPU", "R2_factor"] for ph in PHONES])
        print(f"  {prec}: GPU communality {gpu.min():.3f}-{gpu.max():.3f}; "
              f"CPU communality {cpu.min():.3f}-{cpu.max():.3f} "
              f"(CPU device-specific {(1-cpu.max())*100:.0f}-{(1-cpu.min())*100:.0f}%)")


if __name__ == "__main__":
    main()
