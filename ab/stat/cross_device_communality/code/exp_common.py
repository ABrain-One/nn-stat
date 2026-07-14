"""
Shared data prep for the cross-device latency rank-transfer experiments.

Locked design:
- Device channel = phone x backend, backends {cpu, gpu}; NPU EXCLUDED (GPU-fallback).
- Headline = GPU, fp32, shared models across all 5 phones. int8 = replication.
- Natural log of latency (ns). ns vs ms is an additive constant -> irrelevant to
  variance/Spearman/R²; absolute ms only for reporting.
- Cleaning (decision 7): drop valid=false, non-positive duration, and degenerate
  cells with duration <= 10_000 ns (0.01 ms) [empty band 0.01–0.1 ms separates the
  ~1,000 ns artifacts from real latencies].
- Noise estimator: log-space per-run variance via delta method (std_dev/duration)²,
  i.e. squared coefficient of variation. Reliability of the per-architecture MEAN
  (over m=20 internal runs) uses sigma2_run / m.

Provenance: SM-F926B = PRIOR-context (Z-Fold draft); others = NEW-unpublished.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
RAW = _REPO / "data" / "latency_multidevice_raw.csv"
OUT = _REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
M_RUNS = 20                 # internal benchmark_model iterations
DEGEN_NS = 10_000           # drop duration <= this (0.01 ms)
BACKENDS = ["cpu", "gpu"]   # NPU excluded

SOC = {
    "SM-F926B": "SD888",
    "23106RN0DA": "HelioMT6768",
    "Redmi Note 9 Pro": "SD720G",
    "SM-P613": "SD720G",
    "STK-L21": "Kirin710",
}
PROVENANCE = {d: ("PRIOR-context" if d == "SM-F926B" else "NEW-unpublished") for d in SOC}


def build_channel_long(precision: str) -> pd.DataFrame:
    """Long table: one row per (model, phone, backend) channel, cleaned.
    Columns: model_name, phone, soc, provenance, backend, channel,
             dur (ns mean), std (ns), mean_logT (ln ns), sigma2_run (log-space per-run var)."""
    df = pd.read_csv(RAW)
    df = df[(df.precision == precision) & (df.valid == True)].copy()
    rows = []
    for bk in BACKENDS:
        sub = df[(df[f"{bk}_dur"] > DEGEN_NS)].copy()
        sub = sub[sub[f"{bk}_std"].notna()]
        rec = pd.DataFrame({
            "model_name": sub.model_name,
            "phone": sub.device,
            "backend": bk,
            "dur": sub[f"{bk}_dur"].astype(float),
            "std": sub[f"{bk}_std"].astype(float),
        })
        rows.append(rec)
    long = pd.concat(rows, ignore_index=True)
    long["soc"] = long.phone.map(SOC)
    long["provenance"] = long.phone.map(PROVENANCE)
    long["channel"] = long.phone + " / " + long.backend.str.upper()
    long["mean_logT"] = np.log(long.dur)                      # natural log of mean latency (ns)
    long["sigma2_run"] = (long["std"] / long["dur"]) ** 2     # delta-method log-space per-run var
    return long


def shared_matrix(long: pd.DataFrame, channels: list[str]):
    """Return (M_logT, M_noise, models) for architectures present & clean on ALL given channels.
    M_logT: models x channels mean_logT; M_noise: models x channels sigma2_run."""
    piv_T = long.pivot_table(index="model_name", columns="channel", values="mean_logT")
    piv_N = long.pivot_table(index="model_name", columns="channel", values="sigma2_run")
    piv_T = piv_T[channels]
    piv_N = piv_N[channels]
    keep = piv_T.dropna().index.intersection(piv_N.dropna().index)
    return piv_T.loc[keep], piv_N.loc[keep], list(keep)


def cleaning_report(precision: str) -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df = df[df.precision == precision]
    rows = []
    for bk in BACKENDS:
        for dev in SOC:
            d = df[df.device == dev]
            n_total = len(d)
            n_invalid = int((d.valid != True).sum())
            v = d[d.valid == True]
            n_degen = int((v[f"{bk}_dur"] <= DEGEN_NS).sum() + (v[f"{bk}_dur"] <= 0).sum())
            n_kept = int((v[f"{bk}_dur"] > DEGEN_NS).sum())
            rows.append({"precision": precision, "phone": dev, "backend": bk,
                         "n_files": n_total, "dropped_invalid": n_invalid,
                         "dropped_degenerate": n_degen, "kept": n_kept})
    return pd.DataFrame(rows)
