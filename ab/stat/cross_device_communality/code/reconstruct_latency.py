"""
Reconstruct a tidy multi-device latency table from the committed TFLite
timing JSONs in nn-dataset/ab/nn/stat/run/tflite/{fp32,int8}/<task>/android_<dev>.json

Each JSON = one (model, device, precision) profiling result with, for each of
CPU/GPU/NPU: mean (=*_duration), min, max, std_dev over `iterations` internal
timing iterations. NO individual per-run values are stored. Latency unit = ns.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("NN_DATASET_TFLITE",
            str(Path(__file__).resolve().parents[1] / "data" / "nn_stat_tflite")))
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def soc_of(j: dict) -> str:
    try:
        return j["device_analytics"]["cpu_info"]["arm_architecture"]["hardware"]
    except Exception:
        return "NA"


def main() -> None:
    rows = []
    for prec_dir in ["fp32", "int8"]:
        for f in (ROOT / prec_dir).rglob("android_*.json"):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            task = f.parent.name  # e.g. img-classification_cifar-10_acc_<model>
            rec = {
                "precision": prec_dir,
                "task_dir": task,
                "model_name": j.get("model_name"),
                "device": j.get("device_type"),
                "soc": soc_of(j),
                "os_version": j.get("os_version"),
                "valid": j.get("valid"),
                "iterations": j.get("iterations"),
                "in0": j.get("in_dim_0"), "in1": j.get("in_dim_1"),
                "in2": j.get("in_dim_2"), "in3": j.get("in_dim_3"),
            }
            for u in ["cpu", "gpu", "npu"]:
                rec[f"{u}_dur"] = j.get(f"{u}_duration")
                rec[f"{u}_min"] = j.get(f"{u}_min_duration")
                rec[f"{u}_max"] = j.get(f"{u}_max_duration")
                rec[f"{u}_std"] = j.get(f"{u}_std_dev")
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "latency_multidevice_raw.csv", index=False)
    print(f"Parsed {len(df)} JSON timing files -> latency_multidevice_raw.csv")

    print("\n### device -> SoC map ###")
    print(df.groupby(["device", "soc"]).size().rename("n").reset_index().to_string(index=False))

    print("\n### iterations (internal timing loop) ###")
    print("unique:", sorted(df.iterations.dropna().unique()))

    print("\n### valid flag ###")
    print(df.valid.value_counts(dropna=False).to_dict())

    print("\n### coverage: rows per device x precision ###")
    print(pd.crosstab(df.device, df.precision, margins=True).to_string())

    # distinct models per device/precision
    print("\n### distinct models per device x precision ###")
    cov = df.groupby(["device", "precision"]).model_name.nunique().unstack()
    print(cov.to_string())

    # shared-model overlap across devices (per precision)
    print("\n### models present on ALL 5 devices (shared set) ###")
    for prec in ["fp32", "int8"]:
        sub = df[(df.precision == prec) & (df.valid == True)]
        sets = [set(g.model_name) for _, g in sub.groupby("device")]
        shared = set.intersection(*sets) if sets else set()
        print(f"  {prec}: shared models across all {len(sets)} devices = {len(shared)}")

    print("\n### std_dev availability (per-model timing spread) ###")
    for u in ["cpu", "gpu", "npu"]:
        nn = df[f"{u}_std"].notna().sum()
        pos = (df[f"{u}_dur"] > 0).sum()
        print(f"  {u}: dur>0 = {pos}, std_dev non-null = {nn}")

    print("\n### input resolution (side) distribution ###")
    df["side"] = df["in1"]
    print(df.side.value_counts(dropna=False).sort_index().to_dict())


if __name__ == "__main__":
    main()
