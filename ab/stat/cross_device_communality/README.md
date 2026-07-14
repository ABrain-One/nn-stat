# Cross-device latency rank-transfer — reproducibility repository

Code, derived data, pre-registration lock files, and figures for the
Neurocomputing paper on predicting cross-device latency **rank** transfer from
single-device signal-reliability, via the Kruskal–Kendall (KK) identity

```
rho_S  =  (6/pi) * arcsin( sqrt( R2_{s,d1} * R2_{s,d2} ) / 2 )
```

where `R2_{s,d}` is the communality (shared-signal fraction) of device–channel
`d` from a one-factor latent-speed model of log-latency across architectures.

The headline uses an **internal** 5-SoC phone benchmark (CPU + GPU channels,
fp32 and int8); an **external** replication uses the public nn-Meter dataset.
All predictions were **pre-registered**: KK predictions were written to
immutable lock files and SHA-256-hashed *before* any observed Spearman was
computed. See `HASHES.txt` and [Verifying the pre-registration](#verifying-the-pre-registration).

---

## Repository layout

```
latency-rank-transfer-repro/
├── README.md            this file
├── LICENSE              CC BY 4.0
├── requirements.txt     pinned dependencies
├── HASHES.txt           SHA-256 of the four lock files (sha256sum -c)
├── code/                all analysis + figure scripts
├── data/                revealed CSVs behind every table/figure
│   └── nnmeter/         derived external-replication CSVs
│   └── nnmeter_raw/     (empty) drop nn-Meter *.jsonl here to regenerate external results
├── locks/               the four immutable pre-registration lock CSVs
├── figures/             final PDF + PNG figure files
└── outputs/             (empty) scripts write regenerated results here
```

`data/` and `figures/` hold the **canonical, revealed** artifacts behind the
paper. Re-running the scripts writes fresh copies into `outputs/`, which should
match (the lock files reproduce bit-for-bit — see below).

---

## Dependencies

Tested with **Python 3.10.12**. Key packages (pinned in `requirements.txt`):

| package         | version |
|-----------------|---------|
| numpy           | 2.2.6   |
| pandas          | 2.3.3   |
| scipy           | 1.15.3  |
| matplotlib      | 3.10.8  |
| factor_analyzer | 0.5.1   |
| pingouin        | 0.6.1   |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Figures use the headless `Agg` backend, so no display is required.

---

## Paper element → script → data map

Every table and figure maps to exactly one script and its input/output CSV(s).
Scripts read the shipped `data/` and `locks/` files and write to `outputs/`.

### Internal 5-SoC benchmark

| Paper element | Script (`code/`) | Data / figure |
|---|---|---|
| Raw multi-device latency table (5 SoCs × CPU/GPU, fp32+int8) | `reconstruct_latency.py` † | `data/latency_multidevice_raw.csv` |
| Per-channel communality `R2_{s,c}` (factor + reliability routes); Appendix A factor fit | `exp1_latent_speed.py` | `data/exp1_latent_speed_R2.csv`, `data/exp1_cleaning_report.csv` |
| CPU-vs-GPU communality split figure | `fig_communality_split.py` | `figures/fig_communality_split.{pdf,png}` |
| Pre-registered KK predictions (10 GPU pairs) | `exp2_lock.py` | `locks/predictions_locked.csv` |
| Observed ρ_S, per-pair bootstrap CIs, KK calibration curve/table | `exp2_observed.py` | `data/exp2_observed_comparison.csv`, `data/exp2_rank_order_check.csv`, `figures/exp2_kk_curve.{pdf,png}` |
| §5 baseline comparison (arcsin vs. constant / reliability / Pearson) | `baseline_comparison.py` | `outputs/baseline_comparison.csv` |
| Bivariate-normality diagnostics (Henze–Zirkler / Mardia) | `exp4_normality.py` | `data/exp4_normality.csv`, `figures/exp4_normality_{fp32,int8}.{pdf,png}` |
| Mechanism / residual-correlation (independence-violation) ladder | `exp5_ladder.py` | `data/exp5_ladder.csv`, `data/exp5_ladder_rung_means.csv`, `figures/exp5_ladder.{pdf,png}` |
| Leave-one-family-out generalization | `exp6_lofo.py` | `data/exp6_lofo.csv` |
| Calibration-size subsampling stability (n=50/100/200) | `subsample_communality.py` | `data/subsample_communality.csv`, `data/subsample_communality_summary.csv` |

### External replication (nn-Meter, Appendix B)

| Paper element | Script (`code/`) | Data / figure |
|---|---|---|
| Pooled communalities (4 nn-Meter devices) | `nnmeter_exp1_communality.py` ‡ | `data/nnmeter/nnmeter_communality.csv`, `data/nnmeter/nnmeter_cleaning_report.csv` |
| Pooled KK lock | `nnmeter_lock.py` | `locks/predictions_locked_nnmeter.csv` |
| Pooled observed (Simpson-artifact diagnosis) | `nnmeter_observed.py` ‡ | `data/nnmeter/nnmeter_observed_comparison.csv` |
| Within-space VPU diagnostic | `nnmeter_within_space.py` ‡ | `data/nnmeter/nnmeter_within_space_vpu.csv` |
| Within-space re-lock | `nnmeter_relock_withinspace.py` ‡ | `locks/predictions_locked_nnmeter_withinspace.csv`, `data/nnmeter/nnmeter_withinspace_communality.csv` |
| Within-space reveal | `nnmeter_reveal_withinspace.py` ‡ | `data/nnmeter/nnmeter_withinspace_reveal.csv` |
| Held-out (blind 50/50 split) re-lock | `nnmeter_heldout_lock.py` ‡ | `locks/predictions_locked_nnmeter_heldout.csv`, `data/nnmeter/nnmeter_heldout_split.csv`, `data/nnmeter/nnmeter_heldout_fit_communality.csv` |
| Held-out reveal | `nnmeter_heldout_reveal.py` ‡ | `data/nnmeter/nnmeter_heldout_reveal.csv` |
| External replication figure | `fig_external_replication.py` | `figures/fig_external_replication.{pdf,png}` |

† `reconstruct_latency.py` parses the upstream ABrain `nn-dataset` TFLite timing
JSON tree. That tree is **not** bundled (thousands of files); set
`NN_DATASET_TFLITE=/path/to/nn-dataset/ab/nn/stat/run/tflite` to regenerate.
The output it produces — `data/latency_multidevice_raw.csv` — **is** shipped, so
you can reproduce everything downstream without it.

‡ These scripts read the raw nn-Meter `*.jsonl` files (~1.8 GB, not bundled).
Download `datasets.zip` from the official nn-Meter release
(<https://github.com/microsoft/nn-Meter>) and unzip the per-search-space
`*.jsonl` files into `data/nnmeter_raw/` (see the README there). The **derived**
CSVs in `data/nnmeter/` are shipped, so every reported external number is
inspectable without the download.

---

## Reproduce the results

From the repository root, with the environment active:

```bash
# --- internal benchmark (needs only shipped data/) ---
python code/exp1_latent_speed.py        # communalities  -> outputs/
python code/exp2_lock.py                # KK predictions -> outputs/predictions_locked.csv
python code/exp2_observed.py            # observed + bootstrap CIs + calibration fig
python code/baseline_comparison.py      # Section 5 baseline table
python code/exp4_normality.py           # normality diagnostics + figs
python code/exp5_ladder.py              # residual-correlation ladder + fig
python code/exp6_lofo.py                # leave-one-family-out
python code/subsample_communality.py    # subsampling stability
python code/fig_communality_split.py    # CPU/GPU communality figure

# --- external replication (needs data/nnmeter_raw/*.jsonl downloaded) ---
python code/nnmeter_exp1_communality.py
python code/nnmeter_lock.py
python code/nnmeter_observed.py
python code/nnmeter_within_space.py
python code/nnmeter_relock_withinspace.py
python code/nnmeter_reveal_withinspace.py
python code/nnmeter_heldout_lock.py
python code/nnmeter_heldout_reveal.py
python code/fig_external_replication.py  # reads shipped data/nnmeter/, no download needed
```

Run `exp2_lock.py` before `exp2_observed.py` (observed compares against the lock
regenerated in `outputs/`). External lock scripts likewise precede their reveal
scripts. `reconstruct_latency.py` is optional (its output is shipped).

Seeds are fixed (`SEED = 42`), so bootstrap CIs and random subsamples reproduce
exactly.

---

## Verifying the pre-registration

The four lock CSVs in `locks/` are the sealed KK predictions. Their SHA-256
hashes are printed in the paper and listed in `HASHES.txt`:

```bash
sha256sum -c HASHES.txt        # run from the repository root
```

- `locks/predictions_locked.csv` → `e0979c3c…35b57ee` (paper §5)
- `locks/predictions_locked_nnmeter_heldout.csv` → `fabac5a2…c118f7ec` (paper §9, held-out)
- plus the two intermediate external locks (pooled, within-space).

Regenerating the internal predictions reproduces the lock **bit-for-bit**:
`python code/exp2_lock.py` then
`sha256sum outputs/predictions_locked.csv` yields
`e0979c3c11c127833c7e0eee5d1aebd9777373ae24bd0e7e20c32371335b57ee`.
The reveal scripts hash-check the sealed lock in `locks/` before computing any
observed value.

---

## Data notes

- **Internal benchmark:** 5 phones → SoCs {Snapdragon 888, Helio MT6768,
  Snapdragon 720G ×2, Kirin 710}, CIFAR-10 image-classification models. Latency
  is nanoseconds, mean over 20 internal timing iterations per (model, device,
  precision). Analysis channels = phone × {CPU, GPU}; **NPU is excluded** (it is
  a GPU fallback on these devices). Complete-case n = 491 (fp32) / 503 (int8).
- **External benchmark:** nn-Meter per-search-space latency for 4 real devices
  (Pixel4 CPU, Mi9 Adreno-640 GPU, Pixel3XL Adreno-630 GPU, Myriad VPU).

## Citation

This repository accompanies the paper:

> Faraz Kayani, Sarmad Kayani, Radu Timofte, Dmitry Ignatov.
> "Cross-Device Neural Architecture Latency Rank Transfer from Device
> Communality: Theory and Validation." Manuscript under review, 2026.

A full citation (venue and DOI) will be added here upon publication.

## License

CC BY 4.0 — see `LICENSE`. External nn-Meter data remains under its own license.
