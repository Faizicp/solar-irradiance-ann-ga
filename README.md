# solar-irradiance-ann-ga

PyTorch implementation of a hybrid ANN–GA solar irradiance forecasting model, released
alongside the manuscript *"[paper title]"* (Data and Code Availability section).

## Scope

This repository reproduces the **PyTorch implementation results** subsection of the
manuscript — a full calendar-year (2019, hourly, N=8,760) NSRDB demonstration plus a
ten-day (May 5–14) legacy-baseline comparison. It does **not** currently contain the code
for the paper's main results: the eight-model benchmark suite (SVR, LSTM, Random Forest,
XGBoost, CNN-LSTM, ensembles), the multi-season/multi-weather cross-validation, the
Diebold–Mariano significance tests, the uncertainty-quantification ensemble, or the EV
charging case study. Those were run in a separate environment (see the manuscript,
Section 3.3, for hardware/software details) and are being prepared for release here
incrementally.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

See [`data/README.md`](data/README.md). The ten-day CSVs are included directly; the
177 MB full-year NSRDB extract is not (exceeds GitHub's 100 MB limit) — instructions for
obtaining it from NSRDB are in that file.

## Running

```bash
python3 train_ann_ga.py --nsrdb_csv data/2019-3032554-one_axis.csv
```

Outputs (plots + `ga_statistics.json`) are written to `results/`.

## Known limitations

- No random seed is currently fixed (GA population init/crossover/mutation, the
  train/val split, and network weight init are all unseeded), so point estimates vary
  run-to-run. The manuscript discloses this explicitly (Section 3.3).
- `main()` only builds the full-year `NSRDBDataset`; the `SolarDataset` class (built for
  the ten-day, time-of-day + day-index-only demonstration) and `compute_baseline_metrics()`
  (which reproduces the legacy SVR/LSTM/ANN-GA numbers quoted in the text) exist in the
  script but aren't currently wired into `main()`.

Issues and PRs welcome.
