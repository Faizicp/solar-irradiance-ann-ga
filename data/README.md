# Data

## `2019-3032554-one_axis.csv`

Full calendar year 2019, hourly NSRDB extract for the Peshawar, Pakistan grid cell
(34.01°N, 71.58°E), used by `train_ann_ga.py --nsrdb_csv ...` for the full-year
demonstration. The full file is 177 MB — too large for a single GitHub upload (100 MB
git limit, 25 MB web-uploader limit) — so it is split into 8 parts of ~21 MB each:
`2019-3032554-one_axis.part00.csvpart` ... `2019-3032554-one_axis.part07.csvpart`.

**To reconstruct the original file**, concatenate the parts in order:

```bash
cat 2019-3032554-one_axis.part0*.csvpart > 2019-3032554-one_axis.csv
```

(Verified byte-identical to the original via `cmp` before splitting.)

Alternatively, obtain it directly from NSRDB:

1. Create a free account at https://nsrdb.nrel.gov/
2. Open the NSRDB Data Viewer: https://nsrdb.nrel.gov/data-viewer
3. Enter coordinates 34.01, 71.58 and select **Physical Solar Model v3.2.2**,
   grid-cell identifier **3032554**, year **2019**.
4. Download the CSV and place it at `data/2019-3032554-one_axis.csv`.

## `day1.csv` ... `day10.csv`

Legacy ten-day (May 5–14) daytime-only records, each with four columns of
time-stamped irradiance: actual GHI plus pre-computed SVR, LSTM and an earlier
ANN-GA model's predictions. Used by `SolarDataset` and `compute_baseline_metrics()`
in `train_ann_ga.py`.

## `Scatter Final.csv`

Not read by any function in `train_ann_ga.py`. Kept for provenance; appears to be
a pooled Actual/ANN-GA/SVR/LSTM scatter export from an earlier analysis pass.
