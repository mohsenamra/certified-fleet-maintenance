# Certified fleet maintenance plans from learned failure predictions

Code and result summaries for the paper

> M. Amra, *Certified fleet maintenance plans from learned failure predictions: exact conformal calibration under shared capacity* (manuscript under review, 2026).

A fleet maintenance plan built on machine-learning failure predictions protects itself with a safety margin. This repository calibrates that margin on the outcome of the **executed plan**, with finite-sample, distribution-free guarantees that hold for any predictor:

* **Exact λ-path.** When the margin acts only through per-unit deadlines, the plan is piecewise constant in the margin. Conformal risk control (CRC) and Learn-then-Test (LTT) are computed exactly over a continuous margin, with one planner solve per distinct deadline vector.
* **Abstention.** An abstaining CRC controls the mean risk of certified plans; LTT bounds the probability that an issued certificate is wrong. When the target lies below what the predictor can support (the *floor*), the method abstains.
* **Online control.** A rolling-horizon controller with a long-run guarantee, and a cost-aware variant.

## Contents

| Folder | Case | What it does |
|---|---|---|
| `prototype`, `prototype_v1` | C-MAPSS | first prototypes; v1 builds the RUL predictors (bootstrap ensembles of gradient-boosted regressors) and the scenario pools |
| `prototype_v2` | C-MAPSS | exact λ-path, all calibration rules and baselines, online and cost-aware controllers (simple planner) |
| `prototype_v5` | C-MAPSS | richer planner (multi-period shop visits, spare engines, grouping), exactness checks, failure-cost ratio analysis |
| `prototype_v6` | N-CMAPSS | flight-level predictor, richer planner, cost-tuned and renewal-reward baselines, fault-mode shift |
| `backblaze_v2`, `backblaze_v3`, `backblaze_v4` | Backblaze Drive Stats | failure-risk predictors, EDF planner, CRC/LTT with abstention, forward splits, online rules, stronger predictor (v4) |
| `data` | all | download and preparation scripts (no data are stored here) |
| `paper` | all | scripts that rebuild the tables and figures of the paper from the result summaries |
| `PROTOCOLS.md` | | experiment protocols written before the corresponding runs |

Each experiment folder contains a `RUN_*.bat` file (Windows) with the exact order of the steps; on other systems run the listed Python scripts in the same order. Every `results` folder contains the summaries (`summary_*.txt` / `.json`) produced by the runs reported in the paper; large intermediate files (paths, predictors) are not stored and are recreated by the scripts.

## Data

* **C-MAPSS** and **N-CMAPSS**: NASA Prognostics Data Repository (Saxena et al., 2008; Arias Chao et al., 2021). `data/get_data.py` and `data/get_ncmapss.py` download them; `data/prep_ncmapss.py` aggregates N-CMAPSS to one record per flight. One N-CMAPSS sub-data set (DS08d) is excluded because its file in the public archive is truncated (see `data/ncmapss/skipped_DS08d-010.txt`).
* **Backblaze Drive Stats**: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data. `data/get_backblaze.py` and `data/get_backblaze_v4.py` download the quarters used and build per-model panels.

## Requirements

Python 3.10 or later and the packages in `requirements.txt` (NumPy, SciPy, pandas, scikit-learn, HiGHS through `highspy`, h5py, Matplotlib). The experiments of the paper ran on a laptop with 8 GB of memory: about 4.5 h (C-MAPSS, simple planner), 3.3 h (richer planner), 1.5 h (N-CMAPSS) and 7 h (Backblaze, predictors and evaluation).

## Reproducing the paper

1. Download the data with the scripts in `data`.
2. Run the experiment folders in the order above (`RUN_V1.bat`, `RUN_V2.bat`, `RUN_V5.bat`, `RUN_V6.bat`, `RUN_BACKBLAZE_V3.bat`, `RUN_BACKBLAZE_V3C.bat`, `RUN_BACKBLAZE_V4.bat`). All runs are seeded.
3. Rebuild tables and figures with the scripts in `paper` (usage is given at the top of each script).

## Citation

Please cite the paper (citation details will be added on publication); see `CITATION.cff`.

## License

MIT, see `LICENSE`.
