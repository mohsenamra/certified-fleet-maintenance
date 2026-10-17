# Experiment protocols

These protocols were written in the author's project log **before** the corresponding runs and were not changed after the results were seen. Any deviation from a fixed reading is reported in the paper rather than tuned away.

## P1. Backblaze, stronger predictor (`backblaze_v4`), written 16 October 2026

* **Goal.** A stronger failure-risk predictor. The guarantees do not depend on the predictor; only the floor and the cost do.
* **Data.** Q1 2024 to Q2 2026, the same four drive models, all reported SMART attributes (raw and normalised), per-model panels (`data/get_backblaze_v4.py`).
* **Unchanged evaluation.** Training decision days up to 15 October 2025 (labels up to 12 November 2025), evaluation from 15 November 2025. The planning model, costs, capacities, targets, 100 random-day splits, forward splits and all rules (CRC, LTT with Hoeffding–Bentkus p-values and δ = 0.1, validation tuning, model-based rule, RAIDShield-style rules, online rules) are unchanged. The baseline predictor (v3b/v3c) is reported alongside.
* **Predictor candidates, chosen on training data only.** A: the baseline features on the longer training window. B: extended features (all reported SMART attributes, raw and normalised, changes over 3, 7, 14 and 30 days, age). Both use one histogram gradient-boosting classifier per horizon, bagged over five seeds, made monotone over horizons by a running maximum.
* **Selection.** Internal validation on decision days 1 August to 15 October 2025, with a fit on earlier days whose labels do not reach the validation window. B is chosen only if its mean validation AUC over the four horizons exceeds that of A by more than 0.005. The chosen candidate is refitted on all training days. The choice is made and logged per drive model.
* **Added analyses.** LTT certification as a function of the calibration size n ∈ {30, 60, 90, 120} days; failure-cost sensitivity c_fail ∈ {10, 30, 100} (costs recomputed from stored counts); run time of the path computation.

## P2. Richer planner on C-MAPSS (`prototype_v5`), written 17 October 2026

* **Planner** (`p4v5_lib.RichFleetMILP`, HiGHS, relative gap 0, one thread). K = 20 engines, P = 6 periods of L = 10 cycles. An in-house shop visit occupies one of C lines for D = 2 periods. S0 serviceable spare engines; removed engines return after D (in-house) or D_out = 3 (external) periods; missing spares are leased at 40 per engine-period. A setup cost of 100 is paid in every period with at least one in-house start (grouping). Lost useful life costs 1 per cycle (at the predicted life), an external visit 300, a failure 1,000. The margin enters only through the admissible periods and deferral, so the exact path applies.
* **Checks before the run.** MILP optimum equal to complete enumeration in 200 of 200 small random instances (K = 4, P = 3).
* **Predictions** from the v1 predictors (unchanged). **Scenarios** with a new seed (2027): RICH_ID (in-distribution pool, 1,000 scenarios, C = 2, S0 = 3), RICH_TIGHT (400, C = 1, S0 = 1), S13 and S24 source and target pools (400 each, C = 2, S0 = 3).
* **Exactness.** 200 random (scenario, margin family, margin) direct solves per setting must all agree with the path.
* **Analysis.** The rules, margin families and hindsight grid of `prototype_v2` unchanged. RICH_ID: failure-fraction targets 0.02 and 0.05 (CRC), any-failure targets 0.1 and 0.2 (LTT, δ = 0.1), n = 50 and 100 calibration scenarios, 500 random splits (the rest is the test set). RICH_TIGHT: n = 100, 300 splits. Shift: source to source, naive source to target, recalibration on n = 25 and 50 target scenarios, 300 splits.
* **Reading fixed in advance.** The claim that the calibration layer does not depend on the simple planner holds if (a) exactness is 100%, (b) the CRC mean risk is at most α and the share of LTT splits above α is at most δ + 0.02 in every in-distribution cell, and (c) the σ-rules, the model-based chance constraint and validation tuning behave as with the simple planner, or the paper reports where they do not. The online experiments stay with the simple planner.

## P3. N-CMAPSS (`prototype_v6`), written 17 October 2026, before any planning result

* **Data.** All ten sub-data sets of N-CMAPSS, development and test units alike, because one sub-data set has only 9 to 15 units. Per unit and flight: mean and standard deviation of the 14 measured sensors, mean of the four flight-condition descriptors, flight class, share of the flight in the healthy state, remaining life in flights (`data/prep_ncmapss.py`).
* **Predictor** (`p4v6_lib`). Ridge regression of each sensor on the flight conditions (with squares, pairwise products and the flight class), fitted on healthy flights of the training units; residual window features (mean of the last 3 flights, mean and slope of the last 10) with the flight index and flight class; five bootstrap gradient-boosted regressors, remaining life capped at 80 flights; mean and standard deviation of the ensemble.
* **Settings** (seed 2028). NC_ID: all units, five-fold cross-fitting grouped by unit and stratified by sub-data set; 1,000 fleets of K = 10 distinct units, each observed at a random flight after at least 10 flights. NC_SHIFT: source = sub-data sets with turbine faults (DS01, DS02, DS03, DS07), cross-fitted; target = sub-data sets with fan and compressor faults (DS04, DS05, DS06), predicted by a model trained on all source units; 400 source and 400 target fleets.
* **Planner.** The richer planner with K = 10, P = 6 periods of L = 5 flights, C = 1, D = 2, D_out = 3, S0 = 2, lost life 2 per flight, external visit 300, failure 1,000, setup 100, lease 40. Exactness: 100 direct solves per setting.
* **Rules.** All rules of P2 plus two baselines from the closest fleet planning studies: a cost-tuned additive margin (the margin with the lowest mean calibration cost, no risk constraint; the safety buffer of de Pater et al., 2022, is tuned for cost) and renewal-reward replacement times (Mitici et al., 2023: per engine, the time minimising [c_p + (c_c − c_p) F(t)] / E[min(T, t)] under N(r̂, max(σ, 0.5)²) truncated at zero, c_p = 300, c_c = 1,000), planned with the same planner.
* **Analysis and reading.** As in P2 (targets, splits, 300 replications), with the same criteria fixed in advance.

### Deviation recorded after the data step

One sub-data set (DS08d) could not be read: its file in the public archive is 32 bytes shorter than its internal header states. It was excluded rather than repaired; 99 units from nine sub-data sets remain. DS08d contains all fault modes and was used only in NC_ID, so the shift design is unaffected.
