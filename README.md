# RAAED — Counterfactual Energy Waste Quantification

## Research objective
Estimate unusually high household electricity use relative to a learned routine-aware counterfactual baseline, quantify excess energy in kWh, and identify which available sub-metering category most contributed to the event.

## Important scientific boundaries
- The UCI dataset is one household in Sceaux, France, measured every minute from December 2006 to November 2010. It contains three sub-metering channels. It is not a multi-household sample.
- Sub-meter 1 represents kitchen loads; sub-meter 2 represents laundry-room loads; sub-meter 3 represents an electric water-heater and air-conditioner. These are categories, not appliance-level labels.
- This project estimates excess relative to a modelled baseline. Excess is not automatically proven avoidable waste, a malfunction, or user negligence.
- Synthetic anomaly results measure recovery under controlled injected events only. They must not be reported as real-world anomaly accuracy.
- CEWI is a project-proposed operational index, not an established scientific metric. Its usefulness must be evaluated against baselines.
- Novelty cannot be guaranteed without a systematic literature review. Frame the contribution as a proposed and evaluated framework, not as the first-ever method.

## Windows / VS Code setup
Open this folder in VS Code. In PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
```powershell
python run_pipeline.py
streamlit run app.py
```

First run downloads the UCI ZIP file. If automatic download fails, download the dataset from https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption and put the ZIP in `data/raw/`.

## Research method
1. Convert minute-average global active power (kW) to hourly kWh by summing each minute's kW / 60.
2. Convert sub-metering Wh to hourly kWh by summing Wh and dividing by 1000.
3. Train a one-hour-ahead forecasting model using chronological train/validation/test splits.
4. Calibrate a high-side residual threshold for each hour-of-week using validation residuals and robust median absolute deviation (MAD).
5. Mark unusually high hours when actual minus counterfactual forecast exceeds the routine-aware threshold.
6. Quantify excess energy for flagged hours as positive actual-minus-counterfactual energy, summed across each event.
7. Compare sub-meter category values with category-specific hour-of-week historical baselines to estimate the largest positive contributor.
8. Run a controlled synthetic injection experiment on held-out test data to evaluate whether injected excess is detected and quantified.

## Proposed Counterfactual Energy Waste Index (CEWI)
For an event containing flagged hours t:
- ExcessEnergy = sum(max(0, Actual_t - Counterfactual_t)) kWh.
- Severity_t = max(0, Residual_t - RoutineThreshold_t) / max(RoutineThreshold_t, epsilon).
- CEWI = ExcessEnergy * (1 + mean(Severity_t)).

CEWI is a ranking/triage index, not a physical measurement of avoidable waste. Report ExcessEnergy_kWh separately as the interpretable quantity.

## Files
- `src/data.py`: UCI download and hourly aggregation.
- `src/features.py`: leakage-aware forecasting features and chronological splits.
- `src/modeling.py`: model comparison and counterfactual forecast.
- `src/counterfactual.py`: routine thresholds, event grouping, CEWI, attribution.
- `src/experiments.py`: controlled injection and baseline comparison.
- `app.py`: Streamlit research dashboard.
- `run_pipeline.py`: runs end-to-end pipeline.

## Suggested 3-day priorities
Day 1: dataset, preprocessing, forecasting, validation.
Day 2: counterfactual thresholding, event quantification, attribution, injection tests.
Day 3: dashboard, graphs, results interpretation, literature review, report and viva.
