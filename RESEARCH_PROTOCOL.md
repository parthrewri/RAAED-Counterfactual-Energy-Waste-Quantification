# Research protocol and claims discipline

## Research gap to investigate
Prior studies already combine household/building energy anomaly detection and forecasting. The project should not claim novelty merely from combining these components. The proposed narrower contribution is a reproducible workflow that (i) learns hour-of-week counterfactual consumption, (ii) applies robust high-side residual calibration, (iii) reports event-level excess energy in kWh, and (iv) estimates category-level contributions from sub-meter deviations, with controlled injection tests for quantification.

## Hypotheses
H1: Hour-of-week robust thresholds produce a different and potentially more useful alert profile than one global threshold.
H2: The method can estimate the magnitude of controlled injected excess with measurable error.
H3: Category-level deviations can identify the injected sub-meter category more often than chance in controlled tests.
H4: CEWI can rank events differently from excess energy alone by including normalized threshold exceedance severity.

## Baselines
- Persistence forecasting: next-hour usage equals previous-hour usage.
- Global residual threshold: one high-side threshold calibrated on validation residuals.
- Routine-aware threshold: hour-of-week thresholds calibrated on validation residuals.
- Excess-only ranking: rank events by estimated excess kWh.
- CEWI ranking: rank events by excess kWh multiplied by a severity adjustment.

## Required reporting
- Forecast MAE, RMSE and R² on a chronological test split.
- Alert rate on unmodified test data (not precision/recall; natural anomaly labels are absent).
- Synthetic-injection precision, recall, F1, and excess-quantification MAE.
- Attribution accuracy only when the synthetic injection includes a known sub-meter category label.
- Report all random seeds and injection magnitudes.
- Never describe synthetic labels as real-world ground truth.

## Known limitations
- One household, historical period, France.
- No appliance-level ground truth.
- No independent household validation.
- Consumption above a statistical baseline is not automatically avoidable waste.
- Weather, occupancy, tariff, appliance states and user intent are not present.
- CEWI is a proposed event-ranking index and requires validation against expert-labelled or intervention data.
