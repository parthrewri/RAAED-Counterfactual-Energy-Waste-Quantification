import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

from src.counterfactual import calibrate_routine_thresholds, detect_and_quantify


def run_controlled_injection(test_cf, calibration, fraction=0.05, seed=42):
    """Inject known excess; evaluate detection and incremental quantification."""
    required = {"datetime", "actual_kwh", "baseline_kwh", "residual_kwh"}
    missing = required.difference(test_cf.columns)
    if missing:
        raise ValueError(f"test_cf is missing columns: {sorted(missing)}")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be greater than 0 and less than 1")

    original = test_cf.copy().reset_index(drop=True)
    n = len(original)
    if n < 2:
        raise ValueError("At least two test observations are required.")
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    injected_idx = np.sort(rng.choice(n, size=k, replace=False))

    # Establish pre-existing excess before injecting synthetic consumption.
    original_detected = detect_and_quantify(original, calibration)
    injected = original.copy()
    injected["actual_original_kwh"] = injected["actual_kwh"]
    increments = np.maximum(
        2.0, injected.loc[injected_idx, "actual_kwh"].to_numpy(dtype=float) * 0.75
    )
    injected.loc[injected_idx, "actual_kwh"] = (
        injected.loc[injected_idx, "actual_kwh"].to_numpy(dtype=float) + increments
    )
    injected.loc[injected_idx, "residual_kwh"] = (
        injected.loc[injected_idx, "actual_kwh"].to_numpy(dtype=float)
        - injected.loc[injected_idx, "baseline_kwh"].to_numpy(dtype=float)
    )
    injected["synthetic_truth"] = False
    injected.loc[injected_idx, "synthetic_truth"] = True

    detected = detect_and_quantify(injected, calibration)
    detected["synthetic_truth"] = injected["synthetic_truth"].astype(bool)
    y_true = detected["synthetic_truth"].astype(int)
    y_pred = detected["is_excess_event_hour"].astype(int)

    # Remove pre-existing quantified excess to isolate the injected increment.
    before = original_detected.loc[injected_idx, "quantified_excess_kwh"].to_numpy(dtype=float)
    after = detected.loc[injected_idx, "quantified_excess_kwh"].to_numpy(dtype=float)
    estimated_increment = np.maximum(0.0, after - before)
    quant_mae = mean_absolute_error(increments, estimated_increment)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    metrics = {
        "synthetic_injection_fraction": float(fraction),
        "synthetic_injected_hours": int(k),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "injected_excess_quantification_MAE_kWh": float(quant_mae),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "threshold_method": calibration.get("method", "mad"),
        "threshold_percentile": calibration.get("percentile"),
        "interpretation": "Controlled synthetic stress test only; not real-world anomaly accuracy.",
    }
    return detected, metrics


def run_threshold_sweep(
    validation_residuals, test_cf, percentiles=(95.0, 97.5, 99.0),
    fraction=0.05, seed=42,
):
    """Compare percentile thresholds calibrated on validation data."""
    rows = []
    for percentile in percentiles:
        calibration = calibrate_routine_thresholds(
            validation_residuals, method="percentile", percentile=float(percentile)
        )
        _, metrics = run_controlled_injection(
            test_cf, calibration, fraction=fraction, seed=seed
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        ["f1", "injected_excess_quantification_MAE_kWh"],
        ascending=[False, True],
    ).reset_index(drop=True)
