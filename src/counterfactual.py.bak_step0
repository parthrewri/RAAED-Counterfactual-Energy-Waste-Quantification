import numpy as np
import pandas as pd


GROUPS = ("global", "hour", "hour_weekend", "hour_week")


def routine_group_key(datetimes, group="hour"):
    """Map timestamps to a routine cell. Fewer cells = more samples per cell.

    global: 1 cell | hour: 24 | hour_weekend: 48 | hour_week: 168
    """
    dt = pd.to_datetime(datetimes)
    if group == "global":
        return pd.Series(0, index=dt.index)
    if group == "hour":
        return dt.dt.hour
    if group == "hour_weekend":
        return dt.dt.hour + 24 * (dt.dt.dayofweek >= 5).astype(int)
    if group == "hour_week":
        return dt.dt.dayofweek * 24 + dt.dt.hour
    raise ValueError(f"group must be one of {GROUPS}")


def robust_upper_limit(values, z=3.5):
    """median + z * scaled MAD, computed on SIGNED residuals.

    Residuals must NOT be clipped at zero first: about 59% of residuals are
    <= 0, so clipping forces median = MAD = 0 and a near-zero threshold.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(max(0.0, med + z * 1.4826 * mad) + 1e-6)


def percentile_upper_limit(values, percentile=97.5):
    """Empirical upper percentile of SIGNED residuals (distribution-free)."""
    if not 50.0 < float(percentile) < 100.0:
        raise ValueError("percentile must be greater than 50 and less than 100")
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(max(0.0, np.quantile(x, percentile / 100.0)) + 1e-6)


def _upper_limit(values, method="percentile", z=3.5, percentile=97.5):
    if method == "percentile":
        return percentile_upper_limit(values, percentile=percentile)
    if method == "mad":
        return robust_upper_limit(values, z=z)
    raise ValueError("method must be 'percentile' or 'mad'")


def calibrate_routine_thresholds(
    validation_residuals, z=3.5, method="percentile", percentile=97.5,
    group="hour", min_group_samples=30,
):
    """Calibrate high-side residual thresholds using validation data only.

    Defaults (percentile 97.5, 24 hour-of-day cells) are PROVISIONAL and are
    to be compared against alternatives in the threshold/ablation experiments.
    """
    required = {"datetime", "residual_kwh"}
    missing = required.difference(validation_residuals.columns)
    if missing:
        raise ValueError(f"validation_residuals is missing columns: {sorted(missing)}")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")

    df = validation_residuals.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["residual_kwh"] = pd.to_numeric(df["residual_kwh"], errors="coerce")
    df = df.dropna(subset=["datetime", "residual_kwh"])
    if df.empty:
        raise ValueError("No valid validation residuals were provided.")

    df["cell"] = routine_group_key(df["datetime"], group)
    global_limit = _upper_limit(df["residual_kwh"], method=method, z=z, percentile=percentile)

    limits, group_counts = {}, {}
    for key, cell in df.groupby("cell"):
        count = int(cell["residual_kwh"].notna().sum())
        group_counts[int(key)] = count
        if count >= min_group_samples:
            limits[int(key)] = _upper_limit(
                cell["residual_kwh"], method=method, z=z, percentile=percentile
            )
        else:
            limits[int(key)] = global_limit

    return {
        "limits": limits, "global_limit": float(global_limit),
        "z": float(z), "method": method, "percentile": float(percentile),
        "group": group, "min_group_samples": int(min_group_samples),
        "group_counts": group_counts,
    }


def detect_and_quantify(counterfactuals, calibration):
    """Flag hours above threshold and quantify only the amount above it."""
    required = {"datetime", "residual_kwh"}
    missing = required.difference(counterfactuals.columns)
    if missing:
        raise ValueError(f"counterfactuals is missing columns: {sorted(missing)}")

    df = counterfactuals.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["residual_kwh"] = pd.to_numeric(df["residual_kwh"], errors="coerce")
    dt = df["datetime"]
    group = calibration.get("group", "hour_week")
    df["hour_week"] = dt.dt.dayofweek * 24 + dt.dt.hour
    df["routine_cell"] = routine_group_key(dt, group)
    global_limit = float(calibration.get("global_limit", 0.0))
    df["routine_threshold_kwh"] = (
        df["routine_cell"].map(calibration.get("limits", {})).fillna(global_limit)
    )
    df["positive_excess_kwh"] = df["residual_kwh"].clip(lower=0).fillna(0.0)
    df["excess_above_threshold_kwh"] = (
        df["positive_excess_kwh"] - df["routine_threshold_kwh"]
    ).clip(lower=0.0)
    df["is_excess_event_hour"] = df["excess_above_threshold_kwh"] > 0.0
    df["quantified_excess_kwh"] = df["excess_above_threshold_kwh"]

    # Bounded severity prevents tiny thresholds from exploding the event score.
    denominator = np.maximum(df["routine_threshold_kwh"].to_numpy(dtype=float), 0.1)
    severity = df["excess_above_threshold_kwh"].to_numpy(dtype=float) / denominator
    df["severity"] = np.clip(severity, 0.0, 3.0)
    df["cewi_hour"] = df["quantified_excess_kwh"] * (1.0 + df["severity"])
    return df


def attach_events(detected):
    df = detected.sort_values("datetime").copy()
    flags = df["is_excess_event_hour"].fillna(False).to_numpy(dtype=bool)
    times = pd.to_datetime(df["datetime"])
    event_ids, current, previous_time, active = [], 0, None, False
    for flag, timestamp in zip(flags, times):
        consecutive = (
            previous_time is not None
            and (timestamp - previous_time).total_seconds() <= 3600
        )
        if flag:
            if not active or not consecutive:
                current += 1
            event_ids.append(current)
            active = True
        else:
            event_ids.append(np.nan)
            active = False
        previous_time = timestamp
    df["event_id"] = event_ids
    return df


def routine_category_profiles(hourly_train):
    df = hourly_train.copy()
    df["hour_week"] = df.index.dayofweek * 24 + df.index.hour
    categories = ["sub1_kwh", "sub2_kwh", "sub3_kwh", "unmetered_kwh"]
    missing = set(categories).difference(df.columns)
    if missing:
        raise ValueError(f"hourly_train is missing category columns: {sorted(missing)}")
    profiles = df.groupby("hour_week")[categories].median()
    global_profiles = df[categories].median()
    return {"profiles": profiles, "global": global_profiles, "categories": categories}


def attribute_event_hour(row, profiles):
    """Attribute positive category deviations versus each category's routine profile."""
    timestamp = pd.Timestamp(row["datetime"])
    key = timestamp.dayofweek * 24 + timestamp.hour
    base = profiles["profiles"].loc[key] if key in profiles["profiles"].index else profiles["global"]
    actual = row.get("_hourly_values", {})
    names = {
        "sub1_kwh": "Kitchen", "sub2_kwh": "Laundry room",
        "sub3_kwh": "Water heating / air conditioning",
        "unmetered_kwh": "Other / unmetered loads",
    }
    increments = {
        names[c]: max(0.0, float(actual.get(c, 0.0)) - float(base[c]))
        for c in profiles["categories"]
    }
    total = float(sum(increments.values()))
    shares = {k: (v / total * 100.0 if total > 0 else 0.0) for k, v in increments.items()}
    dominant = max(increments, key=increments.get) if total > 0 else "No category-level excess resolved"
    return {
        "category_excess_kwh": increments,
        "category_shares_pct": shares,
        "largest_positive_contributor": dominant,
        "attributed_excess_total_kwh": total,
    }


def summarize_events(detected):
    df = attach_events(detected)
    flagged = df[df["is_excess_event_hour"]].copy()
    columns = [
        "event_id", "start", "end", "duration_hours", "excess_energy_kwh",
        "CEWI", "max_hourly_excess_kwh",
    ]
    if flagged.empty:
        return df, pd.DataFrame(columns=columns)
    events = flagged.groupby("event_id").agg(
        start=("datetime", "min"), end=("datetime", "max"),
        duration_hours=("datetime", "count"),
        excess_energy_kwh=("quantified_excess_kwh", "sum"),
        CEWI=("cewi_hour", "sum"),
        max_hourly_excess_kwh=("quantified_excess_kwh", "max"),
    ).reset_index()
    return df, events