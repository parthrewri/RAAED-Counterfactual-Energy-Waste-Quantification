from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_hourly
from src.counterfactual import (
    calibrate_routine_thresholds,
    detect_and_quantify,
    routine_category_profiles,
    attribute_event_hour,
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

CATEGORY_NAMES = {
    "sub1_kwh": "Kitchen",
    "sub2_kwh": "Laundry room",
    "sub3_kwh": "Water heating / air conditioning",
    "unmetered_kwh": "Other / unmetered loads",
}


def validate_unmodified_attribution(hourly, valid, test):
    """Check how well positive sub-meter deviations align with quantified excess."""
    cal = calibrate_routine_thresholds(
        valid, method="percentile", percentile=97.5, group="hour"
    )
    det = detect_and_quantify(test, cal)
    flagged = det[det["is_excess_event_hour"]].copy()

    # Strict chronological split: build category profiles using data up to validation end
    profiles = routine_category_profiles(hourly.loc[: valid["datetime"].max()].copy())

    rows = []
    for _, r in flagged.iterrows():
        ts = pd.Timestamp(r["datetime"])
        if ts in hourly.index:
            h_row = hourly.loc[ts]
        else:
            idx = hourly.index.get_indexer([ts], method="nearest")[0]
            h_row = hourly.iloc[idx]

        attr = attribute_event_hour(
            {"datetime": ts, "_hourly_values": h_row.to_dict()}, profiles
        )
        rows.append(
            {
                "datetime": ts,
                "positive_excess_kwh": float(r["positive_excess_kwh"]),
                "quantified_excess_kwh": float(r["quantified_excess_kwh"]),
                "attributed_excess_total_kwh": float(attr["attributed_excess_total_kwh"]),
                "largest_positive_contributor": attr["largest_positive_contributor"],
            }
        )

    df = pd.DataFrame(rows)
    abs_diff_vs_pos = (df["attributed_excess_total_kwh"] - df["positive_excess_kwh"]).abs()
    abs_diff_vs_quant = (df["attributed_excess_total_kwh"] - df["quantified_excess_kwh"]).abs()

    summary = {
        "flagged_hours_evaluated": int(len(df)),
        "mean_positive_excess_kwh": round(float(df["positive_excess_kwh"].mean()), 4),
        "mean_quantified_excess_kwh": round(float(df["quantified_excess_kwh"].mean()), 4),
        "mean_attributed_total_kwh": round(float(df["attributed_excess_total_kwh"].mean()), 4),
        "mae_attributed_vs_positive_kwh": round(float(abs_diff_vs_pos.mean()), 4),
        "mae_attributed_vs_quantified_kwh": round(float(abs_diff_vs_quant.mean()), 4),
        "corr_attributed_vs_positive": round(
            float(df["attributed_excess_total_kwh"].corr(df["positive_excess_kwh"])), 4
        ),
    }
    return df, summary


def validate_controlled_category_injection(hourly, valid, test, fraction=0.05, seed=42):
    """Inject known excess into a specific sub-meter and test Top-1 attribution accuracy."""
    profiles = routine_category_profiles(hourly.loc[: valid["datetime"].max()].copy())
    test_reset = test.copy().reset_index(drop=True)
    n = len(test_reset)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    injected_idx = np.sort(rng.choice(n, size=k, replace=False))
    categories = list(CATEGORY_NAMES.keys())

    records = []
    for i, idx in enumerate(injected_idx):
        row = test_reset.iloc[idx]
        ts = pd.Timestamp(row["datetime"])
        if ts in hourly.index:
            h_vals = hourly.loc[ts].to_dict()
        else:
            nearest = hourly.index.get_indexer([ts], method="nearest")[0]
            h_vals = hourly.iloc[nearest].to_dict()

        # Rotate evenly across all 4 categories
        target_col = categories[i % len(categories)]
        target_name = CATEGORY_NAMES[target_col]
        increment = max(2.0, float(row["actual_kwh"]) * 0.75)

        injected_vals = dict(h_vals)
        injected_vals[target_col] = float(injected_vals.get(target_col, 0.0)) + increment

        attr = attribute_event_hour(
            {"datetime": ts, "_hourly_values": injected_vals}, profiles
        )
        predicted_name = attr["largest_positive_contributor"]

        records.append(
            {
                "datetime": ts,
                "injected_category": target_name,
                "predicted_category": predicted_name,
                "injected_kwh": round(increment, 4),
                "correct": predicted_name == target_name,
            }
        )

    res_df = pd.DataFrame(records)
    overall_acc = float(res_df["correct"].mean()) * 100.0
    per_cat = (
        res_df.groupby("injected_category")["correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    per_cat["accuracy_pct"] = (per_cat["mean"] * 100.0).round(2)
    per_cat = per_cat[["injected_category", "count", "accuracy_pct"]]
    return res_df, round(overall_acc, 2), per_cat


def main():
    valid_path = REPORTS / "validation_residuals.csv"
    test_path = REPORTS / "test_counterfactuals.csv"
    if not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` first.")

    hourly = load_hourly()
    valid = pd.read_csv(valid_path, parse_dates=["datetime"])
    test = pd.read_csv(test_path, parse_dates=["datetime"])

    print("[1/2] Unmodified Flagged-Hour Attribution Consistency (hour, p=97.5)...")
    unmod_df, unmod_summary = validate_unmodified_attribution(hourly, valid, test)
    print("Summary:", unmod_summary)
    print("\nDominant category counts on unmodified flagged hours:")
    print(unmod_df["largest_positive_contributor"].value_counts().to_string())
    unmod_df.to_csv(REPORTS / "unmodified_attribution_audit.csv", index=False)

    print("\n[2/2] Controlled Category-Injection Attribution Accuracy (5% test hours, seed=42)...")
    inj_df, overall_acc, per_cat = validate_controlled_category_injection(
        hourly, valid, test
    )
    print(f"Overall Top-1 Attribution Accuracy: {overall_acc:.2f}% (random chance = 25.00%)")
    print("\nPer-category accuracy:")
    print(per_cat.to_string(index=False))
    inj_df.to_csv(REPORTS / "category_injection_validation.csv", index=False)
    print("\nSaved audit CSVs to reports/.")


if __name__ == "__main__":
    main()