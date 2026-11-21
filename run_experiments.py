from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.counterfactual import (
    GROUPS,
    calibrate_routine_thresholds,
    detect_and_quantify,
    summarize_events,
)
from src.experiments import run_controlled_injection

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def compute_baseline_quantification_mae(test_cf, calibration, fraction=0.05, seed=42):
    """Compare injected excess against above-baseline positive residual (for flagged hours)."""
    original = test_cf.copy().reset_index(drop=True)
    n = len(original)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    injected_idx = np.sort(rng.choice(n, size=k, replace=False))

    original_det = detect_and_quantify(original, calibration)
    injected = original.copy()
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
    injected_det = detect_and_quantify(injected, calibration)

    # Above-baseline excess counted only when the hour is flagged
    before_base = np.where(
        original_det.loc[injected_idx, "is_excess_event_hour"],
        original_det.loc[injected_idx, "positive_excess_kwh"],
        0.0,
    )
    after_base = np.where(
        injected_det.loc[injected_idx, "is_excess_event_hour"],
        injected_det.loc[injected_idx, "positive_excess_kwh"],
        0.0,
    )
    est_base_increment = np.maximum(0.0, after_base - before_base)
    return float(mean_absolute_error(increments, est_base_increment))


def run_ablation_matrix(valid, test, fraction=0.05, seed=42):
    configs = [
        ("mad", 3.5, None),
        ("percentile", None, 95.0),
        ("percentile", None, 97.5),
        ("percentile", None, 99.0),
    ]
    rows = []
    for group in GROUPS:
        for method, z, pct in configs:
            cal = calibrate_routine_thresholds(
                valid,
                z=3.5 if z is None else z,
                method=method,
                percentile=97.5 if pct is None else pct,
                group=group,
            )
            det = detect_and_quantify(test, cal)
            _, events = summarize_events(det)

            flagged_hours = int(det["is_excess_event_hour"].sum())
            alert_rate_pct = 100.0 * flagged_hours / len(det)
            total_excess_kwh = float(det["quantified_excess_kwh"].sum())

            _, syn_metrics = run_controlled_injection(
                test, cal, fraction=fraction, seed=seed
            )
            base_mae = compute_baseline_quantification_mae(
                test, cal, fraction=fraction, seed=seed
            )

            rows.append(
                {
                    "group": group,
                    "method": method,
                    "param": f"z={z}" if method == "mad" else f"p={pct}",
                    "unmodified_flagged_hours": flagged_hours,
                    "unmodified_alert_rate_pct": round(alert_rate_pct, 2),
                    "unmodified_events": len(events),
                    "unmodified_excess_kwh": round(total_excess_kwh, 2),
                    "syn_precision": round(syn_metrics["precision"], 4),
                    "syn_recall": round(syn_metrics["recall"], 4),
                    "syn_f1": round(syn_metrics["f1"], 4),
                    "mae_above_threshold_kwh": round(
                        syn_metrics["injected_excess_quantification_MAE_kWh"], 4
                    ),
                    "mae_above_baseline_kwh": round(base_mae, 4),
                }
            )
    return pd.DataFrame(rows)


def run_cewi_ranking_analysis(valid, test):
    cal = calibrate_routine_thresholds(
        valid, method="percentile", percentile=97.5, group="hour"
    )
    det = detect_and_quantify(test, cal)
    _, events = summarize_events(det)

    if events.empty:
        return events, {}

    df = events.copy()
    df["rank_excess"] = df["excess_energy_kwh"].rank(ascending=False, method="min").astype(int)
    df["rank_cewi"] = df["CEWI"].rank(ascending=False, method="min").astype(int)
    df["rank_shift"] = df["rank_excess"] - df["rank_cewi"]

    # Spearman correlation via Pearson correlation of ranks (no scipy required)
    spearman = float(df["rank_excess"].corr(df["rank_cewi"]))
    changed = int((df["rank_shift"] != 0).sum())
    max_shift = int(df["rank_shift"].abs().max())

    summary = {
        "total_events": int(len(df)),
        "spearman_rank_corr": round(spearman, 4),
        "events_reranked": changed,
        "pct_events_reranked": round(100.0 * changed / len(df), 2),
        "max_rank_shift": max_shift,
    }
    df = df.sort_values("rank_cewi").reset_index(drop=True)
    return df, summary


def main():
    valid_path = REPORTS / "validation_residuals.csv"
    test_path = REPORTS / "test_counterfactuals.csv"
    if not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` first to generate residual CSVs.")

    valid = pd.read_csv(valid_path, parse_dates=["datetime"])
    test = pd.read_csv(test_path, parse_dates=["datetime"])

    print("[1/2] Running Threshold, Routine-Grouping & Quantification Ablation Matrix...")
    ablation_df = run_ablation_matrix(valid, test)
    ablation_out = REPORTS / "threshold_ablation_results.csv"
    ablation_df.to_csv(ablation_out, index=False)
    print(ablation_df.to_string(index=False))
    print(f"\nSaved to: {ablation_out}\n")

    print("[2/2] Running CEWI vs. Excess-Only Ranking Comparison (hour, p=97.5)...")
    ranked_df, rank_summary = run_cewi_ranking_analysis(valid, test)
    rank_out = REPORTS / "cewi_ranking_comparison.csv"
    ranked_df.to_csv(rank_out, index=False)
    print("Ranking summary:", rank_summary)
    print("\nTop 10 events by CEWI:")
    cols = [
        "event_id", "duration_hours", "excess_energy_kwh",
        "CEWI", "rank_excess", "rank_cewi", "rank_shift",
    ]
    print(ranked_df[cols].head(10).round(3).to_string(index=False))
    print(f"\nSaved to: {rank_out}")


if __name__ == "__main__":
    main()