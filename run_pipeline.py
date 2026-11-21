import json
from pathlib import Path
from src.data import load_hourly
from src.modeling import evaluate_forecasters
from src.counterfactual import calibrate_routine_thresholds, detect_and_quantify, summarize_events
from src.experiments import run_controlled_injection

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(exist_ok=True)
    print("[1/5] Load and aggregate UCI data")
    hourly = load_hourly()
    print(f"Hourly rows: {len(hourly):,}")

    print("[2/5] Train forecasting models and select by validation MAE")
    comparison, valid, test, selected = evaluate_forecasters(hourly)
    print(comparison.to_string(index=False))
    print(f"Selected model: {selected}")

    print("[3/5] Calibrate routine-aware counterfactual thresholds")
    calibration = calibrate_routine_thresholds(valid)
    detected = detect_and_quantify(test, calibration)
    detected, events = summarize_events(detected)
    detected.to_csv(REPORTS/"test_excess_detection.csv", index=False)
    events.to_csv(REPORTS/"quantified_events.csv", index=False)

    print("[4/5] Run controlled synthetic excess injection")
    injected, metrics = run_controlled_injection(test, calibration)
    injected.to_csv(REPORTS/"synthetic_injection_results.csv", index=False)
    (REPORTS/"synthetic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[5/5] Done")
    print(f"Detected event hours: {int(detected['is_excess_event_hour'].sum())}")
    print(f"Estimated excess energy in test period: {detected['quantified_excess_kwh'].sum():.3f} kWh")
    print(f"Event count: {len(events)}")
    print("Synthetic stress-test metrics:")
    print(json.dumps(metrics, indent=2))
    print("Now launch: streamlit run app.py")

if __name__ == "__main__":
    main()
