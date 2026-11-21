from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from src.data import load_hourly
from src.counterfactual import calibrate_routine_thresholds, detect_and_quantify, summarize_events, routine_category_profiles, attribute_event_hour

ROOT = Path(__file__).resolve().parent
R = ROOT/"reports"
st.set_page_config(page_title="RAAED | Counterfactual Energy Waste", page_icon="⚡", layout="wide")
st.title("⚡ RAAED — Counterfactual Energy Waste Quantification")
st.caption("Routine-aware baseline • Excess-energy measurement • Category-level attribution")

required = [R/"validation_residuals.csv", R/"test_counterfactuals.csv", ROOT/"models"/"counterfactual_model.joblib"]
if not all(p.exists() for p in required):
    st.warning("Run the research pipeline first: `python run_pipeline.py`")
    st.stop()

hourly = load_hourly()
valid = pd.read_csv(R/"validation_residuals.csv", parse_dates=["datetime"])
test = pd.read_csv(R/"test_counterfactuals.csv", parse_dates=["datetime"])
cal = calibrate_routine_thresholds(valid)
det = detect_and_quantify(test, cal)
det, events = summarize_events(det)
model = joblib.load(ROOT/"models"/"counterfactual_model.joblib")
menu = st.sidebar.radio("Research module", ["Research overview", "Counterfactual timeline", "Waste events", "Category attribution", "Experimental evidence"])

if menu == "Research overview":
    st.subheader("Research outputs")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Selected model", model["model_name"])
    c2.metric("Test hours", f"{len(det):,}")
    c3.metric("Flagged hours", f"{int(det.is_excess_event_hour.sum()):,}")
    c4.metric("Estimated excess", f"{det.quantified_excess_kwh.sum():.2f} kWh")
    st.plotly_chart(px.line(det, x="datetime", y=["actual_kwh","baseline_kwh"], title="Observed consumption vs counterfactual baseline"), use_container_width=True)
    st.info("Estimated excess is consumption above a learned baseline during flagged periods. It is not proof that all excess was avoidable waste.")
elif menu == "Counterfactual timeline":
    st.subheader("Actual consumption against expected counterfactual")
    fig = px.line(det, x="datetime", y=["actual_kwh","baseline_kwh"], title="Held-out test period")
    flagged = det[det.is_excess_event_hour]
    if len(flagged):
        fig.add_scatter(x=flagged.datetime, y=flagged.actual_kwh, mode="markers", name="Flagged high-use hour")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(det, x="datetime", y=["positive_excess_kwh","routine_threshold_kwh"], title="Positive residual and routine-aware threshold"), use_container_width=True)
elif menu == "Waste events":
    st.subheader("Quantified high-consumption events")
    if events.empty:
        st.success("No high-consumption events exceeded the calibrated thresholds in this test period.")
    else:
        st.dataframe(events.round(3), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(events, x="event_id", y="excess_energy_kwh", title="Estimated excess energy by event (kWh)"), use_container_width=True)
        st.caption("CEWI ranks events using estimated excess energy and threshold exceedance severity. ExcessEnergy_kWh remains the interpretable physical quantity.")
elif menu == "Category attribution":
    st.subheader("Which measured category contributed to an event?")
    if events.empty:
        st.info("No detected events are available for attribution.")
    else:
        event_id = st.selectbox("Choose event", events.event_id.astype(int).tolist())
        selected = det[(det.event_id == event_id) & det.is_excess_event_hour].iloc[0]
        ts = pd.Timestamp(selected.datetime)
        row = hourly.loc[ts] if ts in hourly.index else hourly.iloc[hourly.index.get_indexer([ts], method="nearest")[0]]
        profiles = routine_category_profiles(hourly.loc[:valid.datetime.max()].copy())
        attr = attribute_event_hour({"datetime":ts, "_hourly_values":row.to_dict()}, profiles)
        st.metric("Largest positive category deviation", attr["largest_positive_contributor"])
        st.write("Estimated positive category deviations for the selected hour:")
        chart = pd.DataFrame({"Category":list(attr["category_excess_kwh"].keys()), "Excess_kWh":list(attr["category_excess_kwh"].values())})
        st.plotly_chart(px.bar(chart, x="Category", y="Excess_kWh"), use_container_width=True)
        st.caption("Sub-meter categories are broad groups. Attribution is an estimate relative to category routines, not appliance identification. Unmetered loads are retained as a category.")
else:
    st.subheader("Experimental evidence")
    comp = pd.read_csv(R/"forecast_comparison.csv")
    st.write("**Forecasting model comparison**")
    st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
    metrics_path = R/"synthetic_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.write("**Controlled synthetic excess-injection test**")
        st.json(metrics)
        st.warning("These metrics evaluate controlled injected anomalies, not naturally occurring household faults.")
    st.markdown("""
    **Research questions**
    1. Does routine-aware residual calibration identify unusual high-use periods differently from a global threshold?
    2. Can excess energy be estimated in kWh on controlled injected events?
    3. Can sub-metering deviations provide useful category-level attribution?
    4. Does the proposed CEWI rank longer/larger events differently from excess energy alone?
    """)
