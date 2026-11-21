from pathlib import Path
import zipfile
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ZIP_PATH = RAW / "household_power_consumption.zip"

def get_source_file():
    RAW.mkdir(parents=True, exist_ok=True)
    existing = list(RAW.rglob("household_power_consumption.txt"))
    if existing:
        return existing[0]
    if not ZIP_PATH.exists():
        print("Downloading UCI dataset...")
        response = requests.get(ZIP_URL, timeout=180)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        matches = [n for n in archive.namelist() if n.endswith("household_power_consumption.txt")]
        if not matches:
            raise FileNotFoundError("household_power_consumption.txt not found in UCI archive")
        archive.extract(matches[0], RAW)
    found = list(RAW.rglob("household_power_consumption.txt"))
    if not found:
        raise FileNotFoundError("Could not locate extracted UCI text file")
    return found[0]

def load_hourly(force=False):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "hourly.csv"
    if out.exists() and not force:
        return pd.read_csv(out, parse_dates=["datetime"], index_col="datetime")
    path = get_source_file()
    print("Reading raw minute-level file (about 2.1 million rows)...")
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    cols = ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Global_active_power"] = df["Global_active_power"].interpolate(method="time", limit=5)
    df = df.dropna(subset=["Global_active_power"])
    # Global active power is kW averaged over each minute. Sum kW/60 -> kWh.
    hourly = pd.DataFrame(
    index=df["Global_active_power"].resample("h").mean().index
    )
    hourly["energy_kwh"] = df["Global_active_power"].resample("h").sum(min_count=45) / 60.0
    for i in (1, 2, 3):
        hourly[f"sub{i}_kwh"] = df[f"Sub_metering_{i}"].resample("h").sum(min_count=45) / 1000.0
    hourly = hourly.dropna(subset=["energy_kwh"])
    # UCI energy balance: remaining load not covered by the three sub-meters.
    sub_sum = hourly[["sub1_kwh", "sub2_kwh", "sub3_kwh"]].sum(axis=1, min_count=3)
    hourly["unmetered_kwh"] = (hourly["energy_kwh"] - sub_sum).clip(lower=0)
    hourly.index.name = "datetime"
    hourly.to_csv(out)
    print(f"Saved {len(hourly):,} hourly records.")
    return hourly
