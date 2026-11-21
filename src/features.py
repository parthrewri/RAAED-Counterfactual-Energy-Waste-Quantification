import numpy as np
import pandas as pd

TARGET = "energy_kwh"

def create_supervised(hourly):
    df = hourly.sort_index().copy()
    idx = df.index
    X = pd.DataFrame(index=idx)
    X["hour"] = idx.hour
    X["dayofweek"] = idx.dayofweek
    X["month"] = idx.month
    X["weekend"] = (idx.dayofweek >= 5).astype(int)
    X["hour_sin"] = np.sin(2*np.pi*idx.hour/24)
    X["hour_cos"] = np.cos(2*np.pi*idx.hour/24)
    X["dow_sin"] = np.sin(2*np.pi*idx.dayofweek/7)
    X["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
    y = df[TARGET]
    for lag in [1, 2, 24, 48, 168]:
        X[f"lag_{lag}"] = y.shift(lag)
    X["mean_24h_past"] = y.shift(1).rolling(24, min_periods=12).mean()
    X["std_24h_past"] = y.shift(1).rolling(24, min_periods=12).std()
    X["mean_168h_past"] = y.shift(1).rolling(168, min_periods=48).mean()
    out = X.copy()
    out[TARGET] = y
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def split_chronologically(data):
    n = len(data)
    a, b = int(n*.70), int(n*.85)
    return data.iloc[:a].copy(), data.iloc[a:b].copy(), data.iloc[b:].copy()
