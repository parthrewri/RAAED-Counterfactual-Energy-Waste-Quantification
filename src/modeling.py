from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import TARGET, create_supervised, split_chronologically

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

def model_candidates():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=160, max_depth=16,
            min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=250, max_depth=5,
            learning_rate=.04, subsample=.85, colsample_bytree=.9,
            objective="reg:squarederror", random_state=42, n_jobs=-1)
    except Exception:
        models["HistGradient Boosting"] = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=.06, random_state=42)
    return models

def evaluate_forecasters(hourly):
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    data = create_supervised(hourly)
    train, valid, test = split_chronologically(data)
    feats = [c for c in data.columns if c != TARGET]
    Xtr, ytr = train[feats], train[TARGET]
    Xv, yv = valid[feats], valid[TARGET]
    Xt, yt = test[feats], test[TARGET]
    fitted, val_preds, test_preds = {}, {}, {}
    # Persistence baseline is an essential reference: next hour = previous hour.
    val_preds["Persistence"] = valid["lag_1"].to_numpy()
    test_preds["Persistence"] = test["lag_1"].to_numpy()
    fitted["Persistence"] = None
    for name, model in model_candidates().items():
        print(f"Training {name}...")
        model.fit(Xtr, ytr)
        fitted[name] = model
        val_preds[name] = model.predict(Xv)
        test_preds[name] = model.predict(Xt)
    # Select by validation MAE, never by test metrics.
    val_mae = {n: mean_absolute_error(yv, p) for n, p in val_preds.items()}
    selected = min(val_mae, key=val_mae.get)
    rows = []
    for name, pred in test_preds.items():
        rows.append({"Model": name,
            "MAE_kWh": mean_absolute_error(yt, pred),
            "RMSE_kWh": np.sqrt(mean_squared_error(yt, pred)),
            "R2": r2_score(yt, pred)})
    comparison = pd.DataFrame(rows).sort_values("MAE_kWh")
    artifact = {"model_name": selected, "model": fitted[selected], "features": feats}
    joblib.dump(artifact, MODELS / "counterfactual_model.joblib")
    valid_frame = pd.DataFrame({"datetime": valid.index, "actual_kwh": yv.values,
        "baseline_kwh": val_preds[selected], "residual_kwh": yv.values-val_preds[selected]})
    test_frame = pd.DataFrame({"datetime": test.index, "actual_kwh": yt.values,
        "baseline_kwh": test_preds[selected], "residual_kwh": yt.values-test_preds[selected]})
    comparison.to_csv(REPORTS / "forecast_comparison.csv", index=False)
    valid_frame.to_csv(REPORTS / "validation_residuals.csv", index=False)
    test_frame.to_csv(REPORTS / "test_counterfactuals.csv", index=False)
    (REPORTS / "selected_model.json").write_text(json.dumps({
        "selected_by_validation_mae": selected, "validation_mae_kwh": float(val_mae[selected])
    }, indent=2), encoding="utf-8")
    return comparison, valid_frame, test_frame, selected
