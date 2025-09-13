import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SOURCES = {
    "RRSS": {
        "data": BASE_DIR / "BBDD_calls_RRSS.xlsx",
        "model": BASE_DIR / "con_prediction_model_rrss.pkl",
    },
    "FO": {
        "data": BASE_DIR / "BBDD_calls2.xlsx",
        "model": BASE_DIR / "con_prediction_model.pkl",
    },
}

def create_time_features(df):
    df["year"] = df["dat"].dt.year
    df["month"] = df["dat"].dt.month
    df["day"] = df["dat"].dt.day
    df["day_of_week"] = df["dat"].dt.dayofweek
    df["week_of_year"] = df["dat"].dt.isocalendar().week
    df["week_of_month"] = (df["dat"].dt.day - 1) // 7 + 1
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["dat"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["dat"].dt.is_month_end.astype(int)
    df["quarter"] = df["dat"].dt.quarter
    df["day_of_year"] = df["dat"].dt.dayofyear
    df["is_quarter_start"] = df["dat"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["dat"].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df["dat"].dt.is_year_start.astype(int)
    df["is_year_end"] = df["dat"].dt.is_year_end.astype(int)
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_week_of_year"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week_of_year"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df

def add_lag_features(df):
    df = df.sort_values(["cyb", "dat"]).copy()
    grouped = df.groupby("cyb")
    df["lag_1"] = grouped["con"].shift(1)
    df["lag_7"] = grouped["con"].shift(7)
    df["lag_30"] = grouped["con"].shift(30)
    df["lag_14"] = grouped["con"].shift(14)
    df["lag_60"] = grouped["con"].shift(60)
    df["rolling_mean_7"] = grouped["con"].shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)
    df["rolling_std_7"] = grouped["con"].shift(1).rolling(window=7).std().reset_index(level=0, drop=True)
    df["rolling_mean_14"] = grouped["con"].shift(1).rolling(window=14).mean().reset_index(level=0, drop=True)
    df["rolling_std_14"] = grouped["con"].shift(1).rolling(window=14).std().reset_index(level=0, drop=True)
    df["rolling_mean_30"] = grouped["con"].shift(1).rolling(window=30).mean().reset_index(level=0, drop=True)
    df["rolling_mean_60"] = grouped["con"].shift(1).rolling(window=60).mean().reset_index(level=0, drop=True)
    return df

FEATURES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "week_of_month", "is_weekend", "is_month_start", "is_month_end",
    "quarter", "day_of_year", "is_quarter_start", "is_quarter_end",
    "is_year_start", "is_year_end", "sin_day_of_year", "cos_day_of_year",
    "sin_week_of_year", "cos_week_of_year",
    "cyb_encoded", "lag_1", "lag_7", "lag_14", "lag_30", "lag_60",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
    "rolling_mean_30", "rolling_mean_60",
]

def train_model(data_path, model_path):
    df = pd.read_excel(data_path, usecols=["dat", "con", "cyb"])
    df["dat"] = pd.to_datetime(df["dat"], dayfirst=True)
    df = df[df["con"] > 0].copy()

    df = create_time_features(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)

    # Ensure records are ordered chronologically and compute recency weights
    df = df.sort_values("dat").reset_index(drop=True)
    max_date = df["dat"].max()
    df["recency_weight"] = np.exp(-0.01 * (max_date - df["dat"]).dt.days)

    le = LabelEncoder()
    df["cyb_encoded"] = le.fit_transform(df["cyb"])

    # Reserve the last 20% of records for evaluation
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURES]
    y_train = train_df["con"]
    X_test = test_df[FEATURES]
    y_test = test_df["con"]
    sample_weight = train_df["recency_weight"]

    # Hyperparameter tuning with time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "encoder": le,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "params": search.best_params_,
            "last_date": df["dat"].max(),
        }, f)

    print(f"Saved {model_path.name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    for name, paths in SOURCES.items():
        print(f"Training {name} model...")
        train_model(paths["data"], paths["model"])
