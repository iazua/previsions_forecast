import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
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
    return df

def add_lag_features(df):
    df = df.sort_values(["cyb", "dat"]).copy()
    grouped = df.groupby("cyb")
    df["lag_1"] = grouped["con"].shift(1)
    df["lag_7"] = grouped["con"].shift(7)
    df["lag_30"] = grouped["con"].shift(30)
    df["rolling_mean_7"] = grouped["con"].shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)
    df["rolling_std_7"] = grouped["con"].shift(1).rolling(window=7).std().reset_index(level=0, drop=True)
    df["rolling_mean_30"] = grouped["con"].shift(1).rolling(window=30).mean().reset_index(level=0, drop=True)
    return df

FEATURES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "week_of_month", "is_weekend", "is_month_start", "is_month_end",
    "cyb_encoded", "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30",
]

def train_model(data_path, model_path):
    df = pd.read_excel(data_path, usecols=["dat", "con", "cyb"])
    df["dat"] = pd.to_datetime(df["dat"], dayfirst=True)
    df = df[df["con"] > 0].copy()

    df = create_time_features(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["cyb_encoded"] = le.fit_transform(df["cyb"])

    X = df[FEATURES]
    y = df["con"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_split=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "encoder": le,
            "r2": r2,
            "mae": mae,
            "last_date": df["dat"].max(),
        }, f)

    print(f"Saved {model_path.name}: R2={r2:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    for name, paths in SOURCES.items():
        print(f"Training {name} model...")
        train_model(paths["data"], paths["model"])
