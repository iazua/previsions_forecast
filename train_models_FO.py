import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Cargar los datos
DATA_PATH = Path(__file__).resolve().parent / "BBDD_calls_RRSS.xlsx"
data = pd.read_excel(DATA_PATH)

# Limpieza y preprocesamiento
# Convertir la columna de fecha a datetime
data['dat'] = pd.to_datetime(data['dat'], dayfirst=True)

# Filtrar filas donde 'con' es 0 (días no trabajados)
data = data[data['con'] > 0].copy()

# Crear características temporales
def create_time_features(df):
    df['year'] = df['dat'].dt.year
    df['month'] = df['dat'].dt.month
    df['day'] = df['dat'].dt.day
    df['day_of_week'] = df['dat'].dt.dayofweek  # 0 es lunes, 6 es domingo
    df['week_of_year'] = df['dat'].dt.isocalendar().week
    df['week_of_month'] = (df['dat'].dt.day - 1) // 7 + 1
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['dat'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['dat'].dt.is_month_end.astype(int)
    return df

data = create_time_features(data)

# Codificar la variable categórica 'cyb'
le = LabelEncoder()
data['cyb_encoded'] = le.fit_transform(data['cyb'])

# Preparar características y objetivo
features = ['year', 'month', 'day', 'day_of_week', 'week_of_year', 
            'week_of_month', 'is_weekend', 'is_month_start', 
            'is_month_end', 'cyb_encoded']
X = data[features]
y = data['con']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo (Random Forest por su capacidad para capturar relaciones no lineales y estacionalidad)
model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_split=5)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Guardar modelo y encoder
with open('con_prediction_model_rrss.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'encoder': le,
        'r2': r2,
        'mae': mae,
        'last_date': data['dat'].max()
    }, f)

print("Modelo entrenado y guardado exitosamente.")