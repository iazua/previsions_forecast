import streamlit as st
import pandas as pd
import pickle
from datetime import timedelta
import numpy as np
import plotly.graph_objects as go   # Plotly para gráficos interactivos
from pathlib import Path

# Paleta de colores institucional
PRIMARY_COLOR = "#4F2D7F"  # Minsk
DARK_BG_COLOR = "#361860"  # Scarlet Gum
PRIMARY_BG = "#F8F9FA"  # Fondos claros
TABLE_BG_COLOR = DARK_BG_COLOR  # Tablas
ACCENT_COLOR = "#F1AC4B"  # Sandy Brown
WHITE = "#FFFFFF"
BLACK = "#000000"
ACCENT_RGBA = "rgba(241, 172, 75, 0.63)"  # Con opacidad
PRIMARY_RGBA = "rgba(79, 45, 127, 1)"

BASE_DIR = Path(__file__).resolve().parent

# ╭──────────────────────────────────────────────╮
# │ Configuración general                        │
# ╰──────────────────────────────────────────────╯
st.set_page_config(
    page_title="Predicción previsiones FO y RRSS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos globales -----------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --dark-bg: {DARK_BG_COLOR};
        --table-bg: {TABLE_BG_COLOR};
        --primary-bg: {PRIMARY_BG};
        --accent-color: {ACCENT_COLOR};
        --white: {WHITE};
        --black: {BLACK};
    }}

    /* Fondo de la aplicación */
    .stApp {{background-color: var(--dark-bg);}}

    /* DataFrame */
    .stDataFrame div[role="table"] {{background-color: var(--table-bg) !important;color: var(--white);}}
    .stDataFrame th {{background-color: var(--primary-color) !important;color: var(--white);}}

    /* Plotly wrapper */
    .stPlotlyChart div {{background-color: var(--dark-bg) !important;}}

    /* Textos principales */
    h1,h2,h3,h4,h5,h6,p,div,span {{color: var(--white);}}

    /* Botones */
    .stButton>button, .stDownloadButton button {{
        background-color: var(--accent-color);
        color: var(--black);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Logo centrado --------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Logo_Ripley_banco_2.png",
        use_container_width=True,
    )
st.markdown("---")

st.title("📊 Predicción previsiones FO y RRSS")
st.caption("Selecciona la fuente en las opciones de configuración para alternar entre los modelos disponibles.")

# ╭──────────────────────────────────────────────╮
# │ Rutas de modelos y bases                     │
# ╰──────────────────────────────────────────────╯
MODELS = {
    "RRSS": BASE_DIR / "con_prediction_model_rrss.pkl",
    "FO": BASE_DIR / "con_prediction_model.pkl",
}
BASES = {
    "RRSS": BASE_DIR / "BBDD_calls_RRSS.xlsx",
    "FO": BASE_DIR / "BBDD_calls2.xlsx",
}
FORECAST_HORIZON = {"RRSS": 120, "FO": 120}

# ── Nombres de los días --------------------------------------------------------
DAY_NAMES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

# ╭──────────────────────────────────────────────╮
# │ Carga en caché de modelos y datos            │
# ╰──────────────────────────────────────────────╯
@st.cache_resource
def load_all_models(paths):
    return {k: pickle.load(open(v, "rb")) for k, v in paths.items()}

@st.cache_data
def load_historical(path):
    df = pd.read_excel(path)
    df["dat"] = pd.to_datetime(df["dat"], dayfirst=True)
    return df[df["con"] > 0]

models = load_all_models(MODELS)

# ╭──────────────────────────────────────────────╮
# │ Configuración de modelo                      │
# ╰──────────────────────────────────────────────╯
fuente = st.radio(
    "📂 Modelo", list(MODELS), index=0, horizontal=True
)

# ── Datos y modelo activos ──────────────────────
mdata   = models[fuente]
model   = mdata["model"]
le      = mdata["encoder"]
r2      = mdata["r2"]
mae     = mdata["mae"]
last_dt = mdata["last_date"]
periods = FORECAST_HORIZON[fuente]
hist_df = load_historical(BASES[fuente])

# ╭──────────────────────────────────────────────╮
# │ Helpers                                      │
# ╰──────────────────────────────────────────────╯
def create_time_features(df):
    df = df.copy()
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

def prepare_features(df):
    df = create_time_features(df)
    df["cyb_encoded"] = le.transform(df["cyb"])
    df = add_lag_features(df)
    return df

def forecast(df_hist, periods, cyb_dates=None):
    cyb_dates = cyb_dates or set()
    df = df_hist[["dat", "con", "cyb"]].sort_values("dat").copy()
    preds = []
    for _ in range(periods):
        next_date = df["dat"].max() + timedelta(days=1)
        cyb_flag = "SI" if next_date in cyb_dates else "NO"
        new_row = pd.DataFrame({"dat": [next_date], "con": [np.nan], "cyb": [cyb_flag]})
        df = pd.concat([df, new_row], ignore_index=True)
        feat_df = prepare_features(df).dropna()
        features_row = feat_df.loc[feat_df["dat"] == next_date, FEATURES]
        pred_val = model.predict(features_row)[0]
        df.loc[df["dat"] == next_date, "con"] = pred_val
        preds.append({"dat": next_date, "con_pred": int(round(pred_val)), "cyb": cyb_flag})
    return pd.DataFrame(preds)

def make_plot(hist, fut, title):
    fig = go.Figure()
    fig.add_scatter(x=hist["dat"], y=hist["con"], mode="lines", name="Histórico",
                    line=dict(width=2, color=ACCENT_COLOR),
                    hovertemplate="%{x|%d-%m-%Y}<br>Histórico: %{y}<extra></extra>")
    fig.add_scatter(x=fut["dat"], y=fut["con_pred"], mode="lines", name="Predicción",
                    line=dict(width=2, color=PRIMARY_BG),
                    hovertemplate="%{x|%d-%m-%Y}<br>Predicción: %{y}<extra></extra>")
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Valor de 'con'",
        xaxis=dict(rangeslider=dict(visible=True)),
        template="plotly_dark",  # Usa tema oscuro coherente
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig

# ── Predicción ─────────────────────────────────
FEATURES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "week_of_month", "is_weekend", "is_month_start", "is_month_end",
    "cyb_encoded", "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30"
]
future_df = forecast(hist_df, periods)

# ╭──────────────────────────────────────────────╮
# │ Ajustes de predicción                        │
# ╰──────────────────────────────────────────────╯
adjust_box = st.expander("⚙️ Ajustes de predicción", expanded=False)
with adjust_box:
    sel = st.multiselect(
        "Dias Cyber",
        options=future_df["dat"].dt.strftime("%Y-%m-%d"),
    )
    st.info(f"**R²:** {r2:.4f} | **MAE:** {mae:.4f}")

if sel:
    sel_dt = set(pd.to_datetime(sel))
    future_df = forecast(hist_df, periods, cyb_dates=sel_dt)
    adjust_box.success("Predicciones actualizadas ✔️")

# ╭──────────────────────────────────────────────╮
# │ Tabs de salida                               │
# ╰──────────────────────────────────────────────╯
tab1, tab2 = st.tabs([f"📈 Gráfico ({fuente})", f"📋 Tabla ({fuente})"])

with tab1:
    st.plotly_chart(
        make_plot(hist_df, future_df, f"Histórico y predicción ({fuente})"),
        use_container_width=True,
        config={"displaylogo": False},
    )

with tab2:
    table_df = future_df[["dat", "con_pred"]].copy()
    table_df["Fecha"] = table_df["dat"].dt.strftime("%d-%m-%Y")
    table_df["Día de la semana"] = table_df["dat"].dt.dayofweek.map(DAY_NAMES)
    table_df = (
        table_df[["Fecha", "Día de la semana", "con_pred"]]
        .rename(columns={"con_pred": "Valor predicho"})
        .reset_index(drop=True)
    )
    st.dataframe(table_df, use_container_width=True)
    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Descargar (.csv) {fuente}",
        csv,
        f"predicciones_con_{fuente.lower()}.csv",
        mime="text/csv",
    )
