import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from pathlib import Path
import plotly.graph_objects as go  # Plotly para gráficos interactivos

# ╭──────────────────────────────────────────────╮
# │ Configuración general                        │
# ╰──────────────────────────────────────────────╯
st.set_page_config(
    page_title="Predicción previsiones FO y RRSS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos globales -----------------------------------------------------------------
st.markdown(
    """
    <style>
    /* App y sidebar */
    .stApp, .css-1d391kg {background-color:#1a0033;}
    /* DataFrame */
    .stDataFrame div[role="table"]{background-color:#1a0033 !important;color:#FFFFFF;}
    .stDataFrame th{background-color:#330066 !important;color:#FFFFFF;}
    /* Plotly wrapper */
    .stPlotlyChart div{background-color:#1a0033 !important;}
    /* Textos principales */
    h1,h2,h3,h4,h5,h6, p, div, span{color:#FFFFFF;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Logo centrado --------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Logo_Ripley_banco_2.png",
        width=520,
    )
st.markdown("---")

st.title("📊 Predicción previsiones FO y RRSS")
st.caption("Selecciona la fuente en la barra lateral para alternar entre los modelos disponibles.")

# ╭──────────────────────────────────────────────╮
# │ Rutas de modelos y bases                     │
# ╰──────────────────────────────────────────────╯
BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "RRSS": BASE_DIR / "con_prediction_model_rrss.pkl",
    "FO": BASE_DIR / "con_prediction_model.pkl",
}
BASES = {
    "RRSS": BASE_DIR / "BBDD_calls_RRSS.xlsx",
    "FO": BASE_DIR / "BBDD_calls2.xlsx",
}
FORECAST_HORIZON = {"RRSS": 94, "FO": 94}

# ╭──────────────────────────────────────────────╮
# │ Carga en caché de modelos y datos            │
# ╰──────────────────────────────────────────────╯
@st.cache_resource
def load_all_models(paths):
    models = {}
    for k, v in paths.items():
        with open(v, "rb") as f:
            models[k] = pickle.load(f)
    return models


@st.cache_data
def load_historical(path):
    df = pd.read_excel(path, usecols=["dat", "con", "cyb"])
    df["dat"] = pd.to_datetime(df["dat"], dayfirst=True)
    df = df[df["con"] > 0].copy()
    if "cyb" not in df.columns:
        df["cyb"] = "NO"
    return df


models = load_all_models(MODELS)

# ╭──────────────────────────────────────────────╮
# │ Sidebar                                      │
# ╰──────────────────────────────────────────────╯
with st.sidebar:
    fuente = st.radio("📂 Modelo", list(MODELS), index=0)

# ── Datos y modelo activos ──────────────────────
mdata = models[fuente]
model = mdata["model"]
le = mdata["encoder"]
r2 = mdata["r2"]
mae = mdata["mae"]
rmse = mdata["rmse"]
last_dt = mdata["last_date"]
periods = FORECAST_HORIZON[fuente]
hist_df = load_historical(BASES[fuente])
date_options = pd.date_range(last_dt + timedelta(days=1), periods=periods)

# ╭──────────────────────────────────────────────╮
# │ Helpers                                      │
# ╰──────────────────────────────────────────────╯
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


def forecast(model, hist_df, periods, le, cyb_dates):
    df = hist_df[["dat", "con", "cyb"]].copy().sort_values("dat")
    df["cyb"] = df["cyb"].fillna("NO")
    # Ensure the label encoder can handle both expected labels
    if hasattr(le, "classes_"):
        required = np.array(["NO", "SI"], dtype=object)
        le.classes_ = np.unique(np.concatenate([le.classes_.astype(object), required]))
    preds = []
    for _ in range(periods):
        next_date = df["dat"].max() + timedelta(days=1)
        cyb_flag = "SI" if next_date in cyb_dates else "NO"
        df = pd.concat(
            [df, pd.DataFrame({"dat": [next_date], "con": [np.nan], "cyb": [cyb_flag]})],
            ignore_index=True,
        )
        temp = create_time_features(df.copy())
        temp = add_lag_features(temp)
        temp["cyb_encoded"] = le.transform(temp["cyb"])
        X = temp.loc[temp["dat"] == next_date, FEATURES]
        pred = model.predict(X)[0]
        pred_round = int(round(pred))
        df.loc[df["dat"] == next_date, "con"] = pred_round
        preds.append({"dat": next_date, "con_pred": pred_round})
    return pd.DataFrame(preds)


def make_plot(hist, fut, title):
    fig = go.Figure()
    fig.add_scatter(
        x=hist["dat"],
        y=hist["con"],
        mode="lines",
        name="Histórico",
        line=dict(width=2),
        hovertemplate="%{x|%d-%m-%Y}<br>Histórico: %{y}<extra></extra>",
    )
    fig.add_scatter(
        x=fut["dat"],
        y=fut["con_pred"],
        mode="lines",
        name="Predicción",
        line=dict(dash="dash", width=2),
        hovertemplate="%{x|%d-%m-%Y}<br>Predicción: %{y}<extra></extra>",
    )
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


# ╭──────────────────────────────────────────────╮
# │ Sidebar ajustes                              │
# ╰──────────────────────────────────────────────╯
with st.sidebar:
    st.header("")
    sel = st.multiselect('⚙️ Dias Cyber', options=date_options.strftime("%Y-%m-%d"))
    st.info(f"**R²:** {r2:.4f} | **MAE:** {mae:.4f} | **RMSE:** {rmse:.4f}")

cyb_dates = pd.to_datetime(sel)
future_df = forecast(model, hist_df, periods, le, set(cyb_dates))
if sel:
    st.sidebar.success("Predicciones actualizadas ✔️")

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
    st.dataframe(
        future_df[["dat", "con_pred"]]
        .rename(columns={"dat": "Fecha", "con_pred": "Valor predicho"})
        .reset_index(drop=True),
        use_container_width=True,
    )
    csv = (
        future_df[["dat", "con_pred"]]
        .rename(columns={"dat": "Fecha", "con_pred": "Valor predicho"})
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        f"⬇️ Descargar (.csv) {fuente}",
        csv,
        f"predicciones_con_{fuente.lower()}.csv",
        mime="text/csv",
    )

