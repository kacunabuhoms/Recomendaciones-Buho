# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import requests
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Recomendaciones Buho", layout="wide")

# ——————————————————————————————————————————————————————————————————————————
# 1) Definiciones de apoyo: geolocalización, conversión de códigos postales
# ——————————————————————————————————————————————————————————————————————————

# cargamos una sola vez el Excel con coordenadas
@st.cache_data
def load_coords():
    df_coord = pd.read_excel("coordenadas_mx.xlsx")
    postal_codes = df_coord["codigo_postal"].values.reshape(-1, 1)
    coords       = df_coord[["latitud", "longitud"]].values
    tree         = cKDTree(postal_codes)
    return df_coord, tree, coords

df_coord, kdtree, kdcoords = load_coords()

def lookup_coord(cp: int):
    m = df_coord[df_coord["codigo_postal"] == cp]
    if not m.empty:
        return float(m["latitud"].iloc[0]), float(m["longitud"].iloc[0])
    _, idx = kdtree.query([[cp]])
    return tuple(kdcoords[idx[0]])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ     = math.radians(lat2 - lat1)
    Δλ     = math.radians(lon2 - lon1)
    a      = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ——————————————————————————————————————————————————————————————————————————
# 2) Clase de transformación RAW → FEATURES (misma que en entrenamiento)
# ——————————————————————————————————————————————————————————————————————————

class RawToFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in X.iterrows():
            cp_o, cp_d = int(r["cp_origen"]), int(r["cp_destino"])
            fecha      = pd.to_datetime(r["fecha"], dayfirst=True)
            lo, ld     = lookup_coord(cp_o), lookup_coord(cp_d)
            dist       = round(haversine(lo[0], lo[1], ld[0], ld[1]), 2)
            d = {
                "rate":           float(r["rate"]),
                "distance":       dist,
                "origin_state":   r["origin_state"],
                "dest_state":     r["dest_state"],
                "carrier_service": f"{r['paqueteria']}_{r['servicio']}",
                "day_week":       fecha.weekday(),
                "month":          fecha.month
            }
            # cíclicos y flags
            d["day_sin"]    = np.sin(2 * np.pi * d["day_week"] / 7)
            d["day_cos"]    = np.cos(2 * np.pi * d["day_week"] / 7)
            d["month_sin"]  = np.sin(2 * np.pi * d["month"] / 12)
            d["month_cos"]  = np.cos(2 * np.pi * d["month"] / 12)
            d["is_weekend"] = int(d["day_week"] >= 5)
            d["rate_x_dist"]= d["rate"] * d["distance"]
            rows.append(d)
        df_f = pd.DataFrame(rows)
        return df_f.drop(columns=["day_week", "month"])

# preprocessor (igual que en entrenamiento)
numeric_feats     = ["rate", "distance", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend", "rate_x_dist"]
categorical_feats = ["origin_state", "dest_state", "carrier_service"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats),
], remainder="drop")

# ——————————————————————————————————————————————————————————————————————————
# 3) Carga de modelos serializados
# ——————————————————————————————————————————————————————————————————————————

@st.cache_resource
def load_models():
    inc_model  = joblib.load("xgb_pipeline_default_thr.pkl")
    time_model = joblib.load("xgbreg_pipeline.pkl")
    return inc_model, time_model

inc_model, time_model = load_models()

# ——————————————————————————————————————————————————————————————————————————
# 4) Lista de carriers disponibles
# ——————————————————————————————————————————————————————————————————————————

ALL_CARRIERS = [
    "Estafeta", "Paquetexpress", "Fedex", "Afimex",
    "UPS", "DHL", "JT Express", "Buho"
]

# ——————————————————————————————————————————————————————————————————————————
# 5) Interfaz Streamlit
# ——————————————————————————————————————————————————————————————————————————

st.title("📦 Recomendaciones Buho")
st.write("Ingresa los datos para obtener predicción de incidencia y tiempo de entrega.")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        cp_o    = st.text_input("Código postal origen", key="cp_o")
        cp_d    = st.text_input("Código postal destino", key="cp_d")
        rate    = st.number_input("Peso (kg)", min_value=0.0, step=0.1, key="rate")
    with col2:
        servicio = st.selectbox("Servicio", ["Terrestre", "aereo"], key="servicio")
        fecha    = st.date_input("Fecha de envío", key="fecha")
    with col3:
        paq_sel = st.multiselect(
            "Paquetería (opcional)", 
            options=ALL_CARRIERS, 
            help="Deja vacío para evaluar todas"
        )
    submitted = st.form_submit_button("Calcular")

if submitted:
    # construimos raw_input
    if not cp_o or not cp_d:
        st.error("Debe ingresar ambos códigos postales.")
    else:
        raw = {
            "cp_origen": cp_o,
            "cp_destino": cp_d,
            "rate":      rate,
            "servicio":  servicio,
            # convertimos date a string dd/mm/YYYY
            "fecha":     fecha.strftime("%d/%m/%Y")
        }

        # si no seleccionó ninguna paquetería, probamos todas
        carriers_to_run = paq_sel if paq_sel else ALL_CARRIERS
        results = {}

        # para cada carrier candidata:
        for carrier in carriers_to_run:
            raw["paqueteria"] = carrier
            df_raw = pd.DataFrame([raw])

            # 1) Features incidencia
            X_raw_inc  = RawToFeatures().transform(df_raw)
            X_pre_inc  = preprocessor.transform(X_raw_inc)
            inc_pred   = int(inc_model.predict(X_pre_inc)[0])

            # 2) Features tiempo
            raw["incidence"] = inc_pred
            df_time = pd.DataFrame([raw])
            X_raw_time = RawToFeatures().transform(df_time)
            X_pre_time = preprocessor.transform(X_raw_time)
            time_pred  = float(time_model.predict(X_pre_time)[0])

            results[carrier] = {
                "Info": raw.copy(),
                "Model prediction": {
                    "incidence": inc_pred,
                    "delivery_time_bd": round(time_pred, 2)
                }
            }

        st.subheader("Resultados")
        st.json(results)
