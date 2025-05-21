# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime
from scipy.spatial import cKDTree

# ——— Helpers de conversión y geolocalización ———

# 2) Funciones de conversión
def convert_service(service: str) -> str:
    aereo = [
        "1 kg","2 kg","3 kg","4 kg","5 kg","6 kg","7 kg","8 kg","9 kg",
        "10 kg","11 kg","12 kg","13 kg","14 kg","15 kg","16 kg","17 kg",
        "19 kg","20 kg","21 kg","22 kg","24 kg","25 kg","26 kg","27 kg",
        "FEDEX_EXPRESS_SAVER","FEDEX_EXPRESS_SAVER_Z1","FEDEX_EXPRESS_SAVER_Z2",
        "FEDEX_EXPRESS_SAVER_Z3","FEDEX_EXPRESS_SAVER_Z4","FEDEX_EXPRESS_SAVER_Z5",
        "FEDEX_EXPRESS_SAVER_Z6","FEDEX_EXPRESS_SAVER_Z7","FEDEX_EXPRESS_SAVER_Z8",
        "UPS_STANDAR","UPS_SAVER","Standard","standard",
        "STANDARD_ECOMMERCE_Z1","STANDARD_ECOMMERCE_Z2","STANDARD_ECOMMERCE_Z3",
        "STANDARD_ECOMMERCE_Z4","STANDARD_ECOMMERCE_Z5","STANDARD_ECOMMERCE_Z6",
        "STANDARD_ECOMMERCE_Z7","STANDARD_ECOMMERCE_Z8","STANDARD_OVERNIGHT",
        "STANDARD_OVERNIGHT_Z4","STANDARD_OVERNIGHT_Z6",
        "STANDARD_SPECIAL_Z1","STANDARD_SPECIAL_Z2","STANDARD_SPECIAL_Z3",
        "STANDARD_SPECIAL_Z4","STANDARD_SPECIAL_Z5","STANDARD_SPECIAL_Z6",
        "STANDARD_SPECIAL_Z7","STANDARD_Z1","STANDARD_Z2","STANDARD_Z3",
        "STANDARD_Z4","STANDARD_Z5","EXPRESS DOMESTIC","ECONOMY SELECT DOMESTIC",
        "EXPRESS_SPECIAL_Z1","EXPRESS_SPECIAL_Z2","EXPRESS_SPECIAL_Z3",
        "EXPRESS_SPECIAL_Z4","EXPRESS_SPECIAL_Z5","EXPRESS_SPECIAL_Z6",
        "EXPRESS_SPECIAL_Z7","EXPRESS_ECOMMERCE_Z1","EXPRESS_ECOMMERCE_Z2",
        "EXPRESS_ECOMMERCE_Z3","EXPRESS_ECOMMERCE_Z4","EXPRESS_ECOMMERCE_Z5",
        "EXPRESS_ECOMMERCE_Z6","EXPRESS_ECOMMERCE_Z7","EXPRESS_ECOMMERCE_Z8",
        "Terrestre","Dia Sig.","nextday","economico","Metropoli",
        "ground","saver","pickup","SENDEX"
    ]
    return "aereo" if service in aereo else "terrestre"

def convert_carrier(carrier: str) -> str:
    m = {
        "Afimex":"Afimex","buho":"Buho","DHL":"DHL","Estafeta":"Estafeta",
        "FDXM":"Fedex","FEDEX MEXICO":"Fedex","FEDEX":"Fedex","fedex":"Fedex",
        "JTEX":"JT Express","JT Express":"JT Express",
        "Paquetexpress":"Paquetexpress","PAQUETEXPRESS":"Paquetexpress",
        "SENDEX":"Sendex","UPS":"UPS"
    }
    return m.get(carrier, carrier)

def convert_estado(cp: int) -> str:
    estados = [
        {'name':'Ciudad de México','min':1000, 'max':16900},
        {'name':'Aguascalientes',  'min':20000,'max':20997},
        {'name':'Baja California', 'min':21000,'max':22997},
        {'name':'Baja California Sur','min':23000,'max':23997},
        {'name':'Campeche','min':24000,'max':24940},
        {'name':'Coahuila','min':25000,'max':27999},
        {'name':'Colima','min':28000,'max':28989},
        {'name':'Chiapas','min':29000,'max':30997},
        {'name':'Chihuahua','min':31000,'max':33997},
        {'name':'Durango','min':34000,'max':35987},
        {'name':'Guanajuato','min':36000,'max':38997},
        {'name':'Guerrero','min':39000,'max':41998},
        {'name':'Hidalgo','min':42000,'max':43998},
        {'name':'Jalisco','min':44100,'max':49996},
        {'name':'México','min':50000,'max':57950},
        {'name':'Michoacán','min':58000,'max':61998},
        {'name':'Morelos','min':62000,'max':62996},
        {'name':'Nayarit','min':63000,'max':63996},
        {'name':'Nuevo León','min':64000,'max':67996},
        {'name':'Oaxaca','min':68000,'max':71998},
        {'name':'Puebla','min':72000,'max':75997},
        {'name':'Querétaro','min':76000,'max':76998},
        {'name':'Quintana Roo','min':77000,'max':77997},
        {'name':'San Luis Potosí','min':78000,'max':79998},
        {'name':'Sinaloa','min':80000,'max':82996},
        {'name':'Sonora','min':83000,'max':85994},
        {'name':'Tabasco','min':86000,'max':86998},
        {'name':'Tamaulipas','min':87000,'max':89970},
        {'name':'Tlaxcala','min':90000,'max':90990},
        {'name':'Veracruz','min':91000,'max':96998},
        {'name':'Yucatán','min':97000,'max':97990},
        {'name':'Zacatecas','min':98000,'max':99998}
    ]
    for e in estados:
        if e['min'] <= cp <= e['max']:
            return e['name']
    closest = min(estados, key=lambda e: abs(cp - ((e['min']+e['max'])/2)))
    return closest['name']



# Carga del Excel de coordenadas y KD‐Tree
@st.cache(allow_output_mutation=True)
def load_coords():
    dfc = pd.read_excel("coordenadas_mx.xlsx")
    tree = cKDTree(dfc['codigo_postal'].values.reshape(-1,1))
    return dfc, tree

def lookup_coord(cp:int, dfc, tree):
    m = dfc[dfc.codigo_postal==cp]
    if not m.empty:
        return m[['latitud','longitud']].iloc[0].tolist()
    _, idx = tree.query([[cp]])
    return dfc[['latitud','longitud']].values[idx[0]].tolist()

def haversine(lo,ld):
    R=6371.0
    φ1,φ2 = map(math.radians, lo), map(math.radians, ld)
    Δφ = math.radians(ld[0]-lo[0]); Δλ = math.radians(ld[1]-lo[1])
    a = math.sin(Δφ/2)**2 + math.cos(list(φ1)[0])*math.cos(list(φ2)[0])*math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ——— Carga de modelos ———

@st.cache(allow_output_mutation=True)
def load_models():
    inc_model  = joblib.load("xgb_pipeline.pkl")
    time_model = joblib.load("xgbreg_pipeline.pkl")
    return inc_model, time_model

inc_model, time_model = load_models()
dfc, tree = load_coords()

# ——— Interfaz ———

st.title("🦉 Recomendaciones Buho")
st.markdown("Introduce los datos de tu envío para obtener predicción de incidencia y tiempo de entrega.")

with st.form("input_form"):
    cp_origen  = st.text_input("Código Postal Origen", "53580")
    cp_destino = st.text_input("Código Postal Destino", "97000")
    rate       = st.number_input("Peso (kg)",  value=7.0,  min_value=0.1)
    servicio   = st.selectbox("Servicio", ["Terrestre","aereo"])
    fecha_str  = st.date_input("Fecha de Envío", datetime.today())
    carrier    = st.text_input("Paquetería (opcional)", "")
    submitted  = st.form_submit_button("Predecir")

if submitted:
    # 1) Armar DataFrame de raw
    raw = {
      "origin_pc":    int(cp_origen),
      "dest_pc":      int(cp_destino),
      "rate":         rate,
      "service_mode": servicio,
      "carrier":      carrier or "ALL",  # luego filtramos
      "start_date":   fecha_str.strftime("%Y-%m-%d")
    }
    df_raw = pd.DataFrame([raw])

    # 2) Feature engineering inline
    # convertir estados
    df_raw['origin_state'] = df_raw.origin_pc.apply(convert_estado)
    df_raw['dest_state']   = df_raw.dest_pc.apply(convert_estado)
    # coords & distancia
    lo = lookup_coord(int(cp_origen), dfc, tree)
    ld = lookup_coord(int(cp_destino), dfc, tree)
    df_raw['distance'] = round(haversine(lo, ld), 2)
    # day_sin, etc.
    wd = fecha_str.weekday(); m = fecha_str.month
    df_raw['day_sin']   = np.sin(2*np.pi*wd/7)
    df_raw['day_cos']   = np.cos(2*np.pi*wd/7)
    df_raw['month_sin'] = np.sin(2*np.pi*m/12)
    df_raw['month_cos'] = np.cos(2*np.pi*m/12)
    df_raw['is_weekend']= int(wd>=5)
    df_raw['rate_x_dist']= df_raw.rate * df_raw.distance
    df_raw['carrier_service'] = df_raw.carrier + "_" + df_raw.service_mode

    # 3) Evaluar incidencia (si carrier="ALL", iterar sobre lista)
    carriers = [carrier] if carrier and carrier!="ALL" else [
        "Estafeta","Paquetexpress","Fedex","Afimex","UPS","DHL","JT Express","Buho"
    ]
    resultados = {}
    for c in carriers:
        df_try = df_raw.copy()
        df_try['carrier_service'] = c + "_" + df_try.service_mode
        inc = int(inc_model.predict(df_try)[0])
        # 4) Inyectar incidencia y predecir tiempo
        df_try['incidence'] = inc
        time_pred = float(time_model.predict(df_try)[0])
        resultados[c] = {
            "incidence": inc,
            "delivery_time_pred": round(time_pred,2)
        }

    # 5) Mostrar
    st.subheader("📋 Resultados")
    st.json({
        "Info": raw,
        "Predictions": resultados
    })
