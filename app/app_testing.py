import os
import sys
import streamlit as st

# Asegurar que Python pueda encontrar el módulo 'app_src'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


from app_src.core.predictor import predict_new_sample
import pandas as pd
import streamlit as st

# IMPORTAR EL CARGADOR DE DATOS (Asegúrate de usar 'app_src' si renombraste la carpeta)
from app_src.utils.data_loader import load_map_data, load_model_variables, load_shap_data

st.set_page_config(page_title="TFM - Test de Datos", layout="wide")

st.title("🧪 Banco de Pruebas de Carga de Datos")

# Ejecutar las funciones de carga
st.subheader("1. Cargando datos del mapa...")
df_mapa = load_map_data()
if not df_mapa.empty:
    st.success(f"¡Éxito! Datos del mapa cargados. Dimensiones: {df_mapa.shape}")
    st.dataframe(df_mapa.head(3)) # Muestra las 3 primeras filas en una tabla interactiva

st.subheader("2. Cargando variables del modelo...")
df_variables = load_model_variables()
if not df_variables.empty:
    st.success(f"¡Éxito! Variables del modelo cargadas. Dimensiones: {df_variables.shape}")
    st.dataframe(df_variables.head(3))

st.subheader("3. Cargando ecosistema SHAP...")
expected_val_df, df_pred, df_shap = load_shap_data()
if not df_pred.empty and not df_shap.empty:
    st.success("¡Éxito! Archivos SHAP vinculados correctamente.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Pred Info Head:**")
        st.dataframe(df_pred.head(3))
    with col2:
        st.write("**SHAP Info Head:**")
        st.dataframe(df_shap.head(3))
    with col3:
        st.write("**SHAP Expected values:**")
        st.dataframe(expected_val_df.head(3))

# Muestra de prueba (Reemplaza con los nombres reales de tus variables del modelo)
df_test = df_variables.drop(columns=["CC","URB"])
df_test = df_test.iloc[[0]]

from app_src.core.predictor import predict_new_sample

if st.button("Probar Predicción"):
    clase, prob = predict_new_sample(df_test)
    st.write(f"Predicción: {clase} (Probabilidad: {prob:.2%})")