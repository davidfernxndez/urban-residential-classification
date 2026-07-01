import os
import pandas as pd
import streamlit as st
from pyproj import Transformer

# Definir la ruta raíz del proyecto (dos niveles arriba de este archivo: utils -> app_src -> app -> raíz)
# Esto asegura que las rutas funcionen sin importar desde dónde lances Streamlit.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@st.cache_data
def load_map_data() -> pd.DataFrame:
    """Carga los datos descriptivos espaciales para posicionar las urbanizaciones en el mapa."""
    path = os.path.join(BASE_DIR, "data", "processed", "descriptive_data.csv")
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo de mapa en: {path}")
        return pd.DataFrame()
    df_map = pd.read_csv(path)

    # Transform to correct coordinates for the folium map. In this dataset:
    # LON -> X_UTM
    # LAT -> Y_UTM
    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    df_map["Y_UTM"], df_map["X_UTM"] = transformer.transform(
        df_map["LAT"].values,
        df_map["LON"].values
    )    
    return df_map

@st.cache_data
def load_model_variables() -> pd.DataFrame:
    """Carga las variables originales de todas las muestras utilizadas por el modelo XGBoost."""
    path = os.path.join(BASE_DIR, "data", "processed", "model_data.csv")
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo de variables en: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_shap_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga el ecosistema de archivos SHAP locales generados en el experimento.
    Devuelve:
        - expected_value (pd.DataFrame): El valor base/sesgo del explainer de XGBoost.
        - pred_info (pd.DataFrame): Información de las predicciones de las muestras.
        - shap_info (pd.DataFrame): Matriz con los SHAP values de cada variable por muestra.
    """
    base_shap_path = os.path.join(BASE_DIR, "output", "SHAP_local", "XGBoost")
    
    expected_val_path = os.path.join(base_shap_path, "XGBoost_expected_shap_values.csv")
    pred_info_path = os.path.join(base_shap_path, "XGBoost_pred_info.csv")
    shap_info_path = os.path.join(base_shap_path, "XGBoost_shap_info.csv")
    
    # Verificación de existencia de archivos
    for p in [expected_val_path, pred_info_path, shap_info_path]:
        if not os.path.exists(p):
            st.error(f"Falta un archivo crítico de SHAP en la ruta: {p}")
            return 0.0, pd.DataFrame(), pd.DataFrame()
            
    # Lectura de dataframes

    expected_value = pd.read_csv(expected_val_path)
    pred_info = pd.read_csv(pred_info_path)
    shap_info = pd.read_csv(shap_info_path)
    
    return expected_value, pred_info, shap_info