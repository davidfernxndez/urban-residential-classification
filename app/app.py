import os
import sys
import streamlit as st

# Asegurar que Python pueda encontrar el módulo 'src' dentro de la carpeta 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from app_src.ui.explainability_view import render_explainability_interface
from app_src.ui.predict_view import render_predict_interface
from app_src.ui.context_view import render_context_view


# 1. Configuración de la página (Debe ser la primera directiva de Streamlit)
st.set_page_config(
    page_title="TFM - Clasificación de Urbanizaciones",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Análisis de la Fragmentación Urbana en Granada")
st.caption("Plataforma Inteligente para el Estudio del Grado de Cerramiento Residencial mediante Aprendizaje Automático (XGBoost y SHAP)")
st.divider()

st.markdown("""
    <style>
        /* Agranda el texto de los botones de las pestañas superiores */
        button[data-baseweb="tab"] p {
            font-size: 20px !important;  /* Ajusta este valor a tu gusto */
            font-weight: bold !important;
        }
        /* Añade un poco de espacio (padding) para que sean más cómodas de clicar */
        button[data-baseweb="tab"] {
            padding: 12px 24px !important;
        }
    </style>
""", unsafe_allow_html=True)


# Menú de navegación
tab_init, tab_explainability, tab_predict = st.tabs([
    "🏠 Inicio y Contexto", 
    "🗺️ Explorador Geográfico", 
    "🔮 Simulador de Diseño"
])

with tab_init:
    render_context_view()
with tab_explainability:
    render_explainability_interface()
with tab_predict:
    render_predict_interface()
