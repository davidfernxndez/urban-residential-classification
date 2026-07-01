import os
import sys
import streamlit as st

# Asegurar que Python pueda encontrar el módulo 'src' dentro de la carpeta 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar la vista del mapa que acabamos de escribir
from app_src.ui.map_view import render_map_interface

# 1. Configuración de la página (Debe ser la primera directiva de Streamlit)
st.set_page_config(
    page_title="TFM - Clasificación de Urbanizaciones",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Título principal y descripción de la plataforma
st.title("Plataforma de Análisis Urbano y Explicabilidad")
st.markdown("""
Esta plataforma web permite explorar la clasificación de tipologías urbanas de las urbanizaciones residenciales 
y evaluar el comportamiento del modelo **XGBoost** mediante técnicas de Inteligencia Artificial Explicable (**SHAP**).
""")

st.divider()

# 3. Creación de las dos pestañas de la interfaz
tab1, tab2 = st.tabs([
    "🗺️ Interfaz de Explicabilidad (Mapa)", 
    "🤖 Interfaz de Predicción"
])

# 4. Contenido temporal para la pestaña 1
with tab1:
    # Llamamos a la función para renderizar el mapa
    map_data = render_map_interface()

# 5. Contenido temporal para la pestaña 2
with tab2:
    st.header("Simulador de Predicciones en Tiempo Real")
    st.warning("Mensaje temporal: Aquí se configurarán las características de la urbanización ficticia para XGBoost.")