import os
import joblib
import pandas as pd
import streamlit as st

# Definir la ruta raíz del proyecto (tres niveles arriba desde core -> app_src -> app -> raíz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@st.cache_resource
def load_xgboost_artifact() -> dict:
    """
    Carga el artefacto completo del modelo XGBoost (.pkl) desde la carpeta de outputs.
    Devuelve el diccionario con el modelo, el label encoder y el nombre.
    """
    model_path = os.path.join(BASE_DIR, "output", "models", "XGBoost_model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"No se pudo encontrar el artefacto del modelo en la ruta: {model_path}")
        return {}
    
    # Cargar el archivo .pkl utilizando joblib
    artifact = joblib.load(model_path)
    return artifact

def predict_new_sample(input_data: pd.DataFrame) -> tuple[str, float]:
    """
    Toma los datos de una urbanización ficticia, realiza la predicción con XGBoost
    y decodifica la etiqueta para devolver el nombre de la tipología urbana.
    
    Argumentos:
        input_data (pd.DataFrame): Un DataFrame de una sola fila con las variables
                                   estructuradas exactamente como las espera el modelo.
                                   
    Devuelve:
        predicted_label (str): Nombre de la tipología urbana predicha.
        confidence (float): Probabilidad o confianza asignada a esa clase (0.0 a 1.0).
    """
    label_dict = {
        1: "Protegido",
        2: "Controlado",
        3: "Autoaislado",
        4: "Individualista",
        5: "Simbólico"
    }
    # 1. Recuperar el artefacto de la caché
    artifact = load_xgboost_artifact()
    if not artifact:
        return "Error en el modelo", 0.0
    
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    
    # 2. Ejecutar la predicción
    # Obtener el índice numérico de la clase predicha
    prediction_idx = model.predict(input_data)[0]
    
    # Obtener las probabilidades de todas las clases y extraer la de la clase ganadora
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(probabilities[prediction_idx])
    
    # 3. Traducir el índice numérico al nombre de la tipología original
    predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
    predicted_label = label_dict[predicted_label]

    return predicted_label, confidence