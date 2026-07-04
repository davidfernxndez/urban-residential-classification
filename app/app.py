"""
XAI-URBAN Streamlit Application Entry Point.

This file defines the main interface of the XAI-URBAN platform, an explainable AI system
for the classification of urban residential typologies based on morphological features.

The application is structured into three main views:
- Context: description of the problem, dataset, and methodological framework.
- Explainability: interactive geospatial exploration with SHAP-based model interpretation.
- Predictor: simulation tool for predicting new residential developments.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import streamlit as st

# Ensure that streamlit cloud service can read from app_src/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure that streamlit cloud service can read from src/ folder
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app_src.ui.explainability_view import render_explainability_interface
from app_src.ui.predict_view import render_predict_interface
from app_src.ui.context_view import render_context_view
from app_src.appConfig import config

# Page configuration
st.set_page_config(
    page_title="GRANADA_XAI_URBAN",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<div style="
    text-align: left;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-top: 10px;
    color: #1f1f1f;
">
Inteligencia artificial explicable (XAI) para la clasificación de tipologías urbanas
</div>

<div style="
    text-align: left;
    font-size: 14px;
    color: #6b6b6b;
    margin-top: 4px;
    margin-bottom: 18px;
">
Sistema basado en XGBoost y SHAP
</div>
""", unsafe_allow_html=True)

# Author and social networks
st.markdown(f"""
    <div style="
        display: flex;
        justify-content: left;
        align-items: left;
        gap: 14px;
        margin-top: 2px;
        margin-bottom: -2px;
        font-size: 14.5px;
        color: #4a4a4a;
        letter-spacing: 0.2px;
    ">

    <span style="font-weight: 600;">👤 {config.NAME}</span>

    
    <span style="opacity: 0.6;">•</span>
    <a href={config.GITHUB_LINK} target="_blank"
    style="
        text-decoration: none;
        color: #4a4a4a;
        font-weight: 500;
    ">
    💻 GitHub
    </a>
    <span style="opacity: 0.6;">•</span>
    <a href={config.LINKEDIN_LINK} target="_blank"
    style="
        text-decoration: none;
        color: #4a4a4a;
        font-weight: 500;
    ">
    🔗 LinkedIn
    </a>

    </div>
    """, 
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
    <style>
        button[data-baseweb="tab"] p {
            font-size: 18px !important; 
            font-weight: bold !important;
        }
        button[data-baseweb="tab"] {
            padding: 3px 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Navegation Menu
tab_init, tab_explainability, tab_predict = st.tabs([
    " CONTEXTO", 
    "EXPLICABILIDAD", 
    "PREDICTOR"
])

with tab_init:
    render_context_view()
with tab_explainability:
    render_explainability_interface()
with tab_predict:
    render_predict_interface()

# Author and social networks
st.divider()
st.markdown(f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 14px;
        margin-top: 10px;
        margin-bottom: -1px;
        font-size: 40px;
        color: #4a4a4a;
        letter-spacing: 0.2px;
    ">

    <span style="font-weight: 600;">👤 {config.NAME}</span>
    <span style="opacity: 0.6;">•</span>
    <a href={config.GITHUB_LINK} target="_blank"
    style="
        text-decoration: none;
        color: #4a4a4a;
        font-weight: 500;
    ">
    💻 GitHub
    </a>
    <span style="opacity: 0.6;">•</span>
    <a href={config.LINKEDIN_LINK} target="_blank"
    style="
        text-decoration: none;
        color: #4a4a4a;
        font-weight: 500;
    ">
    🔗 LinkedIn
    </a>

    </div>
    """, 
    unsafe_allow_html=True
)