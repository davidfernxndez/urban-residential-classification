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

# Ensure correct display on light and dark backgrounds in any browser
st.markdown(
    """
<style>
    /* ==========================================
       GLOBAL CONFIGURATION
       ========================================== */
    /* Force true light mode at the browser level */
    :root {
        color-scheme: light;
    }

    /* Top decorative line (Streamlit's accent bar) */
    [data-testid="stDecoration"] {
        background: #ff4b4b !important;
    }

    /* ==========================================
       MAIN CONTAINERS & BACKGROUNDS
       ========================================== */
    /* Full application background and default text color */
    .stApp {
        background-color: #ffffff;
        color: #31333f;
    }

    /* Main view container */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }

    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }

    /* Top header area */
    [data-testid="stHeader"] {
        background-color: white !important;
    }

    /* Header text, buttons, and inner elements */
    [data-testid="stHeader"] * {
        color: #31333f !important;
    }

    /* Toolbar (RUNNING / DEPLOY status) */
    [data-testid="stToolbar"] {
        background-color: white !important;
    }

    /* ==========================================
       TYPOGRAPHY & DIVIDERS
       ========================================== */
    /* Global text color override for headings and body text */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #31333f;
    }

    /* Custom Streamlit horizontal dividers (st.divider) */
    hr {
        border: none !important;
        height: 1px !important;
        background-color: #d3d3d3 !important;
        opacity: 1 !important;
    }

    /* ==========================================
       UI ELEMENTS & INPUTS
       ========================================== */
    /* Text inputs and textareas */
    input, textarea {
        background-color: white !important;
        color: black !important;
    }

    /* Expanders (st.expander) */
    details {
        background-color: white;
    }

    /* Full checkbox wrapper text color */
    .stCheckbox {
        color: #31333f !important;
    }

    /* Unchecked checkbox box style */
    .stCheckbox div[role="checkbox"] {
        background-color: white !important;
        border: 2px solid #9e9e9e !important;
    }

    /* Checked checkbox state */
    .stCheckbox div[aria-checked="true"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }

    /* Checkbox hover effect */
    .stCheckbox div[role="checkbox"]:hover {
        border-color: #ff4b4b !important;
    }

    /* ==========================================
       FLOATING COMPONENTS (TOASTS / NOTIFICATIONS)
       ========================================== */
    /* Toast container styling */
    [data-testid="stToast"] {
        background-color: white !important;
        color: #31333f !important;
        border: 1px solid #dcdcdc !important;
    }

    /* Toast inner text */
    [data-testid="stToast"] * {
        color: #31333f !important;
    }

    /* Toast icons */
    [data-testid="stToast"] svg {
        fill: #31333f !important;
    }            
</style>

<meta name="color-scheme" content="light">
""",
unsafe_allow_html=True,
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