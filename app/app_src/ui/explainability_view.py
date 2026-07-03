"""
Streamlit explainability interface: Combines geospatial visualization and SHAP-based local
interpretability in an interactive dashboard.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from app_src.utils.data_loader import load_map_data, load_dataset, load_shap_data
from app_src.appConfig import config


def extract_selected_cc(map_output):
    """
    Extracts the unique identifier (CC) of the residential complex clicked by the user
    by matching the coordinates returned by st_folium against the spatial dataset.

    Parameters
    ----------
        map_output (dict or None): The dictionary returned by the st_folium component.

    Returns
    -------
        str or None: The unique complex identifier ('CC') if a match is found, 
                     None otherwise.
    """
    # Verify if map_output contains a valid click event
    if not map_output or not map_output.get("last_object_clicked"):
        return None

    # Extract latitude and longitude folium coordinates
    click_info = map_output["last_object_clicked"]
    lat_click = click_info.get("lat")
    lon_click = click_info.get("lng")

    if lat_click is None or lon_click is None:
        return None

    # Load map data 
    df_map = load_map_data()

    # Match coordinates using np.isclose to tolerate minor JavaScript decimal rounding errors
    # Note: Folium's 'lat' corresponds to X_UTM, and 'lng' corresponds to Y_UTM
    matched_row = df_map[
        np.isclose(df_map["X_UTM"], lat_click, atol=1e-5) & 
        np.isclose(df_map["Y_UTM"], lon_click, atol=1e-5)
    ]

    if not matched_row.empty:
        return str(matched_row.iloc[0]["CC"])
        
    return None


def get_shap_waterfall(selected_cc, num_features_display=11, figsize=[9, 5.5]):
    """
    This method generates a SHAP waterfall plot to explain a specific prediction of sample identified
    by selected_cc input parameter.
    
    Parameters
    ----------
    selected_cc: str
        CC identifier of the residential complex clicked by the user
    
    num_features_display : int [optional, default=11]
        Maximum number of features displayed in the Waterfall plot.
    
    figsize: list [optional, default=[9, 5.5]]
        Width and height values of the graph.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Object figure that contains the generated visualization.
    """
    # Convert string to int to filter dataframes
    cc_int = int(selected_cc) 
  
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    dataset_df = load_dataset()
    expected_value_df, pred_info_df, shap_values_df = load_shap_data()

    # =========================================================
    # GET FEATURE DATA 
    # =========================================================

    # Filter dataframe for selected cc
    features_values_df = dataset_df[
        dataset_df[id_variable] == cc_int
    ].copy()
    features_values_df = features_values_df.drop(
        columns=[id_variable, target_variable]
    )
    # Convert to 1D-Array
    feature_values = features_values_df.iloc[0].values

    # Get feature names to keep features order
    feature_names = features_values_df.columns

    # =========================================================
    # EXTRACT PREDICTION INFORMATION
    # ========================================================= 

    # Get predicted class
    pred_class = pred_info_df.loc[
        pred_info_df[id_variable] == cc_int,
        f"{target_variable}_pred"
    ].iloc[0]
    pred_class_label = target_label_map[pred_class]

    # Get fold of the class to extract expected value
    fold_idx = pred_info_df.loc[
        pred_info_df[id_variable] == cc_int,
        "fold"
    ].iloc[0]

    # ==========================================================
    # EXTRACT EXPECTED VALUE FOR PREDICTED CLASS (SHAP REFERENCE)
    # ==========================================================
    expected_value = expected_value_df.loc[
        expected_value_df["fold"] == fold_idx,
        f"class_{pred_class}"
    ].iloc[0]

    # ==========================================================
    # GET SHAP VALUES FOR PREDICTED CLASS
    # ==========================================================    
    sample_shap_values_df = shap_values_df[
        (shap_values_df[id_variable] == cc_int) &
        (shap_values_df[target_variable] == pred_class)
    ].copy()

    # Order features according to feature names
    sample_shap_values_df["feature"] = pd.Categorical(
        sample_shap_values_df["feature"],
        categories=feature_names,
    ordered=True
    )
    sample_shap_values_df = sample_shap_values_df.sort_values("feature")

    # Get values as 1D array
    shap_values = sample_shap_values_df["shap_value"].values

    # ==========================================================
    # WATERFALL PLOT
    # ==========================================================    

    # Create explainer object
    exp = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=feature_values,
        feature_names=feature_names
    )

    # Generate SHAP waterfall plot
    shap.plots.waterfall(
        exp,
        max_display=num_features_display,
        show=False
    )
    # Custom plot with matplotlib
    fig = plt.gcf()  
    fig.set_size_inches(figsize[0], figsize[1])  

    # Add grid and adjust marginds
    plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
  
    plt.tight_layout()

    return fig, pred_class_label


def render_descriptive_data(selected_cc, descriptive_df):
    """
    Renders an aesthetic, structured dashboard panel 
    displaying descriptive variables, model prediction, and 
    semantic blocks of morphological binary features for the selected
    residential complex.
    
    Parameters
    ----------
    selected_cc: str
        CC identifier of the residential complex clicked by the user
    
    descriptive_df : pandas.DataFrame
        Dataframe with descriptive information: municipality,
        actual and predicted classes.
    
    Returns
    -------
    None
        Displays the render text in streamlit interface
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE
    class_styles = config.CLASS_STYLES

    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    dataset_df = load_dataset()

    # =========================================================
    # DESCRIPTIVE SECTION
    # =========================================================

    # Selected descriptive data from map dataframe for current
    # residential complex (selected cc)
    row_map = descriptive_df[descriptive_df[id_variable] == selected_cc].iloc[0]

    # Set main title
    st.markdown("### 🔍 Explicabilidad de la predicción con SHAP")
    st.markdown(
        """
        <style>
        /* Label configuration */
        div[data-testid="stMetricLabel"] > div {
            font-size: 1.4rem !important;
            font-weight: 500 !important;
            color: #555555 !important;
        }
        /* Value configuration */
        div[data-testid="stMetricValue"] > div {
            font-size: 1.6rem !important;
            font-weight: bold !important;
            white-space: normal !important; 
            word-wrap: break-word !important; 
            overflow-wrap: break-word !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        st.metric(label="Identificador (CC)", value=selected_cc)
    with col_id2:
        st.metric(label="Municipio", value=str(row_map["MUN"]).replace('_',' '))
        
    st.markdown("---")

    # =========================================================
    # MODEL PREDICTION SECTION
    # =========================================================
    # Select class styles for actual and predicted class
    actual_info = class_styles.get(int(row_map[target_variable]), {"label": "Desconocido", "color": "gray"})
    pred_info = class_styles.get(int(row_map[f"{target_variable}_pred"]), {"label": "Desconocido", "color": "gray"})
    
    # Customize container color based on the accuracy of the prediction.
    is_hit = int(row_map[target_variable]) == int(row_map[f"{target_variable}_pred"])
    badge_bg = "#e8f5e9" if is_hit else "#fff3e0"  
    badge_border = "#2e7d32" if is_hit else "#ef6c00"
    model_evaluation_text = "correcta" if is_hit else "errónea"
    evaluation_icon = "✅" if is_hit else "⚠️"
    

    st.markdown(
        f"""
        <div style='background-color: {badge_bg}; padding: 14px; border-radius: 8px; 
                    border: 1px solid {badge_border}; margin-bottom: 20px;'>
            <p style='margin: 0 0 10px 0; font-size: 1.3rem; font-weight: bold;'>
                {evaluation_icon} Predicción del modelo {model_evaluation_text}
            </p>
            <div style='display: flex; justify-content: space-between; margin-top: 5px; font-size: 1.05rem;'>
                <span><b>Etiqueta Real:</b> <span style='color:{actual_info['color']}; font-weight: bold;'>{actual_info['label']}</span></span>
                <span><b>Predicción:</b> <span style='color:{pred_info['color']}; font-weight: bold;'>{pred_info['label']}</span></span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
 
    # =========================================================
    # FEATURE DATA SECTION
    # =========================================================
    st.markdown("#### 🗂️ Características Morfológicas")

    # Selec feature data from current residential complex (selected cc)
    row = dataset_df[dataset_df[id_variable] == int(selected_cc)].iloc[0]

    # Auxiliary function to format the presence and absence of variables
    def format_binary_feature(label, value):
        icon = "✅ <span style='color: #2e7d32; font-weight: 500;'>" if int(value) == 1 else "❌ <span style='color: #9e9e9e; font-style: italic;'>"

        return f"<div>{icon}{label}</div>"

    # Structural aspects
    with st.expander("🧱 Aspectos Estructurales", expanded=True):
        html_content = "".join([
            format_binary_feature("Calle sin salida (CSS)", row["CSS"]),
            format_binary_feature("Calle en fondo de saco (CFS)", row["CFS"]),
            format_binary_feature("Calle peatonal (CPE)", row["CPE"]),
            format_binary_feature("Comercios internos (LIN)", row["LIN"]),
            format_binary_feature("Comercios externos (LEX)", row["LEX"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # Distance to urban core
    with st.expander("📍 Distancia al Núcleo Urbano", expanded=False):
        if int(row.get("DIS_1", 0)) == 1:
            st.info("⛰️ **Aislado del núcleo urbano**")
        elif int(row.get("DIS_2", 0)) == 1:
            st.info("🏡 **Separadado del núcleo urbano**")
        elif int(row.get("DIS_3", 0)) == 1:
            st.info("🏢 **Integrado en núcleo urbano**")
        else:
            st.warning("No se ha registrado localización.")

    # Enclosure elements
    with st.expander("⛔ Elementos de Cerramiento", expanded=False):
        html_content = "".join([
            format_binary_feature("Verjas (VER)", row["VER"]),
            format_binary_feature("Muros (MUR)", row["MUR"]),
            format_binary_feature("Cadenas (CAD)", row["CAD"]),
            format_binary_feature("Bolardos (BOL)", row["BOL"]),
            format_binary_feature("Arbustos (ARB)", row["ARB"]),
            format_binary_feature("Carteles propiedad privada (CPP)", row["CPP"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # Acces type
    with st.expander("🚪 Puntos de Acceso", expanded=False):
        html_content = "".join([
            format_binary_feature("Entrada por vivienda (PVI)", row["PVI"]),
            format_binary_feature("Entrada por bloque (PBL)", row["PBL"]),
            format_binary_feature("Entrada común (COM)", row["COM"]),
            format_binary_feature("Varias entradas comunes (COMS)", row["COMS"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # Use of public roads
    with st.expander("🚧 Uso de la Vía Pública", expanded=False):
        html_content = "".join([
            format_binary_feature("Dominio público y uso público (PPU)", row["PPU"]),
            format_binary_feature("Dominio privado y uso público restringido (PRE)", row["PRE"]),
            format_binary_feature("Dominio privado y uso privado (PPR)", row["PPR"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # Security
    with st.expander("🛡️ Seguridad y Vigilancia", expanded=False):
        html_content = "".join([
            format_binary_feature("Guardia de seguridad (GSE)", row["GSE"]),
            format_binary_feature("Cámara de seguridad (CSE)", row["CSE"]),
            format_binary_feature("Barrera de seguridad (BSE)", row["BSE"]),
            format_binary_feature("Alarma de seguridad (ASE)", row["ASE"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)


def generate_shap_explication(selected_cc, max_features=4, min_ratio=0.25):
    """
    Generate a text explanation of the predicted class using SHAP values.

    The function analyzes the SHAP contributions associated with the selected
    residential complex and builds a human-readable explanation highlighting
    the most relevant features influencing the prediction.

    Parameters
    ----------
    selected_cc: str
        CC identifier of the residential complex clicked by the user

    max_features : int [optional, default=4]
        Maximum number of relevant features to include in the explanation.

    min_ratio : float [optional, default=0.25]
        Minimum SHAP contribution ratio with respect to the maximum SHAP value
        required for a feature to be included in the explanation.

    Returns
    ----------
    html_text: str
        HTML-formatted text containing the explainability message generated
        from the SHAP values of the predicted class.
    """

    # Convert string to int to filter dataframes
    cc_int = int(selected_cc) 

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE
    target_label_map = config.TARGET_LABEL_MAP
    feature_description = config.FEATURE_DESCRIPTION

    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    dataset_df = load_dataset()
    _, pred_info_df, shap_values_df = load_shap_data()

    # ==========================================================
    # GET SHAP VALUES FOR PREDICTED CLASS
    # ========================================================== 

    # Get predicted class
    pred_class = pred_info_df.loc[
        pred_info_df[id_variable] == cc_int,
        f"{target_variable}_pred"
    ].iloc[0]
    pred_class_label = target_label_map[pred_class]
    
    # Select shap values
    sample_shap_values_df = shap_values_df[
        (shap_values_df[id_variable] == cc_int) &
        (shap_values_df[target_variable] == pred_class)
    ].copy()

    # ==========================================================
    # SELECT MORE RELEVANT FEATURES
    # ==========================================================   
    # Sort by positive SHAP contribution
    sample_shap_values_df = sample_shap_values_df.sort_values("shap_value", ascending=False)

    # Get Max value
    max_shap = sample_shap_values_df["shap_value"].iloc[0]

    # Select more relevance features
    relevant_features = sample_shap_values_df[sample_shap_values_df["shap_value"] >= max_shap * min_ratio]
    relevant_features = relevant_features.head(max_features)

    # Guarantee at least one variable
    if len(relevant_features) == 0:
        relevant_features = sample_shap_values_df.head(1)

    # ==========================================================
    # BUILD EXPLANABILITY TEXT
    # ========================================================== 
    #  
    # List to store explanability sentences
    sentences = []
    for _, row in relevant_features.iterrows():
        
        # Get feature name, description and current value
        feature = row["feature"]
        feature_text = feature_description.get(feature, feature)
        feature_value = dataset_df.loc[dataset_df["CC"]==cc_int, feature].iloc[0]

        if feature_value == 1:
            sentences.append(
                f"tiene <b style='color: #2e7d32;'>{feature_text}</b> "
                f"(<code style='color: #2e7d32; background-color: #e8f5e9;'>{feature}=1</code>)"
            )
        else:
            sentences.append(
                f"no tiene <b style='color: #c62828;'>{feature_text}</b> "
                f"(<code style='color: #c62828; background-color: #ffebee;'>{feature}=0</code>)"
            )

    # Build complete explanability text
    if len(sentences) == 1:
        description = sentences[0]

    elif len(sentences) == 2:
        description = " y ".join(sentences)

    else:
        description = ", ".join(sentences[:-1]) + " y " + sentences[-1]

    html_text = f"""
        <div style='
        background-color: #f8f9fa; 
        border-left: 5px solid #2b5c8f; 
        padding: 18px; 
        border-radius: 4px; 
        margin-top: 10px;
        margin-bottom: 20px;
        line-height: 1.6;
        color: #333333;
        font-size: 1.15rem; /* <-- Aumenta el tamaño de todo el texto base */
        '>
        <span style='font-size: 1.3rem; vertical-align: middle;'>💡</span> 
        El complejo residencial ha sido clasificado en el grado de cerramiento
        <span style='color: #000000; font-weight: bold; font-size: 1.2rem;'>{pred_class_label}</span> 
        debido a que {description}.
        </div>
    """

    return html_text


def render_explainability_interface():
    """
    Render the SHAP-based explainability interface for streamlit platform, including:
    - A Folium-based interactive map with color-coded residential complexes
        according to their predicted enclosure level.
    - A legend describing classification categories.
    - A detail panel that updates dynamically when a user selects a residential
        complex on the map.
    - A SHAP waterfall plot and a human-readable explanation of the model
        prediction for the selected complex.

    Parameters
    -----------
    None

    Returns
    -----------  
    None
    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    class_styles = config.CLASS_STYLES

    # ==========================
    # LOAD REQUIRED DATA
    # ==========================

    # Load underlying spatial dataset containing pre-calculated WGS84 coordinates (X_UTM, Y_UTM)
    df_map = load_map_data()
    if df_map.empty:
        st.warning("The display information on the map (longitude and latitude) is not available.")
        return
    
    # Load predictive information to cross-reference classification labels belongs to.
    _, df_pred, _ = load_shap_data()
    if df_pred.empty:
        st.warning("The prediction information is not available.")
        return   
     
    # Merge predicted label to map dataframe
    df_map["CC"] = df_map["CC"].astype(str)
    df_pred["CC"] = df_pred["CC"].astype(str)
    pred_dict = dict(zip(df_pred["CC"], df_pred["URB_pred"]))
    df_map["URB_pred"] = df_map["CC"].map(pred_dict)

    # =========================================================
    # UI Headers and Descriptions
    # ========================================================= 
    st.subheader("Predicción y explicabilidad del censo de complejos residenciales (área Metropolitana de Granada)")
    st.markdown(
        """
        <p style='color: #475569; font-size: 1.12rem; line-height: 1.5; margin-bottom: 15px;'>
            💡 Desplázese sobre el mapa y sitúe el cursor sobre los complejos residenciales para ver su información.
            <span style='background-color: #e6effa; color: #1e3d59; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Haz clic sobre el marcador</span> 
            para desplegar el análisis de explicabilidad mediante SHAP en el panel inferior.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # ==========================
    # FOLIUM MAP CONFIGURATION
    # ==========================

    # Calculate geographic barycenter (centroid) for adaptive map viewport initializing
    center_x = df_map["X_UTM"].mean()
    center_y = df_map["Y_UTM"].mean()

    # Base Map Customization (Esri Satellite Layer)
    m = folium.Map(
        location=[center_x, center_y], 
        zoom_start=11, 
        max_zoom=22,
        control_scale=True,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )
    # Fallback routine: configure over-zooming. Digital upscaling triggers beyond native zoom (lvl 19)
    # up to max canvas resolution (lvl 22), preventing rendering black screens.
    for child in m._children.values():
        if isinstance(child, folium.TileLayer):
            child.options["max_native_zoom"] = 19
            child.options["max_zoom"] = 22

    # Optimized Spatial Clustering Setup
    # Deactivate cluster pooling automatically at granular levels (zoom >= 16) to expose raw colors
    marker_cluster = marker_cluster = MarkerCluster(
        options={
            'disableClusteringAtZoom': 16,
            'maxClusterRadius': 50
        }
    ).add_to(m)

    # ==========================
    # MARKERS AND TOOLTIPS
    # =========================
    for row in df_map.itertuples():
        # Get sample ID (CC variable), municipality and predicted label
        id = row.CC
        mun = str(row.MUN).replace('_',' ')
        class_value = row.URB_pred

        # Retrieve mapped layout profile
        style = class_styles.get(class_value, {"label": "Desconocido", "color": "gray", "icon": "question-circle-fill"})
        
        # Build clean string templates for interactive tooltips
        tooltip_text = tooltip_text = (
            f"<b>CC:</b> {id}<br>"
            f"<b>Municipio:</b> {mun}<br>"
            f"<b>Grado de cerramiento:</b> {style['label']}"
        )

        # Instantiate responsive node marker linking unique ID to the property name
        folium.Marker(
            location=[row.X_UTM, row.Y_UTM],
            icon=folium.Icon(
                color=style["color"],
                prefix="bi"
            ),
            tooltip=tooltip_text,
            name=id       
        ).add_to(marker_cluster)

    #########################################
    # CANVAS LAYOUT 
    #########################################
    # CSS configuration to prevent the folium map from leaving blank margins below and above it
    st.markdown("""
    <style>

    /* Main container of the folium component */
    div[data-testid="stElementContainer"]:has(iframe) {
        height: 520px !important;
    }

    /* Internal wrapper */
    div[data-testid="stElementContainer"]:has(iframe) > div {
        height: 520px !important;
    }

    /* iframe */
    iframe {
        height: 550px !important;
        display: block;
    }

    </style>
    """, unsafe_allow_html=True)

    # --- ROW 1: MAP (LEFT) AND LEGEND (RIGHT) ---
    # A 5:1 aspect ratio is used so that the map dominates almost the entire screen.

    col_map, col_legend = st.columns([5, 1])
    with col_map:
        # st_folium adapts to the width of the page
        map_output = st_folium(
            m, 
            use_container_width=True, 
            height=550,  
            key="map",
            returned_objects=["last_object_clicked"]
        )

    with col_legend:
        with st.container(border=True):
            st.markdown("<div class='legend-container'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.2rem; font-weight: bold; margin-bottom: 20px; color: #1e3d59;'>Grado de cerramiento</p>", unsafe_allow_html=True)
            hex_colors = {"red": "🔴", "orange": "🟠", "purple": "🟣", "blue": "🔵", "green": "🟢"}

            # Legend style
            for class_key, info in class_styles.items():
                emoji = hex_colors.get(info["color"], "⚪")
                st.markdown(f"{emoji} **{class_key}**: {info['label']}")

            st.markdown("</div>", unsafe_allow_html=True)
    

    # --- EXTRACTION AND PERSISTENCE OF THE ID SELECTED BY THE USER IN THE MAP (CC) ---
    selected_cc = extract_selected_cc(map_output)

    # --- ROW 2: DESCRIPTIVE DATA (LEFT) AND SHAP GRAPH (RIGHT) ---
    if selected_cc:
        st.session_state["selected_urbanizacion"] = selected_cc
        st.toast(f"Complejo residencial seleccionado: {selected_cc}", icon="📌")

        with st.container(border=True):
            col_data, _, col_graph = st.columns([2, 0.1, 3]) 
            with col_data:
                render_descriptive_data(selected_cc, df_map)

            with col_graph:
                # Get SHAP Waterfall GRAPH
                fig_shap, pred_class_label = get_shap_waterfall(selected_cc, num_features_display=11, figsize=[9, 5.5])
                if fig_shap:
                    # Display figure
                    st.markdown(f"""### 📊 Waterfall Plot de la clase predicha: ''{pred_class_label}''""")
                    with st.container(border=True):
                        st.pyplot(fig_shap, clear_figure=True, bbox_inches="tight")
                    
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

                    # Display explainability text
                    explainability_text = generate_shap_explication(selected_cc=selected_cc)
                    st.markdown(explainability_text, unsafe_allow_html=True)
    else:
        # Waiting message
        with st.container(border=True):
            st.markdown(
                """
                <div style='
                    text-align: center; 
                    padding: 80px 20px; 
                    background-color: #f8f9fa;
                    border-radius: 6px;
                '>
                    <div style='font-size: 2.5rem; margin-bottom: 15px;'>📊</div>
                    <h3 style='color: #1e3d59; font-weight: 600; margin-bottom: 10px;'>
                        PANEL DE EXPLICABILIDAD LOCAL CON SHAP
                    </h3>
                    <p style='color: #555555; font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                        Haz clic en cualquier <b>complejo residencial</b> del mapa para cargar su análisis explicativo mediante SHAP.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )