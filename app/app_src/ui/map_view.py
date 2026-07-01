import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from app_src.utils.data_loader import load_map_data, load_model_variables, load_shap_data

import numpy as np
import pandas as pd
import shap


def extract_selected_cc(map_output):
    """
    Extracts the unique identifier (CC) of the residential complex clicked by the user
    by matching the coordinates returned by st_folium against the spatial dataset.

    Args:
        map_output (dict or None): The dictionary returned by the st_folium component.

    Returns:
        str or None: The unique complex identifier ('CC') if a match is found, 
                     None otherwise.
    """
    # Verify if map_output contains a valid click event
    if not map_output or not map_output.get("last_object_clicked"):
        return None

    click_info = map_output["last_object_clicked"]
    lat_click = click_info.get("lat")
    lon_click = click_info.get("lng")

    if lat_click is None or lon_click is None:
        return None

    # Load spatial data (utilizes cache under the hood)
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


def get_shap_waterfall(cc, num_features_display=11, figsize=[10,8]):
    """
    This method displays a SHAP waterfall plot to explain a specific prediction of sample identified
    by id_sample input parameter.
    
    Parameters
    ----------
 
    id_sample: int
        ID value to get sample information using id_variable column in dataframes.
    
    model_name : str
        The name of the algorithm being explained with SHAP.
        This controls conditional logic for SHAP explainers and determines file naming.

    num_features_display : int [optional, default=11]
        Maximum number of features displayed in the Waterfall plot.
    
    figsize: list [optional, default=[10, 18]]
        Width and height values of the graph.

    Returns
    -------
    None
    """
    cc_int = int(cc)  # Convierte el String (ej. "12003") a entero (12003)

    label_dict = {
        1: "Protegido",
        2: "Controlado",
        3: "Autoaislado",
        4: "Individualista",
        5: "Simbólico"
    }    
    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    dataset_df = load_model_variables()
    expected_value_df, pred_info_df, shap_values_df = load_shap_data()

    # =========================================================
    # GET FEATURE DATA 
    # =========================================================

    # Filter dataframe for id_sample
    features_values_df = dataset_df[
        dataset_df["CC"] == cc_int
    ].copy()
    features_values_df = features_values_df.drop(
        columns=["CC", "URB"]
    )
    # Convert to 1D-Array
    feature_values = features_values_df.iloc[0].values

    # Get feature names to keep features order
    feature_names = features_values_df.columns

    # =========================================================
    # EXTRACT PREDICTION INFORMATION
    # ========================================================= 

    # Get actual class
    actual_class = pred_info_df.loc[
        pred_info_df["CC"] == cc_int,
        "URB"
    ].iloc[0]

    # Get predicted class
    pred_class = pred_info_df.loc[
        pred_info_df["CC"] == cc_int,
        "URB_pred"
    ].iloc[0]

    # Get fold of the class to extract expected value
    fold_idx = pred_info_df.loc[
        pred_info_df["CC"] == cc_int,
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
        (shap_values_df["CC"] == cc_int) &
        (shap_values_df["URB"] == pred_class)
    ]

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

    shap.plots.waterfall(
        exp,
        max_display=num_features_display,
        show=False
    )

    fig = plt.gcf()  
    fig.set_size_inches(9, 5.5)  # Ancho 9, Alto 5.5 es la proporción áurea para col_graph
    fig.suptitle(
        "SHAP WATERFALL PLOT",
        fontsize=14,
        color="midnightblue", 
        weight="bold",
        x=0.1, 
        ha="left"
    )
    # 4. Añadir rejilla sutil de fondo para facilitar la lectura de los valores
    plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    
    # 5. Ajustar márgenes de forma estricta
    # Al eliminar subplots_adjust(top=0.82), usamos tight_layout para que las barras 
    # ocupen el 100% del lienzo útil, eliminando por completo el hueco en blanco.
    plt.tight_layout()
    
    # IMPORTANTE: Devolvemos la figura en lugar de pintarla en caliente
    return fig


def render_complex_metadata(selected_cc, df_map, class_styles):
    """
    Renders an aesthetic, structured dashboard panel on the left column 
    displaying identification metrics, model evaluation badges, and 
    semantic blocks of morphological binary features for the selected complex.
    """
    # 1. Recuperar la fila de datos del complejo seleccionado
    row_map = df_map[df_map["CC"] == selected_cc].iloc[0]
    
    # 2. SECCIÓN: Identificación Principal
    st.markdown("### 🎯 Explicabilidad de la predicción con SHAP")
    
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        st.metric(label="Identificador (CC)", value=selected_cc)
    with col_id2:
        st.metric(label="Municipio", value=str(row_map["MUN"]))
        
    st.markdown("---")

    # 3. SECCIÓN: Evaluación de Etiquetas (Real vs Predicha)
    # Extraemos estilos para las clases
    actual_info = class_styles.get(int(row_map["URB"]), {"label": "Desconocido", "color": "gray"})
    pred_info = class_styles.get(int(row_map["URB_pred"]), {"label": "Desconocido", "color": "gray"})
    
    # Lógica de acierto del modelo para el color de fondo del contenedor
    is_hit = int(row_map["URB"]) == int(row_map["URB_pred"])
    badge_bg = "#e8f5e9" if is_hit else "#fff3e0"  # Verde suave si acierta, naranja suave si falla
    badge_border = "#2e7d32" if is_hit else "#ef6c00"
    
    st.markdown(
        f"""
        <div style='background-color: {badge_bg}; padding: 12px; border-radius: 8px; 
                    border: 1px solid {badge_border}; margin-bottom: 20px;'>
            <p style='margin: 0; font-size: 0.9rem; color: #555;'><b>Evaluación del Modelo:</b></p>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span><b>Clase Real:</b> {actual_info['label']}</span>
                <span><b>Predicción:</b> <span style='color:{pred_info['color']}; fontweight:bold;'>{pred_info['label']}</span></span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("#### 📐 Características Morfológicas")
    dataset_df = load_model_variables()
    row = dataset_df[dataset_df["CC"] == int(selected_cc)].iloc[0]
    # Función interna auxiliar para formatear presencia/ausencia estéticamente
    def format_binary_feature(label, value):
        icon = "✅ <span style='color: #2e7d32; font-weight: 500;'>" if int(value) == 1 else "❌ <span style='color: #9e9e9e; font-style: italic;'>"
        end = " (Presente)</span>" if int(value) == 1 else " (Ausente)</span>"
        return f"<div>{icon}{label}{end}</div>"

    # 4. BLOQUE: Aspectos Estructurales
    with st.expander("🧱 Aspectos Estructurales", expanded=True):
        html_content = "".join([
            format_binary_feature("Calles sin salida (CSS)", row["CSS"]),
            format_binary_feature("Calle en fondo de saco (CFS)", row["CFS"]),
            format_binary_feature("Calle peatonal (CPE)", row["CPE"]),
            format_binary_feature("Comercios internos (LIN)", row["LIN"]),
            format_binary_feature("Comercios externos (LEX)", row["LEX"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # 5. BLOQUE: Distancia al Núcleo Urbano (Selección Dinámica)
    with st.expander("📍 Distancia al Núcleo Urbano", expanded=False):
        if int(row.get("DIS_1", 0)) == 1:
            st.info("🏞️ **Urbanización aislada**")
        elif int(row.get("DIS_2", 0)) == 1:
            st.info("🏡 **Urbanización separada**")
        elif int(row.get("DIS_3", 0)) == 1:
            st.info("🏢 **Urbanización integrada en casco urbano**")
        else:
            st.warning("No se ha registrado localización activa.")

    # 6. BLOQUE: Tipo de Cerramiento
    with st.expander("🚧 Elementos de Cerramiento", expanded=False):
        html_content = "".join([
            format_binary_feature("Verjas (VER)", row["VER"]),
            format_binary_feature("Muros (MUR)", row["MUR"]),
            format_binary_feature("Cadenas (CAD)", row["CAD"]),
            format_binary_feature("Bolardos (BOL)", row["BOL"]),
            format_binary_feature("Arbustos (ARB)", row["ARB"]),
            format_binary_feature("Carteles propiedad privada (CPP)", row["CPP"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # 7. BLOQUE: Puntos de Acceso
    with st.expander("🚪 Puntos de Acceso", expanded=False):
        html_content = "".join([
            format_binary_feature("Entrada por vivienda (PVI)", row["PVI"]),
            format_binary_feature("Entrada por bloque (PBL)", row["PBL"]),
            format_binary_feature("Entrada común (COM)", row["COM"]),
            format_binary_feature("Varias entradas comunes (COMS)", row["COMS"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # 8. BLOQUE: Uso de la Vía Pública
    with st.expander("🌐 Uso de la Vía Pública", expanded=False):
        html_content = "".join([
            format_binary_feature("Dominio público y uso público (PPU)", row["PPU"]),
            format_binary_feature("Dominio privado y uso restringido (PRE)", row["PRE"]),
            format_binary_feature("Dominio privado y uso privado (PPR)", row["PPR"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)

    # 9. BLOQUE: Seguridad y Vigilancia
    with st.expander("🛡️ Seguridad y Vigilancia", expanded=False):
        html_content = "".join([
            format_binary_feature("Guardia de seguridad (GSE)", row["GSE"]),
            format_binary_feature("Cámaras de seguridad (CSE)", row["CSE"]),
            format_binary_feature("Barrera de seguridad (BSE)", row["BSE"]),
            format_binary_feature("Alarma de seguridad (ASE)", row["ASE"])
        ])
        st.markdown(html_content, unsafe_allow_html=True)


def generate_shap_explication(selected_cc, max_features=4, min_ratio=0.25):
    """
    """
    feature_descriptions = {
        "CSS": "calles sin salida",
        "CFS": "calles en fondo de saco",
        "CPE": "calles peatonales",
        "LIN": "comercios internos",
        "LEX": "comercios externos",
        "DIS_1": "una ubicación geográfica aislada del núcleo urbano",
        "DIS_2": "una ubicación geográfica separada del núcleo urbano",
        "DIS_3": "una ubicación geográfica integrada en el núcleo urbano",
        "VER": "verjas",
        "MUR": "muros",
        "CAD": "cadenas",
        "BOL": "bolardos",
        "ARB": "arbustos",
        "CPP": "carteles de propiedad privada",
        "PVI": "entrada por vivienda",
        "PBL": "entrada por bloque",
        "COM": "entrada común a la urbanización",
        "COMS": "varias entradas comunes a la urbanización",
        "PPU": "uso y dominio público de la vía",
        "PRE": "dominio privado y uso público restringido de la vía",
        "PPR": "uso y dominio privado de la vía",
        "GSE": "guardia de seguridad",
        "CSE": "cámaras de seguridad",
        "BSE": "barrera de seguridad",
        "ASE": "alarma de seguridad"
    }
    label_dict = {
        1: "Protegido",
        2: "Controlado",
        3: "Autoaislado",
        4: "Individualista",
        5: "Simbólico"
    }  
    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    cc_int = int(selected_cc)  # Convierte el String (ej. "12003") a entero (12003)
    
    dataset_df = load_model_variables()
    _, pred_info_df, shap_values_df = load_shap_data()
    
    # Get predicted class
    pred_class = pred_info_df.loc[
        pred_info_df["CC"] == cc_int,
        "URB_pred"
    ].iloc[0]
    pred_class_label = label_dict[pred_class]

    # ==========================================================
    # GET SHAP VALUES FOR PREDICTED CLASS
    # ==========================================================    
    sample_shap_values_df = shap_values_df[
        (shap_values_df["CC"] == cc_int) &
        (shap_values_df["URB"] == pred_class)
    ]

    # Magnitud absoluta
    sample_shap_values_df["abs_shap"] = sample_shap_values_df["shap_value"].abs()

    # Ordenar por importancia
    sample_shap_values_df = sample_shap_values_df.sort_values("abs_shap", ascending=False)

    # SHAP máximo
    max_shap = sample_shap_values_df["abs_shap"].iloc[0]

    # Selección automática de variables influyentes
    relevantes = sample_shap_values_df[sample_shap_values_df["abs_shap"] >= max_shap * min_ratio]

    # Limitar entre 1 y max_features
    relevantes = relevantes.head(max_features)

    # Garantizar al menos 1
    if len(relevantes) == 0:
        relevantes = sample_shap_values_df.head(1)

    frases = []

    for _, row in relevantes.iterrows():

        feature = row["feature"]
        feature_text = feature_descriptions.get(feature, feature)
        valor = dataset_df.loc[dataset_df["CC"]==cc_int, feature].iloc[0]

        if valor == 1:
            frases.append(
                f"tiene <b style='color: #2e7d32;'>{feature_text}</b> "
                f"(<code style='color: #2e7d32; background-color: #e8f5e9;'>{feature}=1</code>)"
            )
        else:
            frases.append(
                f"no tiene <b style='color: #c62828;'>{feature_text}</b> "
                f"(<code style='color: #c62828; background-color: #ffebee;'>{feature}=0</code>)"
            )

    # Construcción natural del texto
    if len(frases) == 1:
        descripcion = frases[0]

    elif len(frases) == 2:
        descripcion = " y ".join(frases)

    else:
        descripcion = ", ".join(frases[:-1]) + " y " + frases[-1]

    texto_html = f"""
        <div style='
        background-color: #f8f9fa; 
        border-left: 5px solid #2b5c8f; 
        padding: 18px; 
        border-radius: 4px; 
        margin-top: 20px;
        line-height: 1.6;
        color: #333333;
        font-size: 1.15rem; /* <-- Aumenta el tamaño de todo el texto base */
        '>
        <span style='font-size: 1.3rem; vertical-align: middle;'>💡</span> 
        El complejo residencial ha sido clasificado en el grado de cerramiento
        <span style='color: #000000; font-weight: bold; font-size: 1.2rem;'>{pred_class_label}</span> 
        debido a que {descripcion}.
        </div>
    """

    return texto_html


def render_map_interface():
    """
    Renders the interactive map tab showcasing the gated and residential typologies.

    This interface loads geospatial data for residential developments, integrates 
    their classification models' prediction labels, and visualizes them on an Esri 
    World Imagery satellite basemap. 

    The component captures user click events on individual markers to pass the 
    selected complex ID (`CC`) to downstream Explainable AI (XAI) components 
    such as local SHAP explanations.

    Returns:
        dict or None: The `st_folium` output dictionary containing map state events,
                      specifically constrained to track 'last_object_clicked'.
    """

    # =========================================================
    # UI Headers and Descriptions
    # ========================================================= 

    st.subheader("Cartografía de complejos residenciales de acuerdo a su grado de cerramiento")
    st.markdown("""
    Desplázate por el mapa y sitúa el cursor sobre los complejos residenciales para ver su información. 
    Haz **clic sobre un marcador** para cargar su análisis de explicabilidad local mediante SHAP.
    """)

    # Configuration Profiles for Visual Layout
    # Mapping of target enclosure classes to human-readable labels and color codes
    class_styles = {
        1: {"label": "Protegido", "color": "red"},
        2: {"label": "Controlado", "color": "orange"},
        3: {"label": "Autoaislado", "color": "purple"},
        4: {"label": "Individualista", "color": "blue"},
        5: {"label": "Simbólico", "color": "green"}
    }

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

    # Merge predicted label to map dataframe
    df_map["CC"] = df_map["CC"].astype(str)
    df_pred["CC"] = df_pred["CC"].astype(str)
    pred_dict = dict(zip(df_pred["CC"], df_pred["URB_pred"]))
    df_map["URB_pred"] = df_map["CC"].map(pred_dict)

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
        mun = row.MUN
        class_value = row.URB_pred

        # Retrieve mapped layout profile
        style = class_styles.get(class_value, {"label": "Desconocido", "color": "gray", "icon": "question-circle-fill"})
        
        # # Build clean string templates for interactive tooltips
        tooltip_text = tooltip_text = (
            f"<b>CC:</b> {id}<br>"
            f"<b>Municipio:</b> {mun}<br>"
            f"<b>Grado de cerramiento:</b> {style['label']}"
        )

        # Instantiate responsive node marker linking unique ID to the 'name' property
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
    # Estilo para que el mapa no genere espacio en blanco gigante
    st.markdown("""
    <style>

    /* Contenedor principal del componente folium */
    div[data-testid="stElementContainer"]:has(iframe) {
        height: 520px !important;
    }

    /* Wrapper interno */
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
    # --- FILA 1: MAPA (IZQUIERDA) Y LEYENDA (DERECHA) ---
    # Usamos una proporción [5, 1] para que el mapa domine casi toda la pantalla
    col_map, col_legend = st.columns([5, 1])
    with col_map:
        # Forzamos que st_folium se adapte al 100% del ancho de esta columna

        map_output = st_folium(
            m, 
            use_container_width=True, 
            height=550,  
            key="map",
            returned_objects=["last_object_clicked"]
        )

    with col_legend:
        st.markdown("### Leyenda")
        hex_colors = {"red": "🔴", "orange": "🟠", "purple": "🟣", "blue": "🔵", "green": "🟢"}
        
        # Estilo CSS inline para que la leyenda baje un poco y se alinee visualmente con el mapa
        st.markdown("<div style='margin-top: 15px;'><b>Grado de Cerramiento:</b></div>", unsafe_allow_html=True)
        for class_key, info in class_styles.items():
            emoji = hex_colors.get(info["color"], "⚪")
            st.markdown(f"{emoji} **{class_key}**: {info['label']}")
    
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 25px; border: 0; border-top: 1px solid #d3d3d3;'>", unsafe_allow_html=True)

    # --- EXTRACCIÓN Y PERSISTENCIA DEL ID (CC) ---
    selected_cc = extract_selected_cc(map_output)

    # --- FILA 2: METADATOS (IZQUIERDA) Y GRÁFICO SHAP (DERECHA) ---
    if selected_cc:
        st.session_state["selected_urbanizacion"] = selected_cc
        st.toast(f"🎯 Complejo residencial seleccionado: {selected_cc}", icon="📊")
    
        # Creamos una nueva fila de columnas independiente con proporción [2, 3]
        col_meta, col_graph = st.columns([2, 3])
    
        with col_meta:
            with st.container(border=True):
                render_complex_metadata(selected_cc, df_map, class_styles)

        with col_graph:
            # Generamos el gráfico SHAP limpio (sin títulos repetidos)
            fig_shap = get_shap_waterfall(cc=selected_cc, num_features_display=11, figsize=[9, 5.5])
            if fig_shap:
                # Al estar dentro de col_graph, Matplotlib se encajona perfectamente
                # eliminando el espacio en blanco gracias a margin-top negativo controlado
                st.pyplot(fig_shap, clear_figure=True, bbox_inches="tight")

                # -------------------------------------------------------------
                # NUEVO: Generar y mostrar tu texto con el nuevo formato limpio
                # -------------------------------------------------------------
                texto_explicacion = generate_shap_explication(selected_cc=selected_cc)
                st.markdown(texto_explicacion, unsafe_allow_html=True)
    else:
        # Mensaje de espera si el usuario no ha hecho clic
        st.info("💡 Por favor, haz clic en cualquier marcador del mapa para cargar su análisis explicativo SHAP.")

    return map_output