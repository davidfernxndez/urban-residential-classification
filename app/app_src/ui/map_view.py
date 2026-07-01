import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from app_src.utils.data_loader import load_map_data, load_shap_data
from pyproj import Transformer

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

    #################################
    # CANVAS LAYOUT
    #################################
    # Render Grid Layout (Map canvas alongside institutional legend widget)
    col_map, col_legend = st.columns([5, 1]) 
    with col_map:
        map_output = st_folium(
            m, 
            width=1150,  
            height=600,
            key="map",
            returned_objects=["last_object_clicked"]
        )

    with col_legend:
        st.markdown("### Grado de Cerramiento")
        hex_colors = {
            "red": "🔴",
            "orange": "🟠",
            "purple": "🟣",
            "blue": "🔵",
            "green": "🟢" 
        }
        
        # Generate the sidebar color index table dynamically matching class styles
        for class_value, info in class_styles.items():
            emoji = hex_colors.get(info["color"], "⚪")
            st.markdown(f"{emoji} **{class_value}**: {info['label']}")
    
    return map_output