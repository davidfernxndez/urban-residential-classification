"""
Streamlit Simulator interface provides:
- A customizable user interface for configuring residential complexes.
- Prediction of enclosure degree using a trained XGBoost model.
- SHAP-based explainability visualizations and textual explanations.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import shap

from app_src.utils.data_loader import load_dataset, load_model
from app_src.appConfig import config



def custom_instance():
    """
    Generate a form with buttons structured in sections so that
    the user can customize a residential complex through its morphological variables

    Parameters
    ----------

    Returns
    ----------
    df_custom_instance: pandas.DataFrame
        Dataframe containing feature data of the instance customized by the user
    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # =========================================================
    # LOAD REQUIRED DATAFRAMES
    # =========================================================
    dataset_df = load_dataset()
    
    # Get feature order
    dataset_df = dataset_df.drop(
        columns=[id_variable, target_variable]
    )
    feature_order = dataset_df.columns
    
    # =========================================================================
    # CREATE TABS FOR EACH BLOCK OF VARAIBLES
    # =========================================================================

    with st.form(key="prediction_form_id"):
        tab_est, tab_dist, tab_cerr, tab_acc, tab_via, tab_seg = st.tabs([
            "🧱 Aspectos estructurales", 
            "📍 Distancia al Núcleo Urbano",
            "⛔ Elementos de Cerramiento",
            "🚪 Puntos de Acceso", 
            "🚧 Uso de la Vía Pública",
            "🛡️ Seguridad y Vigilancia"
        ])

        # Dictionary to store state of each variable: presence (value 1) or absence (value 0)
        features_data = {}

        # -------------------------------------------------------------------------
        # ASPECTOS ESTRUCTURALES
        # -------------------------------------------------------------------------
        with tab_est:
            st.markdown("<font size='4'><b>Seleccione las características del diseño vial interno y el tipo de comercio presentes en el complejo residencial.</b></font>", unsafe_allow_html=True)
        
            col1, col2 = st.columns(2)
            with col1:
                features_data["CSS"] = 1 if st.checkbox("Calle sin salida (CSS)", value=False) else 0
                features_data["CFS"] = 1 if st.checkbox("Calle en fondo de saco (CFS)", value=False) else 0
                features_data["CPE"] = 1 if st.checkbox("Calle peatonal (CPE)", value=False) else 0
            with col2:
                features_data["LIN"] = 1 if st.checkbox("Comercios internos (LIN)", value=False) else 0
                features_data["LEX"] = 1 if st.checkbox("Comercios externos (LEX)", value=False) else 0
        
        # -------------------------------------------------------------------------
        # DISTANCIA AL NÚCLEO URBANO
        # -------------------------------------------------------------------------
        with tab_dist:
            st.markdown("<font size='4'><b>Seleccione la situación geográfica del complejo residencial respecto al núcleo urbano más cercano.</b></font>", unsafe_allow_html=True)
        
            dis_selection = st.radio(
                label="Solo puede seleccionarse un tipo de situación geográfica:c",
                options=[
                    ("DIS_1", "Aislado (DIS_1)"),
                    ("DIS_2", "Separado (DIS_2)"),
                    ("DIS_3", "Integrado (DIS_3)")
                ],
                format_func=lambda x: x[1] 
            )
            # Only one feature can be activated (value 1)
            features_data["DIS_1"] = 1 if dis_selection[0] == "DIS_1" else 0
            features_data["DIS_2"] = 1 if dis_selection[0] == "DIS_2" else 0
            features_data["DIS_3"] = 1 if dis_selection[0] == "DIS_3" else 0


        # -------------------------------------------------------------------------
        # ELEMENTOS DE CERRAMIENTO
        # -------------------------------------------------------------------------
        with tab_cerr:
            st.markdown("<font size='4'><b>Seleccione los elementos de cerramiento presentes en el complejo residencial.</b></font>", unsafe_allow_html=True)
        
            col_cer1, col_cer2 = st.columns(2)
            with col_cer1:
                features_data["VER"] = 1 if st.checkbox("Verjas (VER)", value=False) else 0
                features_data["MUR"] = 1 if st.checkbox("Muros (MUR)", value=False) else 0
                features_data["CAD"] = 1 if st.checkbox("Cadenas (CAD)", value=False) else 0
            with col_cer2:
                features_data["BOL"] = 1 if st.checkbox("Bolardos (BOL)", value=False) else 0
                features_data["ARB"] = 1 if st.checkbox("Arbustos (ARB)", value=False) else 0
                features_data["CPP"] = 1 if st.checkbox("Carteles de propiedad privada (CPP)", value=False) else 0

        # -------------------------------------------------------------------------
        # PUNTOS DE ACCESO
        # -------------------------------------------------------------------------
        with tab_acc:
            st.markdown("<font size='4'><b>Seleccione los tipos de acceso disponibles en el complejo residencial.</b></font>", unsafe_allow_html=True)
        
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                features_data["PVI"] = 1 if st.checkbox("Entrada por vivienda (PVI)", value=False) else 0
                features_data["PBL"] = 1 if st.checkbox("Entrada por bloque (PBL)", value=False) else 0
            with col_acc2:
                features_data["COM"] = 1 if st.checkbox("Entrada común (COM)", value=False) else 0
                features_data["COMS"] = 1 if st.checkbox("Varias entradas comunes (COMS)", value=False) else 0
                

        # -------------------------------------------------------------------------
        # USO DE LA VÍA
        # -------------------------------------------------------------------------
        with tab_via:
            st.markdown("<font size='4'><b>Seleccione el tipo de uso de la vía pública que existe en el complejo residencial.</b></font>", unsafe_allow_html=True)
            
            via_selection = st.radio(
                label="Solo puede seleccionarse un tipo de uso:",
                options=[
                    ("PPU", "Dominio público y de uso público (PPU)"),
                    ("PRE", "Dominio privado de uso público restringido (PRE)"),
                    ("PPR", "Dominio privado de uso privado (PPR)"),
                    ("NINGUNO", "No tiene viales")
                ],
                format_func=lambda x: x[1]
            )
            # Only one feature can be activated (value 1)
            features_data["PPU"] = 1 if via_selection[0] == "PPU" else 0
            features_data["PRE"] = 1 if via_selection[0] == "PRE" else 0
            features_data["PPR"] = 1 if via_selection[0] == "PPR" else 0

        # -------------------------------------------------------------------------
        # SEGURIDAD Y VIGILANCIA
        # -------------------------------------------------------------------------
        with tab_seg:
            st.markdown("<font size='4'><b>Seleccione los servicios de seguridad existentes en el complejo residencial.</b></font>", unsafe_allow_html=True)
            col_seg1, col_seg2 = st.columns(2)
            with col_seg1:
                features_data["GSE"] = 1 if st.checkbox("Guardia de seguridad (GSE)", value=False) else 0
                features_data["CSE"] = 1 if st.checkbox("Cámaras de seguridad (CSE)", value=False) else 0
            with col_seg2:
                features_data["BSE"] = 1 if st.checkbox("Barrera de seguridad (BSE)", value=False) else 0
                features_data["ASE"] = 1 if st.checkbox("Alarma de seguridad (ASE)", value=False) else 0


        # Button to register the state of the form and request a prediction
        submit_button = st.form_submit_button(
            label="PREDECIR", 
            use_container_width=True,
            type="primary"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if submit_button:
        # Build dataframe with configured variables
        df_custom_instance = pd.DataFrame([features_data])

        # Order features accoring to dataset columns
        df_custom_instance = df_custom_instance.reindex(columns=feature_order)

        return df_custom_instance

    # If user doesn`t click the predictor button returns None`
    return None


def generate_shap_explication(instance_df, shap_values, pred_class_label, max_features=4, min_ratio=0.15):
    """
    Generate a text explanation of the predicted class using SHAP values.

    The function analyzes the SHAP contributions associated with the customized
    residential complex and builds a human-readable explanation highlighting
    the most relevant features influencing the prediction.

    Parameters
    ----------
    instance_df: pandas.DataFrame
        Dataframe containing feature data of the instance customized by the user

    shap_values: numpy array
        1-D array with shap_values of the instance customized by the user
    
    pred_class_label: str
        Label class predicted by the model for the instance customized by the user.

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
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    feature_description = config.FEATURE_DESCRIPTION

    # =========================================================
    # BUILD SHAP VALUES DATAFRAME
    # =========================================================
    instance_shap_df = pd.DataFrame({
        "feature": instance_df.columns,
        "shap_value": shap_values
    })

    # ==========================================================
    # SELECT MORE RELEVANT FEATURES
    # ==========================================================   
    # Sort by positive SHAP contribution
    instance_shap_df = instance_shap_df.sort_values("shap_value", ascending=False)

    # Get Max value
    max_shap = instance_shap_df["shap_value"].iloc[0]

    # Select more relevance features
    relevant_features = instance_shap_df[instance_shap_df["shap_value"] >= max_shap * min_ratio]
    relevant_features = relevant_features.head(max_features)

    # Guarantee at least one variable
    if len(relevant_features) == 0:
        relevant_features = instance_shap_df.head(1)

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
        feature_value = instance_df[feature].iloc[0]

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


def get_waterfall_plots(explainer, pred_idx, instance_df, target_label_map, label_encoder, num_features_display=11, figsize=[9, 5.5]):
    """
    Generate and retrieve SHAP waterfall plots for all classes, separating the 
    predicted class plot from the rest and mapping them to their corresponding labels.

    Parameters
    -----------
    explainer : shap.Explainer
        The trained SHAP explainer instance used to compute SHAP values.
        
    pred_idx : int
        The index of the predicted class for the given instance.
        
    instance_df : pandas.DataFrame
        Dataframe containing the feature data of the specific instance 
        customized by the user.

    target_label_map : dict
        A dictionary mapping encoded or raw target values to human-readable class names.

    label_encoder : sklearn.preprocessing.LabelEncoder
        The fitted label encoder used to inverse transform the numerical class 
        indices back to their original representation.

    num_features_display : int [optional, default=11]
        Maximum number of features displayed in the Waterfall plot.
    
    figsize: list [optional, default=[9, 5.5]]
        Width and height values of the graph.

    Returns
    -----------  
    pred_fig : matplotlib.figure.Figure
        The standalone matplotlib figure object containing the waterfall plot 
        for the predicted class.
    
    pred_shap_values: Numpy array
        Array containing shap values for predicted class

    other_figs : list of tuples
        A list of tuples where each element contains a matplotlib figure object 
        and its associated class name string, representing all other non-predicted classes:
        [(fig_1, class_label_1), (fig_2, class_label_2), ...]
    """    
    # Get number of classes
    num_classes = len(config.TARGET_LABEL_MAP)
    
    # Variables to store data
    other_figs = []
    pred_fig = None    
    pred_shap_values = None

    for class_idx in range(num_classes):
        # Initialize matplotlib figure
        plt.figure()

        # Calculate shap for instance and get expected value for predicted class
        shap_values = explainer.shap_values(instance_df)[:,:,class_idx][0]
        expected_value = explainer.expected_value[class_idx]

        # Create explainer SHAP object
        exp = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=instance_df.iloc[0].values,
            feature_names=instance_df.columns
        )
        # Get waterfall plot
        shap.plots.waterfall(
            exp,
            max_display=num_features_display,
            show=False
        )
        # Customize with matplotlib.pyplot
        fig = plt.gcf()  
        fig.set_size_inches(figsize[0], figsize[1])  

        # Add grid and adjust marginds
        plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout()        

        # Store figures and data
        if class_idx == pred_idx:
            pred_fig = fig
            pred_shap_values = shap_values
        else:
            # Get class label
            class_label = target_label_map[label_encoder.inverse_transform([class_idx])[0]]
            other_figs.append((fig, class_label))
    
    return pred_fig, pred_shap_values, other_figs    


def predict_instance(instance_df):
    """
    Predict the class of a user-defined instance and generate a SHAP-based 
    explainability report, including a waterfall visualization and a textual
    explanation.
    
    Parameters
    -----------
    instance_df: pandas.DataFrame
        Dataframe containing feature data of the instance customized by the user

    num_features_display : int [optional, default=11]
        Maximum number of features displayed in the Waterfall plot.
    
    figsize: list [optional, default=[9, 5.5]]
        Width and height values of the graph.

    Returns
    -----------  
    pred_label: str
        Label class predicted by the model

    probabilities: list
        List of probabilities per class obtained by the model

    pred_fig : matplotlib.figure.Figure
        Matplotlib figure object containing the waterfall plot 
        for the predicted class.

    other_figs : list of tuples
        A list of tuples where each element contains a matplotlib figure object (waterfall plot)
        and its associated class name string, representing all other non-predicted classes:
        [(fig_1, class_label_1), (fig_2, class_label_2), ...]

    html_text: str
        HTML-formatted text containing the explainability message generated
        from the SHAP values of the predicted class.
    """
    # ===================================
    # LOAD CONFIGURATION VARIABLES
    # ===================================
    target_label_map = config.TARGET_LABEL_MAP

    # ==========================
    # LOAD REQUIRED DATA
    # ==========================
    model, label_encoder, explainer, _ = load_model()
    
    # ==========================
    # INSTANCE PREDICTION
    # ==========================
    pred_idx = model.predict(instance_df)[0]
    pred_label = target_label_map[label_encoder.inverse_transform([pred_idx])[0]]
    probabilities = model.predict_proba(instance_df)[0]

    # ==========================
    # GET SHAP WATERFALL PLOTS
    # ==========================
    
    pred_fig, pred_shap_values, other_figs = get_waterfall_plots(explainer, pred_idx, instance_df, target_label_map, label_encoder)

    # Get SHAP text explanation for predicted class
    html_text = generate_shap_explication(instance_df, pred_shap_values, pred_label)

    return pred_label, probabilities, pred_fig, other_figs, html_text


def render_simulator_interface():
    """
    Render the Simulator interface for streamlit platform, including:
    - Button form to allow residential complex customization.
    - Model prediction of customized residential complex with SHAP explainability.
    - Model probabilities of all classes and alternative SHAP waterfall plots.

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

    # =========================================================
    # UI Headers and Descriptions
    # ========================================================= 
    st.subheader("Simulación predictiva de nuevos desarrollos urbanísticos")
    st.markdown(
        """
        <p style='color: #475569; font-size: 1.12rem; line-height: 1.5; margin-bottom: 15px;'>
            💡 Configure las características morfológicas de un complejo residencial mediante el formulario disponible.
            <br><br>
            Pulse el botón
            <span style='background-color: #FF4B4B; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>PREDECIR</span> 
            para obtener el <b>grado de cerramiento</b> estimado, las probabilidades asociadas a cada categoría y el
            análisis de explicabilidad de la decisión mediante SHAP.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # Instantiate button form to allow residential complex customization
    instance_df = custom_instance()

    # Verify if the user has clicked the predict button
    if instance_df is not None:
        # Predict instance and get SHAP explainability
        predicted_label, probabilities, pred_fig_shap, other_figs_shap, explainability_text = predict_instance(instance_df)
        with st.container(border=True):
            col_prediction, _, col_graph = st.columns([2, 0.1, 3])
            
            # =========================================================================
            # PREDICTED LABEL AND PROBABILITY
            # =========================================================================
            with col_prediction:  
                # Customize predicted class according to color styles             
                for _, info in class_styles.items():
                    if info["label"] == predicted_label:
                        border_color = info["hex_code"]
                        break

                # Custom background color
                bg_color = f"rgba({int(border_color.lstrip('#')[0:2], 16)}, {int(border_color.lstrip('#')[2:4], 16)}, {int(border_color.lstrip('#')[4:6], 16)}, 0.08)"

                # Render text with predicted label
                st.markdown(
                    f"""
                    <div style='
                        background-color: {bg_color}; 
                        border-left: 6px solid {border_color}; 
                        padding: 16px; 
                        border-radius: 6px;
                        margin: 15px 0 25px 0;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
                    '>
                        <p style='margin: 0; font-size: 0.95rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;'>
                            Grado de cerramiento predicho
                        </p>
                        <p style='margin: 4px 0 0 0; font-size: 1.6rem; font-weight: bold; color: {border_color};'>
                            {predicted_label}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Plot horizontal bar chart with probabilities
                st.markdown("### 📈 Distribución de probabilidades")
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                total_prob = sum(probabilities)
                # Barras de probabilidad
                for i, prob in enumerate(probabilities):
                    prob_percentage = float(prob) * 100
                    ratio = (prob / total_prob * 100) if total_prob > 0 else (prob * 100)
                    
                    # Get class label and color
                    class_label = class_styles[i+1]["label"]
                    bar_color = class_styles[i+1]["hex_code"]
                    
                    if class_label == predicted_label:
                        # Highlight text of predicted class
                        text_style = f"font-weight: 800; color: {bar_color}; font-size: 1.15rem;"
                        bg_bar_color = f"{bar_color}20" 
                        label_display = f"🎯 {class_label}"
                    else:
                        # Normal text
                        text_style = f"font-weight: 500; color: #475569; font-size: 1.05rem;"
                        bg_bar_color = "#f1f5f9"  
                        label_display = class_label

                    # render progress bar
                    st.markdown(
                        f"""
                        <div style='margin-bottom: 35px;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                                <span style="{text_style}">
                                    {label_display}
                                </span>
                                <span style="font-weight: 600; color: #475569;">
                                    {prob_percentage:.1f}%
                                </span>
                            </div>
                            <div style='
                                background-color: {bg_bar_color};
                                width: 100%;
                                height: 10px;
                                border-radius: 5px;
                                overflow: hidden;
                            '>
                                <div style='
                                    background-color: {bar_color};
                                    width: {ratio}%;
                                    height: 100%;
                                    border-radius: 5px;
                                '></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            # =========================================================================
            # WATERFALL PLOT AND TEXT EXPLAINABILITY
            # =========================================================================
            with col_graph:
                st.markdown(f"""### 📊 Gráfico cascada del grado de cerramiento predicho: *{predicted_label}*""")
                with st.container(border=True):
                    # Plot shap graph
                    st.pyplot(pred_fig_shap, clear_figure=True, bbox_inches="tight")
                                    
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

            # Display explainability text
            st.markdown(explainability_text, unsafe_allow_html=True)
            
            # =========================================================================
            # WATERFALL PLOT OF THE REST OF THE CLASSES
            # =========================================================================
        with st.expander("🔍 Análisis complementario de grados de cerramiento alternativos"):
            st.markdown(
                    """
                    <p style='font-size: 1.1rem; color: #556370; margin-bottom: 20px;'>
                    Consulte la contribución de las variables en los grados de cerramiento alternativos
                    mediante gráficos cascada (<i>waterfall plots</i>) de SHAP. 
                    </p>
                    """, 
                    unsafe_allow_html=True
            )
            # Create a 2x2 matrix to display the four waterfall plots
            fil_0 = st.columns(2)
            fil_1 = st.columns(2)
            cell = [fil_0[0], fil_0[1], fil_1[0], fil_1[1]]

            # Loop to display the figures
            for i, (fig, class_name) in enumerate(other_figs_shap):
                if i >= len(cell):
                    break
                
                # Get label styler
                style_info = next(
                    (info for info in class_styles.values() if info["label"] == class_name), 
                    {"hex_code": "#31333F"}
                )
                current_color = style_info["hex_code"]
                
                # Color background
                current_bg = f"rgba({int(current_color.lstrip('#')[0:2], 16)}, {int(current_color.lstrip('#')[2:4], 16)}, {int(current_color.lstrip('#')[4:6], 16)}, 0.08)"
                # ----------------------------------------

                with cell[i]:
                    # Container for the figure
                    with st.container(border=True):
                        
                        # Set the label as title
                        st.markdown(
                            f"""
                            <div style='
                                background-color: {current_bg}; 
                                border-left: 5px solid {current_color}; 
                                padding: 10px 14px; 
                                border-radius: 6px;
                                margin-bottom: 15px;
                                text-align: left;
                            '>
                                <p style='margin: 2px 0 0 0; font-size: 1.2rem; font-weight: bold; color: {current_color};'>
                                    {class_name}
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Display figure
                        st.pyplot(fig, use_container_width=True, clear_figure=True)          