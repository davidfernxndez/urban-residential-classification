"""
This module contains methods and functions for performing interpretability analysis of the following models:
- Multinomial logistic regression: Using the model coefficients.
- Decision tree: Using impurity reduction and tree visualization (rule extraction).
- Black-box models (SVM, Random Forest, and XGBoost): Using SHAP post-hoc method.
"""

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
from sklearn.model_selection import train_test_split
from dtreeviz import model
from sklearn.preprocessing import LabelEncoder
import ast
import copy

from IPython.display import display

# ==============================================================================
# MULTINOMIAL LOGISTIC REGRESSION
# ==============================================================================

def load_lr_coefs(config, model_path):
    """
    This function loads a serialized Logistic Regression model and its
    associated metadata, extracts model coefficients and odds ratios,
    and structures them into pandas DataFrames with meaningful feature
    and class labels for interpretability analysis.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : Path to input dataset CSV file
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name
        - TARGET_LABEL_MAP: dictionary mapping encoded labels to names

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".

    Returns
    -------
    LR_coef_df : pandas.DataFrame
        DataFrame of logistic regression coefficients (log-odds).
        Rows correspond to classes and columns to features.

    LR_odds_df : pandas.DataFrame
        DataFrame of odds ratios derived from coefficients.
        Computed as exp(coef), with the same structure as LR_coef_df.
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Dataset path
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # Mapping from encoded labels to readable names
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # LOAD TRAINED MODEL ARTIFACTS
    # =========================================================

    LR_artifacts = joblib.load(model_path)

    LR_model = LR_artifacts["model"]
    LR_encoder = LR_artifacts["label_encoder"]
    LR_name = LR_artifacts["name"]

    print(f"{LR_name} model successfully loaded from pkl file.")

    # =========================================================
    # EXTRACT MODEL PARAMETERS
    # =========================================================

    LR_coef = LR_model.coef_
    print(f"Coefficient matrix shape: {LR_coef.shape}")

    # Convert coefficients to odds ratios
    odds_ratios = np.exp(LR_coef)

    # =========================================================
    # LOAD FEATURE NAMES
    # =========================================================

    data_df = pd.read_csv(dataset_path)

    # Remove ID and target columns to retain only predictors
    variable_names = data_df.drop(
        columns=[id_variable, target_variable]
    ).columns

    # =========================================================
    # DECODE CLASS LABELS
    # =========================================================

    class_values = LR_encoder.inverse_transform(LR_model.classes_)

    # Map decoded labels to readable class names
    class_names = [target_label_map[x] for x in class_values]

    # =========================================================
    # BUILD OUTPUT DATAFRAMES
    # =========================================================

    # Coefficient matrix (log-odds)
    LR_coef_df = pd.DataFrame(
        LR_coef,
        columns=variable_names,
        index=class_names
    )

    # Odds ratio matrix
    LR_odds_df = pd.DataFrame(
        odds_ratios,
        columns=variable_names,
        index=class_names
    )

    return LR_coef_df, LR_odds_df


def plot_lr_feature_importance(LR_coef_df, criterion = "mean"):
    """
    Plot a horizontal bar chart of logistic regression coefficients
    order by feature importance according to specified criterion.

    Parameters
    ----------
    LR_coef_df : pandas.DataFrame
        DataFrame containing logistic regression coefficients.
        Rows correspond to classes and columns correspond to features.
    criterion: string [optional, default="mean"]
        Select criterion to get feature importance. Could be "mean" or "max"
    Returns
    -------
    None
        Displays a matplotlib figure.

    """
    if criterion == "mean":
        # Get absolute mean coefficent value across al the classes
        feature_importance = LR_coef_df.abs().mean(axis=0).sort_values(ascending=False)
    elif criterion == "max":
        # Get absolute max coefficent value across al the classes
        feature_importance = LR_coef_df.abs().max(axis=0).sort_values(ascending=False)
    else:
        return print("Input parameter criterion is not available.")
      
    # Horizontal bar plot
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=feature_importance.values, 
        y=feature_importance.index, 
        palette="Blues_r",  
        edgecolor='black'
    )

    # Title and labels
    plt.title('Ranking global de Importancia de Variables\n en Regresión Logística Multinomial', fontsize=14, pad=15)
    plt.xlabel(r'Valor Absoluto Medio del Coeficiente ($|\beta|$)', fontsize=12)
    plt.ylabel('Variables Predictoras', fontsize=12)

    # Grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Text with variables information
    leyenda_texto = (
        "BLOQUES SEMÁNTICOS\n\n"
        "Aspectos estructurales:\n"
        " • CSS: Calles sin salida\n"
        " • CFS: Calle en fondo de saco\n"
        " • CPE: Calle Peatonal\n"
        " • LIN: Comercio Interno\n"
        " • LEX: Comercio Externo\n\n"
        "Distancia:\n"
        " • DIS_1: Urbanización aislada\n"
        " • DIS_2: Urbanización separada\n"
        " • DIS_3: Urbanización integrada\n\n"
        "Tipo de cerramiento:\n"
        " • VER: Verjas       • MUR: Muros\n"
        " • CAD: Cadenas      • BOL: Bolardos\n"
        " • ARB: Arbustos     • CPP: C. prop. privada\n\n"
        "Puntos de acceso:\n"
        " • PVI: Entrada por vivienda\n"
        " • PBL: Entrada por bloque\n"
        " • COM: Entrada común\n"
        " • COMS: Varias entradas comunes\n\n"
        "Uso de la Vía pública:\n"
        " • PPU: Dominio/uso público\n"
        " • PRE: Dominio/uso privado restringido.\n"
        " • PPR: Dominio/uso privado\n\n"
        "Seguridad y Vigilancia:\n"
        " • GSE: Guardia       • CSE: Cámaras\n"
        " • BSE: Barrera       • ASE: Alarma"
    )
    plt.text(
        x=1.1, y=0.5, 
        s=leyenda_texto, 
        transform=plt.gca().transAxes, 
        fontsize=9.5, 
        verticalalignment='center', 
        fontfamily='monospace', 
        bbox=dict(
            boxstyle='round,pad=1', 
            facecolor='#f9f9f9', 
            edgecolor='#cccccc', 
            alpha=1.0
        )
    )


    plt.subplots_adjust(left=0.15, right=0.55, top=0.90, bottom=0.10)
    plt.show()


def plot_lr_class_coefs(LR_coef_df, class_label):
    """
    Plot a horizontal bar chart of logistic regression coefficients
    for a specific class, organized by thematic feature groups.

    This function visualizes the coefficients of a multinomial logistic
    regression model for a given class, grouping variables into predefined
    semantic blocks. It applies a global color normalization to ensure
    comparability across classes and enhances interpretability through
    structured grouping and visual separation.

    Parameters
    ----------
    LR_coef_df : pandas.DataFrame
        DataFrame containing logistic regression coefficients.
        Rows correspond to classes and columns correspond to features.

    class_label : str or int
        The target class for which coefficients will be visualized.

    Returns
    -------
    None
        Displays a matplotlib figure.

    """

    # =========================
    # COEFFICIENTS
    # =========================
    # Compute global min and max values across the entire coefficient
    # matrix to ensure consistent scaling across all plots.
    max = LR_coef_df.to_numpy().max()
    min = LR_coef_df.to_numpy().min()

    # Extract coefficients for the selected class
    coefs = LR_coef_df.loc[class_label]

    # =========================
    # THEMATIC FEATURE GROUPS
    # =========================
    feature_groups = {
        "Aspectos\nestructurales": ["CSS", "CFS", "CPE", "LIN", "LEX"],
        "Distancia": ["DIS_1", "DIS_2", "DIS_3"],
        "Cerramiento": ["VER", "MUR", "CAD", "BOL", "ARB", "CPP"],
        "Acceso": ["PVI", "PBL", "COM", "COMS"],
        "Uso vía\npública": ["PPU", "PRE", "PPR"],
        "Seguridad": ["GSE", "CSE", "BSE", "ASE"]
    }

    # ============================================
    # ORDERING + POSITIONING WITH GROUP SEPARATION
    # ============================================
    ordered_features = []
    group_centers = {}
    y_positions = []
    boundaries = []

    current_pos = 0

    for group, features in feature_groups.items():

        # Sort features within each group by coefficient value
        group_sorted = (
            coefs[features]
            .sort_values()
            .index
            .tolist()
        )

        # Append sorted features to global ordering
        ordered_features.extend(group_sorted)

        # Assign y-axis positions for each feature
        for _ in group_sorted:
            y_positions.append(current_pos)
            current_pos += 1

        # Compute group boundaries and center positions
        # for label group visualization
        start = current_pos - len(group_sorted)
        end = current_pos - 1
        group_centers[group] = (start + end) / 2

        # Store separator line position between groups
        boundaries.append(current_pos)

        # Initial position for the next group
        current_pos += 1

    # Reorder coefficients according to grouped structure
    coefs_sorted = coefs[ordered_features]

    # =========================
    # COLOR MAPPING (CENTERED AT ZERO)
    # =========================
    norm = mpl.colors.TwoSlopeNorm(
        vmin=min,
        vcenter=0,
        vmax=max
    )

    cmap = plt.cm.seismic
    colors = cmap(norm(coefs_sorted.values))

    # =========================
    # BAR PLOT FIGURE
    # =========================
    _, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        y_positions,
        coefs_sorted.values,
        color=colors,
        alpha=0.9
    )

    # Set global x-axis limits for comparability across plots
    ax.set_xlim(min * 1.1 if min < 0 else min * 0.9, max * 1.1)

    # =========================
    # AXES LABELS
    # =========================
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coefs_sorted.index)

    # Vertical reference line at zero
    ax.axvline(0, color="black", linewidth=1)

    # Grid on x-axis for readability
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # =========================
    # GROUP LINES SEPARATORS
    # =========================
    for b in boundaries:
        ax.axhline(b, color="gray", linewidth=1.5, alpha=0.6)

    # =========================
    # GROUP LABELS
    # =========================
    for group, center in group_centers.items():
        ax.text(
            min - 2,  # ligeramente a la izquierda
            center,
            group,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
            color="dimgray",
        )
    
    # =========================
    # TITLE AND STYLING
    # =========================
    ax.set_title(
        r"Influencia de las variables en el grado de cerramiento '{}'".format(class_label),
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.set_xlabel(r"Coeficiente $\beta$", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# DECISION TREE
# ==============================================================================

def plot_tree_feature_importance(config, model_path):
    """
    Plot a horizontal bar chart of decision tree feature importance.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : Path to input dataset CSV file
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".
    Returns
    -------
    None
        Displays a matplotlib figure.

    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Dataset path
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # =========================================================
    # LOAD MODEL FROM PKL FILE
    # =========================================================

    DT_artifacts = joblib.load(model_path)

    DT_model = DT_artifacts["model"]
    DT_name = DT_artifacts["name"]
    print(f"{DT_name} model successfully loaded from pkl file.")


    # =========================================================
    # FEATURE IMPORTANCE BAR PLOT
    # =========================================================

    # Extract feature importance from Decision Tree model
    feature_importance = DT_model.feature_importances_
    
    # Extract feature names from dataset
    data_df = pd.read_csv(dataset_path)
    feature_names = data_df.drop(
        columns=[id_variable, target_variable]
        ).columns
    
    # Sort variables by feature importance value
    sort_index = np.argsort(feature_importance)[::-1]

    # Horizontal bar plot
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=feature_importance[sort_index], 
        y=feature_names[sort_index], 
        palette="Blues_r",  
        edgecolor='black'
    )

    # Title and labels
    plt.title('Ranking de Importancia de variables en Arbol de Decisión', fontsize=14, pad=15)
    plt.xlabel('Feature Importance (normalized)', fontsize=12)
    plt.ylabel('Variables Predictoras', fontsize=12)

    # Grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Text with variables information
    leyenda_texto = (
        "BLOQUES SEMÁNTICOS\n\n"
        "Aspectos estructurales:\n"
        " • CSS: Calles sin salida\n"
        " • CFS: Calle en fondo de saco\n"
        " • CPE: Calle Peatonal\n"
        " • LIN: Comercio Interno\n"
        " • LEX: Comercio Externo\n\n"
        "Distancia:\n"
        " • DIS_1: Urbanización aislada\n"
        " • DIS_2: Urbanización separada\n"
        " • DIS_3: Urbanización integrada\n\n"
        "Tipo de cerramiento:\n"
        " • VER: Verjas       • MUR: Muros\n"
        " • CAD: Cadenas      • BOL: Bolardos\n"
        " • ARB: Arbustos     • CPP: C. prop. privada\n\n"
        "Puntos de acceso:\n"
        " • PVI: Entrada por vivienda\n"
        " • PBL: Entrada por bloque\n"
        " • COM: Entrada común\n"
        " • COMS: Varias entradas comunes\n\n"
        "Uso de la Vía pública:\n"
        " • PPU: Dominio/uso público\n"
        " • PRE: Dominio/uso privado restringido.\n"
        " • PPR: Dominio/uso privado\n\n"
        "Seguridad y Vigilancia:\n"
        " • GSE: Guardia       • CSE: Cámaras\n"
        " • BSE: Barrera       • ASE: Alarma"
    )
    plt.text(
        x=1.1, y=0.5, 
        s=leyenda_texto, 
        transform=plt.gca().transAxes, 
        fontsize=9.5, 
        verticalalignment='center', 
        fontfamily='monospace', 
        bbox=dict(
            boxstyle='round,pad=1', 
            facecolor='#f9f9f9', 
            edgecolor='#cccccc', 
            alpha=1.0
        )
    )


    plt.subplots_adjust(left=0.15, right=0.55, top=0.90, bottom=0.10)
    plt.show()


def get_visual_decision_tree(config, model_path):
    """
    Generate and save a visual representation of a decision tree using dtreeviz.

    This function requires the external 'graphviz' executable to be installed 
    on your system and added to the system's PATH environment variable. 
    Without it, the underlying 'dtreeviz' library will fail to render the 
    tree structure and throw an executable missing error.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - ID_VARIABLE : str
            Column name of the unique identifier.
        - TARGET_VARIABLE : str
            Column name of the target variable.
        - TARGET_LABEL_MAP : dict
            Dictionary mapping original encoded labels to their readable names.
        - IMAGES_DIR : str or Path
            Directory path where the output SVG visualization will be saved.

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".

    Returns
    -------
    viz_tree_view : dtreeviz.utils.DTreeVizRender
        The dtreeviz visualization object containing the rendered tree, 
        suitable for direct display in Jupyter Notebooks.
    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Dataset path
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # Mapping from encoded labels to readable names
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # LOAD MODEL FROM PKL FILE
    # =========================================================
    DT_artifacts = joblib.load(model_path)

    DT_model = DT_artifacts["model"]
    DT_encoder = DT_artifacts["label_encoder"]
    DT_name = DT_artifacts["name"]

    print(f"{DT_name} model successfully loaded from pkl file.")

    # =========================================================
    # PREPARE FEATURE AND TARGET VARIABLES FOR DTREEVIZ
    # =========================================================

    # Get features and target from dataset
    data_df = pd.read_csv(dataset_path)
    X = data_df.drop(
        columns=[id_variable, target_variable]
    ).copy()
    y = data_df[target_variable].copy()

    # Encode target variable to [0,..,K-1] range because the
    # tree was trained with this codification
    y_encoded = DT_encoder.transform(y)

    # Store as int array for dtreeviz internal data types
    y_encoded = np.array(y_encoded, dtype=int)
    
    # Econde target label map in [0,...,K-1] range
    class_names = {
        k - 1: v
        for k, v in target_label_map.items()
    }

    # Get dtreeviz model
    viz_tree = model(
        DT_model,
        X_train=X,
        y_train=y_encoded,
        feature_names=X.columns,
        target_name="Grado de cerramiento",
        class_names=class_names
    )

    # Store dtreeviz model visualization
    viz_tree_view = viz_tree.view(
        title="Estructura del árbol de decisión",
        title_fontsize=18,
    )

    # Save visualization as svg file
    viz_tree_filename = os.path.join(config.IMAGES_DIR, "viz_tree.svg")
    viz_tree_view.save(viz_tree_filename)

    print(f"Decision Tree structure image saved in {viz_tree_filename}")

    return viz_tree


def get_leaf_classes_distribution(config, model_path):
    """
    This method display a table with the class distribution of the
    leaf nodes of a decision tree model.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - TARGET_LABEL_MAP : dict
            Dictionary mapping original encoded labels to their readable names.

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".

    Returns
    -------
    leaf_df : pandas.DataFrame
        DataFrame containing the resulting table
    """    
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    target_label_map = config.TARGET_LABEL_MAP

    # Econde target label map in [0,...,K-1] range
    class_names = {
        k - 1: v
        for k, v in target_label_map.items()
    }

    # =========================================================
    # LOAD MODEL FROM PKL FILE
    # =========================================================
    DT_artifacts = joblib.load(model_path)

    DT_model = DT_artifacts["model"]
    DT_name = DT_artifacts["name"]

    print(f"{DT_name} model successfully loaded from pkl file.")

    # Get tree structure and classes
    tree = DT_model.tree_
    classes = DT_model.classes_

    # Identify leaf nodes
    is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
    leaf_nodes = np.where(is_leaf)[0]

    # List to store leaf data distributions
    leaf_data = []
    for node_id in leaf_nodes:
        # Get number of samples for each class in the leaf
        count_classes = tree.value[node_id][0]
        total_samples = tree.n_node_samples[node_id]
        sum_values_per_leaf = np.sum(count_classes)

        # Calcualte percentage of samples
        if total_samples > 0:
            percentage = (count_classes / sum_values_per_leaf) * 100
        else:
            percentage = np.zeros_like(count_classes)
            
        # Select dominant class of the leaf
        dom_class_index = np.argmax(count_classes)
        dom_class = classes[dom_class_index]
        dom_class_name = class_names.get(dom_class)

        # Create dictionary for current row
        fila = {
            'Leaf_ID': node_id,
            'Total_Muestras': total_samples
        }
        
        # Add samples percentage of each class
        for i, value in enumerate(classes):
            name = class_names.get(value)
            fila[f'{name} (%)'] = round(percentage[i], 2)
            
        # Add dominant class column
        fila['Clase dominante'] = dom_class_name
        
        leaf_data.append(fila)

    # Create final dataframe
    leaf_df = pd.DataFrame(leaf_data)

    # Map internal node_ids to the leaf IDs set in the image
    leaf_id_map = {
        25 : 1,
        26 : 2,
        22 : 3,
        23 : 4,
        24 : 5,
        14 : 6,
        12 : 7,
        15 : 8,
        16 : 9,
        27 : 10,
        28 : 11,
        8  : 12,
        6  : 13,
        17 : 14,
        18 : 15
    }
    leaf_df['Leaf_ID'] = leaf_df['Leaf_ID'].map(leaf_id_map).astype(int)
    leaf_df = leaf_df.sort_values(by="Leaf_ID")

    # Display table
    display(
        leaf_df.style
        .hide(axis="index")
        .format(precision=2)
        .set_caption("Distribución de clases por hoja")
        .set_properties(**{
            'text-align': 'center',
            'border': '1px solid black'
        })
    )

    return leaf_df


# ==============================================================================
# SHAP GLOBAL
# ==============================================================================

def get_shap_svm(config, model_path, background_size=100, kernel_samples=500):
    """
    This function loads a SVM trained model from a serialized artifact, 
    constructs a background dataset for SHAP estimation using stratified sampling, and
    computes SHAP values using the KERNEL SHAP explainer.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - OUTPUT_DIR: str or Path
            Path to the output folder.
        - ID_VARIABLE : str
            Column name of the unique identifier.
        - TARGET_VARIABLE : str
            Column name of the target variable.
        - SEED: int
            Seed for reproducibility

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".

    background_size : int [optional, default=100]
        Number of samples used to construct the SHAP background dataset.

    kernel_samples : int, [optional, default=150]
        Number of Monte Carlo samples used by the Kernel SHAP estimator.
    Returns
    -------
    shap_values : numpy array
        Array of SHAP values with shape (n_samples, n_features, n_classes).
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Paths and folders
    dataset_path = config.DATASET_PATH
    output_dir = config.OUTPUT_DIR

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # Seed for reproducibility
    seed = config.SEED


    # =========================================================
    # LOAD MODEL FROM PKL FILE
    # =========================================================
    artifacts = joblib.load(model_path)

    model = artifacts["model"]
    model_name = artifacts["name"]

    print(f"{model_name} model successfully loaded from pkl file.")

    # =========================================================
    # PREPARE FEATURE AND TARGET VARIABLES
    # =========================================================

    # Get features and target from dataset
    data_df = pd.read_csv(dataset_path)
    X = data_df.drop(
        columns=[id_variable, target_variable]
    ).copy()
    y = data_df[target_variable].copy()

    # =========================================================
    # STRATIFIED SPLIT FOR BACKGROUND DATA
    # =========================================================
    background, _, _, _ = train_test_split(
    X,
    y,
    train_size=background_size,
    stratify=y,
    random_state=seed
    )

    # =========================================================
    # GET SHAP VALUES WITH KERNEL SHAP EXPLAINER
    # =========================================================
    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )
    print("Get shap values with KernelExplainer...")
    shap_values = explainer.shap_values(X, nsamples=kernel_samples)
    print("Shap values calculated succesfully.")
    print(f"Array size: {shap_values.shape}")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    
    # Create output folder to store shap values numpy array
    output_custom_dir = os.path.join(
        output_dir,
        "SHAP_global"
    )
    os.makedirs(output_custom_dir, exist_ok=True)
    print("\nSaving results...")
    print(f"Output directory: {output_custom_dir}")

    np.save(os.path.join(output_custom_dir, f"{model_name}_shap_global.npy"),
            shap_values)
    
    return shap_values


def get_shap_tree(config, model_path):
    """
    This function loads a based-tree trained model from a serialized artifact and 
    computes shap values using TREE SHAP Explainer

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - OUTPUT_DIR: str or Path
            Path to the output folder.
        - ID_VARIABLE : str
            Column name of the unique identifier.
        - TARGET_VARIABLE : str
            Column name of the target variable.

    model_path : str
        Path to the serialized model artifact (joblib .pkl file). This file 
        is expected to contain a dictionary with the trained "model", the 
        "label_encoder", and the model "name".

    Returns
    -------
    shap_values : numpy array
        Array of SHAP values with shape (n_samples, n_features, n_classes).
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Paths and folders
    dataset_path = config.DATASET_PATH
    output_dir = config.OUTPUT_DIR

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # =========================================================
    # LOAD MODEL FROM PKL FILE
    # =========================================================
    artifacts = joblib.load(model_path)

    model = artifacts["model"]
    model_name = artifacts["name"]

    print(f"{model_name} model successfully loaded from pkl file.")

    # =========================================================
    # GET TRAIN DATASET
    # =========================================================
    data_df = pd.read_csv(dataset_path)
    X = data_df.drop(
        columns=[id_variable, target_variable]
    ).copy()

    # =========================================================
    # GET SHAP VALUES WITH TREEE SHAP EXPLAINER
    # =========================================================
    if model_name == "XGBoost":
        # Use internal model attribute of BalancedXGBClassifier wrapper
        explainer = shap.TreeExplainer(model.model)
    else:
        explainer = shap.TreeExplainer(model)

    print("Get shap values with KernelExplainer...")
    shap_values = explainer.shap_values(X)
    print("Shap values calculated succesfully.")
    print(f"Array size: {shap_values.shape}")        

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    
    # Create output folder to store shap values numpy array
    output_custom_dir = os.path.join(
        output_dir,
        "SHAP_global"
    )
    os.makedirs(output_custom_dir, exist_ok=True)
    print("\nSaving results...")
    print(f"Output directory: {output_custom_dir}")

    np.save(os.path.join(output_custom_dir, f"{model_name}_shap_global.npy"),
            shap_values)
    
    return shap_values


def plot_shap_global_feature_importance(config, shap_values, model_name, num_features_display=10, figsize=[10,8]):
    """
    This function computes the global feature importance by averaging
    the absolute SHAP values across all samples and classes, and
    generates a SHAP bar plot.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - ID_VARIABLE : str
            Column name of the unique identifier variable.
        - TARGET_VARIABLE : str
            Column name of the target variable.

    shap_values : numpy.ndarray
        SHAP values array with shape (n_samples, n_features, n_classes)
        where:
        - n_samples is the number of observations,
        - n_features is the number of input variables,
        - n_classes is the number of target classes.

    model_name : str
        Name of the model used for the plot subtitle.

    num_features_display : int [optional, default=10]
        Maximum number of features displayed in the SHAP bar plot.
    
    figsize: list [optional, default=[10,8]]
        Width and height values of the graph.

    Returns
    -------
    None
        Displays the SHAP global feature importance plot.
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Paths and folders
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE 

    # =========================================================
    # GET TRAIN DATA
    # =========================================================
    data_df = pd.read_csv(dataset_path)
    X = data_df.drop(
        columns=[id_variable, target_variable]
    ).copy()

    # =========================================================
    # GLOBAL BAR PLOT
    # =========================================================    
    
    # Calculate SHAP mean values
    shap_values_global = np.mean(np.abs(shap_values), axis=2)
    
    # Create SHAP explainer object
    explainer_values = shap.Explanation(
        values=shap_values_global,
        data = X.values, 
        feature_names= X.columns
    )

    # SHAP bar plot method
    shap.plots.bar(
        explainer_values,
        max_display=num_features_display,
        show_data=False,
        show=False,
    )

    # Custom figsize with matplotlib
    fig = plt.gcf()  
    fig.set_size_inches(figsize[0], figsize[1]) 

    # Main title with Model Name
    fig.suptitle(
        f"{model_name.upper()}",
        fontsize=16,
        color="midnightblue", 
        weight="bold",
        x=0.11, 
        y=1,
        ha="left"
    )
    # Graph title
    fig.text(
        x=0.11,        
        y=0.95,        
        s="Importancia global de variables con SHAP",
        fontsize=14,
        color="midnightblue",
        fontweight="bold",
        ha="left"
    )

    plt.xlabel(
        "Average impact on model output (mean|SHAP value|)",
        fontsize=12,
        fontweight="bold",
        labelpad=10
    )
    plt.ylabel(
        "Variable",
        fontsize=12,
        fontweight="bold",
        labelpad=10        
    )
    plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    
    plt.tight_layout()
    plt.show()


def plot_shap_heatmap(config, shap_values, model_name, figsize=[10,8]):
    """
    This function computes the global feature importance per class by averaging
    the absolute SHAP values across all samples and
    generates a Heat Map.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - ID_VARIABLE : str
            Column name of the unique identifier variable.
        - TARGET_VARIABLE : str
            Column name of the target variable.
        - TARGET_LABEL_MAP: dict
            Dictionary mapping original encoded labels to their readable names.

    shap_values : numpy.ndarray
        SHAP values array with shape (n_samples, n_features, n_classes)
        where:
        - n_samples is the number of observations,
        - n_features is the number of input variables,
        - n_classes is the number of target classes.

    model_name : str
        Name of the model used for the plot subtitle.
 
    figsize: list [optional, default=[10,8]]
        Width and height values of the graph.

    Returns
    -------
    None
        Displays the SHAP Heat Map feature importance plot per class.
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Paths and folders
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE 

    # Mapping from encoded labels to readable names
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # GET FEATURES NAME
    # =========================================================
    data_df = pd.read_csv(dataset_path)
    feature_names = data_df.drop(
        columns=[id_variable, target_variable]
    ).columns

    # =========================================================
    # HEATMAP
    # =========================================================  
    
    # Get mean SHAP value per class
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))

    # Heatmap
    im = ax.imshow(
        mean_shap_values,
        aspect='auto',
        cmap='viridis',          
        interpolation='nearest'
    )

    # Main title with Model Name
    fig.suptitle(
        f"{model_name.upper()}",
        fontsize=16,
        color="midnightblue", 
        weight="bold",
        x=0.11, 
        y=1,
        ha="left"
    )
    # Graph title
    fig.text(
        x=0.11,        
        y=0.94,        
        s="Importancia de variables por clase",
        fontsize=14,
        color="midnightblue",
        fontweight="bold",
        ha="left"
    )

    ax.set_xlabel(
        "Grado de cerramiento",
        fontsize=13,
        fontweight="bold",
        labelpad=12
    )
    ax.set_ylabel(
        "Variables",
        fontsize=13,
        fontweight="bold",
        labelpad=12
    )
    ax.set_xticks(np.arange(len(target_label_map)))
    ax.set_xticklabels(
        list(target_label_map.values()),
        fontsize=11,
        rotation=0
    )
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(
        feature_names,
        fontsize=10
    )
    ax.tick_params(axis='both', which='major', length=0)

    # Add grid between cells to get better visibility
    ax.set_xticks(np.arange(-0.5, len(target_label_map), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Color bar (legend)
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )
    cbar.set_label(
        "Mean(|SHAP value|)",
        fontsize=11,
        labelpad=15
    )
    cbar.ax.tick_params(labelsize=12)

    # Drop spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Show image
    plt.tight_layout()
    plt.show()


def plot_shap_class_dashboard(config, shap_values, class_label, model_name, num_features_display=10, figsize=[18, 10]):
    """
    Generate a side-by-side SHAP analysis visualization for a specific target class.

    This function creates a single figure containing two subplots side by side:
    1. A global feature importance bar plot showing the average impact on the 
       selected class output.
    2. A beeswarm plot displaying the detailed distribution of local impacts 
       (One-vs-Rest) for the selected class.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to the input dataset CSV file.
        - ID_VARIABLE : str
            Column name of the unique identifier variable.
        - TARGET_VARIABLE : str
            Column name of the target variable.
        - TARGET_LABEL_MAP: dict
            Dictionary mapping original encoded labels to their readable names.

    shap_values : numpy.ndarray
        SHAP values array with shape (n_samples, n_features, n_classes)
        where:
        - n_samples is the number of observations,
        - n_features is the number of input variables,
        - n_classes is the number of target classes.

    class_label: str
        Label of the class to analyze.

    model_name : str
        Name of the model used for the plot subtitle.

    num_features_display : int [optional, default=10]
        Maximum number of features displayed in the SHAP bar plot.
    
    figsize: list [optional, default=[18, 10]]
        Width and height values of the graph.

    Returns
    -------
    None
        
    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================

    # Paths and folders
    dataset_path = config.DATASET_PATH

    # Identifier and target variables
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE 

    # Mapping from encoded labels to readable names
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # GET TRAIN DATA
    # =========================================================
    data_df = pd.read_csv(dataset_path)
    X = data_df.drop(
        columns=[id_variable, target_variable]
    ).copy()

    # Econde target label map in [0,...,K-1] range
    class_names = {
        k - 1: v
        for k, v in target_label_map.items()
    }

    # =========================================================
    # EXPLAINER OBJECT
    # =========================================================
    
    # Find class encode number associated to readable name
    class_value = next(k for k, v in class_names.items() if v == class_label)

    # Select SHAP values for current class
    shap_values_class = shap_values[:,:,class_value]

    # Create SHAP explainer object
    explainer_values = shap.Explanation(
        values=shap_values_class,
        data = X.values, 
        feature_names= X.columns
    )

    # =========================================================
    # INITIALIZE SUBPLOTS WITH MATPLOTLIB
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # =========================================================
    # SHAP BAR PLOT (Left graph)
    # =========================================================
    
    plt.sca(ax1) 
    shap.plots.bar(
        explainer_values,
        max_display=num_features_display,
        show_data=False,
        show=False,
    )

    # Custom title and axis
    ax1.set_title("Importancia de variables", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Average impact on model output (mean|SHAP value|)", fontsize=14, fontweight="bold", labelpad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    # =========================================================
    # BEESWARM PLOT (Righ graph)
    # =========================================================
    plt.sca(ax2)
    shap.plots.beeswarm(
        explainer_values,
        max_display=num_features_display,
        color_bar=False,
        show=False
    )

    # Custom title and axis 
    ax2.set_title("Dirección y magnitud del impacto (gráfico Beeswarm)", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Impact on model output (SHAP value)", fontsize=14, fontweight="bold", labelpad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    
    # Color legend
    ax2.scatter([], [], c="blue", label="0 (Ausencia)")
    ax2.scatter([], [], c="red", label="1 (Presencia)")
    ax2.legend(
        title="Valor de la variable",
        loc="lower right",       
        framealpha=0.9,          
        facecolor="white",       
        edgecolor="gray",      
        bbox_to_anchor=(0.98, 0.05),
        fontsize=12,
        title_fontsize=12,
    )

    # =========================================================
    # FORCE GLOBAL FIGSIZE AND TITLE
    # =========================================================
    fig.set_size_inches(figsize[0], figsize[1])

    # Main title with Model Name
    fig.suptitle(
        f"{model_name.upper()}",
        fontsize=18,
        color="midnightblue", 
        weight="bold",
        x=0.12, 
        y=1,
        ha="left"
    )
    # Graph title
    fig.text(
        x=0.12,        
        y=0.96,        
        s=f"Analisis del grado de cerramiento ''{class_label}'' con SHAP",
        fontsize=16,
        color="midnightblue",
        fontweight="bold",
        ha="left"
    )

    plt.tight_layout()
    plt.show()



# ==============================================================================
# SHAP LOCAL
# ==============================================================================

def get_shap_out_of_fold(config, model, model_name):
    """
    This method computes out-of-fold predictions and local SHAP values using outer folds
    of NCV configuration.
 
    For each outer fold, it loads the optimal hyperparameters previously found during 
    the inner CV loop, tunes the provided model, fits it on the outer training 
    set, evaluates it on the outer test set, and extracts local feature importances 
    using SHAP (SHapley Additive exPlanations) values.

    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to input dataset CSV file
        - DATA_FOLDS_DIR : str or Path 
            Directory containing precomputed fold CSV files
        - OUTPUT_DIR : str or Path 
            Directory where results will be saved
        - OUTER_SPLITS : int 
            Number of outer CV folds to load correct files.
        - INNER_SPLITS : int
            Number of inner CV folds to load correct files.
        - OUTER_FOLD_FILENAME: str 
            Name of outer folds file.
        - ID_VARIABLE : str 
            Unique identifier column name 
        - TARGET_VARIABLE : str
            Target variable column name

    model : object
        A scikit-learn compatible model instance.

    model_name : str
        The name of the algorithm being evaluated (Only "XGBoost" and "Logistic_regression" are allowed)
        This controls conditional logic for SHAP explainers and determines file naming.

    Returns
    -------
    all_pred_info_df: pandas.DataFrame:
        Out-of-fold predictions for all samples across all outer folds, including 
        true targets, predicted classes, maximum probabilities, and fold indices.

    all_shap_info_df: pandas.DataFrame:
        Out-of-fold shap values for all samples across al outer folds.
    
    expected_shap_values_df: pandas.DataFrame:
           The baseline/expected SHAP values (explainer base values) for each class 
           across all outer folds.
    """
    # =========================================================
    # VALIDATE MODEL COMPATIBILITY
    # =========================================================
    supported_models = ["XGBoost", "Logistic_Regression"]
    if model_name not in supported_models:
        raise ValueError(
            f"\n" + "!"*80 + "\n"
            f" ERROR: Model '{model_name}' is not currently supported in this method.\n"
            f" Supported models are: {supported_models}\n"
            f" Execution stopped.\n"
            + "!"*80
        )
    
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH
    folds_dir = config.DATA_FOLDS_DIR
    output_dir = config.OUTPUT_DIR

    # Nested Cross Validation splits size
    outer_splits = config.OUTER_SPLITS
    inner_splits = config.INNER_SPLITS

    # Nested Cross Validation folds filenames
    outer_file_name = config.OUTER_FOLD_FILENAME

    # Unique identifier (CC) and target variables 
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # =========================================================
    # LOAD DATASET
    # =========================================================
    dataset_df = pd.read_csv(dataset_path)

    # =========================================================
    # LOAD OUTER NCV FOLDS
    # =========================================================
    outer_folds_df = pd.read_csv(
        os.path.join(folds_dir, outer_file_name)
    )

    # =========================================================
    # LOAD OPTIMAL HYPERPARAMETERS PER NCV FOLD
    # =========================================================
    hyperparam_path = os.path.join(
        output_dir,
        f"{outer_splits}x{inner_splits}_NCV",
        f"{model_name}",
        f"{model_name}_global_metrics.csv")
    hyperparam_df = pd.read_csv(hyperparam_path)

    # =========================================================
    # LABEL ENCODING
    # =========================================================

    # Encode the K classes to the range of values ​​[0,..,K-1] so that
    #  models like XGBoost can work correctly
    label_encoder = LabelEncoder()
    dataset_df[target_variable] = label_encoder.fit_transform(dataset_df[target_variable])

    # =========================================================
    # OUT-OF-FOLD PREDICTION+SHAP LOOP
    # =========================================================  
    pred_info_list = []
    shap_info_list = []
    expected_values_folds = []  

    print("\n" + "="*80)
    print(f" STARTING PROCESS: OUT-OF-FOLD PREDICTIONS & LOCAL SHAP".center(80))
    print("="*80)
    print(f" • Model Name      : {model_name}")
    print(f" • Outer CV Folds  : {outer_splits}")
    print("-"*80)

    for outer_fold_idx in range(outer_splits):

        print(f"\n[ FOLD {outer_fold_idx}/{outer_splits} ]".ljust(80, "-"))

        # =========================================================
        # TRAIN/TEST SET FOR CURRENT FOLD
        # =========================================================  

        # The samples used for the Outer Test Set are those 
        # with the fold idx of the current outer_fold_idx
        outer_test_ids = set(
            outer_folds_df[
                outer_folds_df["outer_fold_idx"]
                == outer_fold_idx
            ][id_variable]
        )

        # The Outer Train set consists of all samples with a different
        # fold idx than the current outer_fold_idx
        outer_train_ids = set(
            outer_folds_df[
                outer_folds_df["outer_fold_idx"]
                != outer_fold_idx
            ][id_variable]
        )

        # Build train/test dataframes with IDs
        train_df = dataset_df[
            dataset_df[id_variable].isin(outer_train_ids)
        ].copy()

        test_df = dataset_df[
            dataset_df[id_variable].isin(outer_test_ids)
        ].copy()

        # Set id_variable as index for data storing procedure
        train_df = train_df.set_index(id_variable)
        test_df = test_df.set_index(id_variable)

        # =====================================================
        # TRAIN / TEST FEATURES & TARGET
        # =====================================================

        X_train = train_df.drop(
            columns=[target_variable]
        )
        y_train = train_df[target_variable]

        X_test = test_df.drop(
            columns=[target_variable]
        )
        y_test = test_df[target_variable]

        # =====================================================
        # SET OPTIMAL HYPERPARAMETERS IN THE MODEL
        # =====================================================        
        best_params_str = hyperparam_df.loc[
            hyperparam_df["outer_fold"] == outer_fold_idx, "best_params"
        ].iloc[0]

        # Convert string to dict
        best_params_dict = ast.literal_eval(best_params_str)

        print("Optimal hyperparameters for current fold:")
        for param_name, param_values in best_params_dict.items():
            print(f"{param_name:<25}: {param_values}")

        # Remove model_ prefix
        clean_params = {
            key.replace("model__", ""): value
            for key, value in best_params_dict.items()
        }

        # Get a model copy and set optimal hyperparameters
        model_tuned = copy.deepcopy(model)
        model_tuned.set_params(**clean_params)

        # =====================================================
        # TRAIN
        # ===================================================== 
        print("\nTraining     | Fitting model... ")
        model_tuned.fit(X_train, y_train)

        # =====================================================
        # PREDICTION
        # =====================================================
        print("Predicting   | Generating out-of-fold predictions... ")
        y_pred = model_tuned.predict(X_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_proba = model_tuned.predict_proba(X_test)
        max_prob = np.max(y_proba, axis=1)

        # Store prediction information for current fold
        pred_info_df = pd.DataFrame({
            id_variable: test_df.index,
            target_variable: label_encoder.inverse_transform(y_test),
            f"{target_variable}_pred": y_pred_decoded,
            "prob": max_prob,
            "fold": outer_fold_idx
        })

        pred_info_list.append(pred_info_df)

        # =====================================================
        # GET SHAP VALUES
        # ===================================================== 
        if model_name == "XGBoost":
            print("SHAP Values  | Explaining via TreeExplainer... ")
            # Create explainer
            explainer = shap.TreeExplainer(model_tuned.model)

            # Get SHAP values
            shap_values = explainer.shap_values(X_test)

            # Get expected shap value for reference
            expected_values = explainer.expected_value
        
        elif model_name == "Logistic_Regression":
            print("SHAP Values  | Explaining via LinearExplainer... ")
            # Create explainer
            explainer = shap.LinearExplainer(model_tuned, X_train)

            # Get SHAP values
            shap_values = explainer.shap_values(X_test)

            # Get expected shap value for reference
            expected_values = explainer.expected_value            


        # Store shap values
        n_samples, n_features, n_classes = shap_values.shape

        ids = np.repeat(X_test.index, n_features * n_classes)
        features = np.tile(np.repeat(X_test.columns, n_classes), n_samples)
        classes = np.tile(np.arange(1, n_classes + 1), n_samples * n_features)

        shap_info_df = pd.DataFrame({
            id_variable: ids,
            "feature": features,
            target_variable: classes,
            "shap_value": shap_values.reshape(-1),
            "fold": outer_fold_idx
        })
        shap_info_list.append(shap_info_df)

        # Store shap expected values
        expected_values_folds.append({
            "fold": outer_fold_idx,
            **{
                f"class_{i+1}": float(ev)
                for i, ev in enumerate(expected_values)
            }
        })

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    print("\n" + "="*80)
    print(" PROCESSING RESULTS & SAVING ".center(80, "="))
    print("="*80)

    # Concatenate fold results
    all_pred_info_df = pd.concat(pred_info_list, ignore_index=True)
    all_shap_info_df = pd.concat(shap_info_list, ignore_index=True)
    expected_shap_values_df = pd.DataFrame(expected_values_folds)

    # Create output folder for SHAP local
    output_custom_dir = os.path.join(
        output_dir,
        f"SHAP_local",
        f"{model_name}"
    )
    os.makedirs(output_custom_dir, exist_ok=True)

    print(f" -> Output directory: {output_custom_dir}")
    print(" -> Exporting CSVs... ", end="", flush=True)

    all_pred_info_df.to_csv(
        os.path.join(output_custom_dir, f"{model_name}_pred_info.csv"),
        index=False
    )
    all_shap_info_df.to_csv(
        os.path.join(output_custom_dir, f"{model_name}_shap_info.csv"),
        index=False
    )
    expected_shap_values_df.to_csv(
        os.path.join(output_custom_dir, f"{model_name}_expected_shap_values.csv"),
        index=False
    )

    print("\n" + "="*80)
    print(" PROCESS COMPLETED SUCCESSFULLY ".center(80, " "))
    print("="*80 + "\n")

    return all_pred_info_df, all_shap_info_df, expected_shap_values_df    


def plot_shap_waterfall(
        config,
        pred_info_df,
        expected_value_df,
        shap_values_df,
        id_sample,
        model_name,
        num_features_display=10,
        figsize=[10,8]):
    """
    This method displays a SHAP waterfall plot to explain a specific sample identified
    by id_sample input parameter.
    
    Parameters
    ----------
    config : object
        Configuration object containing project parameters. Must include:
        - DATASET_PATH : str or Path
            Path to input dataset CSV file
        - ID_VARIABLE : str 
            Unique identifier column name 
        - TARGET_VARIABLE : str
            Target variable column name
        - TARGET_LABEL_MAP: dict
            Dictionary mapping original encoded labels to their readable names.
    
    pred_info_df : pandas.DataFrame
        Dataframe containing out-of-fold predictions information. Must include:
        - id_variable: Sample identifier value
        - target_variable: Actual class value
        - target_variable_pred: Predicted class value
        - fold: Fold where the sample was evaluated

    expected_value_df : pandas.DataFrame
        Dataframe containing the expected SHAP value per class in each fold.
        - id_variable: Sample identifier value
        - feature: Name of the feature
        - target_variable: Class
        - shap_value: SHAP value for this feature and class
        - fold: Fold where the sample was evaluated

    shap_values_df: pandas.DataFrame
        Dataframe containinr out-of-fold shap values. Must include:

    id_sample: int
        ID value to get sample information using id_variable column in dataframes.
    
    model_name : str
        The name of the algorithm being explained with SHAP.
        This controls conditional logic for SHAP explainers and determines file naming.

    num_features_display : int [optional, default=10]
        Maximum number of features displayed in the Waterfall plot.
    
    figsize: list [optional, default=[10, 18]]
        Width and height values of the graph.

    Returns
    -------
    None
        Displays the SHAP local waterfall plot.
    """    
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH

    # Unique identifier (CC) and target variables 
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # Mapping from encoded labels to readable names
    target_label_map = config.TARGET_LABEL_MAP

    # =========================================================
    # LOAD DATASET
    # =========================================================
    dataset_df = pd.read_csv(dataset_path)

    # =========================================================
    # GET FEATURE DATA 
    # =========================================================

    # Filter dataframe for id_sample
    features_values_df = dataset_df[
        dataset_df[id_variable] == id_sample
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

    # Get actual class
    actual_class = pred_info_df.loc[
        pred_info_df[id_variable] == id_sample,
        f"{target_variable}"
    ].iloc[0]

    # Get predicted class
    pred_class = pred_info_df.loc[
        pred_info_df[id_variable] == id_sample,
        f"{target_variable}_pred"
    ].iloc[0]

    # Get fold of the class to extract expected value
    fold_idx = pred_info_df.loc[
        pred_info_df[id_variable] == id_sample,
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
        (shap_values_df[id_variable] == id_sample) &
        (shap_values_df[target_variable] == pred_class)
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

    # Custom figsize with matplotlib
    fig = plt.gcf()  
    fig.set_size_inches(figsize[0], figsize[1]) 

    # Get readable names for classses
    actual_class_label = target_label_map[actual_class]
    pred_class_label = target_label_map[pred_class]

    # Main title with Model Name
    fig.suptitle(
        f"{model_name.upper()}",
        fontsize=16,
        color="midnightblue", 
        weight="bold",
        x=0.05, 
        y=1,
        ha="left"
    )
    # Sample information
    fig.text(
        x=0.05,        
        y=0.95,        
        s=f"Complejo residencial {id_sample}, evaluado en fold  {fold_idx}",
        fontsize=12,
        color="midnightblue",
        fontweight="bold",
        ha="left"
    )
    # Classification information
    fig.text(
        x=0.05, 
        y=0.9, 
        s=f"Clase real: {actual_class_label}   |   Clase predicha: {pred_class_label}",
        fontsize=14,
        color="darkslategray",
        fontweight="bold",
        ha="left", 
    )

    # Margins set to allow space for text
    plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    
    # Adjust top margin to avoid the text enter in the graph
    plt.subplots_adjust(top=0.82)
    plt.show()