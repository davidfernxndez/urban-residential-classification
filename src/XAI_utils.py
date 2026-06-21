import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from dtreeviz import model
import joblib
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