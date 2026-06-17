import os
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib

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
        Configuration object containing:
        - DATASET_PATH : Path to input dataset CSV file
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name
        - TARGET_LABEL_MAP: dictionary mapping encoded labels to names

    model_path : str
        Path to the serialized model artifact (joblib .pkl file).

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