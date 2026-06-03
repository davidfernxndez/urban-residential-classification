"""
Utility functions for Exploratory Data Analysis (EDA).

This module provides reusable tools for the statistical analysis and visualization
of categorical and binary variables performed in notebooks/01_EDA.ipynb.

It includes functions for univariate and bivariate analysis, association testing,
co-occurrence analysis, correlation matrix, and frequency visualization to support
data exploration and interpretation.

It also includes a method to display data points in to a fix map.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import pandas as pd



def plot_fix_map(df, title, figsize=(8,8)):
    '''
    This function represents on a map the residential complexes present in the input dataframe.

    The "EPSG:25830" coordinate system from GeoPandas is used, which georeferences the ETRS89 / UTM zone 30N system. The coordinates used are:
    - Latitude: East coordinates (x-axis).
    - Longitude: West coordinates (y-axis).

    Residential complexes are represented with a marker that indicates their level of enclosure (URB classification)

    Parameters
    ----------
    df: pandas DataFrame
        Data frame containing residential complexes. Must include LAT, LON, and URB columns.
    title: string
        Map title
    figsize: tuple [Optional]
        Size of the map visualization.
    '''
    
    # Check URB, LAT and LON columns exist in the dataframe
    for col in ["URB", "LAT", "LON"]:
        assert col in df.columns, f"Error: '{col}' column is missing."
    
    # Marker style for each URB classification
    styles = {
        "Protegido":      {"color": "red",    "marker": "o"},
        "Controlado":     {"color": "orange",   "marker": "s"},
        "Autoaislado":    {"color": "yellow",  "marker": "^"},
        "Individualista": {"color": "blue", "marker": "D"},
        "Simbólico":      {"color": "green", "marker": "X"},
    }

    # Create column with URB label
    if "URB_label" not in df.columns:
        mapping = {
            1: "Protegido",
            2: "Controlado",
            3: "Autoaislado",
            4: "Individualista",
            5: "Simbólico"
        }
        df["URB"] = df["URB"].astype(int)
        df["URB_label"] = df["URB"].map(mapping)

    # Convert dataframe into a geospatial object
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LAT"], df["LON"]),
        crs="EPSG:25830"
    )
    
    # Conversion of CRS coordinates to Web Mercator (required for display on maps)
    # Converts from CRS units to Web Mercator units
    gdf = gdf.to_crs(epsg=3857)

    # Crate figure to plot the map
    _, ax = plt.subplots(figsize=figsize)

    # Plot data according to URB marker
    for label, style in styles.items():

        subset = gdf[gdf["URB_label"] == label]

        if not subset.empty:
            subset.plot(
                ax=ax,
                color=style["color"],
                marker=style["marker"],
                markersize=50,
                alpha=0.7,
                label=label,
                edgecolor="white",
                linewidth=1.2,
                zorder=3
            )

    # =========================================================
    # ADJUST AUTOMATIC ZOOM
    # =========================================================

    # Get x and y limits
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Calculate center of the map
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Zoom configuration
    base_half_side = 150  # baseline
    dx = xmax - xmin # X max range
    dy = ymax - ymin # y max range
    max_span = max(dx, dy)

    # Get number of data points in the dataframe
    n = len(gdf)
    
    if n == 1:
        # A single residential complex: Zoom 100x100m
        half_side = base_half_side

    else:
        # If there is more than one residential complex,
        #  it is guaranteed that all of them will appear.
        half_side = max(max_span / 2, base_half_side)

        # Avoid zooming in too wide in a few places
        if n < 5:
            half_side = max(half_side, base_half_side * 2)   
        elif n < 20:
            half_side = max(half_side, base_half_side * 4)  

    # Apply Zoom
    ax.set_xlim(cx - half_side, cx + half_side)
    ax.set_ylim(cy - half_side, cy + half_side)

    # =========================================================
    # BASEMAP
    # =========================================================
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    
    # Título y configuraciones visuales
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(title="Grado de cerramiento:")
    ax.set_axis_off() 
    plt.tight_layout()
    plt.show() 


def categorical_bivariate_analysis(
        var_df,
        var_dep,
        var_ind,
        title,
        name_var_dep=None,
        name_var_ind=None,
        figsize=(10, 5)):
    """
    Performs a bivariate statistical and visual analysis between two categorical variables.

    This function evaluates the association between a dependent and an independent 
    categorical variable by conducting Pearson's Chi-squared test of independence 
    and calculating Cramér's V to measure the strength of the relationship. It 
    concludes by rendering a professional side-by-side visualization containing 
    a percentage-based contingency heatmap and a summary statistical card.

    Parameters
    ----------
    var_df : pandas.DataFrame
        The dataset containing the variables to be analyzed.
    var_dep : str
        The column name of the dependent categorical variable in `var_df`.
    var_ind : str
        The column name of the independent categorical variable in `var_df`.
    title : str
        The global title to be displayed on top of the generated visualization.
    name_var_dep : str, optional
        A clean, human-readable display name for the dependent variable to use 
        as the axis label. If None, defaults to `var_dep`.
    name_var_ind : str, optional
        A clean, human-readable display name for the independent variable to use 
        as the axis label. If None, defaults to `var_ind`.
    figsize : tuple of (float, float), default (10, 5)
        Width and height of the figure in inches.

    Returns
    -------
    None
        The function directly displays a Matplotlib/Seaborn plot and does not 
        return any values.

    Notes
    -----
    - The contingency table used for the heatmap displays relative frequencies (percentages).
    """
    # =========================
    # CONTINGENCY TABLE
    # =========================
    tabla = pd.crosstab(var_df[var_dep], var_df[var_ind])

    # =========================
    # TEST CHI-CUADRADO
    # =========================
    chi2, p, _, _ = chi2_contingency(tabla)

    # =========================
    # CRAMER'S V
    # =========================
    n = tabla.values.sum()
    k = min(tabla.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))

    # Strength interpretation
    if cramers_v <= 0.1:
        strength = "Muy débil"
    elif cramers_v <= 0.3:
        strength = "Débil"
    elif cramers_v <= 0.5:
        strength = "Moderada"
    else:
        strength = "Fuerte"

    # =========================
    # Percentage table
    # =========================
    tabla_pct = pd.crosstab(
        var_df[var_ind],
        var_df[var_dep],
        normalize="columns"
    ) * 100

    # ===============================
    # CONTINGENCY TABLE VISUALIZATION
    # ===============================
    fig, (ax_heatmap, ax_text) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.5, 1.2]}
    )

    # Heatmap
    sns.heatmap(
        tabla_pct, 
        annot=True, 
        fmt=".1f", 
        cmap="YlGn", 
        linewidths=1.5,       # Separa las celdas elegantemente
        linecolor="white", 
        cbar=True,           # Opcional: quitar barra si los números ya son claros
        annot_kws={"size": 11, "weight": "bold"}, # Números más legibles
        ax=ax_heatmap
    )

    # Label configuration
    ax_heatmap.set_xlabel(name_var_dep if name_var_dep else var_dep, fontsize=11, labelpad=12, fontweight="semibold")
    ax_heatmap.set_ylabel(name_var_ind if name_var_ind else var_ind, fontsize=11, labelpad=12, fontweight="semibold")
    ax_heatmap.tick_params(labelsize=10)
    ax_heatmap.tick_params(axis='x', rotation=0)

    # Chart Text
    ax_text.axis("off")
    chart_title = "Resultados test $\chi^2$ de Pearson:"
    chart_metrics = (
        f"$\\chi^2$ = {chi2:.3f}\n"
        f"p-value = {p:.5f}\n"
        f"V de Cramer = {cramers_v:.3f}\n\n"
        f"Interpretación:  Relación {strength}"
    )
    ax_text.text(
        0.05, 0.5, f"{chart_title}\n\n{chart_metrics}", 
        fontsize=12,
        va="center",
        ha="left",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#F8FAFC",  
            edgecolor="#E2E8F0",
            linewidth=1.5
        )
    )


    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.05, ha="left", y=0.98)
    plt.tight_layout()
    plt.show()


def analyze_binary_cooccurrence(df, variables, labels_dict=None):
    '''
    Performs a co-occurrence and individual frequency analysis for two binary variables.
    
    This function calculates absolute and relative frequencies for both individual 
    variables and their joint occurrences. It appends a new categorical column to a 
    copy of the DataFrame encoding the occurrence state ("None", "Only Var1", 
    "Only Var2", "Both").
    
    Finally, it displays a side-by-side visualization consisting 
    of a clear Pie Chart for co-occurrence proportions and a Bar Chart
    for individual presence counts.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the binary variables.
    variables : list of str
        A list containing exactly two column names to analyze.
    labels_dict : dict, optional
        A dictionary mapping the extened column names to display in
        chart text.

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with an additional column named 
        'Coocurrencia_<var1>_<var2>' containing the encoded occurrence states.
    '''

    if len(variables) != 2:
        raise ValueError("The list of variables must contain exactly 2 elements.")

    v1, v2 = variables
    df_copy = df.copy()
    total_rows = len(df_copy)

    # =========================================================================
    # INDIVIDUAL FREQ TABLE
    # =========================================================================
    print("=== TABLA DE FRECUENCIAS INDIVIDUALES (Valor = 1) ===")
    abs_v1 = df_copy[v1].sum()
    abs_v2 = df_copy[v2].sum()
    
    # Calculamos el porcentaje respecto al total del dataframe
    pct_v1 = (abs_v1 / total_rows) * 100
    pct_v2 = (abs_v2 / total_rows) * 100

    ind_table = pd.DataFrame({
        "Variable Original": [v1, v2],
        "Frecuencia Absoluta (1)": [abs_v1, abs_v2],
        "Porcentaje del Total (%)": [round(pct_v1, 2), round(pct_v2, 2)]
    }, index=[v1, v2])
    
    print(ind_table)
    print("\n" + "="*50 + "\n")

    # =================================
    # CREATE NEW COOCURRENCE COLUMN
    # =================================
    def code_coocurrence(row):
        val1, val2 = row[v1], row[v2]
        if val1 == 1 and val2 == 1:
            return "Ambos"
        elif val1 == 1 and val2 == 0:
            return f"Solo {v1}"
        elif val1 == 0 and val2 == 1:
            return f"Solo {v2}"
        else:
            return "Ninguno"


    col_name = f"{v1}_{v2}"
    df_copy[col_name] = df_copy.apply(code_coocurrence, axis=1)
    orden_categorias = ["Ninguno", f"Solo {v1}", f"Solo {v2}", "Ambos"]
    df_copy[col_name] = pd.Categorical(
        df_copy[col_name], categories=orden_categorias, ordered=True
    )

    # =================================
    # PRINT COOCURRENCE TABLE
    # =================================
    print(f"=== TABLA DE FRECUENCIAS: {v1} vs {v2} ===")
    freq_abs = df_copy[col_name].value_counts().reindex(orden_categorias, fill_value=0)
    freq_rel = (
        df_copy[col_name]
        .value_counts(normalize=True)
        .reindex(orden_categorias, fill_value=0)
        * 100
    )
    table_freq = pd.DataFrame(
        {"Frecuencia Absoluta": freq_abs, "Porcentaje (%)": freq_rel.round(2)}
    )
    print(table_freq)
    print("-" * 50)

    # =================================
    # CREATE FIGURE (PIE CHART+BAR PLOT)
    # =================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie Chart (Coocurrence)
    data_pie = freq_abs[freq_abs > 0]
    colors = sns.color_palette("Set3", n_colors=len(data_pie))

    axes[0].pie(
        data_pie,
        labels=data_pie.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    axes[0].set_title(
        f"Distribución de coocurrencias\n({v1} y {v2})", fontsize=12, pad=15
    )

    # Bar Plot: Individual distribution
    ind_dist = df[variables].sum()
    sns.barplot(
        x=[v1, v2],
        y=ind_dist.values,
        ax=axes[1],
        palette="magma",
        hue=[v1, v2],
        legend=False,
    )
    axes[1].set_title("Presencia individual en los complejos residenciales", fontsize=12, pad=15)
    axes[1].set_ylabel("Número de complejos residenciales")
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6, color="gray")
    
    # Text Chart to print variables extended name
    if labels_dict is not None:
            lbl1 = labels_dict.get(v1, v1) if labels_dict else v1
            lbl2 = labels_dict.get(v2, v2) if labels_dict else v2
            chart_text = f"{v1}: {lbl1}       {v2}: {lbl2}"

            # Chart style
            chart_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8)

            # Chart position
            fig.text(
                0.5,
                0.0,
                chart_text,
                fontsize=14,
                color="black",
                ha="center",
                va="bottom",
                bbox=chart_props,
            )

    # Margin
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.show()
    
    return df_copy


def plot_univariate_distribution(
        df,
        column_name,
        title=None,
        axis_label=None,
        orientation = "vertical",
        show_values = True,
        dict_label=None,
        order=None,
        figsize=(10,6)
    ):
    """
    Displays the univariate distribution of a categorical variable.

    This function plots a bar chart showing the frequency distribution
    of the specified categorical column, with optional customization
    for ordering, labeling, and orientation.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the categorical column to analyze.
    title : str [optional]
        The title of the generated plot. If None, a default title is used.
    axis_label : str [optional]
        Label used for the main axis (x-axis in vertical mode,
        y-axis in horizontal mode). If None, column_name is used.
    orientation : {"vertical", "horizontal"}, default "vertical"  [optional]
        Determines the orientation of the bar plot.
    show_values : bool, default True  [optional]
        If True, displays the count and percentage values on top
        (or at the end) of each bar.
    dict_label : dict [optional]
        Mapping between original variable values and display names.
    order : list [optional]
        Explicit categorical order for displaying the values of column_name.
    figsize : tuple, default (10, 6)  [optional]
        Figure size of the plot (width, height).
    Returns:
    --------
    None
    """

    # Count frequency and percentage for bar plot
    counts = df[column_name].value_counts()
    # Use order for categorical values
    if order is not None:
        counts = counts.reindex(order)
    percentages = counts / counts.sum() * 100
    
    # Mapping variable names to dict_label naming
    if dict_label is not None:
        counts.index = counts.index.map(dict_label)
        percentages.index = percentages.index.map(dict_label)

    _, ax = plt.subplots(figsize=figsize)
    
    # ======================
    # BAR PLOT ORIENTATION
    # ======================
    if orientation == "vertical":
        sns.barplot(
            x=counts.index,
            y=counts.values,
            palette="Blues_r",
            ax=ax,
            edgecolor="white",
            linewidth=1.5,
            order=counts.index if order is not None else None
        )

        # Labels above bars
        if show_values:
            for bar, cat in zip(ax.patches, counts.index):
                count = counts.loc[cat]
                pct = percentages.loc[cat]

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.03,
                    f"{int(count):,} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="semibold"
                )

    elif orientation == "horizontal":
        sns.barplot(
            x=counts.values,
            y=counts.index,
            palette="Blues_r",
            ax=ax,
            edgecolor="white",
            linewidth=1.5,
            order=counts.index if order is not None else None
        )

        # Labels at the end of the bar
        if show_values:
            for bar, cat in zip(ax.patches, counts.index):
                count = counts.loc[cat]
                pct = percentages.loc[cat]

                ax.text(
                    bar.get_width() + counts.max() * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(count):,} ({pct:.1f}%)",
                    va="center",
                    ha="left",
                    fontsize=12,
                    fontweight="semibold"
                )

    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
            
    
    # ======================
    # STYLE COMMON
    # ======================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if axis_label is not None:
        axis_name = axis_label
    else:
        axis_name = column_name.replace('_',' ')

    if orientation == "vertical":
        ax.set_xlabel(axis_name)
        ax.set_ylabel("Número de complejos residenciales")
    else:
        ax.set_ylabel(axis_name)
        ax.set_xlabel("Número de complejos residenciales")

    ax.set_title(
        title if title else f"Distribución de {column_name}",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20
    )

    plt.grid(axis='y' if orientation == "vertical" else 'x',
             linestyle='--', alpha=0.7, color='#BDC3C7')

    plt.tight_layout()
    plt.show()


def plot_variable_occurrences(df, column_names, title=None, dict_label=None):
    """
    Displays the frequency of occurrence (value 1) across multiple binary variables.

    This function calculates the absolute count and percentage of '1's for a list
    of binary columns, sorts them in descending order, and displays them in a 
    horizontal bar chart.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column_names : list of str
        List of binary column names to analyze and compare.
    title : str, optional
        The title of the generated plot. If None, a default title is used.
    dict_label : dict, optional
        Mapping between technical column names and human-readable labels.

    Returns:
    --------
    None
    """
    # Count frequency and percentage for bar plot
    counts = df[column_names].sum().sort_values(ascending=False)
    total_rows = len(df)
    percentages = (counts / total_rows) * 100

    # Mapping variable names to dict_label naming
    if dict_label is not None:
        counts.index = counts.index.map(dict_label)
        percentages.index = percentages.index.map(dict_label)

    # Horizontal Bar Plot
    _, ax = plt.subplots(figsize=(10, max(5, len(column_names) * 0.6)))
    sns.barplot(
        x=counts.values, 
        y=counts.index, 
        palette="Blues_r", 
        hue=counts.index,
        legend=False,
        ax=ax,
        edgecolor="white",
        linewidth=1.5
    )
    
    # Add label to the right of the bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        pct = percentages.values[i]
        ax.text(
            width + (counts.max() * 0.01),  
            p.get_y() + p.get_height() / 2.,
            f'{int(width):,} ({pct:.1f}%)',
            ha="left", 
            va="center", 
            fontsize=11, 
            fontweight='semibold',
            color='#2C3E50'
        )
    
    # Drop top and right snipes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Label formatting
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel("Número de complejos residenciales", fontsize=12)
    ax.set_ylabel("", fontsize=12)

    # Title
    plot_title = title if title else "Frecuencia de Ocurrencia por Variable (Valor = 1)"
    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=25, loc='left')
    
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7, color='#BDC3C7')
    plt.show()


def phi_correlation(X, thr=0.45, figsize=(10, 8)):
    """
    This function computes the Pearson correlation matrix (equivalent to Phi for
    binary variables), plots an heatmap, and prints a filtered list of variable pairs
    that exceed the specified absolute correlation threshold.

    Parameters:
    -----------
    X : pandas.DataFrame
        The input DataFrame containing variables to analyze.
    thr: float [optional]
        The absolute correlation threshold to filter pairs (Defaults to 0.45)
    figsize: tuple [optional]
        Width and height of the heatmap figure in inche. (Defaults to (10, 8)).

    Returns:
    ----------
        pandas.DataFrame:
            A sorted DataFrame containing the highly correlated pairs  with 
            columns ['Variable_1', 'Variable_2', 'Phi'].
    """
    # =========================
    # PHI CORRELATION MATRIX
    # =========================
    phi_matrix = X.corr(method="pearson")

    # Mask the upper triangle to avoid redundant data visualization
    mask = np.triu(np.ones_like(phi_matrix, dtype=bool))

    _, ax = plt.subplots(figsize=figsize)

    # Heatmap to display correlation matrix
    sns.heatmap(
        phi_matrix,
        mask=mask,
        cmap="RdBu_r",  
        vmin=-1,
        vmax=1,
        center=0,
        square=True,  
        linewidths=0.5,  
        cbar_kws={
            "shrink": 0.7,
            "label": "Phi Correlation Coefficient",
        },
        ax=ax,
    )

    # Title and label formatting
    plt.title(
        "Phi Correlation Matrix Between Independent Variables",
        fontsize=14,
        pad=20,
        fontweight="bold",
        color="black",
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        fontsize=10,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()
    # ================================
    # HIGH CORRELATION PAIRS FILTERING
    # ================================

    # Extract the upper triangle of the matrix (k=1 excludes the diagonal)
    upper = phi_matrix.where(
        np.triu(np.ones(phi_matrix.shape), k=1).astype(bool)
    )

    # Melt the matrix into a long-format DataFrame
    corr_pairs = upper.stack().reset_index()
    corr_pairs.columns = ["Variable_1", "Variable_2", "Phi"]

    # Filter pairs that exceed the absolute threshold and sort them
    high_corr = (
        corr_pairs[corr_pairs["Phi"].abs() > thr]
        .sort_values(by="Phi", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 60)
    print(f"Variables with correlation coefficient |Phi| > {thr}:")
    print("=" * 60)

    if high_corr.empty:
        print(f"No variable pairs found with a correlation higher than {thr}.")
    else:
        # Format the 'Phi' column to 3 decimal places for readability
        print(
            high_corr.to_string(
                formatters={"Phi": "{:,.3f}".format}, index=False
            )
        )

    print("=" * 60)
    
    return high_corr


def plot_boxplot(
    df,
    x,
    y,
    title=None,
    xlabel=None,
    ylabel=None,
    palette="Set2",
    show_outliers=False,
    figsize=(10, 6)
):
    """
    Displays a boxplot for a given categorical x and numerical y variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Categorical variable for the x-axis.
    y : str
        Numerical variable for the y-axis.
    title : str [optional]
        Plot title. If None, a default title is used.
    xlabel : str [optional]
        Label for the x-axis. If None, uses the column name.
    ylabel : str [optional]
        Label for the y-axis. If None, uses the column name.
    palette : str [optional]
        Color palette used for the boxplot. default "Set2"
    show_fliers : bool [optional]
        Whether to display outliers. default False
    figsize : tuple [optional]
        Figure size (width, height). default (10, 6)

    Returns:
    --------
    None
    """

    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        palette=palette,
        legend=False,
        showfliers=show_outliers,
    )

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Labels
    ax.set_xlabel(xlabel if xlabel else x, fontsize=11)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=11)

    # Title
    ax.set_title(
        title if title else f"Boxplot de {y} por {x}",
        fontsize=14,
        weight="bold",
        loc="left",
    )

    sns.despine()
    plt.tight_layout()
    plt.show()