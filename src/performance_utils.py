"""
Performance evaluation and analysis utilities.

This module implements a Nested Cross-Validation framework for the
robust evaluation of machine learning pipelines. Additionally, it
provides functions for aggregating, analyzing, and visualizing
performance metrics, enabling a comprehensive comparison of model
results.
"""
# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==============================================================================
# PERFORMANCE EXPERIMENT USING NESTED CROSS VALIDATION
# ==============================================================================

def performance_experiment(
        config,
        pipeline,
        param_grid,
        experiment_name,
        gridsearch_metric = "f1_macro",
        use_balanced_weights=False
    ):
    """    
    This function performs model evaluation using Nested Cross Validation:
    - Outer loop: estimates generalization performance.
    - Inner loop: performs hyperparameter tuning.
    
    The experiment uses precomputed and fixed fold splits stored in CSV
    files to ensure full reproducibility.
    
    Parameters
    ----------
    config : object
        Configuration object containing:
        - DATASET_PATH : Path to input dataset CSV file
        - DATA_FOLDS_DIR : Directory containing precomputed fold CSV files
        - OUTPUT_DIR : Directory where results will be saved
        - OUTER_SPLITS : Number of outer CV folds
        - INNER_SPLITS : Number of inner CV folds
        - OUTER_FOLD_FILENAME: Name of outer folds file.
        - INNER_FOLD_PREFIX: Prefix for inner fold files.
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name 
    
    pipeline : sklearn.pipeline.Pipeline
        Machine learning pipeline that includes preprocessing steps
        and the final estimator.
    
    param_grid : dict
        Dictionary defining hyperparameter search space for the estimator
        inside the pipeline. Used in GridSearchCV during the inner loop.
    
    experiment_name : str
        Name of the experiment used to label and store output files.
    
    gridsearch_metric: str [optional, default="f1_macro"]
        Metric to optimize in the search of hyperparameters
    
    use_balanced_weights: bool [optional, default="false]
            Indicates if balanced sample weights are computed from the training
            data of each fold.

    Returns
    -------
    global_results_df : pandas.DataFrame
        DataFrame containing aggregated performance metrics per outer fold,
        including macro F1-score, MCC and accuracy. It also includes best
        hyperparameters for each fold.
    
    cm_df : pandas.DataFrame
        DataFrame containing accumulated confusion matrix.
    
    class_report_df : pandas.DataFrame
        DataFrame containing accumulated classification report:
        f1-score, precision and recall values for each class.
    """
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
    inner_prefix = config.INNER_FOLD_PREFIX

    # Unique identifier (CC) and target variables 
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE
    
    # =========================================================
    # LOAD DATASET
    # =========================================================
    dataset_df = pd.read_csv(dataset_path)

    # =========================================================
    # LABEL ENCODING
    # =========================================================

    # Encode the K classes to the range of values ​​[0,..,K-1] so that
    #  models like XGBoost can function correctly
    label_encoder = LabelEncoder()
    dataset_df[target_variable] = label_encoder.fit_transform(dataset_df[target_variable])

    # =========================================================
    # LOAD OUTER FOLDS
    # =========================================================
    outer_folds_df = pd.read_csv(
        os.path.join(folds_dir, outer_file_name)
    )
    
    # List to store global results 
    global_results = []

    # Lists to store labels and predictions for building
    # accumulated confusion matrix
    all_y_test = []
    all_y_pred = []
    
    print("\n" + "=" * 70)
    print("NESTED CROSS-VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Experiment Name      : {experiment_name}")
    print(f"Outer CV Folds       : {outer_splits}")
    print(f"Inner CV Folds       : {inner_splits}")
    print("\nHyperparameter Grid:")
    for param_name, param_values in param_grid.items():
        print(f"{param_name:<25}: {param_values}")
    print("=" * 70)

    # =========================================================
    # OUTER LOOP
    # =========================================================
    start_time = time.time()
    for outer_fold_idx in range(outer_splits):

        print("\n" + "-" * 80)
        print(f"OUTER FOLD [{outer_fold_idx + 1}/{outer_splits}]")
        print("-" * 80)
        
        # =====================================================
        # OUTER TEST SET IDS
        # =====================================================
        # The samples used for the Outer Test Set are those 
        # with the fold idx of the current outer_fold_idx
        outer_test_ids = set(
            outer_folds_df[
                outer_folds_df["outer_fold_idx"]
                == outer_fold_idx
            ][id_variable]
        )

        # =====================================================
        # OUTER TRAIN SET IDS
        # =====================================================
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

        # Reset indexes for compatibility with sklearn gridsearch
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # =====================================================
        # TRAIN / TEST FEATURES & TARGET
        # =====================================================

        X_train = train_df.drop(
            columns=[id_variable, target_variable]
        )
        y_train = train_df[target_variable]

        X_test = test_df.drop(
            columns=[id_variable, target_variable]
        )
        y_test = test_df[target_variable]

        print(
            f"Train samples: {len(X_train):5d} | "
            f"Test samples: {len(X_test):5d}"
        )
        
        # =====================================================
        # CLASS WEIGHTS
        # =====================================================
        # Used for algorithms that dont`t support class_weight
        # internal parameter`
        fit_params = {}        
        if use_balanced_weights:
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
            fit_params["model__sample_weight"] = sample_weights
            print("Using balanced sample weights")

        # =====================================================
        # LOAD INNER FOLDS
        # =====================================================
        inner_folds_df = pd.read_csv(
            os.path.join(
                folds_dir,
                f"{inner_prefix}{outer_fold_idx}.csv"
            )
        )

        # =====================================================
        # MAP CC (ID variable) -> LOCAL TRAIN INDEX
        # =====================================================
        
        # Map CC identifier to the indices of
        # the outer CV training set since sklearn will use the indices
        # of this dataset
        cc_to_local_idx = {
            cc: idx
            for idx, cc in enumerate(train_df[id_variable])
        }

        # =====================================================
        # BUILD CUSTOM INNER CV
        # =====================================================
        
        # List to store tuples of train/val inner sets
        custom_inner_cv = []

        for inner_fold_idx in range(inner_splits):

            # -------------------------------------------------
            # INNER VALIDATION SET IDS
            # -------------------------------------------------

            val_ids = set(
                inner_folds_df[
                    inner_folds_df["inner_fold_idx"]
                    == inner_fold_idx
                ][id_variable]
            )

            # -------------------------------------------------
            # INNER TRAIN SET IDS
            # -------------------------------------------------

            inner_train_ids = set(
                inner_folds_df[
                    inner_folds_df["inner_fold_idx"]
                    != inner_fold_idx
                ][id_variable]
            )

            # Map CC (ID variable) to local index in Outer Train set
            val_idx = np.array([
                cc_to_local_idx[cc]
                for cc in val_ids
            ])
            train_idx = np.array([
                cc_to_local_idx[cc]
                for cc in inner_train_ids
            ])

            custom_inner_cv.append(
                (train_idx, val_idx)
            )

        print(
            f"Inner CV splits successfully loaded "
            f"({inner_splits} folds)"
        )
        
        # =====================================================
        # GRID SEARCH
        # =====================================================
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=custom_inner_cv,
            scoring=gridsearch_metric,
            n_jobs=-1,
            refit=True,
            verbose=1
        )
        print(
            f"Starting GridSearchCV "
            f"(metric='{gridsearch_metric}') ..."
        )
        
        # =====================================================
        # TRAIN
        # =====================================================

        grid_search.fit(X_train, y_train, **fit_params)

        # =====================================================
        # SELECT BEST MODEL
        # =====================================================

        best_model = grid_search.best_estimator_

        # =====================================================
        # OUTER TEST PREDICTION
        # =====================================================

        y_pred = best_model.predict(X_test)

        # =====================================================
        # METRICS MACRO
        # =====================================================
        f1 = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store global results
        global_results.append({
            "model_name": experiment_name,
            "outer_fold": outer_fold_idx,
            "f1_macro": f1,
            "MCC": mcc,
            "accuracy": accuracy,
            "best_params": str(
                grid_search.best_params_
            )
        })
        
        # Store label and prediction (original values)
        y_test_orig = label_encoder.inverse_transform(y_test)
        y_pred_orig = label_encoder.inverse_transform(y_pred)
        all_y_test.extend(y_test_orig)
        all_y_pred.extend(y_pred_orig)

    elapsed_time = time.time() - start_time
    print("\n" + "-" * 80)
    print(f"Total time      : {elapsed_time:.2f} seconds")
    print("-" * 80)
    # =========================================================
    # FORMAT RESULTS
    # =========================================================

    # Store global results in a dataframe
    global_results_df = pd.DataFrame(global_results)

    # Create confusion matrix and save as a dataframe
    class_names = np.unique(all_y_test)
    cm_df = pd.DataFrame(
        confusion_matrix(all_y_test, all_y_pred),
        index = class_names,
        columns= class_names
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    # Classification report
    class_report = classification_report(
        all_y_test,
        all_y_pred,
        output_dict=True
    )
    # Save as a dataframe
    class_report_df = (
        pd.DataFrame(class_report)
        .transpose()
        [["precision", "recall", "f1-score"]]
    )
    class_report_df = class_report_df.drop(
        index=["accuracy", "macro avg", "weighted avg"],
        errors="ignore"
    )
    class_report_df["model_name"] = experiment_name
    class_report_df = class_report_df.reset_index().rename(columns={"index": "class"})

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    
    # Create output folder for the current nested cv configuration
    output_custom_dir = os.path.join(
        output_dir,
        f"outer_folds_{outer_splits}_inner_folds_{inner_splits}",
        f"{experiment_name}"
    )
    os.makedirs(output_custom_dir, exist_ok=True)

    print("\nSaving results...")
    print(f"Output directory: {output_custom_dir}")

    global_results_df.to_csv(
        os.path.join(output_custom_dir, f"{experiment_name}_global_results.csv")
    )
    cm_df.to_csv(
        os.path.join(output_custom_dir, f"{experiment_name}_confusion_matrix.csv")
    )
    class_report_df.to_csv(
        os.path.join(output_custom_dir, f"{experiment_name}_classification_report.csv"),
        index=False
    )

    print("=" * 80)

    return global_results_df, cm_df, class_report_df


# ==============================================================================
# ANALYSIS OF RESULTS
# ==============================================================================

def visualize_model_performance(df, y_lim = [0.8, 1]):
    """
    Generate performance summaries and visualizations for models evaluated
    using Nested Cross-Validation.

    This function computes the mean and standard deviation of the evaluation
    metrics for each model, displays a formatted summary table, and generates:
    - A bar plot showing the average F1-macro and MCC scores for each model.
    - A box plot showing the distribution of F1-macro and MCC scores across
    cross-validation folds for each model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the evaluation results.
        Expected columns: 
        - 'model_name': model identifiers
        - 'outer_fold': Outer fold number
        - 'f1_macro': f1-macro value per outer fold
        - 'MCC': MCC value per outer fold
        - 'accuracy': accuracy value per outer fold

    y_lim : list of float [optional, default=[0.8,1] ] 
        Lower and upper limits of the y-axis for the generated plots.

    Returns
    -------
    pandas.DataFrame
        Summary table containing the mean ± standard deviation of the
        evaluated metrics for each model.
    """

    # =========================================================
    # PRINT TABLE RESULTS (MEAN+-STD)
    # =========================================================
    metrics = ['f1_macro', 'MCC', 'accuracy']
    
    # Format model name
    df['model_name'] = df['model_name'].str.replace('_',' ')
    results_df = df.groupby('model_name')[metrics].agg(['mean', 'std'])
    
    table_df = pd.DataFrame(index=results_df.index)
    for metric in metrics:
        table_df[metric] = results_df[metric].apply(
            lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1
        )
    
    table_df = table_df.reset_index()
    table_df.columns = ['model_name', 'F1-macro', 'MCC', 'Accuracy']
    table_df = table_df.sort_values(by="F1-macro")

    display(
        table_df.style
        .hide(axis="index")
        .set_caption("Rendimiento en Nested Cross Validation (Mean ± Standard Deviation)")
        .set_properties(**{
            'text-align': 'center',
            'border': '1px solid black'
        })
    )

    # =========================================================
    # FORMAT DATAFRAME TO PLOT RESULTS
    # =========================================================
    df_long = df[['model_name', 'f1_macro', 'MCC']].melt(
        id_vars=['model_name'], var_name='metric', value_name='value'
    )
    df_long['metric'] = df_long['metric'].replace({'f1_macro': 'F1-macro', 'MCC': 'MCC'})
    
    # Display in ascending order according to mean F1-macro
    order = table_df['model_name'].tolist()

    # ---------------------------------------------------------
    # BAR PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6)) 
    ax_1 = sns.barplot(
        x='model_name', 
        y='value', 
        hue='metric', 
        data=df_long,
        order = order,
        palette="muted",
        errorbar=None,
        edgecolor="black",
        linewidth = 1.2   
    )
    plt.title('Rendimiento promedio de los modelos\nen Nested Cross Validation', fontsize=14, weight='bold', color='black')
    plt.ylabel('Score', fontsize=12, labelpad=10, color='black')
    plt.xlabel('Modelo', fontsize=12, labelpad=10, color='black')
    plt.ylim(y_lim[0], y_lim[1])
    ax_1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
    ax_1.set_yticks(np.arange(y_lim[0], y_lim[1] + 0.01, 0.02))

    # Plot legend in the bottom
    sns.move_legend(
        ax_1, "upper center",         
        bbox_to_anchor=(0.5, -0.25), 
        ncol=2,                     
        frameon=True, 
        title="Métrica"
    )

    # Drop spines
    ax_1.spines['top'].set_visible(False)
    ax_1.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()                  
    
    # ---------------------------------------------------------
    # BOX PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))  
    ax_2 = sns.boxplot(
        x='model_name', 
        y='value', 
        hue='metric', 
        data=df_long,
        order = order,
        palette="muted",
        linewidth = 1.2   
    )

    plt.title('Distribución del rendimiento de los modelos\nen Nested Cross Validation', fontsize=14, pad=15, weight='bold')
    plt.ylabel('Score', fontsize=12, labelpad=10, color='black')
    plt.xlabel('Modelo', fontsize=12, labelpad=10, color='black')
    plt.ylim(y_lim[0], y_lim[1])
    ax_2.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
    ax_2.set_yticks(np.arange(y_lim[0], y_lim[1] + 0.01, 0.02))

    # Plot legend in the bottom
    sns.move_legend(
        ax_2, "upper center",         
        bbox_to_anchor=(0.5, -0.25), 
        ncol=2,                     
        frameon=True, 
        title="Métrica"
    )

    # Drop spines
    ax_2.spines['top'].set_visible(False)
    ax_2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show() 
    
    return table_df


def plot_confusion_matrix(cm_df, label_map, model_name):
    """
    Plot a normalized confusion matrix as a heatmap.

    The confusion matrix is normalized row-wise so that each row sums
    to 100%, representing the percentage distribution of predicted
    classes for each actual class. 

    Parameters
    ----------
    cm_df : pandas.DataFrame
        Confusion matrix where rows correspond to actual classes and
        columns correspond to predicted classes.

    label_map : dict
        Dictionary mapping class identifiers to readable class names. 

    model_name : str
        Name of the classification model. Displayed in the plot title.

    Returns
    -------
    None
    """
    # Rename label index with mapping dictionary
    cm_df = cm_df.rename(index=label_map, columns=label_map)

    # Normalize confusion matrix
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

    # Plot confusion matrix with a Heat Map
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": "Percentage (%)",
            "shrink": 0.95
        },
        annot_kws={"size": 12}
    )

    ax.set_title(
        f"Matriz de confusión normalizada\nModelo: {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Class", fontsize=12, labelpad=10)

    # Tick styling
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


def compare_class_report(df, model_A, model_B, label_map, y_lim=[0.5, 1]):
    """
    Compare the class-wise performance of two models:
    - model_A represents a transparent (interpretable) model
    - model_B represents a non-interpretable black box model.

    This function generates two visual analyses:
    1. A grouped bar plot showing the F1-score per class for two models.
    2. A heatmap showing the performance gain (Model B - Model A) in precision
       and recall for each class.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing classification metrics per class and model.
        Expected columns:
        - 'class': class labels
        - 'model_name': model identifiers
        - 'precision': precision values per class
        - 'recall': recall values per class
        - 'f1-score': F1-score values per class

    model_A : str
        Name of the model used as reference for comparison.

    model_B : str
        Name of the second model whose performance is compared against model_A.

    label_map : dict
        Dictionary used to map raw class labels to readable names.

    y_lim : list of float [optional, default=[0.5, 1] ]
        Y-axis limits for the F1-score bar plot, defined as [min, max].

    Returns
    -------
    None
    """
    # Work with a copy dataframe
    class_report_df = df.copy()
    # Format model names to replace '_' by ' '
    class_report_df['model_name'] = class_report_df['model_name'].str.replace('_',' ')
    model_A = model_A.replace('_', ' ')
    model_B = model_B.replace('_', ' ')

    # Label mapping to use redeable names
    class_report_df["class"] = class_report_df["class"].astype(int).replace(label_map)
    
    # ---------------------------------------------------------
    # F1-SCORE BAR PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6)) 
    ax_1 = sns.barplot(
        x='class', 
        y='f1-score', 
        hue='model_name', 
        data=class_report_df,
        palette="muted",
        edgecolor="black",
        linewidth = 1.2   
    )
    plt.title('F1-Score por grado de cerramiento y modelo', fontsize=14, weight='bold', color='black')
    plt.ylabel('F1-Score', fontsize=12, labelpad=10, color='black')
    plt.xlabel('Grado de cerramiento', fontsize=12, labelpad=10, color='black')
    plt.ylim(y_lim[0], y_lim[1])
    ax_1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
    ax_1.set_yticks(np.arange(y_lim[0], y_lim[1] + 0.01, 0.05))

    # Plot legend in the bottom
    sns.move_legend(
        ax_1, "upper center",         
        bbox_to_anchor=(0.5, -0.25), 
        ncol=2,                     
        frameon=True, 
        title="Modelo"
    )

    # Drop spines
    ax_1.spines['top'].set_visible(False)
    ax_1.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()  


    # ---------------------------------------------------------
    # PRECISION AND RECALL GAIN
    # ---------------------------------------------------------
    df_pivot = class_report_df.pivot_table(
        index='class',
        columns='model_name',
        values=['precision', 'recall']
    )

    # Calculate the gain of model B with respect to model A
    gain_precision = df_pivot['precision'][model_B] - df_pivot['precision'][model_A]
    gain_recall = df_pivot['recall'][model_B] - df_pivot['recall'][model_A]

    # Store results in dataframe to plot in heatmap
    df_gain = pd.DataFrame({
        'Precision': gain_precision,
        'Recall': gain_recall
    })

    # Order dataframe according label class
    order = [label_map[k] for k in sorted(label_map.keys(), key=int)]
    df_gain = df_gain.reindex(order)

    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(
        df_gain,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 12, "weight": "bold"}
    )

    plt.title(
        f"Ganancia del modelo caja negra ({model_B}) respecto\nal modelo transparente ({model_A})",
        fontsize=14,
        weight='bold',
        pad=12
    )
    plt.xlabel("Métrica", fontsize=12, labelpad=10)
    plt.ylabel("Grado de cerramiento", fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()

