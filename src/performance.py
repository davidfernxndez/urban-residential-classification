"""
This module implements the functions used to perform performance evaluation experiments using Nested Cross Validation.
"""
# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

def performance_experiment(config, pipeline, param_grid, experiment_name):
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
        - NESTED_CV_RESULTS_DIR : Directory where results will be saved
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
    
    Returns
    -------
    macro_results_df : pandas.DataFrame
        DataFrame containing aggregated performance metrics per outer fold,
        including accuracy, macro F1-score, precision, and recall.
    
    class_results_df : pandas.DataFrame
        DataFrame containing per-class performance metrics for each outer fold,
        including precision, recall, F1-score, and support.
    """
    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH
    folds_dir = config.DATA_FOLDS_DIR
    results_dir = config.NESTED_CV_RESULTS_DIR
    
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
    # LOAD OUTER FOLDS
    # =========================================================
    outer_folds_df = pd.read_csv(
        os.path.join(folds_dir, outer_file_name)
    )
    
    # List to store results 
    macro_results = []
    class_results = []
    
    # =========================================================
    # OUTER LOOP
    # =========================================================
    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print("=" * 60)
    for outer_fold_idx in range(outer_splits):
        print(f"\nOUTER FOLD {outer_fold_idx}")
        
        # =====================================================
        # OUTER TEST IDS
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
        # OUTER TRAIN IDS
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

        print("- Outer Train and Test Set correctly selected.")
        
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
        
        # It is necessary to map the identifier CC to the indices of
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

        print("- Inner sets for Inner Cross Validation correctly selected.")
        
        # =====================================================
        # GRID SEARCH
        # =====================================================
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=custom_inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            verbose=1
        )
        print("- Grid Search initiated")
        
        # =====================================================
        # TRAIN
        # =====================================================

        grid_search.fit(X_train, y_train)

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
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        
        # Store global results
        macro_results.append({
            "outer_fold": outer_fold_idx,
            "accuracy": accuracy,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
            "best_params": str(
                grid_search.best_params_
            )
        })
        
        # =====================================================
        # METRICS PER CLASS
        # =====================================================
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
        for class_label in sorted(y_test.unique()):

            class_results.append({
                "outer_fold": outer_fold_idx,
                "class": class_label,
                "precision": report[str(class_label)]["precision"],
                "recall": report[str(class_label)]["recall"],
                "f1": report[str(class_label)]["f1-score"],
                "support": report[str(class_label)]["support"]
            })
        
    # =========================================================
    # RETURN & SAVE RESULTS
    # =========================================================

    macro_results_df = pd.DataFrame(macro_results)
    class_results_df = pd.DataFrame(class_results)
    
    
    macro_results_df.to_csv(
        results_dir / f"{experiment_name}_macro_results.csv",
        index=False
    )
    class_results_df.to_csv(
        results_dir / f"{experiment_name}_class_results.csv",
        index=False
    )


    return macro_results_df, class_results_df
