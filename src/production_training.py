"""
Method to train final model on all available data
"""
# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pandas as pd
import joblib
import time
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ==============================================================================
# TRAIN PRODUCTION MODEL ON ALL AVAILABLE DATA
# ==============================================================================

def train_final_model(config, model, param_grid, model_name, gridsearch_metric="f1_macro", use_balanced_weights=False):
    """    
    Train a final production-ready machine learning model using
    predefined cross-validation folds and hyperparameter optimization.

    The function performs a Grid Search cross-validation using the
    fold assignments stored in the outer folds CSV file. Once the
    best hyperparameter configuration is found, the model is
    automatically retrained on the full dataset using the optimal
    configuration.
    
    The trained model and the LabelEncoder used are stored in a single pkl file.

    Parameters
    ----------
    config : object
        Configuration object containing:
        - DATASET_PATH : Path to input dataset CSV file
        - DATA_FOLDS_DIR : Directory containing precomputed fold CSV files
        - OUTPUT_DIR : Directory where results will be saved
        - OUTER_FOLD_FILENAME: Name of outer folds file.
        - OUTER_SPLTS: Number of splits for CV.
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name 
    
    model : sklearn estimator
        Machine learning estimator that supports sklearn API
    
    param_grid : dict
        Dictionary defining hyperparameter search space for the estimator.
    
    model_name : str 
        Name of the model used for logging and model storage.
    
    gridsearch_metric: str [optional, default="f1_macro"]
        Metric to optimize in the search of hyperparameters
    
    use_balanced_weights: bool [optional, default="false]
        Indicates if balanced sample weights are computed from the training
        data of each fold.

    Returns
    -------
    final_model : sklearn estimator
        Fully trained production model fitted on the entire dataset
        using the best hyperparameter configuration found during
        cross-validation. 
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH
    folds_dir = config.DATA_FOLDS_DIR
    output_dir = config.OUTPUT_DIR
    
    # Cross Validation folds filename
    folds_file_name = config.OUTER_FOLD_FILENAME
    number_of_folds = config.OUTER_SPLITS

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
    #  models like XGBoost can work correctly
    label_encoder = LabelEncoder()
    dataset_df[target_variable] = label_encoder.fit_transform(dataset_df[target_variable])

    # =========================================================
    # LOAD OUTER FOLDS
    # =========================================================
    outer_folds_df = pd.read_csv(
        os.path.join(folds_dir, folds_file_name)
    )

    # =========================================================
    # FEATURES / TARGET
    # =========================================================
    X = dataset_df.drop(columns=[target_variable])
    y = dataset_df[target_variable]
    
    # Reset indexes for compatibility with sklearn gridsearch
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # =====================================================
    # MAP CC (ID variable) -> LOCAL TRAIN INDEX
    # =====================================================
        
    # Map CC identifier to the indices of
    # the training set since sklearn will use the indices
    # of this dataset
    cc_to_local_idx = {
        cc: idx
        for idx, cc in enumerate(X[id_variable])
    }

    print("\n" + "=" * 70)
    print("TRAIN MODEL FOR PRODUCTION ON ALL AVAILABLE DATA")
    print("=" * 70)
    print(f"Model Name      : {model_name}")
    print(f"CV Folds       : {number_of_folds}")
    print("\nHyperparameter Grid:")
    for param_name, param_values in param_grid.items():
        print(f"{param_name:<25}: {param_values}")
    print("=" * 70)

    # =========================================================
    # BUILD CUSTOM CV SPLITS
    # =========================================================
    cv_splits = []

    unique_folds = sorted(outer_folds_df["outer_fold_idx"].unique())

    for fold_idx in unique_folds:

        val_ids = set(
            outer_folds_df[
                outer_folds_df["outer_fold_idx"]
                == fold_idx
            ][id_variable]
        )
  
        train_ids = set(
            outer_folds_df[
                outer_folds_df["outer_fold_idx"]
                != fold_idx
            ][id_variable]
        )
   
        # Map CC (ID variable) to local index in Train set
        val_idx = np.array([
            cc_to_local_idx[cc]
            for cc in val_ids
        ])
        train_idx = np.array([
            cc_to_local_idx[cc]
            for cc in train_ids
        ])

        cv_splits.append(
            (train_idx, val_idx)
        )
    
    # Remove ID column from training features
    X = X.drop(columns=[id_variable])

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================
    # Used for algorithms that dont't support class_weight
    # internal parameter
    fit_params = {}        
    if use_balanced_weights:
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y
        )
        fit_params["sample_weight"] = sample_weights
        print("Using balanced sample weights")
    
    start_time = time.time()
    # =========================================================
    # GRID SEARCH
    # =========================================================
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        scoring=gridsearch_metric,
        cv=cv_splits,
        n_jobs=-1,
        refit=True,
        verbose=1
    )   

    # =========================================================
    # TRAIN GRID SEARCH
    # =========================================================
    grid_search.fit(X, y, **fit_params)

    elapsed_time = time.time() - start_time
    print("\n" + "-" * 80)
    print(f"Total time      : {elapsed_time:.2f} seconds")
    print("-" * 80)

    # =========================================================
    # GET FINAL TRAINED MODEL
    # =========================================================
    final_model = grid_search.best_estimator_

    print(f"Best params for {model_name}:")
    print(grid_search.best_params_) 

    # =========================================================
    # SAVE MODEL
    # =========================================================
    
    # Create output folder for production models
    output_custom_dir = os.path.join(
        output_dir,"models"
    )
    os.makedirs(output_custom_dir, exist_ok=True)

    # Save model and label encoder in one pkl file
    artifacts_to_save = {
        "model": final_model,          
        "label_encoder": label_encoder, 
        "name": model_name
    }
    pkl_path = os.path.join(output_custom_dir, f"{model_name}_model.pkl")
    joblib.dump(artifacts_to_save, pkl_path)

    print("\n" + "-" * 80)
    print(f"Model and Encoder saved in directory: {output_custom_dir}")
    print("-" * 80)
    
    return final_model