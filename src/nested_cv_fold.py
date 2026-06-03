"""
This module contains the functions necessary to generate fold partitions for Nested Cross Validation
and verify that these partitions are correct.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_folds(config):
    """
    Generate CSV files containing the fold assignments required for a
    Nested Cross Validation strategy.

    This function performs a stratified Nested Cross Validation split
    composed of:
        - An outer Stratified K-Fold split for model evaluation.
        - An inner Stratified K-Fold split for hyperparameter tuning.

    The generated fold assignments are stored as CSV files:
        - ``outer_folds.csv``:
            Contains the outer fold index assigned to each sample.
        - ``inner_fold_<outer_fold_idx>.csv``:
            Contains the inner fold assignments corresponding to the
            training subset of each outer fold.

    The function ensures that the identifier column defined in
    ``config.ID_VARIABLE`` contains unique values, since these IDs are
    used to map samples to folds.

    Parameters
    ----------
    config : object
        Configuration object containing:
        - DATASET_PATH : Path to input dataset CSV file
        - DATA_FOLDS_DIR : Directory containing precomputed fold CSV files
        - OUTER_SPLITS : Number of outer CV folds
        - INNER_SPLITS : Number of inner CV folds
        - OUTER_FOLD_FILENAME: Name of outer folds file.
        - INNER_FOLD_PREFIX: Prefix for inner fold files.
        - ID_VARIABLE : Unique identifier column name 
        - TARGET_VARIABLE : Target variable column name 
        - SEED: Seed for reproducibility
    
    Raises
    ------
    ValueError
        If the identifier column specified by ``ID_VARIABLE`` contains
        duplicated values.

    Notes
    -----
    - Stratification is applied in both outer and inner splits to
      preserve class distribution across folds.
    - Inner folds are generated independently for each outer training
      subset.
    - The generated CSV files store only the sample identifiers and
      their associated fold indices.
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH
    folds_dir = config.DATA_FOLDS_DIR
    
    # Nested Cross Validation splits size
    outer_splits = config.OUTER_SPLITS
    inner_splits = config.INNER_SPLITS

    # Output folds filenames
    outer_file_name = config.OUTER_FOLD_FILENAME
    inner_prefix = config.INNER_FOLD_PREFIX

    # Unique identifier (CC) and target variables 
    id_variable = config.ID_VARIABLE
    target_variable = config.TARGET_VARIABLE

    # Seed for reproducibility
    seed = config.SEED

    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================
    os.makedirs(folds_dir, exist_ok=True)
    
    # =========================================================
    # LOAD DATASET
    # =========================================================
    df = pd.read_csv(dataset_path)
    
    # =========================================================
    # CHECK UNIQUE IDS
    # =========================================================
    # It is verified that the ID_variable (CC column) is a unique 
    # identifier in the dataset. This is a necessary condition since
    #  these values ​​will be used to generate the folds.
    if not df[id_variable].is_unique:
        raise ValueError(
            f"Column '{id_variable}' contains duplicated values."
        )

    # =========================================================
    # Select features (X) and target (Y)
    # =========================================================
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # =========================================================
    # OUTER STRATIFIED CV
    # =========================================================
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed
    )

    # List to store Outer folds index
    outer_assignments = []

    # =========================================================
    # GENERATE OUTER + INNER FOLDS
    # =========================================================

    for outer_fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y)
    ):
        print(f"Generating outer fold {outer_fold_idx}\n")

        # The fold index is saved for instances in the Outer Test Set.
        # The Outer Train Set will therefore be the remaining samples that do not have this fold index
        outer_test_df = df.iloc[test_idx]

        # Store ID_variable (CC)
        for cc in outer_test_df[id_variable]:
            outer_assignments.append({
                id_variable: cc,
                "outer_fold_idx": outer_fold_idx
            })

        # =====================================================
        # GENERATE INNER FOLDS FOR CURRENT OUTER FOLD
        # =====================================================

        # Select Outer Train Set
        train_df = df.iloc[train_idx]

        # Select features and target of the Outer Train Set
        X_inner = train_df.drop(columns=[target_variable])
        y_inner = train_df[target_variable]

        # =====================================================
        # INNER STRATIFIED CV
        # =====================================================
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed
        )

        # List to store inner fold index
        inner_assignments = []

        for inner_fold_idx, (_, val_idx) in enumerate(
            inner_cv.split(X_inner, y_inner)
        ):
            print(f"Generating inner fold {inner_fold_idx} for outer fold {outer_fold_idx}")

            inner_val_df = train_df.iloc[val_idx]

            for cc in inner_val_df[id_variable]:
                inner_assignments.append({
                    id_variable: cc,
                    "inner_fold_idx": inner_fold_idx
                })

        # =====================================================
        # SAVE INNER FOLD CSV
        # =====================================================
        inner_fold_df = pd.DataFrame(inner_assignments)

        inner_fold_path = os.path.join(
            folds_dir,
            f"{inner_prefix}{outer_fold_idx}.csv"
        )

        inner_fold_df.to_csv(
            inner_fold_path,
            index=False
        )
        print(f"Save complete inner fold information for outer fold {outer_fold_idx} in inner_fold_{outer_fold_idx}.csv")
        print("-"*30+"\n\n")
    
    # =========================================================
    # SAVE OUTER FOLDS CSV
    # =========================================================
    outer_fold_df = pd.DataFrame(outer_assignments)

    outer_fold_path = os.path.join(
        folds_dir,
        outer_file_name
    )
    outer_fold_df.to_csv(
        outer_fold_path,
        index=False
    )
    
    print("Save complete outer fold information in outer_folds.csv")
    print("\nFold generation completed successfully.")


def validate_folds(
        config,
        outer_col="outer_fold_idx",
        inner_col="inner_fold_idx"
):
    """
    Validates correctness of Nested Cross Validation fold files.

    Parameters
    ----------
    config : object
        Configuration object containing:
        - DATASET_PATH : Path to input dataset CSV file
        - DATA_FOLDS_DIR : Directory containing precomputed fold CSV files
        - ID_VARIABLE : Unique identifier column name  
        - OUTER_FOLD_FILENAME: Name of outer folds file.
        - INNER_FOLD_PREFIX: Prefix for inner fold files.
    outer_col : str
        Column name for outer fold index.
    inner_col : str
        Column name for inner fold index.

    Returns
    -------
    None. Raises AssertionError if validation fails.
    """

    # =========================================================
    # LOAD CONFIGURATION VARIABLES
    # =========================================================
    # Path and folders 
    dataset_path = config.DATASET_PATH
    folds_dir = config.DATA_FOLDS_DIR
        
    # Unique identifier (CC)
    id_col = config.ID_VARIABLE

    # Output filenames
    outer_file_name = config.OUTER_FOLD_FILENAME
    inner_prefix = config.INNER_FOLD_PREFIX

    # =========================================================
    # LOAD DATASET
    # =========================================================
    dataset_df = pd.read_csv(dataset_path)

    # =========================================================
    # LOAD OUTER FOLDS
    # =========================================================

    outer_df = pd.read_csv(
        os.path.join(folds_dir, outer_file_name)
    )    

    assert id_col in dataset_df.columns, "Dataset missing id column"
    assert id_col in outer_df.columns, "Outer file missing id column"
    assert outer_col in outer_df.columns, "Outer file missing outer_fold_idx"


    # =========================================================
    # CHECK OUTER COVERAGE
    # =========================================================
    all_ids = set(dataset_df[id_col])
    outer_ids = set(outer_df[id_col])

    assert all_ids == outer_ids, "Outer folds do not cover full dataset"
    assert outer_df[id_col].is_unique, "Duplicate CC in outer folds"


    # =========================================================
    # VALIDATE EACH INNER FOLD FILE
    # =========================================================
    unique_outer_folds = sorted(outer_df[outer_col].unique())

    for outer_fold_idx in unique_outer_folds:
        # Load Inner csv file as dataframe
        inner_path = os.path.join(folds_dir, f"{inner_prefix}{outer_fold_idx}.csv")
        inner_df = pd.read_csv(inner_path)

        assert id_col in inner_df.columns, f"Missing {id_col} in inner fold {outer_fold_idx}"
        assert inner_col in inner_df.columns, f"Missing {inner_col} in inner fold {outer_fold_idx}"

        # Select Outer train set for this fold
        outer_train_ids = set(outer_df[outer_df[outer_col] != outer_fold_idx][id_col])
        inner_ids = set(inner_df[id_col])


        # Inner must be subset of outer train
        assert inner_ids == outer_train_ids, (
            f"Inner fold {outer_fold_idx} is not consistent with outer train split"
        )

        assert inner_df[id_col].is_unique, f"Duplicate CC in inner fold {outer_fold_idx}"


        # Inner fold distribution
        inner_counts = inner_df[inner_col].value_counts().sort_index()
        assert len(inner_counts) == len(inner_df[inner_col].unique()), \
            f"Inconsistent inner fold indexing in fold {outer_fold_idx}"


    print("\n✅ All Nested CV folds are valid.")