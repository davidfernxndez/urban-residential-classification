"""
Data loading utilities for the Streamlit application.

This module provides cached functions to load datasets, model artifacts,
and SHAP-related data from disk. It ensures efficient data access using
Streamlit caching mechanisms and centralizes all file I/O operations for
the application.
"""

# ==============================================================================
# IMPORTS
# =============================================================================

import os
import pandas as pd
import joblib
import streamlit as st
from pyproj import Transformer
import shap

from app_src.appConfig import config

@st.cache_resource
def load_model() -> dict:
    """
    Load model artifact from pkl file and extract 

    Parameters
    --------
    None

    Returns
    -------
    model: sklearn estimator
        Machine learning estimator that supports sklearn API
    
    label_encoder: sklearn.preprocessing LabelEncoder object
        Encoder used during model training
    
    model_name: str
        Name of the model contained in the pkl artifact
    
    explainer: shap.Treeexplainer
        SHAP explainer object calculates from the model
    """ 
    model_path = config.XGBOOST_MODEL_PATH
    
    if not os.path.exists(model_path):
        st.error(f"Model PKL file not found in: {model_path}")
        return {}
    
    # Load artifact
    artifact = joblib.load(model_path)
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    model_name = artifact["name"]

    # Get SHAP explainer object
    explainer = shap.TreeExplainer(model.model)

    return model, label_encoder, explainer, model_name


@st.cache_data
def load_map_data() -> pd.DataFrame:
    """
    Load descriptive csv file and transform LAT, LON columns to correct
    coordinates for folium map.

    Parameters
    --------
    None

    Returns
    -------
    descriptive_df: pandas.DataFrame:
        Pandas dataframe containing descriptive and map spatial information
    """ 
    path = config.DESCRIPTIVE_DATA_PATH

    if not os.path.exists(path):
        st.error(f"Descriptive data file not found in: {path}")
        return pd.DataFrame()
    
    # Read dataframe
    descriptive_df = pd.read_csv(path)

    # Transform to correct coordinates for the folium map. In this dataset:
    # LON -> X_UTM
    # LAT -> Y_UTM
    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    descriptive_df["Y_UTM"], descriptive_df["X_UTM"] = transformer.transform(
        descriptive_df["LAT"].values,
        descriptive_df["LON"].values
    )    
    return descriptive_df

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Load dataset csv file.

    Parameters
    --------
    None

    Returns
    -------
    dataset_df: pandas.DataFrame:
        Pandas dataframe containing the complete dataset.
    """ 
    path = config.DATASET_PATH
    if not os.path.exists(path):
        st.error(f"Dataset file not found in: {path}")
        return pd.DataFrame()
    
    # Read dataframe
    dataset_df = pd.read_csv(path)

    return dataset_df

@st.cache_data
def load_shap_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the required files to get explainability with SHAP

    Parameters
    --------
    None

    Returns
    -------
    expected_shap_values_df: pandas.DataFrame:
        The baseline/expected SHAP values (explainer base values) for each class 
        across all outer folds.

    pred_info_df: pandas.DataFrame:
        Out-of-fold predictions for all samples across all folds, including 
        true targets, predicted classes, maximum probabilities, and fold indices.
    
    prob_df: pandas.DataFrame:
        Out-of-fold probabilites for all samples across all classes.
    
    shap_info_df: pandas.DataFrame:
        Out-of-fold shap values for all samples across all folds.
    """    
    expected_val_path = config.EXPECTED_VALUE_PATH
    pred_info_path = config.PRED_INFO_PATH
    prob_path = config.PROB_PATH
    shap_info_path = config.SHAP_INFO_PATH
    
    for p in [expected_val_path, pred_info_path, prob_path, shap_info_path]:
        if not os.path.exists(p):
            st.error(f"File not found in: {p}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
    # Read dataframes
    expected_value_df = pd.read_csv(expected_val_path)
    pred_info_df = pd.read_csv(pred_info_path)
    prob_df = pd.read_csv(prob_path)
    shap_info_df = pd.read_csv(shap_info_path)
    
    return expected_value_df, pred_info_df, prob_df, shap_info_df