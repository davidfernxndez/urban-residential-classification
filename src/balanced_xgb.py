"""
This module provides a custom wrapper around XGBoost estimator to seamlessly 
handle class imbalance within Scikit-Learn pipelines, specifically designed 
to avoid data leakage in Nested Cross-Validation.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ==============================================================================
# WRAPPER
# ==============================================================================

class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBClassifier that automatically handles class imbalance.

    This class computes and applies sample weights dynamically during the fitting 
    process based on the target distribution of the current training set. It is 
    specifically designed to prevent data leakage in Nested Cross-Validation and 
    hyperparameter tuning pipelines (e.g., GridSearchCV).

    Attributes:
        kwargs (dict): Dictionary storing all the hyperparameters for XGBClassifier.
        model (XGBClassifier or None): The underlying trained XGBoost model instance.
        classes_ (ndarray): The classes labels discovered during fitting.
    """
    def __init__(self, **kwargs):
        """     
        Initializes the wrapper and dynamically sets parameters as attributes.

        Args:
        **kwargs: Any keyword arguments passed directly to XGBClassifier 
        (e.g., n_estimators, max_depth, learning_rate, objective).
        """
        # Store all dynamic arguments to keep track of hyperparameters
        self.kwargs = kwargs
        self.model = None
            
        # Scikit-Learn requires parameters to be accessible as direct attributes
        # so GridSearchCV can inspect and modify them using 'model__parameter'
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        """
        Returns the hyperparameters of the estimator.

        Required by Scikit-Learn API for cloning and inspecting grid search parameters.

        Args:
            deep (bool): Unused. Included with a default value of True for 
                strict compatibility with the Scikit-Learn estimator API 
                (e.g., for cloning and pipeline inspection).
        Returns:
            dict: Parameter names mapped to their values.
        """
        # Tells Scikit-Learn which parameters are available in this model
        return self.kwargs

    def set_params(self, **params):
        """
        Sets the parameters of this estimator.

        Required by Scikit-Learn API to allow GridSearchCV to update hyperparameters 
        dynamically during the tuning process.
        Args:
            **params: Estimator parameters.
        Returns:
            self: The estimator instance.
        """
        # Update the dictionary and the direct attributes dynamically
        self.kwargs.update(params)

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y):
        """
        Fits the XGBoost classifier with dynamically calculated class weights.

        Sample weights are computed locally using only the 'y' labels provided in 
        this specific call, avoiding data leakage during cross-validation.

        Args:
            X (array-like): Training vectors of shape (n_samples, n_features).
            y (array-like): Target values (class labels) of shape (n_samples,).

        Returns:
            self: The fitted estimator instance.
        """
        # Calculate dynamic sample weights to handle class imbalance without leakage
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
        # Instantiate the native XGBClassifier passing all configured kwargs
        self.model = XGBClassifier(**self.kwargs)
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # Expose classes_ attribute required by Scikit-Learn
        self.classes_ = self.model.classes_

        return self

    def predict(self, X):
        """
        Predicts classes for X using the underlying trained XGBoost model.

        Args:
            X (array-like): Input samples of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels of shape (n_samples,).
        """

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities for X using the underlying trained XGBoost model.

        Args:
            X (array-like): Input samples of shape (n_samples, n_features).

        Returns:
            ndarray: The class probabilities of the input samples of shape 
                (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def __getattr__(self, name):
        """
        Dynamically forwards missing attribute or method accesses to the internal XGBoost model.

        This method is invoked only as a fallback when the requested attribute 
        does not exist within this wrapper class. It enables seamless access to 
        native XGBoost properties and methods (e.g., feature_importances_, 
        n_features_in_, get_booster()).

        Args:
            name (str): The name of the attribute or method being accessed.

        Returns:
            Any: The attribute or method from the underlying XGBClassifier instance.

        Raises:
            AttributeError: If the model is not trained yet or the attribute does 
                not exist in XGBClassifier.
        """        
        if self.model is not None:
            return getattr(self.model, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")