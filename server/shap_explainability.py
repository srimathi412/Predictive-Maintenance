import shap
import numpy as np
import tensorflow as tf

# Fix for SHAP with TF 2.x if needed, generally shap.DeepExplainer works with Keras models
# Ensure strict compatibility prevents 'Tensor' object has no attribute 'numpy' errors in some versions

def explain_model_prediction(model, X_sample, background_data):
    """
    Generates SHAP values for a single instance X_sample using background_data.
    X_sample: (1, time_steps, features)
    background_data: (N, time_steps, features) - e.g. 50 random samples from train set
    """
    # DeepExplainer expects the model and background data
    # Note: DeepExplainer can be slow. GradientExplainer is an alternative.
    explainer = shap.DeepExplainer(model, background_data)
    
    # shap_values is a list for multi-output, array for single output
    shap_values = explainer.shap_values(X_sample)
    
    # For a regression model, it usually returns a list with one array [ (1, time_steps, features) ]
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    return shap_values

def get_top_contributors(shap_values, feature_names, top_n=3):
    """
    Identifies the top contributing sensors based on mean absolute SHAP probability.
    shap_values: (1, time_steps, features)
    """
    # 1. Sum absolute SHAP values across time steps to get importance per feature
    # Shape: (features,)
    feature_importance_scores = np.sum(np.abs(shap_values[0]), axis=0)
    
    # 2. Map to feature names
    importance_dict = dict(zip(feature_names, feature_importance_scores))
    
    # 3. Sort and get top N
    sorted_features = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_features[:top_n]


