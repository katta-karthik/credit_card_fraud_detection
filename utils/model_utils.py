"""
Model utilities for saving, loading, and evaluating machine learning models.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, filepath, use_joblib=True):
    """
    Save a trained model to disk.
    
    Parameters:
    model: Trained model object
    filepath: Path where to save the model
    use_joblib: Whether to use joblib (True) or pickle (False)
    """
    try:
        if use_joblib:
            joblib.dump(model, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filepath, use_joblib=True):
    """
    Load a trained model from disk.
    
    Parameters:
    filepath: Path to the saved model
    use_joblib: Whether to use joblib (True) or pickle (False)
    
    Returns:
    Loaded model object
    """
    try:
        if use_joblib:
            model = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation.
    
    Parameters:
    model: Trained model
    X_test: Test features
    y_test: Test labels
    model_name: Name of the model for display
    
    Returns:
    Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    avg_precision = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return results

def print_evaluation_results(results):
    """
    Print formatted evaluation results.
    
    Parameters:
    results: Dictionary from evaluate_model function
    """
    print(f"\n{'='*50}")
    print(f"{results['model_name']} Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"Specificity:  {results['specificity']:.4f}")
    print(f"F1-Score:     {results['f1_score']:.4f}")
    if results['roc_auc']:
        print(f"ROC AUC:      {results['roc_auc']:.4f}")
    if results['average_precision']:
        print(f"Avg Precision: {results['average_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"TN: {results['true_negatives']:,} | FP: {results['false_positives']:,}")
    print(f"FN: {results['false_negatives']:,} | TP: {results['true_positives']:,}")

def compare_models(results_list):
    """
    Compare multiple model evaluation results.
    
    Parameters:
    results_list: List of result dictionaries from evaluate_model
    
    Returns:
    DataFrame with comparison metrics
    """
    comparison_data = []
    
    for results in results_list:
        comparison_data.append({
            'Model': results['model_name'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'ROC AUC': results['roc_auc'],
            'Avg Precision': results['average_precision']
        })
    
    df = pd.DataFrame(comparison_data)
    return df.round(4)