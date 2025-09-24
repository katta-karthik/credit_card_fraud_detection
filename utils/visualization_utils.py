"""
Visualization utilities for fraud detection project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_confusion_matrix(y_true, y_pred, model_name="Model", normalize=False, figsize=(8, 6)):
    """
    Plot confusion matrix with proper formatting.
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels  
    model_name: Name of the model
    normalize: Whether to normalize the matrix
    figsize: Figure size tuple
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'{model_name} - Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name} - Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """
    Plot ROC curve.
    
    Parameters:
    y_true: True labels
    y_pred_proba: Predicted probabilities
    model_name: Name of the model
    figsize: Figure size tuple
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    y_true: True labels
    y_pred_proba: Predicted probabilities
    model_name: Name of the model
    figsize: Figure size tuple
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = np.trapz(precision, recall)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names, importances, model_name="Model", top_n=20, figsize=(10, 8)):
    """
    Plot feature importance.
    
    Parameters:
    feature_names: List of feature names
    importances: Feature importance values
    model_name: Name of the model
    top_n: Number of top features to show
    figsize: Figure size tuple
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, title="Class Distribution", figsize=(8, 6)):
    """
    Plot class distribution.
    
    Parameters:
    y: Target variable
    title: Plot title
    figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    class_counts = pd.Series(y).value_counts().sort_index()
    colors = ['#2E8B57', '#DC143C']  # SeaGreen for Normal, Crimson for Fraud
    
    bars = plt.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts.values)*0.01, 
                f'{count:,}', ha='center', va='bottom')
    
    # Add percentage labels
    total = sum(class_counts.values)
    for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
        pct = count / total * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{pct:.2f}%', ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df, metric='f1_score', title=None, figsize=(12, 8)):
    """
    Plot comparison of multiple models.
    
    Parameters:
    results_df: DataFrame with model comparison results
    metric: Metric to compare
    title: Plot title
    figsize: Figure size tuple
    """
    if title is None:
        title = f'Model Comparison - {metric.replace("_", " ").title()}'
    
    plt.figure(figsize=figsize)
    
    # Sort by metric value
    sorted_df = results_df.sort_values(metric, ascending=True)
    
    bars = plt.barh(sorted_df['model_name'], sorted_df[metric])
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Model')
    plt.title(title)
    
    # Color bars based on performance
    colors = plt.cm.RdYlGn(sorted_df[metric] / sorted_df[metric].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (model, value) in enumerate(zip(sorted_df['model_name'], sorted_df[metric])):
        plt.text(value + 0.005, i, f'{value:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(train_scores, val_scores, train_sizes, title="Learning Curve", figsize=(10, 6)):
    """
    Plot learning curve.
    
    Parameters:
    train_scores: Training scores
    val_scores: Validation scores  
    train_sizes: Training set sizes
    title: Plot title
    figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()