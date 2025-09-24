# Fraud Detection Project Configuration
"""
Configuration file for the fraud detection project.
Contains all constants, file paths, and settings used across notebooks.
"""

import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
INDIVIDUAL_MODELS_DIR = os.path.join(MODELS_DIR, 'individual')
ENSEMBLE_MODELS_DIR = os.path.join(MODELS_DIR, 'ensemble')

REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Data File Paths
KAGGLE_DATASET_PATH = os.path.join(RAW_DATA_DIR, 'PS_20174392719_1491204439457_log.csv')
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_fraud_data.csv')
SAMPLE_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'sample_fraud_data.csv')

# Model File Paths
MODEL_PATHS = {
    'knn': os.path.join(MODELS_DIR, 'best_knn.pkl'),
    'svm': os.path.join(MODELS_DIR, 'best_svm.pkl'),
    'decision_tree': os.path.join(MODELS_DIR, 'best_dt.pkl'),
    'random_forest': os.path.join(MODELS_DIR, 'best_rf.pkl'),
    'voting_classifier': os.path.join(ENSEMBLE_MODELS_DIR, 'voting_classifier.pkl'),
    'stacking_classifier': os.path.join(ENSEMBLE_MODELS_DIR, 'stacking_classifier.pkl'),
    'ada_boost': os.path.join(ENSEMBLE_MODELS_DIR, 'ada_boost.pkl'),
    'gradient_boost': os.path.join(ENSEMBLE_MODELS_DIR, 'gradient_boost.pkl')
}

# Report File Paths
REPORT_PATHS = {
    'individual_models_comparison': os.path.join(REPORTS_DIR, 'individual_models_comparison.csv'),
    'ensemble_models_comparison': os.path.join(REPORTS_DIR, 'ensemble_models_comparison.csv'),
    'final_model_comparison': os.path.join(REPORTS_DIR, 'model_comparison_final.csv'),
    'deployment_checklist': os.path.join(REPORTS_DIR, 'deployment_checklist.md'),
    'final_evaluation_report': os.path.join(REPORTS_DIR, 'final_evaluation_report.md')
}

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Sample Data Parameters
SAMPLE_SIZE = 10000
FRAUD_RATE = 0.001

# Model Hyperparameters
MODEL_PARAMS = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    },
    'decision_tree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }
}

# Ensemble Parameters
ENSEMBLE_PARAMS = {
    'voting': {
        'voting': 'soft'
    },
    'stacking': {
        'cv': 5,
        'stack_method': 'predict_proba'
    },
    'ada_boost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'gradient_boost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'roc_auc', 'average_precision', 'specificity'
]

# Visualization Settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12
COLOR_PALETTE = 'husl'

# Business Impact Parameters
COST_PARAMETERS = {
    'cost_per_fraud': 1000,  # Average cost of a fraudulent transaction
    'investigation_cost': 50,  # Cost to investigate a flagged transaction
    'false_positive_rate_tolerance': 0.05,  # Maximum acceptable false positive rate
    'minimum_recall': 0.80,  # Minimum required fraud detection rate
}

# Feature Engineering Settings
FEATURE_ENGINEERING = {
    'create_log_features': True,
    'create_ratio_features': True,
    'create_binary_flags': True,
    'handle_categorical': True,
    'scaling_method': 'standard'
}

# Data Quality Thresholds
DATA_QUALITY = {
    'max_missing_threshold': 0.3,  # Maximum missing values per column
    'min_variance_threshold': 0.01,  # Minimum variance for feature selection
    'correlation_threshold': 0.95,  # Maximum correlation between features
}

# Deployment Settings
DEPLOYMENT = {
    'model_format': 'pickle',  # Format for saving models
    'api_timeout': 30,  # API timeout in seconds
    'batch_size': 1000,  # Batch size for prediction
    'monitoring_threshold': 0.05,  # Threshold for model drift detection
}