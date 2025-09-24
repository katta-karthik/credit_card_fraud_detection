"""
Data utilities for loading, preprocessing, and creating sample data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath=None, use_sample=False):
    """
    Load fraud detection dataset.
    
    Parameters:
    filepath: Path to the actual dataset
    use_sample: If True, create sample data when actual data is not available
    
    Returns:
    DataFrame containing the fraud detection data
    """
    if filepath and os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully from {filepath}")
            print(f"Dataset shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            if use_sample:
                print("Creating sample data instead...")
                return create_sample_data()
    else:
        if use_sample:
            print("Creating sample fraud detection dataset...")
            return create_sample_data()
        else:
            print("No data file provided and use_sample is False")
            return None

def create_sample_data(n_samples=10000, fraud_rate=0.001):
    """
    Create a sample fraud detection dataset for demonstration.
    
    Parameters:
    n_samples: Number of samples to generate
    fraud_rate: Proportion of fraudulent transactions
    
    Returns:
    DataFrame with sample fraud detection data
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    # Create normal transactions
    normal_data = {
        'step': np.random.randint(1, 743, n_normal),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_normal, 
                                p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'amount': np.random.lognormal(4, 2, n_normal),
        'nameOrig': [f'C{i}' for i in range(n_normal)],
        'oldbalanceOrg': np.random.lognormal(6, 3, n_normal),
        'newbalanceOrig': np.random.lognormal(6, 3, n_normal),
        'nameDest': [f'M{i}' for i in range(n_normal)],
        'oldbalanceDest': np.random.lognormal(5, 2.5, n_normal),
        'newbalanceDest': np.random.lognormal(5, 2.5, n_normal),
        'isFraud': [0] * n_normal,
        'isFlaggedFraud': [0] * n_normal
    }
    
    # Create fraudulent transactions (different patterns)
    fraud_data = {
        'step': np.random.randint(1, 743, n_fraud),
        'type': np.random.choice(['TRANSFER', 'CASH_OUT'], n_fraud, p=[0.6, 0.4]),
        'amount': np.random.lognormal(8, 1.5, n_fraud),  # Higher amounts for fraud
        'nameOrig': [f'C{i}' for i in range(n_fraud)],
        'oldbalanceOrg': np.random.lognormal(8, 2, n_fraud),
        'newbalanceOrig': np.zeros(n_fraud),  # Often zero for fraud
        'nameDest': [f'C{i}' for i in range(n_fraud)],
        'oldbalanceDest': np.zeros(n_fraud),  # Often zero for fraud
        'newbalanceDest': np.random.lognormal(8, 2, n_fraud),
        'isFraud': [1] * n_fraud,
        'isFlaggedFraud': np.random.choice([0, 1], n_fraud, p=[0.95, 0.05])
    }
    
    # Combine data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = list(normal_data[key]) + list(fraud_data[key])
    
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Sample dataset created:")
    print(f"- Total samples: {len(df)}")
    print(f"- Fraud cases: {df['isFraud'].sum()}")
    print(f"- Fraud rate: {df['isFraud'].mean():.4f}")
    
    return df

def preprocess_data(df, target_column='isFraud'):
    """
    Preprocess the fraud detection dataset.
    
    Parameters:
    df: DataFrame to preprocess
    target_column: Name of the target column
    
    Returns:
    Tuple of (X, y, feature_names, preprocessing_info)
    """
    preprocessing_info = {}
    
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Basic info
    print("Data preprocessing started...")
    print(f"Original shape: {data.shape}")
    
    # Handle missing values
    missing_before = data.isnull().sum().sum()
    data = data.dropna()
    missing_after = data.isnull().sum().sum()
    preprocessing_info['missing_values_removed'] = missing_before - missing_after
    
    # Feature engineering
    if 'amount' in data.columns:
        data['log_amount'] = np.log1p(data['amount'])
        preprocessing_info['log_transformation'] = 'amount'
    
    if 'oldbalanceOrg' in data.columns and 'newbalanceOrig' in data.columns:
        data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
        data['balance_ratio_orig'] = data['newbalanceOrig'] / (data['oldbalanceOrg'] + 1)
    
    if 'oldbalanceDest' in data.columns and 'newbalanceDest' in data.columns:
        data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
        data['balance_ratio_dest'] = data['newbalanceDest'] / (data['oldbalanceDest'] + 1)
    
    # Create binary flags
    if 'newbalanceOrig' in data.columns:
        data['is_zero_balance_orig'] = (data['newbalanceOrig'] == 0).astype(int)
    
    if 'newbalanceDest' in data.columns:
        data['is_zero_balance_dest'] = (data['newbalanceDest'] == 0).astype(int)
    
    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != target_column]
    
    le_dict = {}
    for col in categorical_columns:
        if col in ['nameOrig', 'nameDest']:
            # For name columns, just create a simple encoding
            data[col + '_encoded'] = pd.Categorical(data[col]).codes
            data = data.drop(col, axis=1)
        elif col == 'type':
            # One-hot encode transaction type
            type_dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, type_dummies], axis=1)
            data = data.drop(col, axis=1)
            preprocessing_info['one_hot_encoded'] = col
    
    preprocessing_info['categorical_encodings'] = le_dict
    
    # Separate features and target
    y = data[target_column]
    X = data.drop([target_column], axis=1, errors='ignore')
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    print(f"Processed shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    preprocessing_info['final_shape'] = X.shape
    preprocessing_info['feature_names'] = feature_names
    preprocessing_info['class_distribution'] = y.value_counts().to_dict()
    
    return X, y, feature_names, preprocessing_info

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into train and test sets.
    
    Parameters:
    X: Features
    y: Target
    test_size: Proportion of data for testing
    random_state: Random seed
    stratify: Whether to maintain class distribution
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_param
    )
    
    print(f"Data split completed:")
    print(f"- Training set: {X_train.shape[0]} samples")
    print(f"- Test set: {X_test.shape[0]} samples")
    print(f"- Training fraud rate: {y_train.mean():.4f}")
    print(f"- Test fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test=None, method='standard'):
    """
    Scale features using specified method.
    
    Parameters:
    X_train: Training features
    X_test: Test features (optional)
    method: Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
    Scaled features and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler