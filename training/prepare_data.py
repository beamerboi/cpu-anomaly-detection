"""
Data Preparation Module

Handles loading, preprocessing, and combining datasets
for training anomaly detection models.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NORMAL_DATA_DIR,
    ANOMALY_DATA_DIR,
    TRAINING_SET_PATH,
    FEATURE_NAMES,
    SCALER_PATH,
    TEST_SIZE,
    RANDOM_SEED,
    ensure_directories,
)


def load_csv_files(directory: Path) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory.
    
    Args:
        directory: Path to directory containing CSV files.
    
    Returns:
        Combined DataFrame from all CSV files.
    """
    directory = Path(directory)
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        dataframes.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} samples")
    
    return pd.concat(dataframes, ignore_index=True)


def prepare_training_data(
    output_path: Path = TRAINING_SET_PATH,
    balance_classes: bool = True,
) -> pd.DataFrame:
    """
    Prepare the combined training dataset from normal and anomaly data.
    
    Args:
        output_path: Path to save the combined training set.
        balance_classes: If True, undersample the majority class.
    
    Returns:
        Combined and processed DataFrame.
    """
    ensure_directories()
    
    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)
    
    # Load normal data
    print("\nLoading normal data...")
    try:
        normal_df = load_csv_files(NORMAL_DATA_DIR)
        print(f"  Total normal samples: {len(normal_df)}")
    except FileNotFoundError:
        print("  No normal data found!")
        normal_df = pd.DataFrame()
    
    # Load anomaly data
    print("\nLoading anomaly data...")
    try:
        anomaly_df = load_csv_files(ANOMALY_DATA_DIR)
        print(f"  Total anomaly samples: {len(anomaly_df)}")
    except FileNotFoundError:
        print("  No anomaly data found!")
        anomaly_df = pd.DataFrame()
    
    if normal_df.empty and anomaly_df.empty:
        raise ValueError("No data found in either normal or anomaly directories!")
    
    # Combine datasets
    combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} samples")
    
    # Display class distribution
    if 'label' in combined_df.columns:
        class_counts = combined_df['label'].value_counts()
        print(f"\nClass distribution:")
        print(f"  Normal (0): {class_counts.get(0, 0)}")
        print(f"  Anomaly (1): {class_counts.get(1, 0)}")
        
        # Balance classes if requested
        if balance_classes and len(class_counts) > 1:
            min_class_count = class_counts.min()
            print(f"\nBalancing classes to {min_class_count} samples each...")
            
            balanced_dfs = []
            for label in class_counts.index:
                class_df = combined_df[combined_df['label'] == label]
                if len(class_df) > min_class_count:
                    class_df = class_df.sample(n=min_class_count, random_state=RANDOM_SEED)
                balanced_dfs.append(class_df)
            
            combined_df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"  Balanced dataset: {len(combined_df)} samples")
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Remove unnecessary columns
    columns_to_drop = ['source_file']
    combined_df = combined_df.drop(columns=[c for c in columns_to_drop if c in combined_df.columns])
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nTraining set saved to: {output_path}")
    print(f"  Samples: {len(combined_df)}")
    print(f"  Features: {len(combined_df.columns) - 2}")  # Exclude timestamp and label
    
    return combined_df


def load_and_preprocess(
    data_path: Path = TRAINING_SET_PATH,
    scale_features: bool = True,
    save_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Load training data and preprocess for model training.
    
    Args:
        data_path: Path to the training set CSV.
        scale_features: If True, apply StandardScaler to features.
        save_scaler: If True, save the fitted scaler to file.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    print("=" * 60)
    print("Loading and Preprocessing Data")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Extract features and labels
    # Use only the features defined in config
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    print(f"Using {len(available_features)} features")
    
    X = df[available_features].values
    y = df['label'].values
    
    # Handle missing values
    if np.isnan(X).any():
        print("Warning: Found NaN values, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Handle infinite values
    if np.isinf(X).any():
        print("Warning: Found infinite values, clipping to finite range")
        X = np.clip(X, -1e10, 1e10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y,
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = None
    if scale_features:
        print("\nScaling features with StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if save_scaler:
            ensure_directories()
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler saved to: {SCALER_PATH}")
    
    return X_train, X_test, y_train, y_test, scaler


def get_feature_names() -> List[str]:
    """Get the list of feature names used in the model."""
    # Load training set to get actual available features
    if TRAINING_SET_PATH.exists():
        df = pd.read_csv(TRAINING_SET_PATH, nrows=1)
        return [f for f in FEATURE_NAMES if f in df.columns]
    return FEATURE_NAMES


def main():
    """Prepare training data from collected samples."""
    try:
        # Prepare combined training set
        df = prepare_training_data()
        
        print()
        
        # Load and preprocess
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
        
        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease collect data first:")
        print("  1. Run: python collect_normal_data.py")
        print("  2. Run: python collect_anomaly_data.py")


if __name__ == "__main__":
    main()
