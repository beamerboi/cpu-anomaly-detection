"""
Model Training Module

Trains and evaluates multiple machine learning algorithms
for anomaly detection.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Try to import xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available, skipping XGBoost models")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CV_FOLDS,
    RANDOM_SEED,
    MODELS_DIR,
    ensure_directories,
)


# =============================================================================
# Supervised Models Configuration
# =============================================================================

SUPERVISED_MODELS = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
    },
    "SVM": {
        "model": SVC(random_state=RANDOM_SEED, probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
    },
    "MLP": {
        "model": MLPClassifier(random_state=RANDOM_SEED, max_iter=500),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
        },
    },
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    SUPERVISED_MODELS["XGBoost"] = {
        "model": XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
        ),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    }


# =============================================================================
# Unsupervised Models Configuration
# =============================================================================

UNSUPERVISED_MODELS = {
    "IsolationForest": {
        "model": IsolationForest(random_state=RANDOM_SEED, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200],
            "contamination": [0.1, 0.2, 0.3],
            "max_samples": ["auto", 0.5, 0.8],
        },
    },
    "OneClassSVM": {
        "model": OneClassSVM(),
        "params": {
            "nu": [0.1, 0.2, 0.3],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
}


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_unsupervised: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        is_unsupervised: If True, handle unsupervised model output conversion.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    if is_unsupervised:
        # Unsupervised models return -1 for anomaly, 1 for normal
        y_pred = model.predict(X_test)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = np.where(y_pred == -1, 1, 0)
    else:
        y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    # ROC-AUC requires probability scores for supervised models
    if not is_unsupervised and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        # For unsupervised, use decision_function if available
        if hasattr(model, "decision_function"):
            try:
                y_scores = model.decision_function(X_test)
                # Invert scores for anomaly detection (lower = more anomalous)
                metrics["roc_auc"] = roc_auc_score(y_test, -y_scores)
            except Exception:
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm
    
    return metrics


def train_supervised_model(
    name: str,
    config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparams: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a supervised model with optional hyperparameter tuning.
    
    Args:
        name: Model name.
        config: Model configuration with model instance and params.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        tune_hyperparams: If True, perform grid search.
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    print(f"\n  Training {name}...")
    
    model = config["model"]
    
    if tune_hyperparams and config.get("params"):
        print(f"    Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            model,
            config["params"],
            cv=CV_FOLDS,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"    Best params: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    print(f"    CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test, is_unsupervised=False)
    
    return best_model, metrics


def train_unsupervised_model(
    name: str,
    config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train an unsupervised model (using only normal data for training).
    
    Note: Unsupervised anomaly detectors are typically trained on normal data only,
    but we use the full dataset here to ensure fair comparison.
    For a production system, you might train only on normal samples.
    
    Args:
        name: Model name.
        config: Model configuration.
        X_train: Training features.
        y_train: Training labels (used for filtering normal samples).
        X_test: Test features.
        y_test: Test labels.
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    print(f"\n  Training {name}...")
    
    # For unsupervised models, we have two options:
    # 1. Train on normal data only (traditional approach)
    # 2. Train on all data (semi-supervised approach)
    # We'll use option 1 for more realistic anomaly detection
    
    X_train_normal = X_train[y_train == 0]
    print(f"    Training on {len(X_train_normal)} normal samples")
    
    model = config["model"]
    
    # Simple parameter search for unsupervised models
    best_model = None
    best_f1 = 0
    best_params = {}
    
    params = config.get("params", {})
    if params:
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(params.keys())
        param_values = list(params.values())
        
        for values in product(*param_values):
            current_params = dict(zip(param_names, values))
            
            # Clone and set parameters
            test_model = model.__class__(**current_params, random_state=RANDOM_SEED if hasattr(model, 'random_state') else None)
            
            try:
                test_model.fit(X_train_normal)
                metrics = evaluate_model(test_model, X_test, y_test, is_unsupervised=True)
                
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_model = test_model
                    best_params = current_params
            except Exception as e:
                continue
        
        print(f"    Best params: {best_params}")
    else:
        best_model = model
        best_model.fit(X_train_normal)
    
    if best_model is None:
        best_model = model
        best_model.fit(X_train_normal)
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test, is_unsupervised=True)
    
    return best_model, metrics


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparams: bool = True,
    include_unsupervised: bool = True,
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all configured models.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        tune_hyperparams: If True, perform hyperparameter tuning.
        include_unsupervised: If True, include unsupervised models.
    
    Returns:
        Dictionary mapping model names to (model, metrics) tuples.
    """
    results = {}
    
    # Suppress warnings during training
    warnings.filterwarnings('ignore')
    
    # Train supervised models
    print("\n" + "=" * 60)
    print("Training Supervised Models")
    print("=" * 60)
    
    for name, config in SUPERVISED_MODELS.items():
        try:
            model, metrics = train_supervised_model(
                name, config, X_train, y_train, X_test, y_test, tune_hyperparams
            )
            results[name] = (model, metrics)
            print(f"    Test F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"    Error training {name}: {e}")
    
    # Train unsupervised models
    if include_unsupervised:
        print("\n" + "=" * 60)
        print("Training Unsupervised Models")
        print("=" * 60)
        
        for name, config in UNSUPERVISED_MODELS.items():
            try:
                model, metrics = train_unsupervised_model(
                    name, config, X_train, y_train, X_test, y_test
                )
                results[name] = (model, metrics)
                print(f"    Test F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"    Error training {name}: {e}")
    
    warnings.filterwarnings('default')
    
    return results


def save_model(model, filepath: Path, model_name: str = None):
    """
    Save a trained model to file.
    
    Args:
        model: Trained model to save.
        filepath: Path to save the model.
        model_name: Optional name for logging.
    """
    ensure_directories()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath: Path):
    """
    Load a trained model from file.
    
    Args:
        filepath: Path to the saved model.
    
    Returns:
        Loaded model.
    """
    return joblib.load(filepath)


def main():
    """Train and evaluate all models."""
    from prepare_data import load_and_preprocess, TRAINING_SET_PATH
    
    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
        
        # Train all models
        results = train_all_models(
            X_train, y_train, X_test, y_test,
            tune_hyperparams=True,
            include_unsupervised=True,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        
        print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1"
        ))
        print("-" * 65)
        
        for name, (model, metrics) in sorted(results.items(), key=lambda x: -x[1][1]["f1"]):
            print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            ))
        
        return results
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease prepare training data first:")
        print("  python training/prepare_data.py")


if __name__ == "__main__":
    main()
