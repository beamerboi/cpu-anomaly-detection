"""
Complete Training Pipeline Script

This script runs the entire training pipeline:
1. Prepare and combine data
2. Train all models
3. Compare and select best model
4. Generate reports and visualizations
"""

from training.prepare_data import prepare_training_data, load_and_preprocess
from training.train_models import train_all_models
from training.model_selection import compare_models, select_best_model, generate_report
from config import ensure_directories


def run_training_pipeline(
    skip_data_prep: bool = False,
    tune_hyperparams: bool = True,
    include_unsupervised: bool = True,
):
    """
    Run the complete training pipeline.
    
    Args:
        skip_data_prep: If True, skip data preparation (use existing training set).
        tune_hyperparams: If True, perform hyperparameter tuning.
        include_unsupervised: If True, include unsupervised models.
    """
    ensure_directories()
    
    print("=" * 70)
    print("ANOMALY DETECTION TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Prepare data
    if not skip_data_prep:
        print("\n[Step 1/4] Preparing training data...")
        print("-" * 70)
        prepare_training_data(balance_classes=True)
    else:
        print("\n[Step 1/4] Skipping data preparation (using existing data)")
    
    # Step 2: Load and preprocess
    print("\n[Step 2/4] Loading and preprocessing data...")
    print("-" * 70)
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    
    print(f"\nDataset summary:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Normal in train: {sum(y_train == 0)}")
    print(f"  Anomaly in train: {sum(y_train == 1)}")
    
    # Step 3: Train models
    print("\n[Step 3/4] Training models...")
    print("-" * 70)
    results = train_all_models(
        X_train, y_train, X_test, y_test,
        tune_hyperparams=tune_hyperparams,
        include_unsupervised=include_unsupervised,
    )
    
    # Step 4: Compare and select
    print("\n[Step 4/4] Comparing and selecting best model...")
    print("-" * 70)
    
    # Generate comparison
    compare_models(results)
    
    # Select best model
    best_name, best_model, best_metrics = select_best_model(results)
    
    # Generate report
    generate_report(results, best_name)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest Model: {best_name}")
    print(f"  F1 Score: {best_metrics['f1']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"\nModel saved to: models/best_model.pkl")
    print(f"Scaler saved to: models/scaler.pkl")
    print(f"\nReady for runtime detection!")
    print("  Run: python main.py detect")
    
    return best_name, best_model, best_metrics


if __name__ == "__main__":
    run_training_pipeline()
