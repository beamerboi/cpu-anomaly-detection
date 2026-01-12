"""
Model Selection Module

Compares trained models and selects the best one
based on evaluation metrics.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR,
    BEST_MODEL_PATH,
    ensure_directories,
)
from .train_models import save_model, load_model


def compare_models(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    primary_metric: str = "f1",
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Compare all trained models and generate comparison visualizations.
    
    Args:
        results: Dictionary mapping model names to (model, metrics) tuples.
        primary_metric: Primary metric for ranking (default: f1).
        output_dir: Directory to save visualizations.
    
    Returns:
        DataFrame with comparison results.
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, (model, metrics) in results.items():
        row = {
            "Model": name,
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "ROC_AUC": metrics.get("roc_auc", 0),
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by="F1", ascending=False)
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Generate visualizations
    try:
        _generate_comparison_plots(comparison_df, results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return comparison_df


def _generate_comparison_plots(
    comparison_df: pd.DataFrame,
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    output_dir: Path,
):
    """Generate comparison visualization plots."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Bar chart of all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_bars.png", dpi=150)
    plt.close()
    
    # 2. Heatmap of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = comparison_df.set_index("Model")[metrics]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1)
    ax.set_title("Model Performance Heatmap")
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_heatmap.png", dpi=150)
    plt.close()
    
    # 3. Confusion matrices for top 3 models
    top_models = comparison_df.head(3)["Model"].tolist()
    
    fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(15, 4))
    if len(top_models) == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, top_models):
        model, metrics = results[model_name]
        cm = metrics.get("confusion_matrix")
        
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["Normal", "Anomaly"],
                       yticklabels=["Normal", "Anomaly"])
            ax.set_title(f"{model_name}\n(F1: {metrics['f1']:.3f})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def select_best_model(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    primary_metric: str = "f1",
    min_recall: float = 0.7,
    min_precision: float = 0.5,
    save_path: Path = BEST_MODEL_PATH,
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Select the best model based on specified criteria.
    
    The selection prioritizes:
    1. Meeting minimum recall threshold (catching anomalies is critical)
    2. Meeting minimum precision threshold (avoiding too many false alarms)
    3. Highest primary metric score
    
    Args:
        results: Dictionary mapping model names to (model, metrics) tuples.
        primary_metric: Primary metric for ranking.
        min_recall: Minimum acceptable recall.
        min_precision: Minimum acceptable precision.
        save_path: Path to save the best model.
    
    Returns:
        Tuple of (model_name, model, metrics)
    """
    print("\n" + "=" * 60)
    print("Model Selection")
    print("=" * 60)
    print(f"\nSelection criteria:")
    print(f"  - Minimum recall: {min_recall:.2f}")
    print(f"  - Minimum precision: {min_precision:.2f}")
    print(f"  - Primary metric: {primary_metric}")
    
    # Filter models meeting minimum thresholds
    eligible_models = []
    for name, (model, metrics) in results.items():
        recall = metrics.get("recall", 0)
        precision = metrics.get("precision", 0)
        
        if recall >= min_recall and precision >= min_precision:
            eligible_models.append((name, model, metrics))
            print(f"\n  {name}: ELIGIBLE")
            print(f"    Recall: {recall:.4f}, Precision: {precision:.4f}")
        else:
            print(f"\n  {name}: NOT ELIGIBLE")
            print(f"    Recall: {recall:.4f} (min: {min_recall})")
            print(f"    Precision: {precision:.4f} (min: {min_precision})")
    
    if not eligible_models:
        print("\nNo models meet the minimum criteria. Selecting best overall...")
        # Fall back to best F1 score
        best_name, (best_model, best_metrics) = max(
            results.items(),
            key=lambda x: x[1][1].get(primary_metric, 0)
        )
    else:
        # Select best among eligible
        best_name, best_model, best_metrics = max(
            eligible_models,
            key=lambda x: x[2].get(primary_metric, 0)
        )
    
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {best_metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {best_metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {best_metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {best_metrics.get('f1', 0):.4f}")
    print(f"  ROC AUC:   {best_metrics.get('roc_auc', 0):.4f}")
    
    # Save the best model
    if save_path:
        ensure_directories()
        save_model(best_model, save_path, best_name)
        
        # Save model metadata
        metadata = {
            "model_name": best_name,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else None 
                       for k, v in best_metrics.items() if k != "confusion_matrix"},
            "selection_criteria": {
                "primary_metric": primary_metric,
                "min_recall": min_recall,
                "min_precision": min_precision,
            },
        }
        
        metadata_path = save_path.parent / "best_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved: {metadata_path}")
    
    return best_name, best_model, best_metrics


def generate_report(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    best_model_name: str,
    output_path: Path = None,
):
    """
    Generate a detailed model comparison report.
    
    Args:
        results: Dictionary of model results.
        best_model_name: Name of the selected best model.
        output_path: Path to save the report.
    """
    if output_path is None:
        output_path = MODELS_DIR / "training_report.txt"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ANOMALY DETECTION MODEL TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 70 + "\n\n")
        
        # Sort by F1 score
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1][1].get("f1", 0),
            reverse=True
        )
        
        f.write("{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}\n".format(
            "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"
        ))
        f.write("-" * 75 + "\n")
        
        for name, (model, metrics) in sorted_results:
            marker = " *" if name == best_model_name else ""
            f.write("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}{}\n".format(
                name,
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1", 0),
                metrics.get("roc_auc", 0),
                marker,
            ))
        
        f.write("\n* = Selected best model\n")
        
        f.write("\n\n2. DETAILED METRICS FOR BEST MODEL\n")
        f.write("-" * 70 + "\n\n")
        
        best_model, best_metrics = results[best_model_name]
        f.write(f"Model: {best_model_name}\n\n")
        
        f.write("Confusion Matrix:\n")
        cm = best_metrics.get("confusion_matrix")
        if cm is not None:
            f.write("                 Predicted\n")
            f.write("              Normal  Anomaly\n")
            f.write(f"Actual Normal   {cm[0][0]:5d}    {cm[0][1]:5d}\n")
            f.write(f"       Anomaly  {cm[1][0]:5d}    {cm[1][1]:5d}\n")
        
        f.write("\n\n3. MODEL SELECTION RATIONALE\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"The {best_model_name} model was selected as the best model based on:\n")
        f.write("- Highest F1 score among models meeting recall/precision thresholds\n")
        f.write("- F1 score balances precision and recall, important for anomaly detection\n")
        f.write("- Model successfully identifies anomalies while minimizing false alarms\n")
    
    print(f"Report saved to: {output_path}")


def main():
    """Run model comparison and selection."""
    from prepare_data import load_and_preprocess
    from train_models import train_all_models
    
    print("=" * 60)
    print("Model Selection Pipeline")
    print("=" * 60)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
        
        # Train all models
        results = train_all_models(X_train, y_train, X_test, y_test)
        
        # Compare models
        comparison_df = compare_models(results)
        
        # Select best model
        best_name, best_model, best_metrics = select_best_model(results)
        
        # Generate report
        generate_report(results, best_name)
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease prepare training data first:")
        print("  python training/prepare_data.py")


if __name__ == "__main__":
    main()
