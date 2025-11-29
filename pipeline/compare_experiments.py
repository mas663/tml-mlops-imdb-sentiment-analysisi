"""
MLflow Experiment Comparison & Best Model Selection

This script compares all MLflow experiments and automatically selects
the best model based on F1-score.

Usage:
    python pipeline/compare_experiments.py
"""

import mlflow
import pandas as pd
from pathlib import Path
import json


def get_all_experiments():
    """Get all MLflow experiments with their metrics"""
    client = mlflow.tracking.MlflowClient()
    
    # Get all experiments
    experiments = client.search_experiments()
    
    if not experiments:
        print("❌ No experiments found!")
        return None, None
    
    # Use the first experiment (default experiment)
    experiment_id = experiments[0].experiment_id
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.test_f1_score DESC"]
    )
    
    if not runs:
        print("❌ No runs found!")
        return None, None
    
    return runs, client


def compare_experiments():
    """Compare all experiments and display results"""
    print("\n" + "="*80)
    print("📊 MLFLOW EXPERIMENT COMPARISON")
    print("="*80)
    
    runs, client = get_all_experiments()
    
    if runs is None or client is None:
        print("❌ No runs found!")
        return None, None
    
    # Collect data
    comparison_data = []
    for run in runs:
        data = {
            'run_id': run.info.run_id[:8],
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'model_type': run.data.params.get('model_type', 'N/A'),
            'max_features': run.data.params.get('max_features', 'N/A'),
            'max_iter': run.data.params.get('max_iter', 'N/A'),
            'solver': run.data.params.get('solver', 'N/A'),
            'train_accuracy': run.data.metrics.get('train_accuracy', 0.0),
            'test_accuracy': run.data.metrics.get('test_accuracy', 0.0),
            'test_precision': run.data.metrics.get('test_precision', 0.0),
            'test_recall': run.data.metrics.get('test_recall', 0.0),
            'test_f1_score': run.data.metrics.get('test_f1_score', 0.0),
        }
        comparison_data.append(data)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    print("\n📋 Experiment Comparison Table:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    
    # Find best model
    best_idx = df['test_f1_score'].idxmax()
    best_run = df.iloc[best_idx]
    
    print("\n" + "="*80)
    print("🏆 BEST MODEL SELECTED")
    print("="*80)
    print(f"Run ID:        {best_run['run_id']}")
    print(f"Run Name:      {best_run['run_name']}")
    print(f"Model Type:    {best_run['model_type']}")
    print(f"Max Features:  {best_run['max_features']}")
    print(f"Max Iter:      {best_run['max_iter']}")
    print(f"Solver:        {best_run['solver']}")
    print(f"\n📊 Performance Metrics:")
    print(f"  Train Accuracy: {best_run['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {best_run['test_accuracy']:.4f}")
    print(f"  Test Precision: {best_run['test_precision']:.4f}")
    print(f"  Test Recall:    {best_run['test_recall']:.4f}")
    print(f"  Test F1-Score:  {best_run['test_f1_score']:.4f} ⭐")
    print("="*80)
    
    # Save comparison to file
    output_path = Path("experiment_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Comparison saved to: {output_path}")
    
    # Save best model info
    best_model_info = {
        "best_run_id": best_run['run_id'],
        "best_run_name": best_run['run_name'],
        "model_type": best_run['model_type'],
        "hyperparameters": {
            "max_features": int(best_run['max_features']) if best_run['max_features'] != 'N/A' else None,
            "max_iter": int(best_run['max_iter']) if best_run['max_iter'] != 'N/A' else None,
            "solver": best_run['solver']
        },
        "metrics": {
            "train_accuracy": float(best_run['train_accuracy']),
            "test_accuracy": float(best_run['test_accuracy']),
            "test_precision": float(best_run['test_precision']),
            "test_recall": float(best_run['test_recall']),
            "test_f1_score": float(best_run['test_f1_score'])
        }
    }
    
    best_model_path = Path("best_model_info.json")
    with open(best_model_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    print(f"✅ Best model info saved to: {best_model_path}")
    
    # Generate visualization instructions
    print("\n" + "="*80)
    print("📈 VISUALIZATION OPTIONS")
    print("="*80)
    print("1. MLflow UI (Interactive):")
    print("   mlflow ui --port 5000")
    print("   → Open http://localhost:5000")
    print("   → Compare runs side-by-side")
    print("   → View metrics charts")
    print("\n2. CSV Export:")
    print(f"   → {output_path}")
    print("   → Import to Excel/Google Sheets for custom charts")
    print("\n3. Best Model Info:")
    print(f"   → {best_model_path}")
    print("   → Use for deployment/reporting")
    print("="*80)
    
    return df, best_run


if __name__ == "__main__":
    try:
        df, best_model = compare_experiments()
        print("\n✅ Experiment comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
