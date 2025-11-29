"""
Script untuk membuat summary eksperimen dan memilih best model
"""
import mlflow
import json
from pathlib import Path

def main():
    print("="*60)
    print("MLOPS SENTIMENT ANALYSIS - EXPERIMENT SUMMARY")
    print("="*60)
    
    # Connect to MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name("imdb-sentiment-analysis")
    
    if not experiment:
        print("No experiments found!")
        return
    
    # Get all runs
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["start_time DESC"]
    )
    
    print(f"\n✓ Found {len(runs)} experiment runs\n")
    
    # Collect results
    results = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        
        result = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
            "model_type": params.get("model_type", "Unknown"),
            "max_features": params.get("max_features", "Unknown"),
            "test_accuracy": metrics.get("test_accuracy", 0),
            "test_f1": metrics.get("test_f1_score", 0),
            "train_accuracy": metrics.get("train_accuracy", 0),
        }
        results.append(result)
    
    # Sort by F1 score
    results.sort(key=lambda x: x["test_f1"], reverse=True)
    
    # Display table
    print(f"{'Rank':<6} {'Model Type':<20} {'Features':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*72)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<6} {result['model_type']:<20} {str(result['max_features']):<12} "
              f"{result['test_accuracy']:<12.4f} {result['test_f1']:<12.4f}")
    
    # Best model
    if results:
        best = results[0]
        print("\n" + "="*60)
        print("🏆 BEST MODEL")
        print("="*60)
        print(f"Model Type:       {best['model_type']}")
        print(f"Max Features:     {best['max_features']}")
        print(f"Test Accuracy:    {best['test_accuracy']:.4f} ({best['test_accuracy']*100:.2f}%)")
        print(f"Test F1-Score:    {best['test_f1']:.4f}")
        print(f"Train Accuracy:   {best['train_accuracy']:.4f}")
        print(f"Run ID:           {best['run_id']}")
        print(f"Run Name:         {best['run_name']}")
        print("="*60)
        
        # Save to file
        summary = {
            "best_model": best,
            "all_experiments": results,
            "total_runs": len(results)
        }
        
        with open("experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n✓ Summary saved to: experiment_summary.json")
        print(f"✓ View detailed results in MLflow UI: http://127.0.0.1:5000")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR DEPLOYMENT")
    print("="*60)
    print(f"1. Use best model: {best['model_type']} with {best['max_features']} features")
    print(f"2. Expected accuracy: ~{best['test_accuracy']*100:.1f}%")
    print(f"3. Model files: models/model.pkl, models/vectorizer.pkl")
    print(f"4. Next: Build FastAPI serving endpoint")
    print("="*60)

if __name__ == "__main__":
    main()
