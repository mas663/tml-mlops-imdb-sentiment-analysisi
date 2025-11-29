import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import mlflow

def main():
    print("=" * 60)
    print("MODEL EVALUATION - IMDB Sentiment Analysis")
    print("=" * 60)
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("imdb-sentiment-analysis")
    
    # Get the latest run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("imdb-sentiment-analysis")
    
    if experiment:
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            print(f"\n✓ Using MLflow Run ID: {run_id}")
    
    # Load test data
    print("\n[1/6] Loading test data...")
    test_path = "data/processed/test.csv"
    try:
        df = pd.read_csv(test_path)
        print(f"✓ Test data loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"✗ Error: {test_path} not found!")
        return
    
    X_test = df["review_clean"]
    y_test = df["sentiment"]
    
    # Load models
    print("\n[2/6] Loading trained models...")
    try:
        vectorizer = joblib.load("models/vectorizer.pkl")
        model = joblib.load("models/model.pkl")
        print("✓ Models loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Transform test data
    print("\n[3/6] Transforming test data...")
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Test data transformed. Shape: {X_test_vec.shape}")
    
    # Make predictions
    print("\n[4/6] Making predictions...")
    y_pred = model.predict(X_test_vec)
    print("✓ Predictions completed")
    
    # Calculate metrics
    print("\n[5/6] Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label='positive')
    recall = recall_score(y_test, y_pred, average='binary', pos_label='positive')
    f1 = f1_score(y_test, y_pred, average='binary', pos_label='positive')
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*60}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics to JSON
    print("\n[6/6] Saving metrics...")
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved to: metrics.json")
    
    # Also save simple text format
    with open("metrics.txt", "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall: {recall:.4f}\n")
        f.write(f"f1_score: {f1:.4f}\n")
    print("✓ Metrics saved to: metrics.txt")
    
    # Log to MLflow if run exists
    if experiment and runs:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            mlflow.log_artifact("metrics.json")
            mlflow.log_artifact("metrics.txt")
            print("✓ Metrics logged to MLflow")
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
