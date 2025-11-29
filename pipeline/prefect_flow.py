"""
Prefect Flow for IMDB Sentiment Analysis Pipeline
Orchestrates: Data Preparation → Training → Evaluation
"""

import os
import sys
from pathlib import Path
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import subprocess

# Set Prefect to use ephemeral mode (no server required)
os.environ["PREFECT_API_URL"] = "http://ephemeral-prefect/api"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@task(name="prepare-data", description="Clean and split IMDB dataset", retries=2, retry_delay_seconds=10)
def prepare_data_task():
    """Task 1: Data preparation with train-test split"""
    print("🔄 Starting data preparation...")
    
    # Import and run prepare script
    from pipeline.prepare_data import main as prepare_main
    prepare_main()
    
    print("✅ Data preparation complete")
    return {"status": "success", "stage": "prepare"}


@task(name="train-model", description="Train sentiment analysis model", retries=1, retry_delay_seconds=30)
def train_model_task():
    """Task 2: Model training with MLflow tracking"""
    print("🔄 Starting model training...")
    
    # Import and run train script
    from pipeline.train import main as train_main
    train_main()
    
    print("✅ Model training complete")
    return {"status": "success", "stage": "train"}


@task(name="evaluate-model", description="Evaluate model on test set", retries=1)
def evaluate_model_task():
    """Task 3: Model evaluation with metrics"""
    print("🔄 Starting model evaluation...")
    
    # Import and run evaluate script
    from pipeline.evaluate import main as evaluate_main
    evaluate_main()
    
    print("✅ Model evaluation complete")
    return {"status": "success", "stage": "evaluate"}


@flow(
    name="imdb-sentiment-pipeline",
    description="End-to-end MLOps pipeline for IMDB sentiment analysis",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def sentiment_analysis_pipeline():
    """
    Main Prefect flow orchestrating the entire ML pipeline.
    
    Pipeline stages:
    1. Prepare Data: Clean text, train-test split (80:20)
    2. Train Model: TF-IDF + Logistic Regression with MLflow tracking
    3. Evaluate Model: Metrics on held-out test set
    
    Returns:
        dict: Summary of pipeline execution
    """
    print("=" * 80)
    print("🚀 STARTING IMDB SENTIMENT ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Stage 1: Prepare Data
    prepare_result = prepare_data_task()
    
    # Stage 2: Train Model (depends on prepare)
    train_result = train_model_task()
    
    # Stage 3: Evaluate Model (depends on train)
    evaluate_result = evaluate_model_task()
    
    # Pipeline summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Stages executed: {prepare_result['stage']} → {train_result['stage']} → {evaluate_result['stage']}")
    print("\nCheck results:")
    print("  - Metrics: metrics.txt, metrics.json")
    print("  - Models: models/model.pkl, models/vectorizer.pkl")
    print("  - MLflow UI: http://localhost:5000 (run 'mlflow ui')")
    print("=" * 80)
    
    return {
        "status": "completed",
        "stages": [prepare_result, train_result, evaluate_result]
    }


# Deployment configuration
@flow(name="scheduled-retraining", log_prints=True)
def scheduled_retraining_flow():
    """
    Flow for scheduled model retraining.
    Can be deployed with: prefect deploy
    """
    print("🔄 Starting scheduled retraining...")
    result = sentiment_analysis_pipeline()
    print(f"✅ Retraining complete: {result['status']}")
    return result


if __name__ == "__main__":
    # Run the pipeline
    sentiment_analysis_pipeline()
