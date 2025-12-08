"""
Pipeline orchestration for IMDB sentiment analysis using Prefect.
Manages workflow for data preparation, training, and evaluation.
"""

import sys
from pathlib import Path
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

sys.path.append(str(Path(__file__).parent.parent))


@task(name="prepare-data", description="Clean and split IMDB dataset", retries=2, retry_delay_seconds=10)
def prepare_data_task():
    """Data preparation step: clean text and split train-test"""
    print("Starting data preparation...")
    
    from pipeline.prepare_data import main as prepare_main
    prepare_main()
    
    print("Data preparation complete")
    return {"status": "success", "stage": "prepare"}


@task(name="train-model", description="Train sentiment analysis model", retries=1, retry_delay_seconds=30)
def train_model_task():
    """Model training step with MLflow tracking"""
    print("Starting model training...")
    
    from pipeline.train import main as train_main
    train_main()
    
    print("Model training complete")
    return {"status": "success", "stage": "train"}


@task(name="evaluate-model", description="Evaluate model on test set", retries=1)
def evaluate_model_task():
    """Model evaluation on test set"""
    print("Starting model evaluation...")
    
    from pipeline.evaluate import main as evaluate_main
    evaluate_main()
    
    print("Model evaluation complete")
    return {"status": "success", "stage": "evaluate"}


@flow(
    name="imdb-sentiment-pipeline",
    description="End-to-end MLOps pipeline for IMDB sentiment analysis",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def sentiment_analysis_pipeline():
    """
    Main orchestration flow for ML pipeline.
    
    Pipeline consists of 3 stages:
    1. Data prep: clean text and split 80:20
    2. Training: TF-IDF + LogisticRegression, tracked with MLflow
    3. Evaluation: compute metrics on test set
    
    Returns:
        dict: pipeline execution summary
    """
    print("=" * 80)
    print("STARTING IMDB SENTIMENT ANALYSIS PIPELINE")
    print("=" * 80)
    
    # run data preparation first
    prepare_result = prepare_data_task()
    
    # then train the model
    train_result = train_model_task()
    
    # finally evaluate
    evaluate_result = evaluate_model_task()
    
    # print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Stages: {prepare_result['stage']} → {train_result['stage']} → {evaluate_result['stage']}")
    print("\nResults available at:")
    print("  - Metrics: metrics.txt, metrics.json")
    print("  - Models: models/model.pkl, models/vectorizer.pkl")
    print("  - MLflow UI: http://localhost:5000")
    print("=" * 80)
    
    return {
        "status": "completed",
        "stages": [prepare_result, train_result, evaluate_result]
    }


@flow(name="scheduled-retraining", log_prints=True)
def scheduled_retraining_flow():
    """
    Flow for scheduled retraining (can be deployed with Prefect)
    """
    print("Starting scheduled retraining...")
    result = sentiment_analysis_pipeline()
    print(f"Retraining complete: {result['status']}")
    return result


if __name__ == "__main__":
    sentiment_analysis_pipeline()
